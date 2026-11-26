"""Final experiments: CV with train-tuned baselines, paired per-question results, significance,
and (optionally) calibration logs. Writes results/final/<config>/<tag>/.

    python -m eval.final --config GSM8K_Llama-3-8B-Instruct --lam 2 --judge-ratio 2 --sweep --calibration

Protocol per fold f (5 folds over questions, same split as eval/run.py):
  1. Fit the DP belief model on the training questions (seed + f), exactly as eval/run.py.
  2. Tune each baseline's parameter on the TRAINING questions: simulate every grid value with
     `--tune-episodes` episodes per training question on RNG stream TUNE, pick the best mean reward
     under this run's costs.
  3. Evaluate everything on the TEST questions with `--episodes` episodes per question on RNG stream
     TEST. Episodes use common random numbers, so every policy sees the same sample order for a given
     (question, episode); only judge verdict draws differ between policies that call the judge.
"""
from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import betaln

from eval.common import (PROBE, TEST, TUNE, baseline_grid, fold_assignment, load_data, per_question,
                         run_env, trajectories, tune, episode_rng)
from eval.stats import bootstrap_mean_ci, holm, wilcoxon_p
from policies.dp import DPPolicy, fit_belief
from sim.env import Costs, Env

RULE_NAME = {"fixed_k": "fixed-k", "margin": "margin stop", "margin_abstain": "margin stop+abstain"}
DP_VARIANTS = {  # name -> (use_judge, allow_abstain)
    "DP (resample)": (False, False),
    "DP (+judge)": (True, False),
    "DP (resample, abstain)": (False, True),
    "DP (+judge, abstain)": (True, True),
}
HEADLINE_ORDER = ["first answer", "fixed-k (tuned)", "margin stop (tuned)", "margin stop+abstain (tuned)",
                  "fixed-k (oracle-tuned on test)", "margin stop (oracle-tuned on test)",
                  "margin stop+abstain (oracle-tuned on test)",
                  "DP (resample)", "DP (+judge)", "DP (resample, abstain)", "DP (+judge, abstain)"]
TUNED = ["fixed-k (tuned)", "margin stop (tuned)", "margin stop+abstain (tuned)"]
METRICS = ["reward", "accuracy", "wrong_rate", "abstain_rate", "compute", "resamples", "judges"]
SWEEP_CS = np.geomspace(0.001, 0.2, 9)


def vec_posterior(pol: DPPolicy, pc, jp, jf):
    """Vectorized DPPolicy.posterior."""
    pc = np.clip(np.asarray(pc, float), 1e-12, 1 - 1e-12)
    l1 = betaln(pol.a1 + jp, pol.b1 + jf) - betaln(pol.a1, pol.b1)
    l0 = betaln(pol.a0 + jp, pol.b0 + jf) - betaln(pol.a0, pol.b0)
    return 1.0 / (1.0 + (1 - pc) / pc * np.exp(l0 - l1))


def run_fold(job):
    (config, judge_file, f, folds, lam, jr, c_s, episodes, tune_eps, seed, H, sweep, calib) = job
    t0 = time.time()
    clusters, qids, judge, _ = load_data(config, judge_file)
    fold_of = fold_assignment(len(qids), folds, seed)
    train, test = np.where(fold_of != f)[0], np.where(fold_of == f)[0]
    model = fit_belief(clusters, train, H=H, seed=seed + f, judge=judge, qids=qids)
    base = Costs(c_s=c_s, c_j=jr * c_s, lam=lam, H=H)
    rows = []

    def tag(df, policy, param):
        df = df.copy()
        df["policy"], df["param"], df["fold"] = policy, str(param), f
        return df

    # 1) tune baselines on TRAIN questions only (stream TUNE)
    tr = trajectories(clusters, train, tune_eps, seed, TUNE, f, H)
    chosen = tune(baseline_grid(tr, base, jr))
    chosen["fold"] = f
    chosen["n_train"] = len(train)

    # 2) all baseline grid values on TEST questions (stream TEST)
    te = trajectories(clusters, test, episodes, seed, TEST, f, H)
    grid = baseline_grid(te, base, jr)
    for (rule, p), g in grid.groupby(["rule", "param"], sort=False):
        rows.append(tag(g.drop(columns=["rule", "param"]), f"grid:{rule}", p))
        if rule == "fixed_k" and p == "1":
            rows.append(tag(g.drop(columns=["rule", "param"]), "first answer", 1))
    for r in chosen.itertuples():
        g = grid[(grid.rule == r.rule) & (grid.param == r.param)].drop(columns=["rule", "param"])
        rows.append(tag(g, f"{RULE_NAME[r.rule]} (tuned)", r.param))

    # 3) DP variants at the run's costs (+ optional cost sweep)
    log = [] if calib else None
    for name, (uj, ab) in DP_VARIANTS.items():
        pol = DPPolicy(model, base, use_judge=uj, allow_abstain=ab)
        env = Env(clusters, qids, base, judge)
        do_log = calib and name == "DP (+judge, abstain)"
        belief = (lambda n, a, b, jp, jf, p=pol: (float(model.pc[n, a, b]), float(p.posterior(n, a, b, jp, jf))))
        O, R, J = run_env(env, pol, test, episodes, seed, TEST, f, belief=belief, log=log if do_log else None)
        rows.append(tag(per_question(O, R, J, test, base, jr), name, c_s))
    if sweep:
        for cs in SWEEP_CS:
            cc = Costs(c_s=cs, c_j=jr * cs, lam=lam, H=H)
            for name, (uj, ab) in DP_VARIANTS.items():
                env = Env(clusters, qids, cc, judge)
                O, R, J = run_env(env, DPPolicy(model, cc, uj, ab), test, episodes, seed, TEST, f)
                rows.append(tag(per_question(O, R, J, test, cc, jr), f"sweep:{name}", f"{cs:.6g}"))

    # 4) calibration: vote-only belief at every state of every test trajectory, and judge probes
    cal = None
    if calib:
        pol = DPPolicy(model, base, use_judge=True, allow_abstain=True)
        Qn, E, L = te["a"].shape
        n = np.broadcast_to(np.arange(1, L + 1), (Qn, E, L))
        pc = model.pc[n, te["a"], te["b"]]
        # judge probe: 3 verdicts on the current leader at every state (stream PROBE)
        jp = np.zeros((3, Qn, E, L))
        for i, qi in enumerate(test):
            q = int(qids[qi])
            ppass = np.vectorize(lambda c: judge.prob_pass(q, int(c)))(te["leader"][i]) if judge else \
                np.where(te["correct"][i], 0.8, 0.3)
            for e in range(E):
                u = episode_rng(seed, PROBE, f, qi, e).random((3, L))
                jp[:, i, e] = np.cumsum(u < ppass[e], axis=0)
        post = np.stack([vec_posterior(pol, pc, jp[k], (k + 1) - jp[k]) for k in range(3)])
        cal = {"pc": pc.ravel().astype(np.float32), "correct": te["correct"].ravel(),
               "n": n.ravel().astype(np.int8), "qi": np.repeat(test, E * L).astype(np.int16),
               "probe_post": post.reshape(3, -1).astype(np.float32),
               "probe_jp": jp.reshape(3, -1).astype(np.int8)}
        cal["dp_log"] = pd.DataFrame(log, columns=["fold", "qi", "ep", "n", "a", "b", "jp", "jf", "pc_vote",
                                                   "posterior", "action", "correct"])
    print(f"  fold {f}: {len(train)} train / {len(test)} test, {time.time() - t0:.1f}s", flush=True)
    return pd.concat(rows, ignore_index=True), chosen, cal


def add_oracle_rows(pq: pd.DataFrame) -> pd.DataFrame:
    """'Oracle-tuned on test': parameter chosen by pooled TEST reward (unfair reference)."""
    out = [pq]
    for rule, name in RULE_NAME.items():
        g = pq[pq.policy == f"grid:{rule}"]
        best = g.groupby("param", sort=False).reward.mean().idxmax()
        d = g[g.param == best].copy()
        d["policy"] = f"{name} (oracle-tuned on test)"
        out.append(d)
    return pd.concat(out, ignore_index=True)


def headline_table(pq: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for p in HEADLINE_ORDER:
        d = pq[pq.policy == p]
        r = {"policy": p, "param": "/".join(sorted(d.param.unique(), key=str)), "n_questions": len(d)}
        for m in METRICS:
            r[m] = d[m].mean()
            r[m + "_se"] = d[m].std(ddof=1) / np.sqrt(len(d))
        rows.append(r)
    return pd.DataFrame(rows)


def significance(pq: pd.DataFrame, n_boot=10_000, seed=0) -> pd.DataFrame:
    wide = pq[pq.policy.isin(HEADLINE_ORDER)].pivot_table(index="qi", columns="policy", values="reward")
    pairs = [(dp, b) for dp in DP_VARIANTS for b in TUNED]
    pairs += [("DP (+judge)", "DP (resample)"), ("DP (+judge, abstain)", "DP (resample, abstain)"),
              ("DP (resample, abstain)", "DP (resample)"), ("DP (+judge, abstain)", "DP (+judge)")]
    rows = []
    for a, b in pairs:
        d = (wide[a] - wide[b]).to_numpy()
        mean, lo, hi = bootstrap_mean_ci(d, n_boot, seed)
        rows.append({"policy": a, "vs": b, "mean_diff": mean, "median_diff": float(np.median(d)),
                     "ci_lo": lo, "ci_hi": hi,
                     "wilcoxon_p": wilcoxon_p(d), "n_questions": len(d),
                     "n_nonzero": int((d != 0).sum()), "frac_q_better": float((d > 0).mean()),
                     "frac_q_worse": float((d < 0).mean())})
    s = pd.DataFrame(rows)
    s["holm_p"] = holm(s.wilcoxon_p)
    s["ci_excludes_0"] = (s.ci_lo > 0) | (s.ci_hi < 0)
    return s


def fmt_pm(m, se, d=3):
    return f"{m:.{d}f} ± {se:.{d}f}"


def headline_md(h: pd.DataFrame, meta: dict) -> str:
    lines = [f"Costs: c_s={meta['c_s']}, c_j={meta['judge_ratio']}*c_s, lambda={meta['lam']}, H={meta['H']}; "
             f"judge file {meta['judge_file']}; {meta['episodes']} test episodes/question; "
             f"± = standard error over {int(h.n_questions.max())} questions.", "",
             "| policy | param | accuracy | wrong | abstain | compute | reward |", "|---|---|---|---|---|---|---|"]
    for r in h.itertuples():
        lines.append(f"| {r.policy} | {r.param} | {fmt_pm(r.accuracy, r.accuracy_se)} | "
                     f"{fmt_pm(r.wrong_rate, r.wrong_rate_se)} | {fmt_pm(r.abstain_rate, r.abstain_rate_se)} | "
                     f"{fmt_pm(r.compute, r.compute_se, 2)} | {fmt_pm(r.reward, r.reward_se)} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--judge-file", default=None, help="file in data/processed (default <CONFIG>_judge.csv)")
    ap.add_argument("--tag-suffix", default="", help="appended to the output folder name, e.g. _judge7b")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--lam", type=float, default=2.0)
    ap.add_argument("--judge-ratio", type=float, default=2.0)
    ap.add_argument("--c-s", type=float, default=0.01)
    ap.add_argument("--episodes", type=int, default=100, help="test episodes per question")
    ap.add_argument("--tune-episodes", type=int, default=100, help="tuning episodes per training question")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sweep", action="store_true", help="also run the DP c_s sweep (for frontier figures)")
    ap.add_argument("--calibration", action="store_true")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--out", default="results/final")
    args = ap.parse_args()

    t0 = time.time()
    _, qids, judge, judge_path = load_data(args.config, args.judge_file)
    if not judge_path.exists():
        raise SystemExit(f"judge file {judge_path} not found")
    out = Path(args.out) / args.config / f"lam{args.lam:g}_jr{args.judge_ratio:g}{args.tag_suffix}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"{args.config} lam={args.lam:g} jr={args.judge_ratio:g} judge={judge_path.name} -> {out}", flush=True)
    jobs = [(args.config, args.judge_file, f, args.folds, args.lam, args.judge_ratio, args.c_s, args.episodes,
             args.tune_episodes, args.seed, 20, args.sweep, args.calibration) for f in range(args.folds)]
    with Pool(min(args.workers, args.folds)) as pool:
        res = pool.map(run_fold, jobs)

    pq = add_oracle_rows(pd.concat([r[0] for r in res], ignore_index=True))
    pq["qid"] = qids[pq.qi.to_numpy()]
    pq.to_csv(out / "per_question.csv", index=False)
    chosen = pd.concat([r[1] for r in res], ignore_index=True)
    chosen["policy"] = chosen.rule.map(RULE_NAME) + " (tuned)"
    chosen[["fold", "policy", "param", "tune_reward", "n_train"]].to_csv(out / "tuned_params.csv", index=False)

    meta = dict(config=args.config, lam=args.lam, judge_ratio=args.judge_ratio, c_s=args.c_s, H=20,
                folds=args.folds, episodes=args.episodes, tune_episodes=args.tune_episodes, seed=args.seed,
                judge_file=judge_path.name, judge_false_pass=judge.false_pass, judge_false_fail=judge.false_fail,
                n_questions=int(len(qids)), runtime_s=None)
    h = headline_table(pq)
    h.to_csv(out / "headline.csv", index=False)
    s = significance(pq)
    s.to_csv(out / "significance.csv", index=False)

    if args.calibration:
        cal = [r[2] for r in res]
        np.savez_compressed(out / "calibration_states.npz",
                            **{k: np.concatenate([c[k] for c in cal], axis=-1)
                               for k in ("pc", "correct", "n", "qi", "probe_post", "probe_jp")})
        pd.concat([c["dp_log"] for c in cal]).to_csv(out / "calibration_dp_states.csv.gz", index=False)

    meta["runtime_s"] = round(time.time() - t0, 1)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    md = headline_md(h, meta)
    (out / "headline.md").write_text(md)
    print(md)
    print(chosen[["fold", "policy", "param", "tune_reward"]].to_string(index=False))
    print(s[["policy", "vs", "mean_diff", "ci_lo", "ci_hi", "wilcoxon_p", "holm_p"]].round(4).to_string(index=False))
    print(f"done in {meta['runtime_s']}s")


if __name__ == "__main__":
    main()
