"""Task 5: why the 3B judge does not help.

    python -m eval.judge_analysis [--judge-suffix _judge]

Writes results/final/judge_analysis/:
  per_question.csv    pass@1, top-answer vote share (contestedness), tercile, judge pass rates
  by_tercile.csv      judge accuracy / false-pass / false-fail by contestedness tercile (+95% bootstrap CIs)
  informativeness.csv likelihood ratios of a judge verdict vs. one more vote, by vote state
  consistency.csv     how often the 8 verdicts on one answer all agree

This is a descriptive analysis of the data (no policy is fit here), so it uses all questions.
Informativeness is measured over the states of simulated 21-sample trajectories (100 per question,
RNG stream ANALYSIS) at attempts n = 2..10, which is where the DP makes its decisions.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from eval.common import ANALYSIS, DATA, load_data, trajectories

CONFIGS = ["GSM8K_Llama-3-8B-Instruct", "MATH_Llama-3-8B-Instruct"]
OUT = Path("results/final/judge_analysis")


def question_table(config, clusters, jdf):
    Q, N = clusters.shape
    rows = []
    for qi in range(Q):
        r = clusters[qi]
        v, c = np.unique(r[r >= 0], return_counts=True)
        rows.append({"config": config, "qid": qi, "pass_at_1": float((r == 0).mean()),
                     "top_share": float(c.max() / N) if len(c) else 0.0,
                     "top_is_correct": bool(len(c) and v[c.argmax()] == 0)})
    q = pd.DataFrame(rows)
    q["tercile"] = pd.qcut(q.top_share.rank(method="first"), 3, labels=["most contested", "middle", "least contested"])
    j = jdf.assign(rate=jdf.n_pass / jdf.n_total)
    q = q.merge(j[j.cluster == 0][["qid", "rate"]].rename(columns={"rate": "judge_pass_correct"}), on="qid", how="left")
    w = j[j.cluster != 0].groupby("qid").rate.mean().rename("judge_pass_wrong_mean")
    return q.merge(w, on="qid", how="left")


def _rates(j):
    c, w = j[j.is_correct], j[~j.is_correct]
    tp, fn = c.n_pass.sum(), (c.n_total - c.n_pass).sum()
    fp, tn = w.n_pass.sum(), (w.n_total - w.n_pass).sum()
    fpr = fp / (fp + tn) if fp + tn else np.nan
    fnr = fn / (tp + fn) if tp + fn else np.nan
    return {"accuracy": (tp + tn) / (tp + tn + fp + fn), "false_pass": fpr, "false_fail": fnr,
            "balanced_accuracy": 1 - (fpr + fnr) / 2, "lr_pass": (1 - fnr) / fpr if fpr else np.nan,
            "n_correct_answers": len(c), "n_wrong_answers": len(w)}


def by_tercile(q, jdf, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    j = jdf[jdf.n_total > 0].merge(q[["qid", "tercile", "top_share"]], on="qid")
    rows = []
    for t, g in [("all", j)] + list(j.groupby("tercile", observed=True)):
        r = {"tercile": t, "n_questions": g.qid.nunique(),
             "top_share_min": g.top_share.min(), "top_share_max": g.top_share.max(), **_rates(g)}
        qs = g.qid.unique()
        by_q = {k: v for k, v in g.groupby("qid")}
        boots = []
        for _ in range(n_boot):
            s = pd.concat([by_q[k] for k in rng.choice(qs, len(qs))])
            boots.append(_rates(s))
        b = pd.DataFrame(boots)
        for m in ("accuracy", "false_pass", "false_fail", "balanced_accuracy"):
            r[m + "_lo"], r[m + "_hi"] = np.nanquantile(b[m], [0.025, 0.975])
        rows.append(r)
    return pd.DataFrame(rows)


def informativeness(config, clusters, judge, episodes=100, seed=0, n_range=(2, 10), lam=2.0):
    Q = clusters.shape[0]
    tr = trajectories(clusters, np.arange(Q), episodes, seed, ANALYSIS, 0)
    A, B, C, LD, DR = tr["a"], tr["b"], tr["correct"], tr["leader"], tr["draw"]
    p = np.zeros(A.shape)
    for qi in range(Q):
        p[qi] = np.vectorize(lambda c: judge.prob_pass(qi, int(c)))(LD[qi])
    nxt = np.zeros(A.shape, bool)
    nxt[..., :-1] = (DR[..., 1:] == LD[..., :-1]) & (LD[..., :-1] != -1)
    n = np.broadcast_to(np.arange(1, A.shape[-1] + 1), A.shape)
    sel = (n >= n_range[0]) & (n <= n_range[1])
    cats = {"unanimous (b=0)": (B == 0),
            "clear lead (a-b>=2, b>=1)": (B >= 1) & (A - B >= 2),
            "split (a-b<=1, b>=1)": (B >= 1) & (A - B <= 1),
            "all states": np.ones(A.shape, bool)}
    thr = lam / (1 + lam)                           # submit iff P(correct) > lam / (1 + lam)
    rows = []
    for name, m in cats.items():
        m = m & sel
        c = C[m]
        pi = c.mean()
        pc_, pw_ = p[m][c].mean(), p[m][~c].mean()
        vc, vw = nxt[m][c].mean(), nxt[m][~c].mean()

        def mi(a1, a0):                              # mutual information I(correct; binary signal), bits
            h = lambda x: 0.0 if x in (0, 1) else -(x * np.log2(x) + (1 - x) * np.log2(1 - x))
            return h(pi * a1 + (1 - pi) * a0) - (pi * h(a1) + (1 - pi) * h(a0))

        post = lambda lr: pi * lr / (pi * lr + 1 - pi)
        rows.append({
            "config": config, "state": name, "n_states": int(m.sum()), "n_questions": int(m.any((1, 2)).sum()),
            "p_leader_correct": pi,
            "judge_p_pass_if_correct": pc_, "judge_p_pass_if_wrong": pw_,
            "judge_lr_pass": pc_ / pw_, "judge_lr_fail": (1 - pc_) / (1 - pw_),
            "vote_p_leader_if_correct": vc, "vote_p_leader_if_wrong": vw,
            "vote_lr_agree": vc / vw, "vote_lr_disagree": (1 - vc) / (1 - vw),
            "judge_mi_bits": mi(pc_, pw_), "vote_mi_bits": mi(vc, vw),
            "posterior_after_pass": post(pc_ / pw_), "posterior_after_fail": post((1 - pc_) / (1 - pw_)),
            "posterior_after_agree_vote": post(vc / vw), "posterior_after_disagree_vote": post((1 - vc) / (1 - vw)),
            "submit_threshold_lam2": thr,
        })
    return pd.DataFrame(rows)


def consistency(config, jdf):
    j = jdf[jdf.n_total > 0]
    unanimous = (j.n_pass == 0) | (j.n_pass == j.n_total)
    return pd.DataFrame([{"config": config, "answers": len(j), "frac_all_8_agree": unanimous.mean(),
                          "frac_correct_all_fail": ((j.n_pass == 0) & j.is_correct).sum() / j.is_correct.sum(),
                          "frac_wrong_all_pass": ((j.n_pass == j.n_total) & ~j.is_correct).sum() / (~j.is_correct).sum()}])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-suffix", default="_judge", help="_judge (3B) or _judge7b")
    ap.add_argument("--episodes", type=int, default=100)
    args = ap.parse_args()
    out = OUT if args.judge_suffix == "_judge" else OUT.parent / f"judge_analysis{args.judge_suffix}"
    out.mkdir(parents=True, exist_ok=True)
    Qs, T, I, K = [], [], [], []
    for cfg in CONFIGS:
        jf = f"{cfg}{args.judge_suffix}.csv"
        if not (DATA / jf).exists():
            print(f"skip {cfg}: {jf} missing")
            continue
        clusters, qids, judge, _ = load_data(cfg, jf)
        jdf = pd.read_csv(DATA / jf)
        q = question_table(cfg, clusters, jdf)
        t = by_tercile(q, jdf)
        t.insert(0, "config", cfg)
        Qs.append(q), T.append(t), K.append(consistency(cfg, jdf))
        I.append(informativeness(cfg, clusters, judge, args.episodes))
        print(cfg, "\n", t[["tercile", "n_questions", "top_share_min", "top_share_max", "accuracy", "false_pass",
                            "false_fail", "balanced_accuracy"]].round(3).to_string(index=False))
        print(I[-1][["state", "n_states", "p_leader_correct", "judge_lr_pass", "judge_lr_fail", "vote_lr_agree",
                     "vote_lr_disagree", "judge_mi_bits", "vote_mi_bits"]].round(3).to_string(index=False))
    pd.concat(Qs).to_csv(out / "per_question.csv", index=False)
    pd.concat(T).to_csv(out / "by_tercile.csv", index=False)
    pd.concat(I).to_csv(out / "informativeness.csv", index=False)
    pd.concat(K).to_csv(out / "consistency.csv", index=False)
    print(pd.concat(K).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
