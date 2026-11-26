"""K-fold cross-validated comparison of the DP policies against fixed-rule baselines.

    python -m eval.run --config SYNTH
    python -m eval.run --config GSM8K_Llama-3-8B-Instruct --lam 2 --judge-ratio 2

Writes results/<config>/: summary.csv, headline.md, and four figures.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from policies.baselines import fixed_k, margin, margin_abstain
from policies.dp import DPPolicy, fit_belief
from sim.env import (ABSTAIN, JUDGE, RESAMPLE, SUBMIT, Costs, Env, JudgeTable, evaluate, load_config,
                     summarize)

# Categorical slots in fixed order (validated reference palette), plus marker shapes so
# identity never relies on color alone.
STYLE = {
    "first answer":      ("#52514e", "*"),
    "fixed-k majority":  ("#2a78d6", "o"),
    "margin stop":       ("#eb6834", "s"),
    "DP (resample)":     ("#1baf7a", "^"),
    "DP (+judge)":       ("#6250d6", "D"),
    "margin stop (abstain)": ("#eda100", "v"),
}
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def run(config, folds=5, lam=2.0, judge_ratio=2.0, c_s=0.01, episodes=100, seed=0, H=20, judge_file=None):
    clusters, qids, judge = load_config(config)
    if judge_file is not None:          # e.g. <CONFIG>_judge7b.csv; relative to data/processed/
        path = Path(judge_file)
        path = path if path.exists() else Path(__file__).resolve().parent.parent / "data" / "processed" / judge_file
        judge = JudgeTable.from_csv(path)
    Q = len(qids)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(Q)
    fold_of = np.empty(Q, int)
    for f, chunk in enumerate(np.array_split(perm, folds)):
        fold_of[chunk] = f

    sweep_cs = np.geomspace(0.001, 0.2, 9)
    rows, models = [], []
    for f in range(folds):
        t0 = time.time()
        train, test = np.where(fold_of != f)[0], np.where(fold_of == f)[0]
        model = fit_belief(clusters, train, H=H, seed=seed + f, judge=judge, qids=qids)
        models.append(model)
        base = Costs(c_s=c_s, c_j=judge_ratio * c_s, lam=lam, H=H)

        def add(name, param, policy, costs):
            env = Env(clusters, qids, costs, judge, seed=seed * 1000 + f)
            s = summarize(evaluate(env, policy, test, episodes), judge_ratio)
            rows.append({"fold": f, "policy": name, "param": param, **s})

        add("first answer", 1, fixed_k(1), base)
        for k in (2, 3, 5, 7, 9, 13, 17, 21):
            add("fixed-k majority", k, fixed_k(k), base)
        for m in (1, 2, 3, 4, 5, 6, 8):
            add("margin stop", m, margin(m), base)
        for m in (2, 3, 4, 5, 6):
            add("margin stop (abstain)", m, margin_abstain(m, k_max=10), base)
        for cs in sweep_cs:
            cc = Costs(c_s=cs, c_j=judge_ratio * cs, lam=lam, H=H)
            add("DP (resample)", cs, DPPolicy(model, cc, use_judge=False, allow_abstain=False), cc)
            add("DP (+judge)", cs, DPPolicy(model, cc, use_judge=True, allow_abstain=False), cc)
            add("DP (+judge, abstain)", cs, DPPolicy(model, cc, use_judge=True, allow_abstain=True), cc)
            add("DP (resample, abstain)", cs, DPPolicy(model, cc, use_judge=False, allow_abstain=True), cc)
        # Headline operating point: everything scored under the same default costs
        for name, pol in [("HEADLINE DP (+judge, abstain)", DPPolicy(model, base, True, True)),
                          ("HEADLINE DP (+judge)", DPPolicy(model, base, True, False)),
                          ("HEADLINE DP (resample, abstain)", DPPolicy(model, base, False, True)),
                          ("HEADLINE DP (resample)", DPPolicy(model, base, False, False))]:
            add(name, c_s, pol, base)
        print(f"fold {f}: {len(train)} train / {len(test)} test questions, {time.time() - t0:.1f}s", flush=True)
    return pd.DataFrame(rows), models, dict(lam=lam, judge_ratio=judge_ratio, c_s=c_s, H=H)


def aggregate(df):
    g = df.groupby(["policy", "param"])
    m = g.mean(numeric_only=True).drop(columns="fold")
    se = g.sem(numeric_only=True).drop(columns="fold").add_suffix("_se")
    return m.join(se).reset_index()


def _style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color="#e6e5e1", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED)


def plot_frontier(agg, metric, ylabel, title, path, include_abstain=False):
    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor=SURFACE)
    for name, (col, mk) in STYLE.items():
        key = name
        if include_abstain and name.startswith("DP"):
            key = name[:-1] + ", abstain)"
        if not include_abstain and name == "margin stop (abstain)":
            continue
        d = agg[agg.policy == key].sort_values("compute")
        if d.empty:
            continue
        single = len(d) == 1
        ax.errorbar(d.compute, d[metric], yerr=d[metric + "_se"], color=col, marker=mk,
                    markersize=13 if single else 7, zorder=5 if single else 2,
                    linewidth=0 if single else 2, capsize=0, elinewidth=1, label=key,
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
    _style(ax, title, "average compute per question (resample-equivalents)", ylabel)
    ax.legend(frameon=False, labelcolor=INK, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_policy_map(model, costs, path, n=8):
    """What the resample-only DP does at each (leader votes, runner-up votes) after n attempts."""
    pol = DPPolicy(model, costs, use_judge=False, allow_abstain=True)
    acts = {SUBMIT: 0, RESAMPLE: 1, ABSTAIN: 2}
    cols = ["#1baf7a", "#86b6ef", "#e0dfda"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), facecolor=SURFACE)
    for ax, nn in zip(axes, (4, n)):
        grid = np.full((nn + 1, nn + 1), np.nan)
        for a in range(1, nn + 1):
            for b in range(0, min(a, nn - a) + 1):
                if model.visits[nn, a, b] == 0:      # unreachable / never seen in training
                    continue
                grid[b, a] = acts[pol.value(nn, a, b, 0, 0, 0)[1]]
                ax.text(a, b, f"{model.pc[nn, a, b]:.2f}", ha="center", va="center", fontsize=7, color=INK)
        from matplotlib.colors import ListedColormap
        ax.imshow(grid, origin="lower", cmap=ListedColormap(cols), vmin=0, vmax=2, aspect="auto")
        _style(ax, f"after {nn} attempts", "leader votes a", "runner-up votes b")
        ax.grid(False)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=c, label=l) for c, l in zip(cols, ["submit", "resample", "abstain"])],
               loc="upper right", frameon=False, ncol=3)
    fig.suptitle("DP decision by vote counts (cell text = P(leader correct))", x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def headline(agg, path, meta):
    base = agg[agg.policy.isin(["first answer", "fixed-k majority", "margin stop", "margin stop (abstain)"])].copy()
    base["label"] = base.policy + " (" + base.param.astype(str) + ")"
    hl = agg[agg.policy.str.startswith("HEADLINE")].copy()
    hl["label"] = hl.policy.str.replace("HEADLINE ", "")
    t = pd.concat([hl, base])[["label", "accuracy", "wrong_rate", "abstain_rate", "compute", "reward"]]
    lines = [f"Costs: c_s={meta['c_s']}, c_j={meta['judge_ratio']}*c_s, lambda={meta['lam']}, H={meta['H']}",
             "", "| policy | accuracy | wrong | abstain | compute | reward |", "|---|---|---|---|---|---|"]
    for r in t.sort_values("reward", ascending=False).itertuples():
        lines.append(f"| {r.label} | {r.accuracy:.3f} | {r.wrong_rate:.3f} | {r.abstain_rate:.3f} | "
                     f"{r.compute:.2f} | {r.reward:.3f} |")
    Path(path).write_text("\n".join(lines) + "\n")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="SYNTH")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--lam", type=float, default=2.0)
    ap.add_argument("--judge-ratio", type=float, default=2.0, help="c_j / c_s")
    ap.add_argument("--c-s", type=float, default=0.01)
    ap.add_argument("--episodes", type=int, default=100, help="episodes per test question")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--judge-file", default=None,
                    help="judge CSV in data/processed/ (default <CONFIG>_judge.csv), e.g. <CONFIG>_judge7b.csv")
    args = ap.parse_args()

    suffix = "" if args.judge_file in (None, f"{args.config}_judge.csv") else \
        "_" + Path(args.judge_file).stem.replace(args.config + "_", "")
    out = Path("results") / args.config / f"lam{args.lam:g}_jr{args.judge_ratio:g}{suffix}"
    out.mkdir(parents=True, exist_ok=True)
    df, models, meta = run(args.config, args.folds, args.lam, args.judge_ratio, args.c_s, args.episodes, args.seed,
                         judge_file=args.judge_file)
    df.to_csv(out / "raw.csv", index=False)
    agg = aggregate(df)
    agg.to_csv(out / "summary.csv", index=False)
    plot_frontier(agg, "accuracy", "accuracy", "Accuracy vs compute (no abstaining)", out / "accuracy_vs_compute.png")
    plot_frontier(agg, "wrong_rate", "wrong-answer rate", "Wrong answers vs compute (abstaining allowed)",
                  out / "wrong_vs_compute.png", include_abstain=True)
    plot_frontier(agg, "reward", "mean reward", "Reward vs compute", out / "reward_vs_compute.png", include_abstain=True)
    plot_policy_map(models[0], Costs(c_s=args.c_s, c_j=args.judge_ratio * args.c_s, lam=args.lam),
                    out / "dp_policy_map.png")
    print(headline(agg, out / "headline.md", meta))


if __name__ == "__main__":
    main()
