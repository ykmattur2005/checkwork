"""Tables (Markdown + LaTeX), calibration metrics and all final figures from results/final/.

    python -m eval.report

Reads results/final/<config>/<run>/ (written by eval/final.py) and results/final/judge_analysis/.
Writes results/final/tables/*.{md,tex}, results/final/calibration/*.csv, results/final/figures/*.{pdf,png}.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from eval.common import fold_assignment, load_data
from eval.figstyle import (INK, MUTED, POLICY_STYLE, SPINE, legend_right, policy_map, policy_map_figure, save, style_ax,
                           style_of)
from eval.final import HEADLINE_ORDER
from eval.stats import brier, calibration_bins, ece
from policies.dp import DPPolicy, fit_belief
from sim.env import Costs

ROOT = Path("results/final")
TAB = ROOT / "tables"
CAL = ROOT / "calibration"
CONFIGS = {"GSM8K_Llama-3-8B-Instruct": "gsm8k", "MATH_Llama-3-8B-Instruct": "math"}
PRETTY = {"gsm8k": "GSM8K", "math": "MATH"}
LAMS = [1.2, 2.0, 5.0]
JRS = [0.5, 1.0, 2.0]


# ----------------------------------------------------------------------------- helpers
def run_dir(cfg, lam, jr, suffix=""):
    return ROOT / cfg / f"lam{lam:g}_jr{jr:g}{suffix}"


def tex_escape(s: str) -> str:
    s = str(s)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"),
                 ("±", r"$\pm$"), ("λ", r"$\lambda$"), ("≥", r"$\geq$"), ("≤", r"$\leq$"), ("−", "--"),
                 ("<=", r"$\leq$"), (">=", r"$\geq$")]:
        s = s.replace(a, b)
    return s


def write_table(df: pd.DataFrame, name: str, caption: str, source: str, align=None):
    """Markdown (with source line) + booktabs LaTeX."""
    TAB.mkdir(parents=True, exist_ok=True)
    cols = list(df.columns)
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    (TAB / f"{name}.md").write_text(f"{caption}\n\nSource: `{source}`\n\n" + "\n".join(md) + "\n")
    align = align or ("l" + "r" * (len(cols) - 1))
    tex = [r"\begin{table}[t]", r"\centering", r"\small", rf"\caption{{{tex_escape(caption)}}}",
           rf"\label{{tab:{name}}}", rf"\begin{{tabular}}{{{align}}}", r"\toprule",
           " & ".join(tex_escape(c) for c in cols) + r" \\", r"\midrule"]
    tex += [" & ".join(tex_escape(v) for v in r) + r" \\" for r in df.itertuples(index=False)]
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB / f"{name}.tex").write_text("\n".join(tex) + "\n")
    return "\n".join(md)


def pm(m, s, d=3):
    return f"{m:.{d}f} ± {s:.{d}f}"


def agg(pq: pd.DataFrame) -> pd.DataFrame:
    g = pq.groupby(["policy", "param"], sort=False)
    m = g[["reward", "accuracy", "wrong_rate", "abstain_rate", "compute"]].mean()
    s = g[["reward", "accuracy", "wrong_rate", "abstain_rate", "compute"]].sem().add_suffix("_se")
    return m.join(s).reset_index()


# ----------------------------------------------------------------------------- tables
def headline_tables():
    out = {}
    for cfg, short in CONFIGS.items():
        for lam in LAMS:
            for jr in JRS:
                d = run_dir(cfg, lam, jr)
                if not (d / "headline.csv").exists():
                    continue
                h = pd.read_csv(d / "headline.csv")
                t = pd.DataFrame({
                    "policy": h.policy, "param": h.param,
                    "accuracy": [pm(a, b) for a, b in zip(h.accuracy, h.accuracy_se)],
                    "wrong": [pm(a, b) for a, b in zip(h.wrong_rate, h.wrong_rate_se)],
                    "abstain": [pm(a, b) for a, b in zip(h.abstain_rate, h.abstain_rate_se)],
                    "compute": [pm(a, b, 2) for a, b in zip(h.compute, h.compute_se)],
                    "reward": [pm(a, b) for a, b in zip(h.reward, h.reward_se)]})
                name = f"headline_{short}_lam{lam:g}_jr{jr:g}"
                cap = (f"{PRETTY[short]}, λ={lam:g}, c_j/c_s={jr:g}, c_s=0.01: test-set means over questions "
                       f"± standard error (n={int(h.n_questions.max())} questions, 100 episodes each). "
                       "param = chosen parameter per fold (tuned) or c_s (DP); fixed-k param is k, "
                       "margin params are m or (m, k_max). Compute in resample-equivalents.")
                out[name] = write_table(t, name, cap, str(d / "headline.csv"), align="llrrrrr")
    return out


def tuned_param_tables():
    for cfg, short in CONFIGS.items():
        rows = []
        for lam in LAMS:
            t = pd.read_csv(run_dir(cfg, lam, 2) / "tuned_params.csv")
            for f, g in t.groupby("fold"):
                r = {"λ": f"{lam:g}", "fold": f}
                for p in g.itertuples():
                    r[p.policy.replace(" (tuned)", "")] = p.param
                rows.append(r)
        write_table(pd.DataFrame(rows), f"tuned_params_{short}",
                    f"{PRETTY[short]}: baseline parameters chosen on each fold's training questions "
                    "(c_j/c_s=2; baselines do not use the judge, so the choice is the same for every c_j).",
                    f"{ROOT}/{cfg}/lam*_jr2/tuned_params.csv")


def verdict(r):
    ci = r.ci_lo > 0 or r.ci_hi < 0
    w = r.holm_p < 0.05
    return {(True, True): "yes", (True, False): "mean only", (False, True): "median only",
            (False, False): "no"}[(ci, w)]


def sig_table(s: pd.DataFrame):
    return pd.DataFrame({
        "policy": s.policy, "vs": s.vs,
        "mean diff [95% CI]": [f"{m:+.3f} [{a:+.3f}, {b:+.3f}]" for m, a, b in zip(s.mean_diff, s.ci_lo, s.ci_hi)],
        "median diff": [f"{m:+.3f}" for m in s.median_diff],
        "Wilcoxon p": [f"{p:.1e}" if p < 1e-3 else f"{p:.3f}" for p in s.wilcoxon_p],
        "Holm p": [f"{p:.1e}" if p < 1e-3 else f"{p:.3f}" for p in s.holm_p],
        "q better/worse": [f"{a:.0%}/{b:.0%}" for a, b in zip(s.frac_q_better, s.frac_q_worse)],
        "significant": [verdict(r) for r in s.itertuples()]})


def significance_tables():
    for cfg, short in CONFIGS.items():
        for lam in LAMS:
            d = run_dir(cfg, lam, 2)
            s = pd.read_csv(d / "significance.csv")
            write_table(sig_table(s), f"significance_{short}_lam{lam:g}",
                        f"{PRETTY[short]}, λ={lam:g}, c_j/c_s=2: paired per-question reward differences "
                        "(policy minus vs). 95% percentile bootstrap CI over questions (10,000 resamples); "
                        "two-sided Wilcoxon signed-rank p, Holm-adjusted within the table. 'significant': "
                        "yes = CI excludes 0 and Holm p<0.05; 'mean only' / 'median only' = only one test rejects.",
                        str(d / "significance.csv"), align="llrrrrrl")
        rows = []
        for jr in JRS:
            s = pd.read_csv(run_dir(cfg, 2, jr) / "significance.csv")
            s = s[s.policy.str.contains("judge") & s.vs.str.startswith("DP (resample")]
            t = sig_table(s)
            t.insert(0, "c_j/c_s", f"{jr:g}")
            rows.append(t)
        write_table(pd.concat(rows), f"judge_value_{short}",
                    f"{PRETTY[short]}, λ=2: value of adding the judge to the DP at three judge costs "
                    "(same tests as the significance tables; Holm adjustment within each c_j run).",
                    f"{ROOT}/{cfg}/lam2_jr*/significance.csv", align="lllrrrrrl")


def lambda_table():
    pols = ["first answer", "fixed-k (tuned)", "margin stop (tuned)", "margin stop+abstain (tuned)",
            "DP (resample)", "DP (+judge)", "DP (resample, abstain)", "DP (+judge, abstain)"]
    for cfg, short in CONFIGS.items():
        rows = []
        for p in pols:
            r = {"policy": p}
            for lam in LAMS:
                h = pd.read_csv(run_dir(cfg, lam, 2) / "headline.csv").set_index("policy")
                r[f"reward λ={lam:g}"] = pm(h.loc[p, "reward"], h.loc[p, "reward_se"])
                r[f"abstain λ={lam:g}"] = f"{h.loc[p, 'abstain_rate']:.2f}"
            rows.append(r)
        write_table(pd.DataFrame(rows), f"reward_vs_lambda_{short}",
                    f"{PRETTY[short]}: test reward (± SE over questions) and abstain rate by penalty λ "
                    "(c_s=0.01, c_j/c_s=2).", f"{ROOT}/{cfg}/lam*_jr2/headline.csv")


# ----------------------------------------------------------------------------- calibration
def calibration():
    CAL.mkdir(parents=True, exist_ok=True)
    metrics, bins = [], []
    for cfg, short in CONFIGS.items():
        d = run_dir(cfg, 2, 2)
        z = np.load(d / "calibration_states.npz")
        log = pd.read_csv(d / "calibration_dp_states.csv.gz")
        y = z["correct"].astype(float)
        judged = log[(log.jp + log.jf) > 0]
        sets = [
            ("vote-only pc[n,a,b], all trajectory states n=1..21", z["pc"], y),
            ("vote-only pc[n,a,b], trajectory states n<=10", z["pc"][z["n"] <= 10], y[z["n"] <= 10]),
            ("vote-only pc[n,a,b], DP-visited states", log.pc_vote, log.correct),
            ("posterior (votes+judge), DP-visited states", log.posterior, log.correct),
            ("vote-only pc[n,a,b], DP-visited states with >=1 verdict", judged.pc_vote, judged.correct),
            ("posterior (votes+judge), DP-visited states with >=1 verdict", judged.posterior, judged.correct),
        ] + [(f"posterior after {k + 1} probe verdict{'s' if k else ''}, all trajectory states",
              z["probe_post"][k], y) for k in range(3)]
        for name, p, yy in sets:
            p, yy = np.asarray(p, float), np.asarray(yy, float)
            if len(p) == 0:
                continue
            base = yy.mean()
            metrics.append({"config": short, "belief": name, "n_states": len(p), "frac_correct": base,
                            "ECE": ece(p, yy), "Brier": brier(p, yy), "Brier_const": brier(np.full(len(p), base), yy)})
            b = calibration_bins(p, yy)
            b.insert(0, "belief", name)
            b.insert(0, "config", short)
            bins.append(b)
    m, b = pd.DataFrame(metrics), pd.concat(bins)
    m.to_csv(CAL / "metrics.csv", index=False)
    b.to_csv(CAL / "bins.csv", index=False)
    t = m.assign(**{"n_states": m.n_states.map(lambda x: f"{x:,}"),
                    "frac_correct": m.frac_correct.map(lambda x: f"{x:.3f}"),
                    "ECE": m.ECE.map(lambda x: f"{x:.3f}"), "Brier": m.Brier.map(lambda x: f"{x:.3f}"),
                    "Brier_const": m.Brier_const.map(lambda x: f"{x:.3f}")})
    t = t.rename(columns={"Brier_const": "Brier (base rate)", "frac_correct": "P(correct)"})
    write_table(t, "calibration", "Calibration of the belief model on held-out folds (λ=2, c_j/c_s=2). "
                "Trajectory states: every attempt n of 100 test episodes per question; DP-visited: decision states "
                "of DP (+judge, abstain); probe: k simulated judge verdicts on the leader at every trajectory state. "
                "ECE uses 10 equal-width bins; Brier (base rate) = Brier of always predicting the mean.",
                str(CAL / "metrics.csv"), align="llrrrrr")
    return m, b


# ----------------------------------------------------------------------------- figures
def frontier_series(cfg, include_abstain):
    pq = pd.read_csv(run_dir(cfg, 2, 2) / "per_question.csv")
    a = agg(pq)
    series = [("fixed-k", a[a.policy == "grid:fixed_k"]), ("margin stop", a[a.policy == "grid:margin"]),
              ("DP (resample)", a[a.policy == "sweep:DP (resample)"]), ("DP (+judge)", a[a.policy == "sweep:DP (+judge)"])]
    if include_abstain:
        series += [("margin stop+abstain", a[a.policy == "grid:margin_abstain"]),
                   ("DP (resample, abstain)", a[a.policy == "sweep:DP (resample, abstain)"]),
                   ("DP (+judge, abstain)", a[a.policy == "sweep:DP (+judge, abstain)"])]
    first = a[a.policy == "first answer"]
    return series, first, a


def plot_frontier(cfg, short, metric, ylabel, title, name, include_abstain):
    series, first, a = frontier_series(cfg, include_abstain)
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    for label, d in series:
        col, mk = POLICY_STYLE[label]
        d = d.sort_values("compute")
        y, ye = d[metric], d[metric + "_se"]
        scatter = label == "margin stop+abstain"
        ax.errorbar(d.compute, y, yerr=ye, color=col, marker=mk, linestyle="none" if scatter else "-",
                    markersize=5.5 if scatter else 6, alpha=0.85 if scatter else 1, elinewidth=0.8, capsize=0,
                    markeredgecolor="white", markeredgewidth=0.8, label=label + (" (all m, k_max)" if scatter else ""),
                    zorder=2 if scatter else 3)
    col, mk = POLICY_STYLE["first answer"]
    ax.errorbar(first.compute, first[metric], yerr=first[metric + "_se"], color=col,
                marker=mk, markersize=11, linestyle="none", markeredgecolor="white", markeredgewidth=0.8,
                label="first answer", zorder=4)
    style_ax(ax, title, "compute per question (resample-equivalents, c_j = 2 c_s)", ylabel)
    ax.set_xlim(left=-0.4)
    legend_right(ax)
    save(fig, f"{name}_{short}")


def frontier_figures():
    for cfg, short in CONFIGS.items():
        P = PRETTY[short]
        plot_frontier(cfg, short, "accuracy", "accuracy", f"{P}: accuracy vs compute (no abstaining)",
                      "accuracy_vs_compute", include_abstain=False)
        plot_frontier(cfg, short, "wrong_rate", "wrong-answer rate",
                      f"{P}: wrong-answer rate vs compute", "wrong_vs_compute", include_abstain=True)
        # reward vs compute: accuracy - λ·wrong (λ=2) BEFORE compute cost, so points from different c_s
        # in the DP sweep are comparable; its SE comes from per-question values.
        pq = pd.read_csv(run_dir(cfg, 2, 2) / "per_question.csv")
        pq["gross"] = pq.accuracy - 2.0 * pq.wrong_rate
        g = pq.groupby(["policy", "param"], sort=False).gross.agg(["mean", "sem"]).reset_index()
        series, first, a = frontier_series(cfg, True)
        a = a.merge(g.rename(columns={"mean": "gross", "sem": "gross_se"}), on=["policy", "param"])
        fig, ax = plt.subplots(figsize=(6.6, 3.6))
        for label, pol in [("fixed-k", "grid:fixed_k"), ("margin stop", "grid:margin"),
                           ("DP (resample)", "sweep:DP (resample)"), ("DP (+judge)", "sweep:DP (+judge)"),
                           ("margin stop+abstain", "grid:margin_abstain"),
                           ("DP (resample, abstain)", "sweep:DP (resample, abstain)"),
                           ("DP (+judge, abstain)", "sweep:DP (+judge, abstain)"), ("first answer", "first answer")]:
            d = a[a.policy == pol].sort_values("compute")
            col, mk = POLICY_STYLE[label]
            scatter = label in ("margin stop+abstain", "first answer")
            ax.errorbar(d.compute, d.gross, yerr=d.gross_se, color=col, marker=mk,
                        linestyle="none" if scatter else "-", markersize=11 if label == "first answer" else 6,
                        alpha=0.85 if label == "margin stop+abstain" else 1, elinewidth=0.8, capsize=0,
                        markeredgecolor="white", markeredgewidth=0.8,
                        label=label + (" (all m, k_max)" if label == "margin stop+abstain" else ""))
        style_ax(ax, f"{P}: reward vs compute (abstaining allowed)",
                 "compute per question (resample-equivalents, c_j = 2 c_s)",
                 "accuracy − 2·wrong (before compute cost)")
        ax.axhline(0, color=SPINE, linewidth=0.8, zorder=1)
        ax.set_xlim(left=-0.4)
        legend_right(ax)
        save(fig, f"reward_vs_compute_{short}")


def lambda_figure():
    """Two panels with their own y-scale (abstaining policies live near 0, the others far below on MATH)."""
    groups = [("policies that must answer", ["first answer", "fixed-k (tuned)", "margin stop (tuned)",
                                             "DP (resample)", "DP (+judge)"]),
              ("policies that may abstain", ["margin stop+abstain (tuned)", "DP (resample, abstain)",
                                             "DP (+judge, abstain)"])]
    for cfg, short in CONFIGS.items():
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        handles = []
        for ax, (title, pols) in zip(axes, groups):
            offs = np.linspace(-0.03, 0.03, len(pols))
            for p, o in zip(pols, offs):
                col, mk = style_of(p)
                ys, es = [], []
                for lam in LAMS:
                    h = pd.read_csv(run_dir(cfg, lam, 2) / "headline.csv").set_index("policy")
                    ys.append(h.loc[p, "reward"]), es.append(h.loc[p, "reward_se"])
                handles.append(ax.errorbar(np.log(LAMS) + o, ys, yerr=es, color=col, marker=mk, capsize=2,
                                           elinewidth=0.9, markeredgecolor="white", markeredgewidth=0.8, label=p))
            ax.set_xticks(np.log(LAMS))
            ax.set_xticklabels([f"{l:g}" for l in LAMS])
            ax.axhline(0, color=SPINE, linewidth=0.8, zorder=1)
            style_ax(ax, f"{PRETTY[short]}: {title}", "penalty λ for a wrong answer (log scale)",
                     "mean test reward per question (± SE)" if ax is axes[0] else None)
        fig.legend(handles=handles, labels=[h.get_label() for h in handles], frameon=False, loc="upper left",
                   bbox_to_anchor=(1.0, 0.92), labelcolor=INK)
        fig.tight_layout()
        save(fig, f"reward_vs_lambda_{short}")


def policy_map_figures():
    for cfg, short in CONFIGS.items():
        clusters, qids, judge, _ = load_data(cfg)
        fold_of = fold_assignment(len(qids), 5, 0)
        train = np.where(fold_of != 0)[0]
        model = fit_belief(clusters, train, H=20, seed=0, judge=judge, qids=qids)    # = fold 0 of eval/final.py
        pol = DPPolicy(model, Costs(c_s=0.01, c_j=0.02, lam=2.0), use_judge=True, allow_abstain=True)
        policy_map_figure(pol, model, f"{PRETTY[short]}: DP (+judge, abstain) action by vote counts (fold-0 model, "
                          "λ=2, c_s=0.01, c_j=0.02); cell text = P(leader correct)", f"policy_map_{short}")


def reliability_figures(bins: pd.DataFrame, metrics: pd.DataFrame):
    panels = [
        ("held-out belief", [("vote-only pc[n,a,b], all trajectory states n=1..21", "#0b0b0b", "o", "-"),
                              ("posterior (votes+judge), DP-visited states", POLICY_STYLE["DP (+judge, abstain)"][0],
                               "P", "-")]),
        ("after k judge verdicts (probe)", [("vote-only pc[n,a,b], all trajectory states n=1..21", "#a3a29c", "o", "--"),
                                            ("posterior after 1 probe verdict, all trajectory states",
                                             POLICY_STYLE["DP (+judge)"][0], "D", "-"),
                                            ("posterior after 3 probe verdicts, all trajectory states",
                                             "#2e2380", "s", "-")]),
    ]
    short_label = {"vote-only pc[n,a,b], all trajectory states n=1..21": "votes only, all states",
                   "posterior (votes+judge), DP-visited states": "votes + judge, states the DP visits",
                   "posterior after 1 probe verdict, all trajectory states": "votes + 1 verdict",
                   "posterior after 3 probe verdicts, all trajectory states": "votes + 3 verdicts"}
    for short in CONFIGS.values():
        fig, axes = plt.subplots(2, 2, figsize=(9.2, 5.6), gridspec_kw={"height_ratios": [3, 1.3]}, sharex=True)
        for j, (title, sets) in enumerate(panels):
            ax, axc = axes[0, j], axes[1, j]
            ax.plot([0, 1], [0, 1], color=SPINE, linewidth=1, zorder=1)
            w = 0.8 / len(sets)
            for i, (name, col, mk, ls) in enumerate(sets):
                b = bins[(bins.config == short) & (bins.belief == name)]
                mt = metrics[(metrics.config == short) & (metrics.belief == name)].iloc[0]
                bb = b[b["count"] > 0]
                ax.plot(bb.mean_pred, bb.frac_correct, color=col, marker=mk, linestyle=ls, markeredgecolor="white",
                        markeredgewidth=0.8, label=f"{short_label[name]}\n(ECE {mt.ECE:.3f}, Brier {mt.Brier:.3f})")
                axc.bar(b.lo + 0.05 + (i - (len(sets) - 1) / 2) * w * 0.1, b["count"].clip(lower=0.8), width=w * 0.1,
                        color=col, alpha=0.9 if ls == "-" else 0.5)
            style_ax(ax, f"{PRETTY[short]}: {title}", None, "fraction correct" if j == 0 else None)
            ax.set_xlim(0, 1), ax.set_ylim(0, 1)
            ax.legend(frameon=False, loc="upper left", fontsize=7.5, labelcolor=INK)
            style_ax(axc, None, "predicted P(leader correct), 10 bins", "states per bin" if j == 0 else None)
            axc.set_yscale("log")
        fig.tight_layout()
        save(fig, f"reliability_{short}")


def judge_figure():
    t = pd.read_csv(ROOT / "judge_analysis" / "by_tercile.csv")
    order = ["most contested", "middle", "least contested", "all"]
    # neutral inks + hatching: these are judge metrics, not policies, so they must not reuse policy colors
    mets = [("false_pass", "false-pass rate", "#3a3a38", ""), ("false_fail", "false-fail rate", "#9b9a94", ""),
            ("balanced_accuracy", "balanced accuracy", "#ffffff", "////")]
    for cfg, short in CONFIGS.items():
        d = t[t.config == cfg].set_index("tercile").loc[order]
        fig, ax = plt.subplots(figsize=(6.6, 3.4))
        x = np.arange(len(order))
        w = 0.26
        for i, (m, lab, col, hatch) in enumerate(mets):
            y = d[m].to_numpy()
            err = np.vstack([y - d[m + "_lo"], d[m + "_hi"] - y])
            ax.bar(x + (i - 1) * w, y, width=w - 0.03, color=col, label=lab, yerr=err, capsize=2, hatch=hatch,
                   edgecolor="#3a3a38", linewidth=0.6,
                   error_kw={"elinewidth": 0.8, "ecolor": INK})
        labels = [f"{o}\n(top share {d.loc[o, 'top_share_min']:.2f}–{d.loc[o, 'top_share_max']:.2f})"
                  if o != "all" else "all questions" for o in order]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color=SPINE, linewidth=0.8, linestyle="--", zorder=1)
        style_ax(ax, f"{PRETTY[short]}: 3B judge error by question contestedness",
                 "contestedness tercile (vote share of the most common answer)", "rate (95% bootstrap CI)")
        ax.grid(axis="x", visible=False)
        legend_right(ax)
        save(fig, f"judge_by_contestedness_{short}")


def judge_tables():
    t = pd.read_csv(ROOT / "judge_analysis" / "by_tercile.csv")
    t = pd.DataFrame({
        "dataset": t.config.str.split("_").str[0], "tercile": t.tercile, "questions": t.n_questions,
        "top share": [f"{a:.2f}–{b:.2f}" for a, b in zip(t.top_share_min, t.top_share_max)],
        "answers (correct/wrong)": [f"{a}/{b}" for a, b in zip(t.n_correct_answers, t.n_wrong_answers)],
        "accuracy": [f"{a:.2f} [{b:.2f}, {c:.2f}]" for a, b, c in zip(t.accuracy, t.accuracy_lo, t.accuracy_hi)],
        "false-pass": [f"{a:.2f} [{b:.2f}, {c:.2f}]" for a, b, c in zip(t.false_pass, t.false_pass_lo, t.false_pass_hi)],
        "false-fail": [f"{a:.2f} [{b:.2f}, {c:.2f}]" for a, b, c in zip(t.false_fail, t.false_fail_lo, t.false_fail_hi)],
        "balanced acc.": [f"{a:.2f} [{b:.2f}, {c:.2f}]" for a, b, c in
                          zip(t.balanced_accuracy, t.balanced_accuracy_lo, t.balanced_accuracy_hi)]})
    write_table(t, "judge_terciles", "3B judge (Qwen2.5-3B-Instruct, 8 verdicts per answer) by contestedness tercile "
                "(vote share of each question's most common answer over 10k samples). Verdict-weighted rates, "
                "95% bootstrap CIs over questions.", str(ROOT / "judge_analysis" / "by_tercile.csv"), align="llrlrllll")
    i = pd.read_csv(ROOT / "judge_analysis" / "informativeness.csv")
    i = pd.DataFrame({
        "dataset": i.config.str.split("_").str[0], "vote state (n=2..10)": i.state, "states": i.n_states.map(lambda x: f"{x:,}"),
        "P(correct)": i.p_leader_correct.map(lambda x: f"{x:.2f}"),
        "judge LR pass/fail": [f"{a:.2f} / {b:.2f}" for a, b in zip(i.judge_lr_pass, i.judge_lr_fail)],
        "vote LR agree/disagree": [f"{a:.2f} / {b:.2f}" for a, b in zip(i.vote_lr_agree, i.vote_lr_disagree)],
        "MI judge (bits)": i.judge_mi_bits.map(lambda x: f"{x:.3f}"), "MI vote (bits)": i.vote_mi_bits.map(lambda x: f"{x:.3f}"),
        "P(correct) after pass/fail": [f"{a:.2f} / {b:.2f}" for a, b in zip(i.posterior_after_pass, i.posterior_after_fail)]})
    write_table(i, "judge_informativeness",
                "How much one judge verdict vs. one more vote tells us about whether the current leader is correct, "
                "by vote state. LR = P(signal | leader correct) / P(signal | leader wrong); MI = mutual information "
                "between the binary signal and correctness. A vote 'agrees' if the next sample equals the leader. "
                "Judge costs 2× a vote in the main runs.", str(ROOT / "judge_analysis" / "informativeness.csv"),
                align="llrrrrrrr")


def judge7b_table():
    """3B vs 7B judge at λ=2 (only if the 7B runs exist; see colab/judge_7b.ipynb)."""
    rows = []
    for cfg, short in CONFIGS.items():
        for jr in JRS:
            for judge, suf in (("3B", ""), ("7B", "_judge7b")):
                d = run_dir(cfg, 2, jr, suf)
                if not (d / "headline.csv").exists():
                    continue
                h = pd.read_csv(d / "headline.csv").set_index("policy")
                s = pd.read_csv(d / "significance.csv")
                for p, base in (("DP (+judge)", "DP (resample)"), ("DP (+judge, abstain)", "DP (resample, abstain)")):
                    r = s[(s.policy == p) & (s.vs == base)].iloc[0]
                    rows.append({"dataset": PRETTY[short], "c_j/c_s": f"{jr:g}", "judge": judge, "policy": p,
                                 "reward": pm(h.loc[p, "reward"], h.loc[p, "reward_se"]),
                                 "judge calls": f"{h.loc[p, 'judges']:.2f}",
                                 "gain vs no judge [95% CI]": f"{r.mean_diff:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]"})
    if any(r["judge"] == "7B" for r in rows):
        write_table(pd.DataFrame(rows), "judge7b_comparison", "3B vs 7B judge at λ=2: DP reward with the judge and "
                    "its gain over the same DP without the judge.", f"{ROOT}/*/lam2_jr*_judge7b/", align="lllllrrl")
        return True
    return False


def main():
    for d in (TAB, CAL):
        d.mkdir(parents=True, exist_ok=True)
    headline_tables()
    tuned_param_tables()
    significance_tables()
    lambda_table()
    judge_tables()
    print("7B judge comparison:", judge7b_table())
    m, b = calibration()
    print(m.round(4).to_string(index=False))
    frontier_figures()
    lambda_figure()
    policy_map_figures()
    reliability_figures(b, m)
    judge_figure()
    print("tables:", sorted(p.name for p in TAB.glob("*.tex")))
    print("figures:", sorted(p.name for p in (ROOT / "figures").glob("*.pdf")))


if __name__ == "__main__":
    main()
