"""One visual style for every final figure: fixed color + marker per policy, recessive axes,
legends outside the data, PDF + PNG output."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from sim.env import ABSTAIN, JUDGE, RESAMPLE, SUBMIT

# Validated categorical palette (dataviz reference, slots 1-7, light mode) + neutral for the trivial baseline.
POLICY_STYLE = {
    "first answer":           ("#52514e", "*"),
    "fixed-k":                ("#2a78d6", "o"),
    "margin stop":            ("#eb6834", "s"),
    "DP (resample)":          ("#1baf7a", "^"),
    "margin stop+abstain":    ("#eda100", "v"),
    "DP (+judge, abstain)":   ("#e87ba4", "P"),
    "DP (resample, abstain)": ("#008300", "X"),
    "DP (+judge)":            ("#6250d6", "D"),
}
ACTION_STYLE = {SUBMIT: ("submit", "#1baf7a", "#0b0b0b"), RESAMPLE: ("resample", "#86b6ef", "#0b0b0b"),
                JUDGE: ("judge", "#6250d6", "#ffffff"), ABSTAIN: ("abstain", "#e0dfda", "#0b0b0b")}
INK, MUTED, GRID, SPINE, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#c3c2b7", "#ffffff"
FIG = Path("results/final/figures")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 8.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
    "lines.linewidth": 1.8, "lines.markersize": 6,
})


def base_name(policy: str) -> str:
    for suf in (" (tuned)", " (oracle-tuned on test)"):
        policy = policy.replace(suf, "")
    return policy


def style_of(policy: str):
    return POLICY_STYLE[base_name(policy)]


def style_ax(ax, title=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title, loc="left", color=INK)
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(SPINE)
    ax.tick_params(colors=MUTED)


def legend_right(fig_or_ax, handles=None, labels=None, **kw):
    kw = {"frameon": False, "loc": "upper left", "bbox_to_anchor": (1.01, 1.0), "labelcolor": INK, **kw}
    if handles is None:
        return fig_or_ax.legend(**kw)
    return fig_or_ax.legend(handles, labels, **kw)


def save(fig, name: str, folder: Path = FIG):
    folder.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(folder / f"{name}.{ext}", dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return folder / f"{name}.png"


def policy_map(fig, axes, pol, model, ns, jp=0, jf=0, annotate=True):
    """DP action at each (a, b) after n attempts (no judge calls yet unless jp/jf given).
    Cells never reached in the training simulations are left blank."""
    codes = {SUBMIT: 0, RESAMPLE: 1, JUDGE: 2, ABSTAIN: 3}
    cmap = ListedColormap([ACTION_STYLE[a][1] for a in (SUBMIT, RESAMPLE, JUDGE, ABSTAIN)])
    for ax, n in zip(axes, ns):
        grid = np.full((n + 1, n + 1), np.nan)
        txt = []
        for a in range(1, n + 1):
            for b in range(0, min(a, n - a) + 1):
                if model.visits[n, a, b] == 0:
                    continue
                act = pol.value(n, a, b, jp, jf, jp + jf)[1]
                grid[b, a] = codes[act]
                txt.append((a, b, pol.posterior(n, a, b, jp, jf), ACTION_STYLE[act][2]))
        ax.imshow(grid, origin="lower", cmap=cmap, vmin=-0.5, vmax=3.5, aspect="auto")
        if annotate:
            for a, b, p, c in txt:
                ax.text(a, b, f"{p:.2f}", ha="center", va="center", fontsize=6.5 if n <= 6 else 5.5, color=c)
        ax.set_xlim(0.5, n + 0.5)
        ax.set_ylim(-0.5, n // 2 + 0.5)
        ax.set_xticks(range(1, n + 1))
        ax.set_yticks(range(0, n // 2 + 1))
        style_ax(ax, f"n = {n}", "leader votes a", "runner-up votes b" if ax is axes[0] else None)
        ax.grid(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_visible(False)
    handles = [Patch(facecolor=ACTION_STYLE[a][1], edgecolor=SPINE, label=ACTION_STYLE[a][0])
               for a in (SUBMIT, RESAMPLE, JUDGE, ABSTAIN)]
    return handles


def policy_map_figure(pol, model, title, name, ns=(2, 3, 4, 5, 6, 8)):
    """Two rows: no judge verdicts yet / after one FAIL verdict on the current leader."""
    fig = plt.figure(figsize=(12, 6.2))
    gs = fig.add_gridspec(2, len(ns), width_ratios=ns, height_ratios=[1, 1], hspace=0.75, wspace=0.25,
                          left=0.05, right=0.88, top=0.84, bottom=0.08)
    rows = []
    for r in range(2):
        rows.append([fig.add_subplot(gs[r, i]) for i in range(len(ns))])
    h = policy_map(fig, rows[0], pol, model, ns)
    policy_map(fig, rows[1], pol, model, ns, jp=0, jf=1)
    fig.text(0.05, 0.895, "(a) no judge verdicts yet", color=INK, fontsize=10, weight="bold")
    fig.text(0.05, 0.445, "(b) after one FAIL verdict on the current leader", color=INK, fontsize=10, weight="bold")
    fig.legend(handles=h, loc="upper left", bbox_to_anchor=(0.895, 0.84), frameon=False, title="DP action",
               labelcolor=INK)
    fig.suptitle(title, x=0.05, y=0.97, ha="left", color=INK, fontsize=10.5)
    return save(fig, name)
