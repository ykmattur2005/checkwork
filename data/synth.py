"""Synthetic stand-in for a processed Monkey Business config (same file format).

Lets you develop and test everything before the real data is downloaded.

    python -m data.synth --name SYNTH --questions 128
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "processed"


def make(n_questions=128, n_samples=10_000, seed=0, judge_reps=8, top_k=3):
    rng = np.random.default_rng(seed)
    clusters = np.empty((n_questions, n_samples), dtype=np.int32)
    judge_rows = []
    for q in range(n_questions):
        p = rng.beta(0.6, 0.4)                       # per-question success rate (U-shaped, like real LLMs)
        k_wrong = rng.integers(1, 30)                # number of distinct wrong answers
        w = rng.dirichlet(np.full(k_wrong, 0.3))     # concentrated -> wrong answers often coincide
        p_unparse = rng.uniform(0, 0.05)
        probs = np.concatenate([[p], (1 - p) * (1 - p_unparse) * w, [(1 - p) * p_unparse]])
        labels = np.concatenate([np.arange(k_wrong + 1), [-1]])
        draw = rng.choice(labels, size=n_samples, p=probs)
        # relabel wrong clusters by frequency (1 = most common wrong answer), as prep.py does
        vals, cnts = np.unique(draw[draw > 0], return_counts=True)
        remap = {v: i + 1 for i, v in enumerate(vals[np.argsort(-cnts)])}
        clusters[q] = [remap.get(x, x) for x in draw]
        # judge: per-answer pass probability (heterogeneous)
        freq = pd.Series(clusters[q][clusters[q] >= 0]).value_counts()
        for cid in freq.index[:top_k]:
            pp = rng.beta(8, 2) if cid == 0 else rng.beta(2, 6)
            judge_rows.append({"qid": q, "cluster": int(cid),
                               "n_pass": int(rng.binomial(judge_reps, pp)), "n_total": judge_reps})
    return clusters, pd.DataFrame(judge_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="SYNTH")
    ap.add_argument("--questions", type=int, default=128)
    ap.add_argument("--samples", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    clusters, judge = make(args.questions, args.samples, args.seed)
    np.savez_compressed(OUT / f"{args.name}.npz", clusters=clusters, qids=np.arange(args.questions))
    judge.to_csv(OUT / f"{args.name}_judge.csv", index=False)
    print(f"saved {clusters.shape}, pass@1={(clusters == 0).mean():.3f}, judge rows={len(judge)}")


if __name__ == "__main__":
    main()
