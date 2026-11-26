"""Shared machinery for the final experiments (results/final/).

* Common random numbers (CRN): every episode (question qi, episode e) gets its own RNG seeded by
  (seed, stream, fold, qi, e). All policies evaluated on the same stream therefore see the SAME
  sample order for a given (qi, e), which makes per-question comparisons paired. Tuning uses
  stream TUNE and test evaluation uses stream TEST, so tuning and test simulations never share draws.
* Fast trajectory evaluation of resample-only rules (fixed-k, margin, margin+abstain): we draw the
  full 21-sample order once per episode with exactly the call Env.reset makes, track (a, b, leader)
  after every attempt, and score every parameter value of every rule from those arrays. This gives
  results identical to running Env (tested in tests/test_final.py).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sim.env import ABSTAIN, JUDGE, RESAMPLE, SUBMIT, Costs, Env, JudgeTable, VoteTracker

TEST, TUNE, ANALYSIS, PROBE = 0, 1, 2, 3          # RNG streams

DATA = Path(__file__).resolve().parent.parent / "data" / "processed"

# Baseline grids (task 1)
FIXED_K_GRID = [1, 2, 3, 5, 7, 9, 13, 17, 21]
MARGIN_GRID = [1, 2, 3, 4, 5, 6, 7, 8]
MARGIN_ABSTAIN_GRID = [(m, k) for m in range(2, 9) for k in (5, 10, 15, 20)]


def load_data(config: str, judge_file: str | None = None):
    z = np.load(DATA / f"{config}.npz")
    path = DATA / (judge_file or f"{config}_judge.csv")
    judge = JudgeTable.from_csv(path) if path.exists() else None
    return z["clusters"], z["qids"], judge, path


def fold_assignment(Q: int, folds: int = 5, seed: int = 0) -> np.ndarray:
    """Identical to eval/run.py so folds match the earlier results."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(Q)
    fold_of = np.empty(Q, int)
    for f, chunk in enumerate(np.array_split(perm, folds)):
        fold_of[chunk] = f
    return fold_of


def episode_rng(seed: int, stream: int, fold: int, qi: int, ep: int) -> np.random.Generator:
    return np.random.default_rng([seed, stream, fold, int(qi), ep])


# ----------------------------------------------------------------------------- trajectories
def trajectories(clusters, qidx, episodes, seed, stream, fold, H=20) -> dict:
    """Vote statistics after each attempt n = 1..H+1 for every (question, episode).

    Returns arrays of shape (len(qidx), episodes, H+1): a, b, correct (leader correct), leader
    (cluster id of the leader, -1 if it is an unparseable answer), draw (cluster id drawn).
    """
    N = clusters.shape[1]
    L = min(N, H + 1)
    shape = (len(qidx), episodes, L)
    A = np.zeros(shape, np.int16)
    B = np.zeros(shape, np.int16)
    C = np.zeros(shape, bool)
    LEAD = np.zeros(shape, np.int32)
    DRAW = np.zeros(shape, np.int32)
    for i, qi in enumerate(qidx):
        row = clusters[qi]
        for e in range(episodes):
            order = episode_rng(seed, stream, fold, qi, e).choice(N, size=L, replace=False)
            t = VoteTracker()
            for k, x in enumerate(row[order]):
                t.add(int(x))
                A[i, e, k], B[i, e, k], C[i, e, k] = t.a, t.b, t.leader_correct
                LEAD[i, e, k] = t.leader if isinstance(t.leader, (int, np.integer)) else -1
                DRAW[i, e, k] = x
    return {"a": A, "b": B, "correct": C, "leader": LEAD, "draw": DRAW, "qidx": np.asarray(qidx)}


def rule_outcomes(traj, kind: str, param):
    """Score one resample-only rule on trajectories.

    Returns (outcome, resamples) arrays of shape (Q, E): outcome 1 correct, -1 wrong, 0 abstain.
    Semantics match policies/baselines.py exactly (including the H budget).
    """
    A, B, C = traj["a"].astype(int), traj["b"].astype(int), traj["correct"]
    Qn, E, L = A.shape
    n = np.arange(1, L + 1)
    if kind == "fixed_k":
        stop = np.full((Qn, E), min(param, L) - 1)
        submit = np.ones((Qn, E), bool)
    elif kind in ("margin", "margin_abstain"):
        m, k_max = (param, 21) if kind == "margin" else param
        hit = (A - B) >= m
        forced = (n >= k_max) | (n == L)                 # n >= k_max or checks_left <= 0
        cond = hit | forced[None, None, :]
        stop = cond.argmax(-1)
        submit = np.take_along_axis(hit, stop[..., None], -1)[..., 0] if kind == "margin_abstain" \
            else np.ones((Qn, E), bool)
    else:
        raise ValueError(kind)
    corr = np.take_along_axis(C, stop[..., None], -1)[..., 0]
    outcome = np.where(submit, np.where(corr, 1, -1), 0)
    return outcome, stop


def per_question(outcome, resamples, judges, qidx, costs: Costs, judge_ratio: float) -> pd.DataFrame:
    """Per-question means over episodes. outcome/resamples/judges: (Q, E)."""
    reward = np.where(outcome == 1, 1.0, np.where(outcome == -1, -costs.lam, 0.0)) \
        - costs.c_s * resamples - costs.c_j * judges
    return pd.DataFrame({
        "qi": np.asarray(qidx),
        "reward": reward.mean(1),
        "accuracy": (outcome == 1).mean(1),
        "wrong_rate": (outcome == -1).mean(1),
        "abstain_rate": (outcome == 0).mean(1),
        "compute": (resamples + judge_ratio * judges).mean(1),
        "resamples": resamples.mean(1),
        "judges": judges.mean(1),
        "episodes": outcome.shape[1],
    })


def baseline_grid(traj, costs: Costs, judge_ratio: float) -> pd.DataFrame:
    """Per-question results for every baseline parameter on the given trajectories."""
    out = []
    z = np.zeros(traj["a"].shape[:2])
    specs = [("fixed_k", k) for k in FIXED_K_GRID] + [("margin", m) for m in MARGIN_GRID] + \
            [("margin_abstain", p) for p in MARGIN_ABSTAIN_GRID]
    for kind, p in specs:
        o, r = rule_outcomes(traj, kind, p)
        d = per_question(o, r, z, traj["qidx"], costs, judge_ratio)
        d["rule"], d["param"] = kind, str(p)
        out.append(d)
    return pd.concat(out, ignore_index=True)


def tune(grid: pd.DataFrame) -> pd.DataFrame:
    """Pick, for each rule, the parameter with the highest mean reward over the questions in `grid`.
    Ties go to the earlier (cheaper) grid entry."""
    rows = []
    for rule, g in grid.groupby("rule", sort=False):
        means = g.groupby("param", sort=False).reward.mean()
        rows.append({"rule": rule, "param": means.idxmax(), "tune_reward": float(means.max())})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- env episodes (DP)
def run_env(env: Env, policy, qidx, episodes, seed, stream, fold, belief=None, log=None):
    """Run policy with CRN seeding. Returns (outcome, resamples, judges) arrays of shape (Q, E).

    If `log` is a list and `belief` is a callable (n, a, b, jp, jf) -> (pc_vote, posterior),
    one row per decision state is appended: the beliefs, the action and whether the leader is correct.
    """
    Qn = len(qidx)
    O = np.zeros((Qn, episodes), int)
    R = np.zeros((Qn, episodes), int)
    J = np.zeros((Qn, episodes), int)
    for i, qi in enumerate(qidx):
        for e in range(episodes):
            env.rng = episode_rng(seed, stream, fold, qi, e)
            obs = env.reset(qi)
            while True:
                act = policy(obs)
                if log is not None:
                    pv, post = belief(obs["n"], obs["a"], obs["b"], obs["jp"], obs["jf"])
                    log.append((fold, int(qi), e, obs["n"], obs["a"], obs["b"], obs["jp"], obs["jf"],
                                pv, post, act, env.tracker.leader_correct))
                obs, _, done = env.step(act)
                if done:
                    break
            O[i, e] = {"correct": 1, "wrong": -1, "abstain": 0}[env.outcome]
            R[i, e] = env.tracker.n - 1
            J[i, e] = env.n_judge
    return O, R, J
