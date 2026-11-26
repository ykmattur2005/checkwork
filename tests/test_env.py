import numpy as np
import pytest
from collections import Counter

from sim.env import (ABSTAIN, JUDGE, LEAD, OTHER, RESAMPLE, RUNNER, SUBMIT, SWITCH, Costs, Env,
                     JudgeTable, VoteTracker, evaluate, summarize)
from policies.baselines import fixed_k, margin


def test_tracker_categories_and_ties():
    t = VoteTracker()
    assert t.add(5) == LEAD and (t.a, t.b) == (1, 0)
    assert t.add(0) == RUNNER and (t.a, t.b) == (1, 1) and t.leader == 5   # tie keeps leader
    assert t.add(7) == OTHER and (t.a, t.b) == (1, 1)
    assert t.add(0) == SWITCH and (t.a, t.b) == (2, 1) and t.leader_correct
    assert t.add(0) == LEAD and (t.a, t.b) == (3, 1)
    assert t.add(5) == RUNNER and (t.a, t.b) == (3, 2)


def test_unparseable_never_accumulates():
    t = VoteTracker()
    for _ in range(5):
        t.add(-1)
    assert (t.a, t.b, t.n) == (1, 1, 5)


def test_tracker_matches_bruteforce():
    rng = np.random.default_rng(1)
    for _ in range(300):
        seq = rng.integers(0, 4, size=15)
        t = VoteTracker()
        for i, x in enumerate(seq):
            t.add(int(x))
            c = Counter(seq[: i + 1].tolist())
            vals = sorted(c.values(), reverse=True)
            assert t.a == vals[0] and t.b == (vals[1] if len(vals) > 1 else 0)
            assert c[t.leader] == t.a


def test_no_sample_reused_and_budget():
    clusters = np.arange(30).reshape(1, 30)          # every sample distinct
    env = Env(clusters, np.array([0]), Costs(H=20), seed=0)
    env.reset(0)
    for _ in range(20):
        env.step(RESAMPLE)
    assert len(set(env.order[: env.ptr].tolist())) == 21
    with pytest.raises(ValueError):
        env.step(RESAMPLE)
    _, r, done = env.step(ABSTAIN)
    assert done and r == 0


def test_judge_resets_on_leader_change():
    clusters = np.array([[1, 0, 0] * 10])
    env = Env(clusters, np.array([0]), Costs(), JudgeTable({(0, 1): 0.0, (0, 0): 1.0}), seed=3)
    obs = env.reset(0)
    while env.tracker.leader != 1:                     # make wrong answer the leader first
        obs = env.reset(0)
    obs, _, _ = env.step(JUDGE)
    assert obs["jf"] == 1
    while env.tracker.leader == 1 and obs["checks_left"] > 0:
        obs, _, _ = env.step(RESAMPLE)
    if env.tracker.leader == 0:
        assert obs["jp"] == obs["jf"] == 0 and obs["n_judge"] == 1


def test_fixed_k_matches_direct_majority():
    """Sanity check from the plan: env majority vote == majority vote computed directly."""
    rng = np.random.default_rng(0)
    Q, N, k = 40, 2000, 5
    p = rng.uniform(0.2, 0.9, Q)
    clusters = np.where(rng.random((Q, N)) < p[:, None], 0, rng.integers(1, 4, (Q, N))).astype(np.int32)
    env = Env(clusters, np.arange(Q), Costs(), seed=1)
    df = evaluate(env, fixed_k(k), range(Q), episodes_per_q=200)
    # direct Monte Carlo of the same rule
    hits = []
    for q in range(Q):
        for _ in range(200):
            t = VoteTracker()
            for x in clusters[q, rng.choice(N, k, replace=False)]:
                t.add(int(x))
            hits.append(t.leader_correct)
    assert abs((df.outcome == "correct").mean() - np.mean(hits)) < 0.02
    s = summarize(df, 2.0)
    assert s["resamples"] == k - 1 and s["abstain_rate"] == 0


def test_margin_stops():
    clusters = np.zeros((1, 100), dtype=np.int32)
    env = Env(clusters, np.array([0]), Costs(), seed=0)
    df = evaluate(env, margin(3), [0], 5)
    assert (df.resamples == 2).all() and (df.outcome == "correct").all()
