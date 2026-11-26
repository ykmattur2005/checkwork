import numpy as np
import pandas as pd
import pytest

from eval.common import (TEST, TUNE, baseline_grid, episode_rng, per_question, rule_outcomes, run_env,
                         trajectories, tune)
from eval.stats import bootstrap_mean_ci, brier, calibration_bins, ece, holm, wilcoxon_p
from policies.baselines import fixed_k, margin, margin_abstain
from sim.env import Costs, Env, JudgeTable


def _clusters(seed=0, Q=12, N=500):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.2, 0.9, Q)
    return np.where(rng.random((Q, N)) < p[:, None], 0, rng.integers(-1, 4, (Q, N))).astype(np.int32)


@pytest.mark.parametrize("kind,param,policy", [
    ("fixed_k", 1, fixed_k(1)), ("fixed_k", 5, fixed_k(5)), ("fixed_k", 21, fixed_k(21)),
    ("margin", 3, margin(3)), ("margin", 8, margin(8)),
    ("margin_abstain", (3, 10), margin_abstain(3, 10)), ("margin_abstain", (6, 5), margin_abstain(6, 5)),
    ("margin_abstain", (2, 20), margin_abstain(2, 20))])
def test_fast_rules_match_env_exactly(kind, param, policy):
    """Trajectory scoring == running the real policy in Env with the same CRN seeds, episode by episode."""
    cl = _clusters()
    q = np.arange(len(cl))
    traj = trajectories(cl, q, 15, seed=3, stream=TEST, fold=0)
    o_fast, r_fast = rule_outcomes(traj, kind, param)
    env = Env(cl, q, Costs(), seed=0)
    o_env, r_env, j_env = run_env(env, policy, q, 15, 3, TEST, 0)
    assert np.array_equal(o_fast, o_env) and np.array_equal(r_fast, r_env) and (j_env == 0).all()


def test_streams_differ_but_are_reproducible():
    a = episode_rng(0, TEST, 0, 5, 1).random()
    assert a == episode_rng(0, TEST, 0, 5, 1).random()
    assert a != episode_rng(0, TUNE, 0, 5, 1).random()


def test_tuning_uses_train_questions_only():
    """Training questions reward big k; test questions are all-correct (k=1 best). The tuned k must be
    the training optimum, and must not change when the test questions' data change."""
    rng = np.random.default_rng(0)
    Q, N = 20, 400
    cl = np.where(rng.random((Q, N)) < 0.6, 0, rng.integers(1, 30, (Q, N))).astype(np.int32)
    train, test = np.arange(10), np.arange(10, 20)
    costs = Costs(c_s=0.001, lam=2)
    choose = lambda c: tune(baseline_grid(trajectories(c, train, 30, 0, TUNE, 0), costs, 2.0))
    ch = choose(cl).set_index("rule").param
    assert int(ch["fixed_k"]) > 1
    cl2 = cl.copy()
    cl2[test] = 0                                     # make the test questions trivially easy
    assert choose(cl2).set_index("rule").param.equals(ch)
    # and it is the argmax of the train grid
    grid = baseline_grid(trajectories(cl, train, 30, 0, TUNE, 0), costs, 2.0)
    fk = grid[grid.rule == "fixed_k"].groupby("param").reward.mean()
    assert fk.idxmax() == ch["fixed_k"]


def test_per_question_reward():
    o = np.array([[1, -1, 0, 1]])
    r = np.array([[0, 2, 4, 1]])
    j = np.array([[0, 1, 0, 0]])
    d = per_question(o, r, j, [7], Costs(c_s=0.01, c_j=0.03, lam=2), judge_ratio=3)
    assert d.reward.iloc[0] == pytest.approx((1 - 2 + 0 + 1 - 0.01 * 7 - 0.03) / 4)
    assert d.compute.iloc[0] == pytest.approx((7 + 3) / 4)


def test_bootstrap_known_cases():
    m, lo, hi = bootstrap_mean_ci(np.full(30, 0.25), n_boot=2000)
    assert m == lo == hi == pytest.approx(0.25)
    d = np.random.default_rng(1).normal(1.0, 1.0, 400)
    m, lo, hi = bootstrap_mean_ci(d, n_boot=5000)
    se = d.std(ddof=1) / 20
    assert lo < 1.0 < hi and lo > 0
    assert hi - lo == pytest.approx(2 * 1.96 * se, rel=0.1)


def test_wilcoxon_known_case():
    # all 5 differences positive, no ties: exact two-sided p = 2 / 2^5
    assert wilcoxon_p([1, 2, 3, 4, 5]) == pytest.approx(2 / 32)
    assert wilcoxon_p([0, 0, 0]) == 1.0
    # symmetric differences -> p = 1
    assert wilcoxon_p([-1, 1, -2, 2, -3, 3]) == pytest.approx(1.0)


def test_holm():
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_calibration_binning():
    pred = np.array([0.0, 0.05, 0.15, 0.95, 1.0, 1.0])
    y = np.array([0, 1, 0, 1, 1, 0])
    t = calibration_bins(pred, y)
    assert list(t["count"]) == [2, 1, 0, 0, 0, 0, 0, 0, 0, 3]    # 1.0 lands in the last bin
    assert t.loc[0, "mean_pred"] == pytest.approx(0.025) and t.loc[0, "frac_correct"] == 0.5
    assert t.loc[9, "frac_correct"] == pytest.approx(2 / 3)
    expected = (2 * abs(0.025 - 0.5) + 1 * 0.15 + 3 * abs((0.95 + 2) / 3 - 2 / 3)) / 6
    assert ece(pred, y) == pytest.approx(expected)
    assert brier(pred, y) == pytest.approx(np.mean((pred - y) ** 2))
    perfect = np.repeat([0.0, 1.0], 50)
    assert ece(perfect, perfect) == 0 and brier(perfect, perfect) == 0
