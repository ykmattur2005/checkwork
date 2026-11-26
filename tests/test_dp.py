import numpy as np

from data.synth import make
from policies.baselines import fixed_k
from policies.dp import DPPolicy, fit_belief
from sim.env import ABSTAIN, JUDGE, RESAMPLE, SUBMIT, Costs, Env, JudgeTable, evaluate

clusters, judge_df = make(n_questions=60, n_samples=3000, seed=1)
judge = JudgeTable({(int(r.qid), int(r.cluster)): r.n_pass / r.n_total for r in judge_df.itertuples()}, 0.25, 0.2)
qids = np.arange(60)
model = fit_belief(clusters, range(60), episodes_per_q=150, judge=judge, qids=qids)


def test_belief_probabilities_valid():
    assert np.all((model.pc >= 0) & (model.pc <= 1))
    s = model.T.sum(-1)
    assert np.allclose(s[model.T.sum(-1) > 0], 1)


def test_more_agreement_means_more_confidence():
    # unanimous 5/5 should be more trustworthy than a split 3-2
    assert model.pc[5, 5, 0] > model.pc[5, 3, 2]
    assert model.pc[10, 10, 0] > model.pc[2, 2, 0] - 0.02


def test_free_checks_keep_checking_and_expensive_checks_stop():
    free = DPPolicy(model, Costs(c_s=0.0, c_j=1e9, lam=2), use_judge=False)
    assert free({"n": 1, "a": 1, "b": 0, "jp": 0, "jf": 0, "n_judge": 0}) == RESAMPLE
    pricey = DPPolicy(model, Costs(c_s=10, c_j=20, lam=2), use_judge=True)
    assert pricey({"n": 1, "a": 1, "b": 0, "jp": 0, "jf": 0, "n_judge": 0}) in (SUBMIT, ABSTAIN)


def test_judge_fail_lowers_belief():
    p = DPPolicy(model, Costs())
    assert p.posterior(3, 2, 1, 0, 1) < p.posterior(3, 2, 1, 0, 0) < p.posterior(3, 2, 1, 1, 0)


def test_huge_penalty_abstains_when_unsure():
    p = DPPolicy(model, Costs(c_s=5, c_j=5, lam=1000))
    assert p({"n": 1, "a": 1, "b": 0, "jp": 0, "jf": 0, "n_judge": 0}) == ABSTAIN


def test_dp_value_matches_simulation_in_sample():
    """DP's predicted value at the start state ~= simulated mean reward (same data used to fit)."""
    costs = Costs(c_s=0.02, c_j=0.04, lam=2)
    pol = DPPolicy(model, costs)
    env = Env(clusters, qids, costs, judge, seed=5)
    df = evaluate(env, pol, range(60), episodes_per_q=60)
    v0 = pol.value(1, 1, 0, 0, 0, 0)[0]
    assert abs(df.reward.mean() - v0) < 0.06, (df.reward.mean(), v0)


def test_dp_beats_fixed_k_on_reward_in_sample():
    costs = Costs(c_s=0.02, c_j=0.04, lam=2)
    env = Env(clusters, qids, costs, judge, seed=6)
    dp = evaluate(env, DPPolicy(model, costs), range(60), 40).reward.mean()
    best_fixed = max(evaluate(env, fixed_k(k), range(60), 40).reward.mean() for k in (1, 3, 5, 9))
    assert dp > best_fixed
