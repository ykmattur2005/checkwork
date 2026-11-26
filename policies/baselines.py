"""Fixed-rule baselines. A policy is any callable obs -> action."""
from sim.env import ABSTAIN, RESAMPLE, SUBMIT


def fixed_k(k: int):
    """Majority vote over k attempts (k=1 is 'submit the first answer')."""
    def policy(obs):
        if obs["n"] < k and obs["checks_left"] > 0:
            return RESAMPLE
        return SUBMIT
    policy.__name__ = f"fixed_k={k}"
    return policy


def margin(m: int, k_max: int = 21):
    """Keep sampling until the leader is ahead of the runner-up by m votes (or k_max attempts)."""
    def policy(obs):
        if obs["a"] - obs["b"] >= m or obs["n"] >= k_max or obs["checks_left"] <= 0:
            return SUBMIT
        return RESAMPLE
    policy.__name__ = f"margin={m}"
    return policy


def margin_abstain(m: int, k_max: int = 10):
    """Margin stopping, but if the leader is still not m votes ahead after k_max attempts, abstain."""
    def policy(obs):
        if obs["a"] - obs["b"] >= m:
            return SUBMIT
        if obs["n"] >= k_max or obs["checks_left"] <= 0:
            return ABSTAIN
        return RESAMPLE
    policy.__name__ = f"margin_abstain={m}"
    return policy
