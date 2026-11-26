"""Simulator: replays stored samples as RESAMPLE results and cached judge verdicts as JUDGE results."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

RESAMPLE, JUDGE, SUBMIT, ABSTAIN = "resample", "judge", "submit", "abstain"

# Vote outcome categories (see VoteTracker.add)
LEAD, RUNNER, SWITCH, OTHER = 0, 1, 2, 3


class VoteTracker:
    """Tracks vote counts. Leader changes only when another answer gets STRICTLY more votes.

    add() returns how the compressed state (a, b) changed:
      LEAD   -> (a+1, b)        vote for the leader
      RUNNER -> (a, b+1)        runner-up count went up, leader unchanged
      SWITCH -> (a+1, a)        another answer overtook the leader
      OTHER  -> (a, b)          vote for some answer below the runner-up
    """

    def __init__(self):
        self.counts: dict = {}
        self.leader = None
        self.n = self.a = self.b = 0
        self._unique = 0

    def add(self, cluster: int) -> int:
        if cluster == -1:                      # unparseable -> a fresh answer every time
            self._unique += 1
            cluster = ("unparsed", self._unique)
        self.n += 1
        if self.leader is None:
            self.leader, self.a, self.counts[cluster] = cluster, 1, 1
            return LEAD
        c = self.counts.get(cluster, 0) + 1
        self.counts[cluster] = c
        if cluster == self.leader:
            self.a = c
            return LEAD
        if c > self.a:
            self.b, self.a, self.leader = self.a, c, cluster
            return SWITCH
        if c > self.b:
            self.b = c
            return RUNNER
        return OTHER

    @property
    def leader_correct(self) -> bool:
        return self.leader == 0


@dataclass
class Costs:
    c_s: float = 0.01      # resample cost
    c_j: float = 0.02      # judge cost
    lam: float = 2.0       # penalty for a wrong submission
    H: int = 20            # max checks per episode


@dataclass
class JudgeTable:
    """Per-(question, answer) judge pass probability, with pooled fallback rates."""
    p_pass: dict = field(default_factory=dict)
    false_pass: float = 0.3     # P(pass | wrong)
    false_fail: float = 0.2     # P(fail | correct)

    @classmethod
    def from_csv(cls, path, qids=None):
        df = pd.read_csv(path)
        df = df[df.n_total > 0]
        if qids is not None:
            df = df[df.qid.isin(set(qids))]
        p = {(int(r.qid), int(r.cluster)): r.n_pass / r.n_total for r in df.itertuples()}
        rate = df.n_pass / df.n_total
        correct = df.cluster == 0
        fp = float(rate[~correct].mean()) if (~correct).any() else 0.3
        ff = float(1 - rate[correct].mean()) if correct.any() else 0.2
        return cls(p, fp, ff)

    def prob_pass(self, qid: int, cluster) -> float:
        if isinstance(cluster, (int, np.integer)) and (qid, int(cluster)) in self.p_pass:
            return self.p_pass[(qid, int(cluster))]
        return 1 - self.false_fail if cluster == 0 else self.false_pass


def load_config(name: str, root: Path | None = None):
    root = root or Path(__file__).resolve().parent.parent / "data" / "processed"
    z = np.load(root / f"{name}.npz")
    judge_path = root / f"{name}_judge.csv"
    judge = JudgeTable.from_csv(judge_path) if judge_path.exists() else None
    return z["clusters"], z["qids"], judge


class Env:
    def __init__(self, clusters: np.ndarray, qids: np.ndarray, costs: Costs,
                 judge: JudgeTable | None = None, seed: int = 0):
        self.clusters, self.qids, self.costs = clusters, qids, costs
        self.judge = judge or JudgeTable()
        self.rng = np.random.default_rng(seed)

    def reset(self, qi: int):
        """qi indexes rows of `clusters`."""
        self.qi = qi
        N = self.clusters.shape[1]
        self.order = self.rng.choice(N, size=min(N, self.costs.H + 1), replace=False)
        self.ptr = 0
        self.tracker = VoteTracker()
        self.jp = self.jf = self.n_judge = 0
        self.cost = 0.0
        self.done = False
        self._draw()
        return self.obs()

    def _draw(self):
        lead_before = self.tracker.leader
        self.tracker.add(int(self.clusters[self.qi, self.order[self.ptr]]))
        self.ptr += 1
        if self.tracker.leader != lead_before:
            self.jp = self.jf = 0

    @property
    def checks_used(self) -> int:
        return (self.tracker.n - 1) + self.n_judge

    def obs(self) -> dict:
        t = self.tracker
        return {"n": t.n, "a": t.a, "b": t.b, "jp": self.jp, "jf": self.jf,
                "n_judge": self.n_judge, "checks_left": self.costs.H - self.checks_used}

    def step(self, action: str):
        assert not self.done
        c = self.costs
        if action in (RESAMPLE, JUDGE) and self.checks_used >= c.H:
            raise ValueError("check budget exhausted; must SUBMIT or ABSTAIN")
        if action == RESAMPLE:
            self.cost += c.c_s
            self._draw()
            return self.obs(), -c.c_s, False
        if action == JUDGE:
            self.cost += c.c_j
            self.n_judge += 1
            p = self.judge.prob_pass(int(self.qids[self.qi]), self.tracker.leader)
            if self.rng.random() < p:
                self.jp += 1
            else:
                self.jf += 1
            return self.obs(), -c.c_j, False
        self.done = True
        if action == ABSTAIN:
            self.outcome = "abstain"
            return self.obs(), 0.0, True
        if action == SUBMIT:
            ok = self.tracker.leader_correct
            self.outcome = "correct" if ok else "wrong"
            return self.obs(), (1.0 if ok else -c.lam), True
        raise ValueError(action)


def run_episode(env: Env, policy, qi: int) -> dict:
    obs = env.reset(qi)
    total = 0.0
    while True:
        a = policy(obs)
        obs, r, done = env.step(a)
        total += r
        if done:
            return {"qi": qi, "outcome": env.outcome, "cost": env.cost, "reward": total,
                    "resamples": env.tracker.n - 1, "judges": env.n_judge}


def evaluate(env: Env, policy, question_idx, episodes_per_q: int = 100) -> pd.DataFrame:
    rows = [run_episode(env, policy, qi) for qi in question_idx for _ in range(episodes_per_q)]
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, judge_ratio: float) -> dict:
    """judge_ratio = c_j / c_s, so compute is reported in 'resample-equivalents'."""
    return {
        "accuracy": (df.outcome == "correct").mean(),
        "wrong_rate": (df.outcome == "wrong").mean(),
        "abstain_rate": (df.outcome == "abstain").mean(),
        "compute": (df.resamples + judge_ratio * df.judges).mean(),
        "resamples": df.resamples.mean(),
        "judges": df.judges.mean(),
        "reward": df.reward.mean(),
    }
