"""Exact dynamic programming over the observable statistics (n, a, b, jp, jf, n_judge).

Belief model (fit on TRAINING questions only):
  * P(leader correct | n, a, b)            -- empirical, smoothed. Captures wrong answers that
                                              coincide, because it is measured, not assumed.
  * P(vote outcome | n, a, b, leader correct) for outcomes LEAD / RUNNER / SWITCH / OTHER
  * Judge: a correct answer's per-answer pass probability ~ Beta(a1, b1), a wrong answer's
    ~ Beta(a0, b0). Repeated verdicts on the same answer are therefore correlated
    (Beta-Binomial), so a second judge call is worth less than the first.
  * Judge verdicts and votes are assumed independent given correctness, so
        P(correct | n,a,b,jp,jf)  ∝  P(correct | n,a,b) * BetaBinomialLik(jp, jf | correct).

Because the belief is a function of observable counts, the POMDP collapses to an MDP over
those counts, which we solve exactly by backward induction (memoized recursion).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.special import betaln

from sim.env import (ABSTAIN, JUDGE, LEAD, OTHER, RESAMPLE, RUNNER, SUBMIT, SWITCH, Costs,
                     JudgeTable, VoteTracker)


@dataclass
class BeliefModel:
    pc: np.ndarray        # [n, a, b] -> P(leader correct)
    T: np.ndarray         # [c, n, a, b, outcome] -> prob
    visits: np.ndarray    # [n, a, b] -> training visit count (for diagnostics)
    judge_beta: tuple     # ((a0, b0) for wrong, (a1, b1) for correct)


def _prior_outcomes(n, a, b):
    """Rough 'votes proportional to counts' prior used only for smoothing."""
    p = np.zeros(4)
    p[LEAD] = a / n
    if b == a:
        p[SWITCH] = b / n
    else:
        p[RUNNER] = b / n
    rest = max(0.0, 1 - p.sum())
    if b == 0:
        p[RUNNER] += rest
    else:
        p[OTHER] += rest
    return p / p.sum()


def fit_judge_beta(judge: JudgeTable | None, train_qids) -> tuple:
    def mom(vals, fallback_mean):
        vals = np.asarray(vals, float)
        if len(vals) < 3:
            m, conc = fallback_mean, 5.0
        else:
            m = float(np.clip(vals.mean(), 0.02, 0.98))
            v = vals.var()
            conc = m * (1 - m) / v - 1 if v > 0 else 200.0
            conc = float(np.clip(conc, 2.0, 200.0))
        return (m * conc, (1 - m) * conc)

    if judge is None:
        return ((1.5, 3.5), (4.0, 1.0))
    train = set(int(q) for q in train_qids)
    wrong = [p for (q, c), p in judge.p_pass.items() if q in train and c != 0]
    right = [p for (q, c), p in judge.p_pass.items() if q in train and c == 0]
    return (mom(wrong, judge.false_pass), mom(right, 1 - judge.false_fail))


def fit_belief(clusters, train_idx, H=20, episodes_per_q=400, kappa=2.0, seed=0,
               judge: JudgeTable | None = None, qids=None) -> BeliefModel:
    rng = np.random.default_rng(seed)
    S = H + 2
    cnt = np.zeros((S, S, S))
    corr = np.zeros((S, S, S))
    tc = np.zeros((2, S, S, S, 4))
    N = clusters.shape[1]
    for qi in train_idx:
        draws = clusters[qi][rng.integers(0, N, size=(episodes_per_q, H + 1))]
        for ep in draws:
            t = VoteTracker()
            t.add(int(ep[0]))
            for k in range(1, H + 2):
                n, a, b, c = t.n, t.a, t.b, int(t.leader_correct)
                cnt[n, a, b] += 1
                corr[n, a, b] += c
                if k == H + 1:
                    break
                tc[c, n, a, b, t.add(int(ep[k]))] += 1

    # P(correct | n,a,b): shrink toward the pooled estimate for the same margin a-b
    g = corr.sum() / cnt.sum()
    pc = np.full((S, S, S), g)
    margin_c, margin_n = np.zeros(S), np.zeros(S)
    for n in range(1, S):
        for a in range(1, n + 1):
            for b in range(0, min(a, n - a) + 1):
                margin_c[a - b] += corr[n, a, b]
                margin_n[a - b] += cnt[n, a, b]
    f = (margin_c + kappa * g) / (margin_n + kappa)
    for n in range(1, S):
        for a in range(1, n + 1):
            for b in range(0, min(a, n - a) + 1):
                pc[n, a, b] = (corr[n, a, b] + kappa * f[a - b]) / (cnt[n, a, b] + kappa)

    T = np.zeros_like(tc)
    for c in (0, 1):
        for n in range(1, S):
            for a in range(1, n + 1):
                for b in range(0, min(a, n - a) + 1):
                    T[c, n, a, b] = (tc[c, n, a, b] + kappa * _prior_outcomes(n, a, b)) / (
                        tc[c, n, a, b].sum() + kappa)

    beta = fit_judge_beta(judge, qids[train_idx] if qids is not None else train_idx)
    return BeliefModel(pc, T, cnt, beta)


class DPPolicy:
    def __init__(self, model: BeliefModel, costs: Costs, use_judge=True, allow_abstain=True, j_max=6):
        self.m, self.c = model, costs
        self.use_judge, self.allow_abstain, self.j_max = use_judge, allow_abstain, j_max
        (self.a0, self.b0), (self.a1, self.b1) = model.judge_beta
        self.value = lru_cache(maxsize=None)(self._value)
        self.__name__ = "DP" + ("+judge" if use_judge else "") + ("" if allow_abstain else " (no abstain)")

    def posterior(self, n, a, b, jp, jf) -> float:
        pi = self.m.pc[n, a, b]
        if jp == jf == 0:
            return pi
        l1 = betaln(self.a1 + jp, self.b1 + jf) - betaln(self.a1, self.b1)
        l0 = betaln(self.a0 + jp, self.b0 + jf) - betaln(self.a0, self.b0)
        return 1.0 / (1.0 + (1 - pi) / pi * np.exp(l0 - l1))

    def _value(self, n, a, b, jp, jf, nj):
        c = self.c
        pi = self.posterior(n, a, b, jp, jf)
        best = (pi - (1 - pi) * c.lam, SUBMIT)
        if self.allow_abstain and best[0] < 0:
            best = (0.0, ABSTAIN)
        checks_left = c.H - (n - 1) - nj
        if checks_left <= 0:
            return best

        # RESAMPLE
        ev = 0.0
        for corr, w in ((1, pi), (0, 1 - pi)):
            if w == 0:
                continue
            T = self.m.T[corr, n, a, b]
            nxt = {LEAD: (a + 1, b, False), RUNNER: (a, b + 1, False),
                   SWITCH: (a + 1, a, True), OTHER: (a, b, False)}
            for o, (a2, b2, sw) in nxt.items():
                if T[o] > 0:
                    j = (0, 0) if sw else (jp, jf)
                    ev += w * T[o] * self.value(n + 1, a2, b2, *j, nj)[0]
        q = ev - c.c_s
        if q > best[0]:
            best = (q, RESAMPLE)

        # JUDGE
        if self.use_judge and nj < self.j_max:
            pp = pi * (self.a1 + jp) / (self.a1 + self.b1 + jp + jf) + \
                 (1 - pi) * (self.a0 + jp) / (self.a0 + self.b0 + jp + jf)
            q = (pp * self.value(n, a, b, jp + 1, jf, nj + 1)[0]
                 + (1 - pp) * self.value(n, a, b, jp, jf + 1, nj + 1)[0] - c.c_j)
            if q > best[0]:
                best = (q, JUDGE)
        return best

    def __call__(self, obs):
        return self.value(obs["n"], obs["a"], obs["b"], obs["jp"], obs["jf"], obs["n_judge"])[1]
