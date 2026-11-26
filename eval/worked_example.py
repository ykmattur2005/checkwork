"""Task 7: worked example on GSM8K question 74 (correct 240; most common wrong answer 300).

    python -m eval.worked_example

Uses the DP model of the CV fold in which q74 is a TEST question (fit only on the other 4 folds, with
the same seed as eval/final.py) and replays the first 3 test episodes (same CRN seeds as eval/final.py)
for DP (+judge, abstain) and fixed-k=9 majority at lambda=2, c_s=0.01, c_j=0.02.
Writes results/final/worked_example/q74.md and the policy-map figure.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eval.common import TEST, DATA, episode_rng, fold_assignment, load_data
from eval.figstyle import INK, legend_right, policy_map, policy_map_figure, save
from policies.baselines import fixed_k
from policies.dp import DPPolicy, fit_belief
from sim.env import JUDGE, RESAMPLE, Costs, Env

CFG, QID, SEED, LAM, JR, CS = "GSM8K_Llama-3-8B-Instruct", 74, 0, 2.0, 2.0, 0.01
OUT = Path("results/final/worked_example")


def answer_names():
    names = {}
    for line in open(DATA / f"{CFG}_reps.jsonl"):
        d = json.loads(line)
        if d["qid"] == QID:
            m = re.search(r"#### (\-?[0-9\.\,]+)", d["solution"])
            names[d["cluster"]] = m.group(1).replace(",", "") if m else f"cluster {d['cluster']}"
            question = d["question"]
    return names, question


def trace(env, pol, model, qi, ep, names, dp=None):
    env.rng = episode_rng(SEED, TEST, FOLD, qi, ep)
    obs = env.reset(qi)
    rows = []
    name = lambda c: names.get(c, "unparseable" if not isinstance(c, (int, np.integer)) else f"other wrong #{c}")
    first = name(env.tracker.leader)
    while True:
        n, a, b, jp, jf = obs["n"], obs["a"], obs["b"], obs["jp"], obs["jf"]
        belief = dp.posterior(n, a, b, jp, jf) if dp else model.pc[n, a, b]
        act = pol(obs)
        leader = name(env.tracker.leader)
        obs, r, done = env.step(act)
        if act == RESAMPLE:
            res = f"drew {name(int(env.clusters[qi, env.order[env.ptr - 1]]))}"
        elif act == JUDGE:
            res = "PASS" if obs["jp"] > jp else "FAIL"
        else:
            res = f"{env.outcome} (reward {r:+.2f} before costs)"
        rows.append(f"| {len(rows) + 1} | {n} | {a}-{b} | {leader} | {jp}/{jf} | {belief:.3f} | {act} | {res} |")
        if done:
            total = (1.0 if env.outcome == "correct" else -LAM if env.outcome == "wrong" else 0.0) - env.cost
            return first, rows, env.outcome, total, env.tracker.n - 1, env.n_judge


def main():
    global FOLD
    OUT.mkdir(parents=True, exist_ok=True)
    clusters, qids, judge, _ = load_data(CFG)
    fold_of = fold_assignment(len(qids), 5, SEED)
    qi = int(np.where(qids == QID)[0][0])
    FOLD = int(fold_of[qi])
    train = np.where(fold_of != FOLD)[0]
    assert qi not in train
    model = fit_belief(clusters, train, H=20, seed=SEED + FOLD, judge=judge, qids=qids)
    costs = Costs(c_s=CS, c_j=JR * CS, lam=LAM)
    dp = DPPolicy(model, costs, use_judge=True, allow_abstain=True)
    names, question = answer_names()
    row = clusters[qi]
    v, c = np.unique(row, return_counts=True)
    top = sorted(zip(c, v), reverse=True)[:4]
    jt = pd.read_csv(DATA / f"{CFG}_judge.csv").query("qid == @QID")

    L = [f"# Worked example: GSM8K question {QID}", "",
         f"**Question.** {question}", "",
         f"Correct answer {names[0]}; the most common wrong answer {names[1]} divides by the 4 roommates and "
         f"forgets Jenna herself.", "",
         "Answer frequencies over the 10,000 stored samples (data/processed/GSM8K_Llama-3-8B-Instruct.npz): " +
         ", ".join(f"{names.get(int(k), f'cluster {k}')}: {n / len(row):.1%}" for n, k in top) + ".", "",
         "3B judge verdicts on the representative solutions (data/processed/GSM8K_Llama-3-8B-Instruct_judge.csv): " +
         ", ".join(f"{names[int(r.cluster)]}: {r.n_pass}/{r.n_total} PASS" for r in jt.itertuples()) +
         ". The judge rejects the correct answer as often as the wrong ones, so its verdict carries no "
         "information on this question.", "",
         f"**Model.** Belief model of CV fold {FOLD}, fit on the {len(train)} questions of the other four folds "
         f"(q{QID} excluded). Costs: λ = {LAM:g}, c_s = {CS}, c_j = {JR * CS:g}. Judge Beta priors fit on the training "
         f"questions: wrong answers Beta({model.judge_beta[0][0]:.2f}, {model.judge_beta[0][1]:.2f}), "
         f"correct answers Beta({model.judge_beta[1][0]:.2f}, {model.judge_beta[1][1]:.2f}).", "",
         "Episodes are test episodes 0-2 of eval/final.py (same seeds), so both policies see the same sample "
         "order. Belief = the DP's P(leader correct) given votes and judge verdicts (for fixed-k it is shown "
         "for reference only; fixed-k does not use it). Votes column = a-b (leader-runner-up); judge column = "
         "PASS/FAIL verdicts on the current leader.", ""]
    for ep in range(3):
        for label, pol, use_dp in [("DP (+judge, abstain)", dp, dp), ("fixed-k = 9 majority", fixed_k(9), None)]:
            env = Env(clusters, qids, costs, judge)
            first, rows, outcome, total, nres, nj = trace(env, pol, model, qi, ep, names, use_dp)
            L += [f"### Episode {ep}: {label}", "",
                  f"First sample: {first}. Outcome: **{outcome}**, {nres} resamples, {nj} judge calls, "
                  f"episode reward {total:+.2f}.", "",
                  "| step | n | votes a-b | leader | judge P/F | belief | action | result |",
                  "|---|---|---|---|---|---|---|---|", *rows, ""]

    pq = pd.read_csv(f"results/final/{CFG}/lam2_jr2/per_question.csv").query("qid == @QID")
    sel = pq[pq.policy.isin(["DP (+judge, abstain)", "DP (resample, abstain)", "DP (resample)"]) |
             ((pq.policy == "grid:fixed_k") & (pq.param.astype(str) == "9"))]
    L += [f"### All 100 test episodes for q{QID} (results/final/{CFG}/lam2_jr2/per_question.csv)", "",
          "| policy | accuracy | wrong | abstain | compute | reward |", "|---|---|---|---|---|---|"]
    for r in sel.itertuples():
        nm = "fixed-k = 9" if r.policy.startswith("grid") else r.policy
        L.append(f"| {nm} | {r.accuracy:.2f} | {r.wrong_rate:.2f} | {r.abstain_rate:.2f} | {r.compute:.2f} | {r.reward:+.3f} |")

    # policy map for this model: no judge evidence vs. after one FAIL verdict
    policy_map_figure(dp, model, f"DP (+judge, abstain) policy of the GSM8K fold-{FOLD} model used for q{QID} "
                      f"(λ={LAM:g}, c_s={CS}, c_j={JR * CS:g}); cell text = P(leader correct)",
                      "worked_example_q74_policy_map")
    L += ["", f"Policy map: results/final/figures/worked_example_q74_policy_map.pdf "
          f"(top: no judge verdicts; bottom: after one FAIL verdict)."]

    # text version of the policy map (states reachable with n <= 6), for the write-up
    L += ["", "| n | a-b | P(correct) | action | P(correct) after 1 FAIL | action after 1 FAIL |", "|---|---|---|---|---|---|"]
    for n in range(1, 7):
        for a in range(1, n + 1):
            for b in range(0, min(a, n - a) + 1):
                if model.visits[n, a, b] == 0:
                    continue
                L.append(f"| {n} | {a}-{b} | {dp.posterior(n, a, b, 0, 0):.3f} | {dp.value(n, a, b, 0, 0, 0)[1]} | "
                         f"{dp.posterior(n, a, b, 0, 1):.3f} | {dp.value(n, a, b, 0, 1, 1)[1]} |")
    (OUT / "q74.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
