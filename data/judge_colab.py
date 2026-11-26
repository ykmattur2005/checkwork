"""Run a small open LLM as a judge over the representative answers. Meant for Google Colab (GPU).

In Colab:
    !pip install -q vllm pandas
    # upload data/processed/<CONFIG>_reps.jsonl, then
    !python judge_colab.py --reps GSM8K_Llama-3-8B-Instruct_reps.jsonl --out GSM8K_Llama-3-8B-Instruct_judge.csv

Each (question, answer) is judged R times at temperature > 0 so that repeated JUDGE calls
give fresh, noisy observations in the simulator. If verdicts were deterministic, a second
judge call would carry no information and the policy could never learn to judge twice.
"""
from __future__ import annotations

import argparse
import json
import re

import pandas as pd

PROMPT = """You are grading a student's solution to a math problem.

Problem:
{question}

Student solution:
{solution}

Check the reasoning and the final answer carefully. Is the final answer correct?
Reply with a single word: CORRECT or INCORRECT."""


def parse_verdict(text: str) -> int | None:
    t = text.strip().upper()
    if re.search(r"\bINCORRECT\b", t):
        return 0
    if re.search(r"\bCORRECT\b", t):
        return 1
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--repeats", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-solution-chars", type=int, default=4000)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    rows = [json.loads(l) for l in open(args.reps)]
    llm = LLM(model=args.model, max_model_len=4096)
    tok = llm.get_tokenizer()
    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": PROMPT.format(question=r["question"],
                                                      solution=r["solution"][: args.max_solution_chars])}],
            tokenize=False, add_generation_prompt=True)
        for r in rows
    ]
    sp = SamplingParams(n=args.repeats, temperature=args.temperature, max_tokens=8)
    outs = llm.generate(prompts, sp)

    records = []
    for r, o in zip(rows, outs):
        v = [parse_verdict(c.text) for c in o.outputs]
        v = [x for x in v if x is not None]
        records.append({"qid": r["qid"], "cluster": r["cluster"], "n_pass": sum(v), "n_total": len(v),
                        "is_correct": r["is_correct"]})
    df = pd.DataFrame(records)
    df.to_csv(args.out, index=False)

    # Sanity check: judge error rates against the labels.
    d = df[df.n_total > 0]
    pr = d.n_pass / d.n_total
    print(f"false-pass rate (wrong answers judged CORRECT): {pr[~d.is_correct].mean():.3f}")
    print(f"false-fail rate (right answers judged INCORRECT): {1 - pr[d.is_correct].mean():.3f}")
    print(f"unparseable verdicts: {1 - d.n_total.sum() / (len(df) * args.repeats):.2%}")


if __name__ == "__main__":
    main()
