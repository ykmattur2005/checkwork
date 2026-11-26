"""Download Monkey Business configs and compress them into answer-cluster arrays.

Run on your own machine (needs internet + `pip install datasets`):

    python -m data.prep --config GSM8K_Llama-3-8B-Instruct
    python -m data.prep --config MATH_Llama-3-8B-Instruct

Outputs:
    data/processed/<config>.npz        clusters (Q, N) int32, qids (Q,)
    data/processed/<config>_reps.jsonl representative sample text for the top answers
                                       of each question (input to judge_colab.py)
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent / "processed"

# ---------------------------------------------------------------- extraction
GSM_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
MATH_FINAL_RE = re.compile(r"[Ff]inal [Aa]nswer:\s*The final answer is\s*(.*?)\.?\s*I hope it is correct", re.S)


def _last_boxed(s: str) -> str | None:
    i = s.rfind("\\boxed")
    if i < 0:
        return None
    j = s.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(s)):
        if s[k] == "{":
            depth += 1
        elif s[k] == "}":
            depth -= 1
            if depth == 0:
                return s[j + 1 : k]
    return None


def extract_gsm8k(sample: str) -> str | None:
    # Only look at the first answer block: models sometimes keep generating new questions.
    m = GSM_RE.search(sample)
    if not m:
        return None
    return normalize_number(m.group(1))


def normalize_number(x: str) -> str | None:
    x = x.replace(",", "").rstrip(".").strip()
    try:
        f = float(x)
    except ValueError:
        return x or None
    return str(int(f)) if f.is_integer() else repr(f)


def extract_math(sample: str) -> str | None:
    m = MATH_FINAL_RE.search(sample)
    ans = m.group(1) if m else _last_boxed(sample)
    if ans is None:
        return None
    return normalize_math(ans)


def normalize_math(s: str) -> str | None:
    s = s.strip().strip("$").strip()
    b = _last_boxed(s)
    if b is not None:
        s = b
    for a, c in [("\\left", ""), ("\\right", ""), ("\\!", ""), ("\\,", ""), ("\\;", ""),
                 ("\\dfrac", "\\frac"), ("\\tfrac", "\\frac"), ("^\\circ", ""), ("^{\\circ}", ""),
                 ("\\%", ""), ("\\$", ""), ("\\text{ }", ""), (" ", "")]:
        s = s.replace(a, c)
    s = re.sub(r"\\text\{(.*?)\}", r"\1", s)
    s = s.rstrip(".")
    if re.fullmatch(r"-?[0-9\.,]+", s):
        return normalize_number(s)
    return s or None


EXTRACTORS = {"GSM8K": extract_gsm8k, "MATH": extract_math}


# ---------------------------------------------------------------- clustering
def cluster_question(samples: list[str], is_corrects: list[bool], extract) -> tuple[np.ndarray, list[int]]:
    """Return cluster ids per sample (0 = correct, 1.. = wrong answers by frequency, -1 = unparseable)
    and, for representatives, one sample index per cluster id (index into samples)."""
    wrong_keys = []
    for s, c in zip(samples, is_corrects):
        wrong_keys.append(None if c else extract(s))
    counts = Counter(k for k, c in zip(wrong_keys, is_corrects) if not c and k is not None)
    ranking = {k: i + 1 for i, (k, _) in enumerate(counts.most_common())}
    ids = np.empty(len(samples), dtype=np.int32)
    rep: dict[int, int] = {}
    for i, (k, c) in enumerate(zip(wrong_keys, is_corrects)):
        cid = 0 if c else (ranking[k] if k is not None else -1)
        ids[i] = cid
        if cid >= 0 and cid not in rep:
            rep[cid] = i
    return ids, rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="e.g. GSM8K_Llama-3-8B-Instruct")
    ap.add_argument("--max-samples", type=int, default=10_000)
    ap.add_argument("--top-k-reps", type=int, default=3, help="answers per question to send to the judge")
    args = ap.parse_args()

    from datasets import load_dataset  # imported here so the rest of the repo works without it

    task = args.config.split("_")[0]
    extract = EXTRACTORS[task]
    ds = load_dataset("ScalingIntelligence/monkey_business", args.config)["test"]

    OUT.mkdir(parents=True, exist_ok=True)
    all_ids, qids = [], []
    n_unparsed = 0
    with open(OUT / f"{args.config}_reps.jsonl", "w") as f:
        for qi, row in enumerate(ds):
            samples = row["samples"][: args.max_samples]
            corr = row["is_corrects"][: args.max_samples]
            ids, rep = cluster_question(samples, corr, extract)
            all_ids.append(ids)
            qids.append(qi)
            n_unparsed += int((ids == -1).sum())
            freq = Counter(ids[ids >= 0].tolist())
            for cid, _ in freq.most_common(args.top_k_reps):
                f.write(json.dumps({"qid": qi, "cluster": cid, "question": row["question"],
                                    "solution": samples[rep[cid]], "is_correct": cid == 0}) + "\n")
            print(f"q{qi:3d}  pass@1={np.mean(corr):.3f}  distinct wrong answers={ids.max()}", flush=True)

    clusters = np.stack(all_ids)
    np.savez_compressed(OUT / f"{args.config}.npz", clusters=clusters, qids=np.array(qids))
    print(f"\nsaved {clusters.shape} to {OUT / (args.config + '.npz')}")
    print(f"overall pass@1 = {(clusters == 0).mean():.3f}   unparseable wrong samples = {n_unparsed / clusters.size:.3%}")


if __name__ == "__main__":
    main()
