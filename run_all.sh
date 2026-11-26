#!/usr/bin/env bash
# Reproduces every number, table and figure in results/final/ (the experiments section).
#   conda activate checkwork && bash run_all.sh            # everything (~15-20 min on 8+ cores)
#   bash run_all.sh experiments|analysis|report|test       # one stage
# Inputs: data/processed/ (never modified). Outputs: results/final/.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
STAGE=${1:-all}
CONFIGS=(GSM8K_Llama-3-8B-Instruct MATH_Llama-3-8B-Instruct)

if [[ $STAGE == all || $STAGE == experiments ]]; then
  for CFG in "${CONFIGS[@]}"; do
    # penalty grid at judge_ratio 2 (lambda=2 also gets the DP cost sweep and calibration logs)
    $PY -m eval.final --config "$CFG" --lam 1.2 --judge-ratio 2
    $PY -m eval.final --config "$CFG" --lam 2   --judge-ratio 2 --sweep --calibration
    $PY -m eval.final --config "$CFG" --lam 5   --judge-ratio 2
    # judge-cost grid at lambda=2
    $PY -m eval.final --config "$CFG" --lam 2   --judge-ratio 0.5 --sweep
    $PY -m eval.final --config "$CFG" --lam 2   --judge-ratio 1   --sweep
    # stronger judge (Qwen2.5-7B-Instruct-AWQ), only if colab/judge_7b.ipynb has been run
    if [[ -f "data/processed/${CFG}_judge7b.csv" ]]; then
      for JR in 0.5 1 2; do
        $PY -m eval.final --config "$CFG" --lam 2 --judge-ratio $JR --sweep \
            --judge-file "${CFG}_judge7b.csv" --tag-suffix _judge7b
      done
    else
      echo "NOTE: data/processed/${CFG}_judge7b.csv missing; skipping 7B-judge runs (see colab/judge_7b.ipynb)"
    fi
  done
fi

if [[ $STAGE == all || $STAGE == analysis ]]; then
  $PY -m eval.judge_analysis            # task 5: judge accuracy by contestedness, likelihood ratios
  if ls data/processed/*_judge7b.csv >/dev/null 2>&1; then
    $PY -m eval.judge_analysis --judge-suffix _judge7b
  fi
  $PY -m eval.worked_example            # task 7: GSM8K q74 traces + policy map
fi

if [[ $STAGE == all || $STAGE == report ]]; then
  $PY -m eval.report                    # tables (md + tex), calibration metrics, all final figures
  $PY -m eval.build_doc                 # results/final/EXPERIMENTS.md from eval/experiments_template.md
fi

if [[ $STAGE == all || $STAGE == test ]]; then
  $PY -m pytest -q
fi
