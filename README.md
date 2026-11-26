# When Should an LLM Agent Check Its Work?

AA228/CS238 final project. An agent answering a math question can **resample** (draw another answer), **judge** (ask a second model whether the leading answer is correct), **submit**, or **abstain**. We model this as a POMDP that trades accuracy against compute, and solve it exactly by dynamic programming over vote and verdict counts.

## Method

- **Data.** Stored samples from the Monkey Business dataset (Llama-3-8B-Instruct, 10,000 samples per question) on GSM8K (127 questions) and MATH (128). Resampling replays these samples, so no model runs during evaluation.
- **Judge.** Qwen2.5-3B-Instruct, 8 verdicts per answer for each question's three most common answers.
- **Belief model.** P(leading answer correct | votes) estimated from training questions, with a Beta-Binomial model of judge verdicts.
- **Policy.** Exact backward induction over the count state (attempts, leader votes, runner-up votes, judge verdicts).
- **Evaluation.** 5-fold cross-validation over questions. Baselines (fixed-k voting, margin stopping, margin stopping with abstention) are tuned on training folds. Policies are compared with paired bootstrap confidence intervals and Wilcoxon signed-rank tests.

## Layout

```
data/      dataset download and preprocessing, judge script
sim/       simulator
policies/  baselines and dynamic-programming policy
eval/      experiments, statistics, figures
tests/     unit tests
colab/     notebook for an optional 7B judge run
```

## Reproduce

```bash
pip install numpy pandas scipy matplotlib pytest datasets
python -m data.prep --config GSM8K_Llama-3-8B-Instruct   # skip if data/processed/ is present
python -m data.prep --config MATH_Llama-3-8B-Instruct
bash run_all.sh                                           # writes results/final/
python -m pytest -q
```
