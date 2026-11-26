3B judge (Qwen2.5-3B-Instruct, 8 verdicts per answer) by contestedness tercile (vote share of each question's most common answer over 10k samples). Verdict-weighted rates, 95% bootstrap CIs over questions.

Source: `results/final/judge_analysis/by_tercile.csv`

| dataset | tercile | questions | top share | answers (correct/wrong) | accuracy | false-pass | false-fail | balanced acc. |
|---|---|---|---|---|---|---|---|---|
| GSM8K | all | 127 | 0.12–1.00 | 124/253 | 0.65 [0.60, 0.69] | 0.26 [0.20, 0.31] | 0.55 [0.47, 0.63] | 0.60 [0.55, 0.64] |
| GSM8K | most contested | 43 | 0.12–0.79 | 40/89 | 0.64 [0.57, 0.71] | 0.22 [0.14, 0.31] | 0.66 [0.52, 0.78] | 0.56 [0.49, 0.63] |
| GSM8K | middle | 42 | 0.79–0.97 | 42/84 | 0.57 [0.49, 0.65] | 0.34 [0.23, 0.45] | 0.61 [0.46, 0.75] | 0.53 [0.45, 0.61] |
| GSM8K | least contested | 42 | 0.97–1.00 | 42/80 | 0.72 [0.65, 0.79] | 0.22 [0.14, 0.31] | 0.38 [0.26, 0.52] | 0.70 [0.62, 0.77] |
| MATH | all | 128 | 0.02–0.98 | 78/306 | 0.74 [0.70, 0.78] | 0.14 [0.10, 0.18] | 0.74 [0.63, 0.82] | 0.56 [0.52, 0.62] |
| MATH | most contested | 43 | 0.02–0.19 | 13/116 | 0.83 [0.77, 0.89] | 0.11 [0.05, 0.17] | 0.70 [0.44, 0.93] | 0.59 [0.48, 0.73] |
| MATH | middle | 42 | 0.20–0.40 | 24/102 | 0.72 [0.65, 0.79] | 0.13 [0.07, 0.20] | 0.91 [0.79, 0.99] | 0.48 [0.43, 0.54] |
| MATH | least contested | 43 | 0.42–0.98 | 41/88 | 0.67 [0.61, 0.73] | 0.19 [0.11, 0.27] | 0.65 [0.50, 0.79] | 0.58 [0.52, 0.65] |
