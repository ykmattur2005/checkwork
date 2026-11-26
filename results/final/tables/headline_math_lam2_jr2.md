MATH, λ=2, c_j/c_s=2, c_s=0.01: test-set means over questions ± standard error (n=128 questions, 100 episodes each). param = chosen parameter per fold (tuned) or c_s (DP); fixed-k param is k, margin params are m or (m, k_max). Compute in resample-equivalents.

Source: `results/final/MATH_Llama-3-8B-Instruct/lam2_jr2/headline.csv`

| policy | param | accuracy | wrong | abstain | compute | reward |
|---|---|---|---|---|---|---|
| first answer | 1 | 0.264 ± 0.027 | 0.736 ± 0.027 | 0.000 ± 0.000 | 0.00 ± 0.00 | -1.208 ± 0.082 |
| fixed-k (tuned) | 13/9 | 0.370 ± 0.036 | 0.630 ± 0.036 | 0.000 ± 0.000 | 10.41 ± 0.17 | -0.994 ± 0.107 |
| margin stop (tuned) | 2/3 | 0.387 ± 0.037 | 0.613 ± 0.037 | 0.000 ± 0.000 | 10.35 ± 0.52 | -0.941 ± 0.113 |
| margin stop+abstain (tuned) | (3, 5)/(4, 5) | 0.148 ± 0.025 | 0.036 ± 0.009 | 0.815 ± 0.025 | 3.75 ± 0.04 | 0.038 ± 0.033 |
| fixed-k (oracle-tuned on test) | 13 | 0.378 ± 0.036 | 0.622 ± 0.036 | 0.000 ± 0.000 | 12.00 ± 0.00 | -0.987 ± 0.109 |
| margin stop (oracle-tuned on test) | 2 | 0.387 ± 0.036 | 0.613 ± 0.036 | 0.000 ± 0.000 | 9.62 ± 0.51 | -0.936 ± 0.112 |
| margin stop+abstain (oracle-tuned on test) | (3, 5) | 0.169 ± 0.028 | 0.040 ± 0.009 | 0.791 ± 0.027 | 3.71 ± 0.04 | 0.052 ± 0.036 |
| DP (resample) | 0.01 | 0.390 ± 0.037 | 0.610 ± 0.037 | 0.000 ± 0.000 | 11.82 ± 0.36 | -0.948 ± 0.114 |
| DP (+judge) | 0.01 | 0.390 ± 0.037 | 0.610 ± 0.037 | 0.000 ± 0.000 | 11.85 ± 0.36 | -0.948 ± 0.113 |
| DP (resample, abstain) | 0.01 | 0.152 ± 0.027 | 0.026 ± 0.009 | 0.821 ± 0.028 | 3.80 ± 0.20 | 0.061 ± 0.034 |
| DP (+judge, abstain) | 0.01 | 0.151 ± 0.027 | 0.025 ± 0.009 | 0.824 ± 0.028 | 3.87 ± 0.20 | 0.062 ± 0.034 |
