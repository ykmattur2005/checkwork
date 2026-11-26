MATH, λ=1.2, c_j/c_s=2, c_s=0.01: test-set means over questions ± standard error (n=128 questions, 100 episodes each). param = chosen parameter per fold (tuned) or c_s (DP); fixed-k param is k, margin params are m or (m, k_max). Compute in resample-equivalents.

Source: `results/final/MATH_Llama-3-8B-Instruct/lam1.2_jr2/headline.csv`

| policy | param | accuracy | wrong | abstain | compute | reward |
|---|---|---|---|---|---|---|
| first answer | 1 | 0.264 ± 0.027 | 0.736 ± 0.027 | 0.000 ± 0.000 | 0.00 ± 0.00 | -0.619 ± 0.060 |
| fixed-k (tuned) | 13/9 | 0.364 ± 0.035 | 0.636 ± 0.035 | 0.000 ± 0.000 | 8.81 ± 0.14 | -0.488 ± 0.078 |
| margin stop (tuned) | 2 | 0.387 ± 0.036 | 0.613 ± 0.036 | 0.000 ± 0.000 | 9.62 ± 0.51 | -0.445 ± 0.083 |
| margin stop+abstain (tuned) | (3, 5) | 0.169 ± 0.028 | 0.040 ± 0.009 | 0.791 ± 0.027 | 3.71 ± 0.04 | 0.084 ± 0.032 |
| fixed-k (oracle-tuned on test) | 9 | 0.363 ± 0.035 | 0.637 ± 0.035 | 0.000 ± 0.000 | 8.00 ± 0.00 | -0.482 ± 0.077 |
| margin stop (oracle-tuned on test) | 2 | 0.387 ± 0.036 | 0.613 ± 0.036 | 0.000 ± 0.000 | 9.62 ± 0.51 | -0.445 ± 0.083 |
| margin stop+abstain (oracle-tuned on test) | (3, 5) | 0.169 ± 0.028 | 0.040 ± 0.009 | 0.791 ± 0.027 | 3.71 ± 0.04 | 0.084 ± 0.032 |
| DP (resample) | 0.01 | 0.380 ± 0.036 | 0.620 ± 0.036 | 0.000 ± 0.000 | 9.04 ± 0.37 | -0.454 ± 0.082 |
| DP (+judge) | 0.01 | 0.381 ± 0.036 | 0.619 ± 0.036 | 0.000 ± 0.000 | 9.07 ± 0.37 | -0.453 ± 0.082 |
| DP (resample, abstain) | 0.01 | 0.194 ± 0.031 | 0.046 ± 0.012 | 0.760 ± 0.031 | 3.90 ± 0.16 | 0.101 ± 0.036 |
| DP (+judge, abstain) | 0.01 | 0.195 ± 0.031 | 0.048 ± 0.012 | 0.757 ± 0.031 | 3.98 ± 0.17 | 0.097 ± 0.036 |
