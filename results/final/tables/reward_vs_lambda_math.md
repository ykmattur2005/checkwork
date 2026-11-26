MATH: test reward (± SE over questions) and abstain rate by penalty λ (c_s=0.01, c_j/c_s=2).

Source: `results/final/MATH_Llama-3-8B-Instruct/lam*_jr2/headline.csv`

| policy | reward λ=1.2 | abstain λ=1.2 | reward λ=2 | abstain λ=2 | reward λ=5 | abstain λ=5 |
|---|---|---|---|---|---|---|
| first answer | -0.619 ± 0.060 | 0.00 | -1.208 ± 0.082 | 0.00 | -3.417 ± 0.164 | 0.00 |
| fixed-k (tuned) | -0.488 ± 0.078 | 0.00 | -0.994 ± 0.107 | 0.00 | -2.825 ± 0.225 | 0.00 |
| margin stop (tuned) | -0.445 ± 0.083 | 0.00 | -0.941 ± 0.113 | 0.00 | -2.762 ± 0.227 | 0.00 |
| margin stop+abstain (tuned) | 0.084 ± 0.032 | 0.79 | 0.038 ± 0.033 | 0.82 | 0.015 ± 0.023 | 0.90 |
| DP (resample) | -0.454 ± 0.082 | 0.00 | -0.948 ± 0.114 | 0.00 | -2.781 ± 0.227 | 0.00 |
| DP (+judge) | -0.453 ± 0.082 | 0.00 | -0.948 ± 0.113 | 0.00 | -2.780 ± 0.226 | 0.00 |
| DP (resample, abstain) | 0.101 ± 0.036 | 0.76 | 0.061 ± 0.034 | 0.82 | 0.046 ± 0.029 | 0.89 |
| DP (+judge, abstain) | 0.097 ± 0.036 | 0.76 | 0.062 ± 0.034 | 0.82 | 0.046 ± 0.028 | 0.89 |
