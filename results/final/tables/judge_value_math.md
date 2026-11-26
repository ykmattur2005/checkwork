MATH, λ=2: value of adding the judge to the DP at three judge costs (same tests as the significance tables; Holm adjustment within each c_j run).

Source: `results/final/MATH_Llama-3-8B-Instruct/lam2_jr*/significance.csv`

| c_j/c_s | policy | vs | mean diff [95% CI] | median diff | Wilcoxon p | Holm p | q better/worse | significant |
|---|---|---|---|---|---|---|---|---|
| 0.5 | DP (+judge) | DP (resample) | +0.002 [-0.011, +0.015] | +0.000 | 0.990 | 1.000 | 36%/45% | no |
| 0.5 | DP (+judge, abstain) | DP (resample, abstain) | +0.010 [-0.003, +0.025] | -0.000 | 0.507 | 1.000 | 33%/56% | no |
| 1 | DP (+judge) | DP (resample) | +0.002 [-0.005, +0.010] | +0.000 | 0.068 | 0.364 | 18%/44% | no |
| 1 | DP (+judge, abstain) | DP (resample, abstain) | +0.008 [-0.003, +0.019] | -0.000 | 0.481 | 0.481 | 30%/52% | no |
| 2 | DP (+judge) | DP (resample) | +0.000 [-0.003, +0.004] | +0.000 | 0.078 | 0.403 | 7%/23% | no |
| 2 | DP (+judge, abstain) | DP (resample, abstain) | +0.001 [-0.003, +0.004] | +0.000 | 0.067 | 0.403 | 11%/34% | no |
