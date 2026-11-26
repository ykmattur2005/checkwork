GSM8K, λ=2: value of adding the judge to the DP at three judge costs (same tests as the significance tables; Holm adjustment within each c_j run).

Source: `results/final/GSM8K_Llama-3-8B-Instruct/lam2_jr*/significance.csv`

| c_j/c_s | policy | vs | mean diff [95% CI] | median diff | Wilcoxon p | Holm p | q better/worse | significant |
|---|---|---|---|---|---|---|---|---|
| 0.5 | DP (+judge) | DP (resample) | +0.002 [-0.004, +0.008] | +0.000 | 0.936 | 1.000 | 31%/35% | no |
| 0.5 | DP (+judge, abstain) | DP (resample, abstain) | +0.002 [-0.024, +0.027] | +0.000 | 0.904 | 1.000 | 45%/40% | no |
| 1 | DP (+judge) | DP (resample) | +0.002 [-0.002, +0.007] | +0.000 | 0.795 | 1.000 | 15%/22% | no |
| 1 | DP (+judge, abstain) | DP (resample, abstain) | -0.001 [-0.025, +0.021] | +0.000 | 0.647 | 1.000 | 39%/46% | no |
| 2 | DP (+judge) | DP (resample) | -0.000 [-0.003, +0.003] | +0.000 | 0.447 | 1.000 | 10%/17% | no |
| 2 | DP (+judge, abstain) | DP (resample, abstain) | -0.005 [-0.023, +0.010] | +0.000 | 0.911 | 1.000 | 24%/31% | no |
