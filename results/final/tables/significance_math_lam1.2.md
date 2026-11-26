MATH, λ=1.2, c_j/c_s=2: paired per-question reward differences (policy minus vs). 95% percentile bootstrap CI over questions (10,000 resamples); two-sided Wilcoxon signed-rank p, Holm-adjusted within the table. 'significant': yes = CI excludes 0 and Holm p<0.05; 'mean only' / 'median only' = only one test rejects.

Source: `results/final/MATH_Llama-3-8B-Instruct/lam1.2_jr2/significance.csv`

| policy | vs | mean diff [95% CI] | median diff | Wilcoxon p | Holm p | q better/worse | significant |
|---|---|---|---|---|---|---|---|
| DP (resample) | fixed-k (tuned) | +0.035 [+0.017, +0.054] | +0.011 | 0.009 | 0.035 | 55%/45% | yes |
| DP (resample) | margin stop (tuned) | -0.008 [-0.021, +0.004] | +0.000 | 0.990 | 1.000 | 49%/49% | no |
| DP (resample) | margin stop+abstain (tuned) | -0.538 [-0.657, -0.419] | -0.735 | 1.3e-11 | 1.6e-10 | 34%/66% | yes |
| DP (+judge) | fixed-k (tuned) | +0.035 [+0.017, +0.055] | +0.011 | 0.007 | 0.035 | 55%/45% | yes |
| DP (+judge) | margin stop (tuned) | -0.008 [-0.020, +0.005] | +0.000 | 0.959 | 1.000 | 49%/49% | no |
| DP (+judge) | margin stop+abstain (tuned) | -0.537 [-0.657, -0.419] | -0.752 | 1.6e-11 | 1.7e-10 | 35%/65% | yes |
| DP (resample, abstain) | fixed-k (tuned) | +0.589 [+0.486, +0.694] | +0.728 | 1.5e-14 | 2.4e-13 | 77%/23% | yes |
| DP (resample, abstain) | margin stop (tuned) | +0.546 [+0.428, +0.666] | +0.675 | 4.6e-11 | 4.6e-10 | 65%/35% | yes |
| DP (resample, abstain) | margin stop+abstain (tuned) | +0.016 [+0.002, +0.030] | +0.018 | 3.6e-07 | 2.9e-06 | 76%/24% | yes |
| DP (+judge, abstain) | fixed-k (tuned) | +0.586 [+0.483, +0.691] | +0.735 | 1.7e-14 | 2.6e-13 | 77%/23% | yes |
| DP (+judge, abstain) | margin stop (tuned) | +0.543 [+0.424, +0.663] | +0.675 | 5.1e-11 | 4.6e-10 | 65%/35% | yes |
| DP (+judge, abstain) | margin stop+abstain (tuned) | +0.013 [-0.002, +0.027] | +0.018 | 1.8e-06 | 1.3e-05 | 76%/24% | median only |
| DP (+judge) | DP (resample) | +0.001 [-0.001, +0.002] | +0.000 | 0.216 | 0.647 | 9%/23% | no |
| DP (+judge, abstain) | DP (resample, abstain) | -0.003 [-0.007, +0.000] | +0.000 | 9.1e-05 | 5.4e-04 | 9%/47% | median only |
| DP (resample, abstain) | DP (resample) | +0.554 [+0.440, +0.670] | +0.705 | 7.6e-12 | 1.1e-10 | 66%/34% | yes |
| DP (+judge, abstain) | DP (+judge) | +0.550 [+0.437, +0.667] | +0.739 | 1.1e-11 | 1.4e-10 | 66%/34% | yes |
