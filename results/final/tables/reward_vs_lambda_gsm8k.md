GSM8K: test reward (± SE over questions) and abstain rate by penalty λ (c_s=0.01, c_j/c_s=2).

Source: `results/final/GSM8K_Llama-3-8B-Instruct/lam*_jr2/headline.csv`

| policy | reward λ=1.2 | abstain λ=1.2 | reward λ=2 | abstain λ=2 | reward λ=5 | abstain λ=5 |
|---|---|---|---|---|---|---|
| first answer | 0.497 ± 0.057 | 0.00 | 0.314 ± 0.078 | 0.00 | -0.373 ± 0.156 | 0.00 |
| fixed-k (tuned) | 0.609 ± 0.058 | 0.00 | 0.487 ± 0.080 | 0.00 | 0.037 ± 0.160 | 0.00 |
| margin stop (tuned) | 0.662 ± 0.059 | 0.00 | 0.547 ± 0.081 | 0.00 | 0.131 ± 0.162 | 0.00 |
| margin stop+abstain (tuned) | 0.671 ± 0.053 | 0.09 | 0.601 ± 0.064 | 0.14 | 0.478 ± 0.071 | 0.30 |
| DP (resample) | 0.658 ± 0.059 | 0.00 | 0.537 ± 0.082 | 0.00 | 0.118 ± 0.162 | 0.00 |
| DP (+judge) | 0.657 ± 0.060 | 0.00 | 0.537 ± 0.082 | 0.00 | 0.121 ± 0.162 | 0.00 |
| DP (resample, abstain) | 0.665 ± 0.055 | 0.08 | 0.605 ± 0.059 | 0.19 | 0.502 ± 0.072 | 0.28 |
| DP (+judge, abstain) | 0.661 ± 0.054 | 0.09 | 0.600 ± 0.061 | 0.18 | 0.502 ± 0.073 | 0.28 |
