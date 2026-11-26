Calibration of the belief model on held-out folds (λ=2, c_j/c_s=2). Trajectory states: every attempt n of 100 test episodes per question; DP-visited: decision states of DP (+judge, abstain); probe: k simulated judge verdicts on the leader at every trajectory state. ECE uses 10 equal-width bins; Brier (base rate) = Brier of always predicting the mean.

Source: `results/final/calibration/metrics.csv`

| config | belief | n_states | P(correct) | ECE | Brier | Brier (base rate) |
|---|---|---|---|---|---|---|
| gsm8k | vote-only pc[n,a,b], all trajectory states n=1..21 | 266,700 | 0.846 | 0.014 | 0.094 | 0.130 |
| gsm8k | vote-only pc[n,a,b], trajectory states n<=10 | 127,000 | 0.828 | 0.009 | 0.101 | 0.142 |
| gsm8k | vote-only pc[n,a,b], DP-visited states | 68,479 | 0.725 | 0.008 | 0.159 | 0.199 |
| gsm8k | posterior (votes+judge), DP-visited states | 68,479 | 0.725 | 0.009 | 0.159 | 0.199 |
| gsm8k | vote-only pc[n,a,b], DP-visited states with >=1 verdict | 5,232 | 0.634 | 0.052 | 0.229 | 0.232 |
| gsm8k | posterior (votes+judge), DP-visited states with >=1 verdict | 5,232 | 0.634 | 0.033 | 0.229 | 0.232 |
| gsm8k | posterior after 1 probe verdict, all trajectory states | 266,700 | 0.846 | 0.019 | 0.094 | 0.130 |
| gsm8k | posterior after 2 probe verdicts, all trajectory states | 266,700 | 0.846 | 0.024 | 0.095 | 0.130 |
| gsm8k | posterior after 3 probe verdicts, all trajectory states | 266,700 | 0.846 | 0.028 | 0.096 | 0.130 |
| math | vote-only pc[n,a,b], all trajectory states n=1..21 | 268,800 | 0.356 | 0.010 | 0.153 | 0.229 |
| math | vote-only pc[n,a,b], trajectory states n<=10 | 128,000 | 0.323 | 0.008 | 0.147 | 0.219 |
| math | vote-only pc[n,a,b], DP-visited states | 61,871 | 0.422 | 0.017 | 0.173 | 0.244 |
| math | posterior (votes+judge), DP-visited states | 61,871 | 0.422 | 0.017 | 0.173 | 0.244 |
| math | vote-only pc[n,a,b], DP-visited states with >=1 verdict | 621 | 0.565 | 0.043 | 0.244 | 0.246 |
| math | posterior (votes+judge), DP-visited states with >=1 verdict | 621 | 0.565 | 0.044 | 0.242 | 0.246 |
| math | posterior after 1 probe verdict, all trajectory states | 268,800 | 0.356 | 0.014 | 0.152 | 0.229 |
| math | posterior after 2 probe verdicts, all trajectory states | 268,800 | 0.356 | 0.019 | 0.153 | 0.229 |
| math | posterior after 3 probe verdicts, all trajectory states | 268,800 | 0.356 | 0.024 | 0.154 | 0.229 |
