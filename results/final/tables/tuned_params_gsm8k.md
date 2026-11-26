GSM8K: baseline parameters chosen on each fold's training questions (c_j/c_s=2; baselines do not use the judge, so the choice is the same for every c_j).

Source: `results/final/GSM8K_Llama-3-8B-Instruct/lam*_jr2/tuned_params.csv`

| λ | fold | fixed-k | margin stop | margin stop+abstain |
|---|---|---|---|---|
| 1.2 | 0 | 7 | 2 | (2, 10) |
| 1.2 | 1 | 7 | 2 | (3, 10) |
| 1.2 | 2 | 7 | 3 | (3, 10) |
| 1.2 | 3 | 5 | 2 | (3, 10) |
| 1.2 | 4 | 5 | 2 | (3, 10) |
| 2 | 0 | 9 | 3 | (3, 10) |
| 2 | 1 | 7 | 2 | (4, 10) |
| 2 | 2 | 7 | 3 | (3, 10) |
| 2 | 3 | 5 | 2 | (4, 10) |
| 2 | 4 | 7 | 2 | (4, 10) |
| 5 | 0 | 9 | 3 | (6, 10) |
| 5 | 1 | 9 | 3 | (7, 10) |
| 5 | 2 | 13 | 3 | (6, 10) |
| 5 | 3 | 13 | 3 | (7, 10) |
| 5 | 4 | 7 | 3 | (7, 10) |
