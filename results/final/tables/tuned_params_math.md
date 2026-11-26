MATH: baseline parameters chosen on each fold's training questions (c_j/c_s=2; baselines do not use the judge, so the choice is the same for every c_j).

Source: `results/final/MATH_Llama-3-8B-Instruct/lam*_jr2/tuned_params.csv`

| λ | fold | fixed-k | margin stop | margin stop+abstain |
|---|---|---|---|---|
| 1.2 | 0 | 9 | 2 | (3, 5) |
| 1.2 | 1 | 9 | 2 | (3, 5) |
| 1.2 | 2 | 13 | 2 | (3, 5) |
| 1.2 | 3 | 9 | 2 | (3, 5) |
| 1.2 | 4 | 9 | 2 | (3, 5) |
| 2 | 0 | 9 | 2 | (3, 5) |
| 2 | 1 | 13 | 2 | (4, 5) |
| 2 | 2 | 13 | 3 | (3, 5) |
| 2 | 3 | 13 | 2 | (3, 5) |
| 2 | 4 | 9 | 2 | (3, 5) |
| 5 | 0 | 17 | 3 | (5, 5) |
| 5 | 1 | 21 | 2 | (5, 5) |
| 5 | 2 | 21 | 3 | (5, 5) |
| 5 | 3 | 21 | 3 | (5, 5) |
| 5 | 4 | 17 | 3 | (5, 5) |
