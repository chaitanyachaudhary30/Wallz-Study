### Exact ground truth (all non-terminal solved states)

| Model | States | Value-opt % | Rank-opt % | Sign agree % | 3-class value % | MAE | Legacy protocol |
|---|---:|---:|---:|---:|---:|---:|---|
| release 3x3 | 138646 | 78.0 | 75.5 | 82.9 | 65.4 | 0.4458 | sign 86.8% (old 96.5%) |
| final 3x3 s42 | 138646 | 96.0 | 91.1 | 97.7 | 89.5 | 0.1762 | sign 98.1% (old 96.5%) |
| final 3x3 s43 | 138646 | 96.2 | 93.3 | 97.9 | 89.7 | 0.1825 | sign 97.6% (old 96.5%) |
| final 3x3 s44 | 138646 | 96.1 | 92.0 | 97.8 | 89.6 | 0.1645 | sign 98.6% (old 96.5%) |
| final 3x3 s45 | 138646 | 96.5 | 93.2 | 97.7 | 86.4 | 0.18 | sign 98.4% (old 96.5%) |
| final 3x3 s46 | 138646 | 96.0 | 93.0 | 97.8 | 86.7 | 0.1882 | sign 98.6% (old 96.5%) |
| release 4x4 | 4350 | 79.1 | 65.8 | 84.5 | 67.7 | 0.4575 | value-opt 75.7% / rank-opt 64.3% (old 93.3 / 70.7) |
| final 4x4 s42 | 4350 | 96.3 | 91.0 | 97.5 | 94.1 | 0.0947 | value-opt 95.0% / rank-opt 89.0% (old 93.3 / 70.7) |

### Matches (opening_plies 2, both seatings, runner cap 200)

| Model | Opponent | Games | Win % | Draw % | Trunc % | Loss % | Mean plies | Win % as P0 | Win % as P1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| release 3x3 | random | 200 | 78.5 | 0.5 | 0.0 | 21.0 | 24.37 | 80.0 | 77.0 |
| release 3x3 | greedy | 200 | 24.0 | 16.5 | 0.0 | 59.5 | 8.44 | 29.0 | 19.0 |
| release 3x3 | minimax:depth=2 | 100 | 18.0 | 10.0 | 0.0 | 72.0 | 8.54 | 36.0 | 0.0 |
| release 3x3 | exact | 100 | 18.0 | 0.0 | 0.0 | 82.0 | 6.68 | 36.0 | 0.0 |
| final 3x3 s42 | random | 200 | 99.0 | 0.5 | 0.0 | 0.5 | 10.27 | 99.0 | 99.0 |
| final 3x3 s42 | greedy | 200 | 64.0 | 21.5 | 0.0 | 14.5 | 9.22 | 81.0 | 47.0 |
| final 3x3 s42 | minimax:depth=2 | 100 | 70.0 | 12.0 | 0.0 | 18.0 | 17.28 | 100.0 | 40.0 |
| final 3x3 s42 | exact | 100 | 28.0 | 44.0 | 0.0 | 28.0 | 22.66 | 56.0 | 0.0 |
| final 3x3 s43 | random | 200 | 99.0 | 0.0 | 0.0 | 1.0 | 10.01 | 100.0 | 98.0 |
| final 3x3 s43 | greedy | 200 | 64.0 | 21.5 | 0.0 | 14.5 | 9.64 | 81.0 | 47.0 |
| final 3x3 s43 | minimax:depth=2 | 100 | 60.0 | 22.0 | 0.0 | 18.0 | 20.22 | 80.0 | 40.0 |
| final 3x3 s43 | exact | 100 | 28.0 | 44.0 | 0.0 | 28.0 | 11.97 | 56.0 | 0.0 |
| final 3x3 s44 | random | 200 | 99.5 | 0.0 | 0.0 | 0.5 | 12.46 | 100.0 | 99.0 |
| final 3x3 s44 | greedy | 200 | 64.0 | 21.5 | 0.0 | 14.5 | 9.11 | 81.0 | 47.0 |
| final 3x3 s44 | minimax:depth=2 | 100 | 60.0 | 18.0 | 4.0 | 18.0 | 38.08 | 84.0 | 36.0 |
| final 3x3 s44 | exact | 100 | 28.0 | 44.0 | 0.0 | 28.0 | 16.5 | 56.0 | 0.0 |
| final 3x3 s45 | random | 200 | 99.0 | 0.0 | 0.0 | 1.0 | 10.26 | 100.0 | 98.0 |
| final 3x3 s45 | greedy | 200 | 76.0 | 9.5 | 0.0 | 14.5 | 6.83 | 81.0 | 71.0 |
| final 3x3 s45 | minimax:depth=2 | 100 | 60.0 | 22.0 | 0.0 | 18.0 | 21.6 | 80.0 | 40.0 |
| final 3x3 s45 | exact | 100 | 28.0 | 44.0 | 0.0 | 28.0 | 26.74 | 56.0 | 0.0 |
| final 3x3 s46 | random | 200 | 98.5 | 0.5 | 0.0 | 1.0 | 10.04 | 99.0 | 98.0 |
| final 3x3 s46 | greedy | 200 | 64.0 | 21.5 | 0.0 | 14.5 | 10.38 | 81.0 | 47.0 |
| final 3x3 s46 | minimax:depth=2 | 100 | 72.0 | 0.0 | 0.0 | 28.0 | 27.9 | 80.0 | 64.0 |
| final 3x3 s46 | exact | 100 | 28.0 | 44.0 | 0.0 | 28.0 | 13.52 | 56.0 | 0.0 |
| release 4x4 | random | 200 | 53.5 | 21.0 | 0.0 | 25.5 | 27.2 | 37.0 | 70.0 |
| release 4x4 | greedy | 200 | 0.0 | 6.5 | 0.0 | 93.5 | 9.03 | 0.0 | 0.0 |
| release 4x4 | minimax:depth=2 | 100 | 0.0 | 13.0 | 0.0 | 87.0 | 10.27 | 0.0 | 0.0 |
| release 4x4 | exact | 100 | 0.0 | 0.0 | 0.0 | 100.0 | 8.6 | 0.0 | 0.0 |
| final 4x4 s42 | random | 200 | 95.0 | 1.0 | 0.0 | 4.0 | 9.45 | 90.0 | 100.0 |
| final 4x4 s42 | greedy | 200 | 50.0 | 22.0 | 0.0 | 28.0 | 11.99 | 44.0 | 56.0 |
| final 4x4 s42 | minimax:depth=2 | 100 | 39.0 | 21.0 | 0.0 | 40.0 | 13.78 | 20.0 | 58.0 |
| final 4x4 s42 | exact | 100 | 29.0 | 27.0 | 0.0 | 44.0 | 14.04 | 0.0 | 58.0 |

### 3x3, five seeds (aggregate_3x3_seeds.json)

| Metric | Mean % | Min % | Max % |
|---|---:|---:|---:|
| value_optimal_rate | 96.2 | 96.0 | 96.5 |
| rank_optimal_rate | 92.5 | 91.1 | 93.3 |
| sign_agreement | 97.8 | 97.7 | 97.9 |
| legacy_sign_agreement | 98.2 | 97.6 | 98.6 |
| greedy_win_rate | 66.4 | 64.0 | 76.0 |
| minimax:depth=2_win_rate | 64.4 | 60.0 | 72.0 |
| exact_win_rate | 28.0 | 28.0 | 28.0 |
| random_win_rate | 99.0 | 98.5 | 99.5 |
