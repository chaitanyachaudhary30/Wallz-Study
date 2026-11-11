### DQN 5x5, seed 42, 500040 steps (planned 2M), larger-board match results

| Opponent | Games | Distinct | W / D / T / L % | Win % [95% Wilson] | Win % as P0 / P1 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| random | 100 | 100 | 98 / 2 / 0 / 0 | 98.0 [93.0, 99.5] | 98 / 98 |
| greedy | 100 | 100 | 66 / 28 / 0 / 6 | 66.0 [56.3, 74.5] | 72 / 60 |
| minimax:depth=2 | 100 | 100 | 73 / 10 / 0 / 17 | 73.0 [63.6, 80.7] | 74 / 72 |

Sources: final_dqn/5x5/final_dqn_5x5_s42_500k_eval.json, final_dqn/distinct_reanalysis.json. No exact solver on 5x5: no claim about optimal play.
