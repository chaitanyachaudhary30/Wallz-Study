### Minimax vs Greedy (final engine)

| Run | Minimax spec | N | Opening plies | Games | Win rate (95% Wilson) | Draw rate | Truncation rate | Mean plies |
| :--- | :--- | :---: | :---: | ---: | :---: | ---: | ---: | ---: |
| `final_minimax/vs_greedy_N5_depth1_diag_summary` | `minimax:depth=1` | 5 | 2 | 20 | 50.0% [29.9%, 70.1%] | 10.0% | 0.0% | 16.5 |
| `final_minimax/vs_greedy_N5_depth3_diag_summary` | `minimax:depth=3` | 5 | 2 | 20 | 25.0% [11.2%, 46.9%] | 0.0% | 0.0% | 12.6 |
| `final_minimax/vs_greedy_N5_open4_depth1_diag_summary` | `minimax:depth=1` | 5 | 4 | 100 | 45.0% [35.6%, 54.8%] | 9.0% | 0.0% | 21.5 |
| `final_minimax/vs_greedy_N5_open4_depth3_diag_summary` | `minimax:depth=3` | 5 | 4 | 100 | 45.0% [35.6%, 54.8%] | 7.0% | 0.0% | 20.7 |
| `final_minimax/vs_greedy_N5_open4_distinct_summary` | `minimax:depth=2` | 5 | 4 | 100 | 39.0% [30.0%, 48.8%] | 1.0% | 0.0% | 18.5 |
| `final_minimax/vs_greedy_N5_summary` | `minimax:depth=2` | 5 | 2 | 100 | 35.0% [26.4%, 44.7%] | 0.0% | 0.0% | 11.6 |
| `final_minimax/vs_greedy_N9_timed_r2_summary` | `minimax:depth=2,time_limit=1.0` | 9 | 2 | 20 | 70.0% [48.1%, 85.5%] | 0.0% | 0.0% | 21.9 |
| `final_minimax/vs_greedy_N9_timed_summary` | `minimax:depth=2,time_limit=1.0` | 9 | 2 | 20 | 70.0% [48.1%, 85.5%] | 0.0% | 0.0% | 21.9 |
| `final_minimax_hop/vs_greedy_N5_default_repro_summary` | `minimax:depth=2` | 5 | 4 | 100 | 39.0% [30.0%, 48.8%] | 1.0% | 0.0% | 18.5 |
| `final_minimax_hop/vs_greedy_N5_hop_d1_summary` | `minimax:depth=1,eval=hop` | 5 | 4 | 100 | 50.0% [40.4%, 59.6%] | 0.0% | 0.0% | 12.6 |
| `final_minimax_hop/vs_greedy_N5_hop_d2_summary` | `minimax:depth=2,eval=hop` | 5 | 4 | 100 | 50.0% [40.4%, 59.6%] | 2.0% | 0.0% | 13.4 |
| `final_minimax_hop/vs_greedy_N5_hop_d3_summary` | `minimax:depth=3,eval=hop` | 5 | 4 | 100 | 50.0% [40.4%, 59.6%] | 4.0% | 0.0% | 13.3 |
| `final_minimax_hop/vs_greedy_N9_hop_timed_summary` | `minimax:depth=2,time_limit=1.0,eval=hop` | 9 | 2 | 20 | 50.0% [29.9%, 70.1%] | 0.0% | 0.0% | 17.8 |
| `final_minimax_hop/vs_minimax_N5_hop_d2_vs_default_d2_summary` | `minimax:depth=2,eval=hop` | 5 | 4 | 100 | 54.0% [44.3%, 63.4%] | 2.0% | 0.0% | 15.4 |
