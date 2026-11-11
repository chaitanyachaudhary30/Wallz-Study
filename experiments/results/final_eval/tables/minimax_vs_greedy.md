### Minimax vs Greedy, distinct games (final engine)

| Run | Role | Spec | N | Opening plies | Matches | Distinct games | Minimax W-L-D-T | Win % [95% Wilson] | Mean plies |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `final_minimax/N5` | earlier design: 2-ply openings, 9 distinct (superseded by N5_open4_distinct) | `minimax:depth=2` | 5 | 2 | 100 | 18 | 7-11-0-0 | 38.9 [20.3, 61.4] | 11.2 |
| `final_minimax/N5_open4_distinct` | main (Member 1) | `minimax:depth=2` | 5 | 4 | 100 | 100 | 39-60-1-0 | 39.0 [30.0, 48.8] | 18.5 |
| `final_minimax/N9_timed` | larger board, timed, load-affected | `minimax:depth=2,time_limit=1.0` | 9 | 2 | 20 | 16 | 11-5-0-0 | 68.8 [44.4, 85.8] | 21.6 |
| `final_minimax/N5_depth1_diag` | diagnostic | `minimax:depth=1` | 5 | 2 | 20 | 16 | 8-7-1-0 | 50.0 [28.0, 72.0] | 14.0 |
| `final_minimax/N5_depth3_diag` | diagnostic | `minimax:depth=3` | 5 | 2 | 20 | 16 | 4-12-0-0 | 25.0 [10.2, 49.5] | 12.4 |
| `final_minimax/N5_open4_depth1_diag` | diagnostic | `minimax:depth=1` | 5 | 4 | 100 | 100 | 45-46-9-0 | 45.0 [35.6, 54.8] | 21.5 |
| `final_minimax/N5_open4_depth3_diag` | diagnostic | `minimax:depth=3` | 5 | 4 | 100 | 100 | 45-48-7-0 | 45.0 [35.6, 54.8] | 20.7 |
| `final_minimax_hop/N5_hop_d1` | new contribution (hop scorer) | `minimax:depth=1,eval=hop` | 5 | 4 | 100 | 100 | 50-50-0-0 | 50.0 [40.4, 59.6] | 12.6 |
| `final_minimax_hop/N5_hop_d2` | new contribution (hop scorer) | `minimax:depth=2,eval=hop` | 5 | 4 | 100 | 100 | 50-48-2-0 | 50.0 [40.4, 59.6] | 13.4 |
| `final_minimax_hop/N5_hop_d3` | new contribution (hop scorer) | `minimax:depth=3,eval=hop` | 5 | 4 | 100 | 100 | 50-46-4-0 | 50.0 [40.4, 59.6] | 13.3 |
| `final_minimax_hop/N5_default_repro` | diagnostic (reproduction check of N5_open4_distinct) | `minimax:depth=2` | 5 | 4 | 100 | 100 | 39-60-1-0 | 39.0 [30.0, 48.8] | 18.5 |
| `final_minimax_hop/N9_hop_timed` | new contribution (hop scorer), timed, load-affected | `minimax:depth=2,time_limit=1.0,eval=hop` | 9 | 2 | 20 | 16 | 8-8-0-0 | 50.0 [28.0, 72.0] | 17.6 |

Counts are over distinct games (both agents are deterministic). Rows marked diagnostic are checks, not results. `vs_greedy_N9_timed_r2` replays the same 20 N9 games (not listed).
