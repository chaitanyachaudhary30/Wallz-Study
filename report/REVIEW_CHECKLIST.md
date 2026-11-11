# Review checklist (human reviewer)

## 1. Files to open

1. `report/final_report.md` (the report; `report/final_report.tex` is the same content, not compiled)
2. `experiments/results/final_eval/tables/` (all tables; `manifest.json` lists each table's inputs)
3. `experiments/results/STALE.md` (which older results are not used, and why)
4. Member sections: `docs/report_sections/member1_minimax.md`, `member3_eval.md`, `member4_dqn.md`
5. Stage reports: `docs/reports/stage1_engine_freeze.md`, `stage3_greedy_demo_docs.md`

## 2. Ten key numbers to check (open the file, find the field)

| # | Claim in the report | File | Field |
|---|---|---|---|
| 1 | N3 K1 M1 A1B1C1 is a DRAW, 150,710 states | `experiments/results/final_eval/solve_N3_K1_M1.json` | `cells[7].root_status`, `.states` |
| 2 | N4 K1 M1 C-on cells are capped (UNKNOWN) | `experiments/results/final_eval/solve_N4_K1_M1.json` | `cells[*].complete`, `stop_reason` |
| 3 | Ablation primary: B effect on win rate -0.175, survives Holm | `experiments/results/final_eval/ablation_N5_K3_M10_open4/summary.json` | `effects.x_goal_win.effects.B` |
| 4 | Ablation primary: 640 matches, 640/640 replays verified | same file | `design.n_matches`, `replay_verification` |
| 5 | Minimax d2 vs Greedy N5: 39-60-1 over 100 distinct games | `experiments/results/final_minimax/opening_maps.json` | `results.N5_open4_distinct.distinct_game_results.overall` |
| 6 | Hop scorer +11 points at depth 2, 95% +1 to +21 | `experiments/results/final_minimax_hop/paired_d2_default_vs_hop.json` | `results.difference_b_minus_a`, `paired_bootstrap95` |
| 7 | Greedy vs Random N5: 498 of 500 | `experiments/results/final_greedy/random_N5_summary.json` | `results.overall.greedy_goal_wins` |
| 8 | DQN 5x5 vs Greedy: 66% win [56.3, 74.5], 100 distinct games | `experiments/results/final_dqn/distinct_reanalysis.json` | `results["5x5/final_dqn_5x5_s42_500k_eval.json"].by_opponent.greedy.distinct` |
| 9 | DQN 3x3 multi-seed: win vs Greedy mean 0.650, SD 0.056 | `experiments/results/final_eval/multi_seed_3x3.json` | `aggregate.vs.greedy.win_rate` |
| 10 | DQN 3x3 value-optimal 96.0-96.5% over all solved states | `experiments/results/final_dqn/aggregate_3x3_seeds.json` | `value_optimal_rate` |

Also check: no table or sentence mixes Section 5a (3x3/4x4 exact) with 5b (5x5/9x9 matches),
and draw rate and truncation rate are always separate columns.

## 3. Rerun

```
python scripts/make_tables.py                     # regenerates final_eval/tables/ (no randomness)
python scripts/report_md_to_tex.py                # regenerates report/final_report.tex
python -m pytest tests/test_solver_final_rules.py -q
python -m pytest tests/test_multi_seed_aggregation.py -q
python -m pytest tests/test_stage1_engine_freeze.py -q
python -m pytest tests/web/test_engine_sync.py -q
```

`tests/conftest.py` needs torch, or set `WALLZ_SKIP_DQN=1`. Three test files need `hypothesis`.

## 4. Demo checks

Start: `python -m web.backend.app --port 5000` and `npm --prefix web/frontend run dev`
(`docs/web_demo.md`). Then check:

1. A HOP is offered only across a wall the opponent placed, and only once.
2. A BLOCK lasts exactly K plies (set K = 2, count the plies).
3. Only the latest wall can be moved with REPLACE.
4. A wall that would cut off a pawn is refused.
5. At the ply cap the banner says "truncated", not "draw".
6. In the Arena, wins, draws and truncations add up to the number of games.
