# Stage 2 report: exact solver, miniature solves, ablation statistics (Member 3, Tasks 1-4 and 6)

- **Base:** `main` = `70e2351` (Stage 1 engine freeze). No engine rule was changed.
- **Branch:** `bot3-chaitanya/solver-stats`.
- **Commits that produced results:** `f930e8e` (N = 3 and N = 4, M = 0 solves), `8944f16`
  (M = 1 solves, cross-checks, 6-ply ablation), `4ef8124` (4-ply primary and 2-ply
  pilot-comparison ablations), `87fd485` (`analysis.json`). PR #2 (merged) held the first
  round; this follow-up PR holds the distinct-opening rework. Every
  result JSON records the commit and a dirty flag (all `false`) in `run.source` (and in
  `source` for solver, cross-check and ablation files), plus the command, Python version
  (3.12.5), platform and UTC date. The engine, solver and match-play code are the same in all these
  commits; they differ only in analysis code, the opening picker and docs.
- **Machine:** Windows 11, Python 3.12.5, numpy 2.2.0, pytest 9.2.0. Heavy jobs ran under the
  shared `HEAVY.lock`. Peak memory of the largest solve cell was about 1.0 GB resident, read
  from the OS process list during the run (not recorded in a result file).
- **Rule of this report:** exact small-board results (Sections 1-2) and 5x5 match results
  (Section 4) are kept apart. No cell the solver did not finish is called exact.

## Code changes (all small)

| File | Change | Why |
|---|---|---|
| `evaluation/state_graph.py` | one shared tuple per distinct action | memory on 400k-state graphs; same values |
| `evaluation/ablation.py` | `solver_sweep(keep_solutions=False)` for `solve --abc all` | peak memory = one cell's graph, not eight |
| `evaluation/match_runner.py` | `run_metadata()` (commit, dirty, command, Python, platform, date) | provenance in every result |
| `scripts/run_experiments.py` | provenance taken at program start; full CLI config saved; `rule_draw` response; effects resampled per **distinct opening** (`effects`) next to per seed (`effects_by_seed`); `replicas` check; `ablation --distinct-openings N` with a recorded `opening_map` | see Section 4 |
| `evaluation/ablation.py` | `first_distinct_openings()`: first seed of each distinct opening, scanning seeds in order | Section 4.2 |
| `evaluation/metrics.py` | `draw_rate`, `x_goal_loss_rate` (+ Wilson), `sd_length_plies`; `unit_table(unit='opening')` | separate draw and truncation columns; replica seeds |
| `scripts/crosscheck_solvers.py` (new) | retrograde vs `agents/exact_solver.py`, canonical-merge check | Task 1 |
| `scripts/stage2_analysis.py` (new) | builds `final_eval/analysis.json` from the result files | every table below |
| `scripts/mutation_check_solver_tests.py` (new) | proves the new tests fail on two broken rules | Task 1 |
| `scripts/run_multi_seed_3x3.py` (rewritten), `tests/test_multi_seed_aggregation.py` (new, 4 tests) | no training by default, truncation column, t-intervals, provenance | Task 3 preparation |
| `tests/test_solver_final_rules.py` (new, 18 tests), `tests/test_metrics_holm.py` (+1 test) | | Task 1, opening units |

## 1. Task 1: solver validation against the final engine

Command: `python -m unittest -v tests.test_solver_final_rules` -> 18 tests, OK, 21 s
(`final_eval/test_solver_final_rules.txt`). Hand-built 3x3 positions; value and rank are for the
side to move.

| Area | Test | What is checked | Result |
|---|---|---|---|
| terminal win | `test_terminal_win_state` | P0 on goal row, P1 to move: terminal `win_p0`, LOSS rank 0 for P1, +1 for P0, no actions | pass |
| terminal loss | `test_terminal_loss_state` | P1 on row 0, P0 to move: `win_p1`, LOSS rank 0 | pass |
| win / forced loss | `test_win_in_one_and_forced_loss_in_two` | WIN rank 1 with the goal move as only rank-optimal action; P1 to move LOSS rank 2, every action leads to a WIN-in-1 for P0 | pass |
| no legal action | `test_no_legal_action_draw` | fixture `no_legal_draw.json`: empty legal list, both paths open, terminal `draw_no_legal_actions`, `draw_type` `no_legal_actions` | pass |
| cyclic draw | `test_cyclic_draws` | pawn-interference cycle (DRAW, rank None, `cycle`); full N3 A1B0C0 K1 M1 root DRAW `cycle`; every node satisfies the negamax equations; value-optimal play revisits a position and never ends | pass |
| block expiry | `test_block_active_on_last_ply_then_expires` | K = 2: timer-1 block expires after P1's ply (P1 LOSS 2, same as no block); timer 2 keeps the cell shut on P0's ply (P1 WIN 3); K = 1 timer-1 block stops the mover on its last ply (WIN 1 -> LOSS 2) | pass |
| block timers | `test_block_timers_on_every_edge` | every edge of N3 C K2 M0: child timers = parent - 1, removed at 0, new block = K, block flag consumed | pass |
| wall inventory | `test_wall_inventory` | same position LOSS 2 with 0 walls, WIN 5 with 1 wall (rank-optimal `WALL (1,0,H)`); whole N3 M2 game: WALL only with stock, costs exactly 1, nothing else costs | pass |
| latest-wall REPLACE | `test_latest_wall_replace` | A off LOSS 2, A on WIN 5 by `REPLACE (1,0,H)`; with an older and a latest wall, every REPLACE edge moves only the latest wall, costs nothing, older wall never moves | pass |
| hop availability and ownership | `test_hop_needs_unused_hop_and_opponent_wall` | opponent's wall + unused hop: WIN 1 by HOP; own wall, or hop used: WIN 3 and no HOP edge anywhere; rule B off: no HOP | pass |
| hop edges | `test_hop_edges_cross_an_opponent_wall_once` | with REPLACE on (owner survives REPLACE): every HOP edge has hop unused, `wall_owner_on_edge == opponent`, hop consumed, pawn lands on target | pass |
| no walls, no hop | `test_no_walls_means_no_hop` | M = 0: B on and B off give the same graph size and root | pass |
| canonical key | `test_owner_kept_in_key_only_while_a_hop_remains` | owner-swapped walls: different keys while a hop remains, same key and same value/rank once hops are used | pass |
| engine is the only rule source | `test_every_edge_is_an_engine_transition` | `apply_action` spied: called exactly once per edge; every node's edges equal `get_legal_actions()`; each edge re-applied on a fresh engine gives the child's `position_key` and winner; all 5 action types seen | pass |
| no rule code in solver | `test_solver_code_has_no_rule_logic` | AST: `state_graph.py`/`retrograde.py` read no engine internals (pawns, walls, blocks, path checks); `retrograde.py` does not import the engine | pass |
| independent reference | `test_matches_bellman_reference` | DFS over raw keys + value iteration (no shared code) on N3 B M1, N3 C K2 M0 and a REPLACE mid-game root: every raw key's value and rank equal | pass |
| legacy solver | `test_legacy_exact_solver_agrees` | `agents/exact_solver.py` vs retrograde on 4 configs: status and distance per raw key | pass |
| canonical merge | `test_canonical_merge_keeps_value_and_rank` | raw vs canonical graphs on 3 configs: every merged group has one (value, rank) | pass |

**Mutation check** (`python scripts/mutation_check_solver_tests.py`,
`final_eval/mutation_check_solver_tests.txt`): with wall owners dropped from the key, 4 of 18
tests fail; with HOP allowed across any wall, 3 of 18 fail. So the tests can see both errors.

**Cross-check rerun** (`python scripts/crosscheck_solvers.py -N 3 -K 1 -M <m> --out ...`,
`final_eval/crosscheck_N3_K1_M0.json`, `crosscheck_N3_K1_M1.json`): all 16 cells complete,
0 status mismatches, 0 rank mismatches, 0 merged groups with unequal value or rank.
Note: `agents/exact_solver.py` is now an adapter over the same `StateGraph`/`retrograde` code
with raw keys, so this rerun checks canonical merging, not a second algorithm. The independent
check is the Bellman reference above.

| N3 K1 M1 cell | raw keys | canonical nodes | nodes with merged raw keys |
|---|---:|---:|---:|
| A0B0C0 | 1,953 | 1,341 | 612 |
| A0B0C1 | 23,764 | 16,696 | 7,068 |
| A0B1C0 | 5,013 | 4,893 | 120 |
| A0B1C1 | 61,148 | 59,240 | 1,908 |
| A1 cells | equal | equal | 0 (with A on and M = 1 each wall is a latest wall, nothing to merge) |

Engine note (not a bug in reachable play): `get_legal_actions()` assumes a valid position. A
hand-built root where a player already had no path listed a REPLACE that `apply_action` then
rejected. All test roots were checked for open paths; every graph built from the real start
position is consistent (`test_every_edge_is_an_engine_transition`).

## 2. Task 2: the 8 miniature configurations

Commands (caps as in the pilot; `--max-edges` default 20,000,000):

```
python scripts/run_experiments.py solve -N 3 -K 1 -M 0 --abc all --max-states 400000 --max-seconds 900 --out experiments/results/final_eval/solve_N3_K1_M0.json
python scripts/run_experiments.py solve -N 3 -K 1 -M 1 --abc all --max-states 400000 --max-seconds 900 --out experiments/results/final_eval/solve_N3_K1_M1.json
python scripts/run_experiments.py solve -N 4 -K 1 -M 0 --abc all --max-states 400000 --max-seconds 900 --out experiments/results/final_eval/solve_N4_K1_M0.json
python scripts/run_experiments.py solve -N 4 -K 1 -M 1 --abc all --max-states 400000 --max-seconds 900 --out experiments/results/final_eval/solve_N4_K1_M1.json
python scripts/stage2_analysis.py
```

Root value is for player 0, who moves first. Rank = plies to the goal under optimal play.
Seconds = build + solve. Source: `final_eval/analysis.json` `solve_tables`.

### 2.1 N = 3, K = 1, M = 0 (`solve_N3_K1_M0.json`)

| Cell | Complete | States | Edges | Root | Rank | Seconds |
|---|---|---:|---:|---|---:|---:|
| A0B0C0 | yes | 45 | 82 | WIN | 3 | 0.00 |
| A0B0C1 | yes | 642 | 3,120 | WIN | 5 | 0.16 |
| A0B1C0 | yes | 45 | 82 | WIN | 3 | 0.01 |
| A0B1C1 | yes | 642 | 3,120 | WIN | 5 | 0.17 |
| A1B0C0 | yes | 45 | 82 | WIN | 3 | 0.00 |
| A1B0C1 | yes | 642 | 3,120 | WIN | 5 | 0.17 |
| A1B1C0 | yes | 45 | 82 | WIN | 3 | 0.01 |
| A1B1C1 | yes | 642 | 3,120 | WIN | 5 | 0.17 |

### 2.2 N = 3, K = 1, M = 1 (`solve_N3_K1_M1.json`)

| Cell | Complete | States | Edges | Root | Rank | Seconds |
|---|---|---:|---:|---|---:|---:|
| A0B0C0 | yes | 1,341 | 3,180 | WIN | 7 | 0.14 |
| A0B0C1 | yes | 16,696 | 79,710 | WIN | 9 | 3.50 |
| A0B1C0 | yes | 4,893 | 9,140 | WIN | 5 | 0.49 |
| A0B1C1 | yes | 59,240 | 242,204 | WIN | 7 | 11.48 |
| A1B0C0 | yes | 3,861 | 16,598 | DRAW | - | 0.77 |
| A1B0C1 | yes | 46,886 | 314,980 | DRAW | - | 14.60 |
| A1B1C0 | yes | 12,821 | 51,510 | DRAW | - | 2.51 |
| A1B1C1 | yes | 150,710 | 956,644 | DRAW | - | 50.61 |

### 2.3 N = 4, K = 1, M = 0 (`solve_N4_K1_M0.json`)

| Cell | Complete | States | Edges | Root | Rank | Seconds |
|---|---|---:|---:|---|---:|---:|
| A0B0C0 | yes | 176 | 400 | LOSS | 6 | 0.02 |
| A0B0C1 | yes | 4,512 | 40,472 | LOSS | 8 | 1.83 |
| A0B1C0 | yes | 176 | 400 | LOSS | 6 | 0.02 |
| A0B1C1 | yes | 4,512 | 40,472 | LOSS | 8 | 1.80 |
| A1B0C0 | yes | 176 | 400 | LOSS | 6 | 0.02 |
| A1B0C1 | yes | 4,512 | 40,472 | LOSS | 8 | 1.82 |
| A1B1C0 | yes | 176 | 400 | LOSS | 6 | 0.02 |
| A1B1C1 | yes | 4,512 | 40,472 | LOSS | 8 | 1.71 |

### 2.4 N = 4, K = 1, M = 1 (`solve_N4_K1_M1.json`)

| Cell | Complete | States | Edges | Root | Rank | Seconds |
|---|---|---:|---:|---|---:|---:|
| A0B0C0 | yes | 27,860 | 92,308 | LOSS | 10 | 4.72 |
| A0B0C1 | **capped** (400k states) | 400,004 | 3,197,444 | UNKNOWN | - | 154.39 |
| A0B1C0 | yes | 149,360 | 371,404 | LOSS | 8 | 18.56 |
| A0B1C1 | **capped** (400k states) | 400,010 | 2,266,293 | UNKNOWN | - | 107.45 |
| A1B0C0 | yes | 98,504 | 1,248,040 | DRAW | - | 55.18 |
| A1B0C1 | **capped** (400k states) | 400,005 | 3,713,862 | UNKNOWN | - | 174.00 |
| A1B1C0 | yes | 372,160 | 4,662,904 | DRAW | - | 211.45 |
| A1B1C1 | **capped** (400k states) | 400,003 | 2,770,373 | UNKNOWN | - | 139.36 |

The four C-on cells of N = 4, M = 1 are **not exhaustively solved**. Their roots are UNKNOWN;
their partial WIN/LOSS labels are sound but cover only a small share of the graph (for example
A1B1C1: 531 WIN, 187 LOSS, 399,285 UNKNOWN; `solve_N4_K1_M1.json` `value_counts`).

### 2.5 Comparison with the pilot (`analysis.json` `pilot_comparison`)

Every **B-off** cell (16 of 16 across the four settings, including the complete and the capped
N = 4, M = 1 cells) has the same completeness, states, edges, root value and rank as the
pilot. This is the built-in check: the HOP change cannot touch a game without B.

Every **B-on** difference (pilot -> final):

| Setting | Cell | States | Edges | Root |
|---|---|---|---|---|
| N3 M0 | A0B1C0, A1B1C0 | 186 -> 45 | 398 -> 82 | LOSS 2 -> WIN 3 |
| N3 M0 | A0B1C1, A1B1C1 | 2,592 -> 642 | 13,600 -> 3,120 | LOSS 4 -> WIN 5 |
| N3 M1 | A0B1C0 | 5,730 -> 4,893 | 14,824 -> 9,140 | LOSS 4 -> WIN 5 |
| N3 M1 | A0B1C1 | 68,292 -> 59,240 | 346,500 -> 242,204 | LOSS 6 -> WIN 7 |
| N3 M1 | A1B1C0 | 16,554 -> 12,821 | 72,450 -> 51,510 | DRAW, unchanged |
| N3 M1 | A1B1C1 | 192,072 -> 150,710 | 1,335,672 -> 956,644 | DRAW, unchanged |
| N4 M0 | A0B1C0, A1B1C0 | 712 -> 176 | 2,112 -> 400 | WIN 3 -> LOSS 6 |
| N4 M0 | A0B1C1, A1B1C1 | 18,080 -> 4,512 | 177,248 -> 40,472 | WIN 5 -> LOSS 8 |
| N4 M1 | A0B1C0 | 114,488 -> 149,360 | 452,284 -> 371,404 | WIN 5 -> LOSS 8 |
| N4 M1 | A1B1C0 | capped (400,000) -> **complete** 372,160 | 5,165,610 -> 4,662,904 | UNKNOWN -> DRAW |
| N4 M1 | A0B1C1 | capped both (400,007 -> 400,010) | 2,364,837 -> 2,266,293 | UNKNOWN both |
| N4 M1 | A1B1C1 | capped both (400,001 -> 400,003) | 2,332,450 -> 2,770,373 | UNKNOWN both |

Why:
1. **M = 0.** There is never a wall, and the final HOP needs an opponent's wall, so HOP is
   never legal. B-on cells now equal B-off cells exactly. The pilot's 2-cell jump needed no
   wall, so it added positions and changed the root: on 3x3 it turned player 0's win into a
   loss, on 4x4 a loss into a win.
2. **M = 1, fewer edges.** A hop now needs an opponent's wall next to the pawn; the old jump
   needed nothing. Edges drop in every B-on cell that completed in both runs, and in N4 A1B1C0.
3. **M = 1, state counts move both ways.** Fewer positions are reachable without the free jump,
   but `canonicalize_key` now keeps wall owners while a hop remains, which splits positions the
   pilot merged. On 3x3 the first effect wins (fewer states); on N4 A0B1C0 the second wins
   (149,360 vs 114,488). The files do not separate the two effects.
4. **Root values.** Without the free sprint, B no longer changes who wins: every complete B-on
   root now has the same value as its B-off neighbour, often with a different rank (N3 M1 A0B1C0 WIN 5 vs
   A0B0C0 WIN 7; N4 M1 A0B1C0 LOSS 8 vs A0B0C0 LOSS 10).

### 2.6 Exact factorial effects (`analysis.json` `exact_effects`)

Effects use the +/-1 coding over the 8 cells (`evaluation/metrics.py` `factorial_effects`).
They are exact: no sampling. Root value: WIN = +1, DRAW = 0, LOSS = -1 for player 0. Only
settings with all 8 cells complete get effects; **N = 4, M = 1 gets none** (4 cells capped).

| Effect | N3 M1 root value | N3 M1 log10 states | N3 M1 branching | N3 M1 draw fraction | N3 M0 log10 states | N4 M0 log10 states | N4 M0 branching |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | -1.00 | +0.433 | +2.403 | +0.311 | 0 | 0 | 0 |
| B | 0 | +0.535 | -0.484 | -0.058 | 0 | 0 | 0 |
| C | 0 | +1.083 | +1.870 | +0.038 | +1.154 | +1.409 | +6.363 |
| AB | 0 | -0.021 | +0.166 | -0.064 | 0 | 0 | 0 |
| AC | 0 | -0.006 | -0.183 | +0.003 | 0 | 0 | 0 |
| BC | 0 | -0.007 | -0.039 | -0.018 | 0 | 0 | 0 |
| ABC | 0 | -0.000 | +0.005 | -0.015 | 0 | 0 | 0 |

With M = 0, A (nothing to move) and B (nothing to hop) are exactly zero, as they must be. With
M = 1 on 3x3 only A changes the root value (first-player WIN -> DRAW). The pilot's reading
"without A, B flips the root" (`docs/person3_evaluation.md` 6.1) came from the old jump. State counts stay almost multiplicative (log-scale interactions
within 0.021 of zero).

## 3. Task 3: multi-seed experiments: **prepared, not run (pending Bot 2 checkpoints)**

The old `scripts/run_multi_seed_3x3.py` **trained DQN** whenever
`checkpoints/dqn_3x3_s<seed>/final.pt` was missing (none exist): `scripts/train_dqn.py --steps 50000`
for each of seeds 42-46, up to 4 in parallel. DQN training belongs to Bot 2, so it was not run.
Conductor decision: Bot 2 trains the 5 seeds under the final rules into
`checkpoints/final/3x3_s<seed>/final.pt`; this study runs in a later integration item.

Problems in the old script, all fixed in this PR:

| Old behaviour | Now |
|---|---|
| `truncated` counted as a draw (`'draw' in cat or cat == 'truncated'`) | outcomes come from `evaluation/metrics.py` `outcome_for`; win, loss, draw, truncated and forfeits are separate columns |
| trained missing seeds automatically | `--no-train` is the default: a missing checkpoint exits with an error naming the seeds; `--train` (sequential) only on request |
| checkpoint path `checkpoints/dqn_3x3_s<seed>/final.pt` | `--checkpoint-dir` (default `checkpoints/final`), pattern `3x3_s{seed}/final.pt` |
| `max_plies=100` hidden in the code | `--max-plies` with default 100 (the study's old value, kept and stated); recorded in the output; the 200 of other experiments is unchanged |
| `std` = `np.std` (population); IQM and bootstrap CI over 5 values | sample SD (n - 1), 95% t-interval mean +/- t(0.975, n-1) * SD / sqrt(n) (2.776 for 5 seeds), min, max, leave-one-seed-out means and the largest shift; pooled Wilson intervals for rates |
| no provenance | commit, dirty flag, command, Python, platform, date, torch version, full CLI config, seeds, rules and SHA-256 of every checkpoint |
| `from scripts.validate_dqn import validate_cell` (that function does not exist) | `evaluate_cell_x16` from `scripts/validate_dqn.py` |
| exact solve cached in `experiments/cache/sol_N3_A1B1C1_K1_M1.pkl` (could hold an old-HOP solve) | solved fresh each run (about 1 minute) |

The old `gate_x11_passed` flag (`ci_low >= 0.90 or iqm >= 0.90`) was dropped: the gate
belongs to the DQN member, and the raw sign-agreement statistics are kept.

The aggregation needs no torch and is tested on synthetic per-seed results:
`python -m unittest tests.test_multi_seed_aggregation` -> 4 tests pass (truncation column,
t-interval and leave-one-out, 5-seed aggregate with pooled Wilson, missing checkpoint exits
without training). A smoke run loaded `checkpoints/release/dqn_3x3/model.pt` and played 2 games
through the new code path. The exact-solve and X-16 steps were not run end to end.

Command for the later item:

```
python scripts/run_multi_seed_3x3.py --checkpoint-dir checkpoints/final --seeds 42 43 44 45 46 --matches 100 --max-plies 100 --out experiments/results/final_eval/multi_seed_3x3.json
```

## 4. Task 4: 2^3 factorial match ablation on the final engine

### 4.1 What the pilot design could and could not show

- **All 8 cells tested:** yes, in the pilot and here (`summary.json` `cells`, 8 labels).
- **Final rules:** the pilot ran at `7e2d6da`, before the HOP change, the Greedy changes and the
  Minimax fix. It is stale (`experiments/results/STALE.md`).
- **Enough matches?** No. Both agents are deterministic (Greedy without a seed breaks ties by
  index; Minimax without a time limit is a fixed search). With `opening_plies = 2` on 5x5 there
  are only **9** possible openings in total, and the pilot's seeds 0-9 drew only **8** of them
  (`analysis.json` `opening_counts_N5`). Two pilot seeds were exact replicas, and resampling
  seeds treated them as independent, so the pilot's effective sample was 8 openings, not 10
  seeds. The pilot's own `matches.jsonl` was in the git-ignored `experiments/logs/` and is not
  on this machine, so N = 8 comes from the seed -> opening map for seeds 0-9
  (`analysis.json` `opening_counts_N5.pilot_seed_to_opening_plies2`: seed 3 repeats seed 1,
  seed 8 repeats seed 4). More seeds at 2 plies cannot add evidence.

### 4.2 Design (conductor decision)

- **Opening length.** The smallest `opening_plies` with at least 30 distinct openings on N5 is
  **4**: 2 plies give 9 openings, 4 plies give 99 (seeds 0-1999, `analysis.json`
  `opening_counts_N5`). 4 plies is 2 pawn moves each, so no opening reaches a goal row.
- **Openings picked.** Seeds are scanned 0, 1, 2, ... and only the first seed of each distinct
  action sequence is kept (`--distinct-openings`, `evaluation/ablation.py`
  `first_distinct_openings`). No replica seed is played. The seed -> opening map is in each
  `summary.json` `opening_map`.
- **Primary:** N5 K3 M10, `minimax:depth=2` vs `greedy`, both seatings, all 8 cells,
  `max_plies 200`, `opening_plies 4`, **40 distinct openings** (seeds 0-9, 12, 14-24, 26, 28-31,
  33-35, 37, 38, 40-44, 47, 50, 52): 640 matches, 195 s (`ablation_open4_stdout.txt`).
- **Direct comparison with the pilot:** `opening_plies 2`, all **9** distinct openings
  (seeds 0, 1, 2, 4, 5, 6, 7, 9, 20): 144 matches.
- **Sensitivity check, kept from the earlier round:** `opening_plies 6`, seeds 10-49
  (40 distinct openings), 640 matches (`ablation_N5_K3_M10_open6/`). It was the primary under
  the previous decision and is reported only to show how much the results depend on the
  opening length. The earlier 2-ply run with 30 seeds (`ablation_N5_K3_M10_open2/`,
  merged in PR #2) is kept unchanged but superseded: its 480 matches replay the same 144 distinct
  games as the pilot comparison (`replicas.groups_with_differing_replicas = 0`), and its
  win-rate `effects` equal the pilot comparison's (`experiments/results/STALE.md`).
- **Unit of analysis:** the distinct opening, paired across the 8 cells and both seatings.
  10,000 bootstrap draws over openings, percentile 95% intervals, two-sided bootstrap p-values,
  Holm over the 7 effects of each response. A p-value printed 0.0000 means no bootstrap draw fell
  on the other side of zero (p < 0.0002).

```
python scripts/run_experiments.py ablation --x minimax:depth=2 --y greedy -N 5 -K 3 -M 10 --distinct-openings 40 --opening-plies 4 --max-plies 200 --n-boot 10000 --out-dir experiments/results/final_eval/ablation_N5_K3_M10_open4 --log-dir experiments/results/final_eval/logs
python scripts/run_experiments.py verify-replay experiments/results/final_eval/logs/ablation_N5_K3_M10_open4/matches.jsonl
python scripts/run_experiments.py ablation --x minimax:depth=2 --y greedy -N 5 -K 3 -M 10 --distinct-openings 40 --opening-plies 2 --max-plies 200 --n-boot 10000 --out-dir experiments/results/final_eval/ablation_N5_K3_M10_open2_pilot_comparison --log-dir experiments/results/final_eval/logs
python scripts/run_experiments.py verify-replay experiments/results/final_eval/logs/ablation_N5_K3_M10_open2_pilot_comparison/matches.jsonl
```

(`--distinct-openings 40` at 2 plies stops at the 9 that exist.)

**Replays:** 640/640 (4 plies), 144/144 (2 plies) and 640/640 (6 plies) verified
(`verify_replay.txt` in each folder, and `replay_verification` in each `summary.json`).
No forfeits, no truncations. `replicas.distinct_games` equals the match count in all three runs.

### 4.3 Primary results (`ablation_N5_K3_M10_open4/summary.json`)

Rates are for Minimax (the focus agent). W-L-D-T = goal wins, goal losses, rule draws, truncations.

| Cell | Games | W-L-D-T | Win rate (95% Wilson) | Draw rate (95% Wilson) | Truncation rate | Mean plies (SD) |
|---|---:|---|---|---|---:|---|
| A0B0C0 | 80 | 57-17-6-0 | 0.7125 (0.605-0.800) | 0.0750 (0.035-0.154) | 0 | 24.1 (11.8) |
| A0B0C1 | 80 | 51-22-7-0 | 0.6375 (0.528-0.734) | 0.0875 (0.043-0.170) | 0 | 26.2 (13.2) |
| A0B1C0 | 80 | 46-32-2-0 | 0.5750 (0.466-0.677) | 0.0250 (0.007-0.087) | 0 | 15.3 (7.8) |
| A0B1C1 | 80 | 40-40-0-0 | 0.5000 (0.393-0.607) | 0.0000 (0.000-0.046) | 0 | 16.3 (7.4) |
| A1B0C0 | 80 | 50-13-17-0 | 0.6250 (0.515-0.723) | 0.2125 (0.137-0.314) | 0 | 41.9 (36.1) |
| A1B0C1 | 80 | 51-16-13-0 | 0.6375 (0.528-0.734) | 0.1625 (0.097-0.258) | 0 | 34.4 (21.8) |
| A1B1C0 | 80 | 34-45-1-0 | 0.4250 (0.323-0.534) | 0.0125 (0.002-0.067) | 0 | 15.3 (9.4) |
| A1B1C1 | 80 | 33-46-1-0 | 0.4125 (0.311-0.522) | 0.0125 (0.002-0.067) | 0 | 18.2 (18.8) |

Totals: 362 wins, 231 losses, 47 draws, 0 truncated (`analysis.json`). Draw types are in
`summary.json` `draw_breakdown`: A-off draws are mostly no-legal-action, the A1B0 draws are
almost all repetition (16 of 17 and 13 of 13).

Effects on Minimax's goal-win rate (40 openings, `effects.x_goal_win`):

| Effect | Estimate | 95% CI | Raw p | Holm p | Survives Holm |
|---|---:|---|---:|---:|---|
| A | -0.081 | (-0.147, -0.019) | 0.0116 | 0.0696 | no |
| B | -0.175 | (-0.259, -0.088) | 0.0000 | 0.0000 | **yes** |
| C | -0.037 | (-0.072, -0.006) | 0.0222 | 0.1110 | no |
| AB | -0.037 | (-0.078, +0.003) | 0.0918 | 0.2754 | no |
| AC | +0.037 | (+0.000, +0.075) | 0.0586 | 0.2344 | no |
| BC | -0.006 | (-0.047, +0.034) | 0.8278 | 1.0000 | no |
| ABC | -0.006 | (-0.031, +0.019) | 0.7156 | 1.0000 | no |

Effects on rule-draw rate (`effects.rule_draw`): only B survives Holm (-0.122, CI
(-0.172, -0.075)). A (+0.053) and AB (-0.053) have raw intervals that exclude zero but Holm
p 0.1668.

Effects on game length in plies (`effects.length_plies`): A +7.0, B -15.4, AB -6.0, AC -1.9,
BC +2.3 and ABC +2.8 survive Holm; C does not (-0.4, Holm p 0.5974).

Effects on Minimax's goal-loss rate (`effects.x_goal_loss`): B (+0.297, CI (+0.231, +0.359))
and AB (+0.091, CI (+0.034, +0.147), Holm p 0.0072) survive Holm. Win, loss and draw rates sum
to 1 here (no truncations, no forfeits), so the AB loss effect is balanced by AB's draw effect
(-0.053) and win effect (-0.037), neither of which survives Holm.

**Reading.** On the win rate only B survives Holm: it lowers Minimax's win rate against
Greedy by about 0.18, lowers the draw rate by about 0.12, raises the loss rate by about 0.30 and
shortens games by about 15 plies. AB raises Minimax's loss rate by about 0.09. A (-0.08) and
C (-0.04) have raw 95% win-rate intervals that exclude zero but do not survive Holm. The
**pilot's AB interaction (+0.11) is not supported**: it is -0.037 here (Holm p 0.28) and -0.042
in the 2-ply comparison.

### 4.4 Pilot comparison and sensitivity to opening length

Win-rate effects, estimate (95% CI). Bold = survives Holm.

| Effect | Pilot (`7e2d6da`, 10 seeds = 8 openings, seed bootstrap) | 2 plies, 9 openings (direct comparison) | **4 plies, 40 openings (primary)** | 6 plies, 40 openings (sensitivity) |
|---|---|---|---|---|
| A | -0.04 (-0.26, 0.19) | -0.014 (-0.181, +0.097) | -0.081 (-0.147, -0.019) | -0.037 (-0.103, +0.025) |
| B | -0.01 (-0.15, 0.13) | **-0.403 (-0.472, -0.306)** | **-0.175 (-0.259, -0.088)** | **-0.413 (-0.497, -0.328)** |
| C | -0.06 (-0.19, 0.08) | -0.069 (-0.194, +0.014) | -0.037 (-0.072, -0.006) | **-0.088 (-0.150, -0.028)** |
| AB | +0.11 (0.03, 0.20) | -0.042 (-0.111, +0.028) | -0.037 (-0.078, +0.003) | +0.044 (-0.000, +0.091) |
| AC | +0.01 (-0.04, 0.06) | -0.042 (-0.083, +0.000) | +0.037 (+0.000, +0.075) | -0.006 (-0.028, +0.013) |
| BC | -0.06 (-0.15, 0.03) | +0.014 (-0.028, +0.056) | -0.006 (-0.047, +0.034) | **+0.056 (+0.016, +0.109)** |
| ABC | -0.04 (-0.21, 0.11) | +0.042 (+0.000, +0.083) | -0.006 (-0.031, +0.019) | +0.013 (-0.012, +0.037) |

Sources: `person3_pilot/ablation_N5_minimax2_vs_greedy/summary.json` (no Holm there),
`final_eval/ablation_N5_K3_M10_open2_pilot_comparison/`, `.../ablation_N5_K3_M10_open4/`,
`.../ablation_N5_K3_M10_open6/`, all `summary.json` `effects.x_goal_win`.

- The pilot's results do not carry over to the final engine. With the same 2-ply openings, B
  goes from -0.01 to -0.40. The HOP change, the Greedy changes and the Minimax fix all came in
  between; these files cannot say which one caused it.
- **The size of B depends on the opening length** (-0.40, -0.18, -0.41 at 2, 4 and 6 plies),
  and C and BC are supported only at 6 plies. The sign of B is the same in all three; its size
  and the small effects are not stable. Effects here describe these two agents on these opening
  distributions, not the mechanics in general.
- In the 2-ply comparison only B survives Holm.

### 4.5 Power (`analysis.json` `ablations.*.mde_win_rate_80pct_power`)

An effect is a mean of 4 cells minus a mean of 4 cells, each cell a mean over n independent
openings, so Var(effect) = sd^2 / (2n). With the worst case sd = 0.5 per opening and the
pairing ignored, the smallest effect on win rate found with 80% power is
(z<sub>1-alpha/2</sub> + 0.842) * 0.5 / sqrt(2n):

| Design | n (openings) | MDE at alpha 0.05 | MDE at alpha 0.05/7 (Holm's first step) |
|---|---:|---:|---:|
| primary, 4 plies | 40 | 0.157 | 0.197 |
| pilot comparison, 2 plies | 9 | 0.330 | 0.416 |

These bounds are conservative: the paired bootstrap standard errors of the primary run give
MDEs of 0.04-0.12 depending on the effect (`mde_80pct_from_bootstrap_se`). A (-0.081) and
C (-0.037) are below their own bootstrap MDEs (0.092 and 0.047), which fits their raw but not
Holm-level evidence. For the 2-ply design the worst-case bound is 0.33; its paired bootstrap
MDEs are 0.06-0.20 by effect, so it can still find large effects such as B (-0.403). All
bootstrap-based MDEs are estimated from the same data, so treat them as rough.

## 5. Task 5 inputs (not done here)

`scripts/make_tables.py` `generate_all_tables()` reads, first match wins where noted:
- ablation (first existing): `experiments/results/person3_pilot/ablation_N5_minimax2_vs_greedy/summary.json`
  (**stale**), then `experiments/results/ablation_5x5_k3/summary.json` (does not exist);
- agreement (each existing): `person3_pilot/agreement_N3_K1_M1.json` (stale),
  `agreement/agreement_3x3.json` (stale), `dqn/agreement_N4.json` (stale, Bot 2);
- benchmark: `dqn/benchmark_N3.json`, `dqn/benchmark_N4.json` (stale, Bot 2);
- solves: every `experiments/results/solves/solve_*.json` (M = 0, still valid; not `final_eval/`);
- greedy: `experiments/results/greedy/summary.json` (partly stale).

So Task 5 must point `make_tables.py` at `final_eval/ablation_N5_K3_M10_open4/summary.json` and
`final_eval/solve_*.json`, and read `effects` (per opening), not `effects_by_seed`.

## 6. Tests run (one file at a time)

| File | Result |
|---|---|
| `tests.test_solver_final_rules` (new) | 18 pass |
| `tests.test_retrograde` | 17 pass |
| `tests.test_exact_solver` | 9 pass |
| `tests.test_match_runner` | 17 pass (1 new: `test_first_distinct_openings`) |
| `tests.test_metrics_holm` | 5 pass (1 new) |
| `tests.test_multi_seed_aggregation` (new) | 4 pass |
| `tests.test_golden_regression` | pass |
| `tests.test_agents` | pass |
| `pytest tests/test_edge_invariants.py` | 4 pass |

## 7. Independent check of this report

A fresh reviewer that saw only `git diff 70e2351..HEAD` and `final_eval/` tried to find one
number that does not follow from the files, or one unsupported significance claim. It
re-read every table in Sections 2 and 4 against the JSON, recomputed all 16 pilot-vs-final
diffs and the four MDE values from the formula, checked the Holm flags, the replica counts,
the capped-cell labels and that `truncated` is never merged into draws. **No wrong number and
no unsupported claim found.** It flagged two wording points, both fixed here: where the commit
is stored in `analysis.json`, and that the 1.0 GB peak memory is not in a file. Not checked by
it: the `logs/*.jsonl` match logs and a rerun of the bootstrap.

A second fresh reviewer checked the redone Section 4 (distinct openings) the same way. It
recomputed the Wilson intervals, the per-cell mean plies from the primary `matches.jsonl`, the
Holm steps, the MDE values, and both seed lists with `first_distinct_openings` (they equal
`design.opening_seeds` and every `opening_map` entry). It found no wrong number but four points,
all fixed: one length rounded half-even (A0B1C0 15.25 printed 15.2, now 15.3; ABC +2.85, now
+2.8); "only B affects who wins" ignored the AB effect on the loss rate (now reported); the
2-ply "nothing under 0.33" was only the worst-case bound (now given next to the paired
estimates); and "the pilot's intervals were likely too narrow" cannot be tested without the
pilot's log (now worded as such). Not checked by it: a rerun of the bootstrap.

## 8. Not done, and limits

- Task 3 prepared but not run: it waits for Bot 2's 5 checkpoints (later integration item).
  Its exact-solve and X-16 steps were not exercised end to end. Task 5 not started (later item).
- Agent-vs-exact agreement (`agreement` subcommand) was not rerun; the pilot agreement files stay stale.
- N = 4, M = 1 C-on cells are capped at 400k states: no exact claim, no factorial effects there.
- The 5x5 ablation is one opponent pairing, one depth, one board size; its results say nothing
  about optimal play.
