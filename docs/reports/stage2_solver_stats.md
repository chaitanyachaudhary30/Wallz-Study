# Stage 2 report: exact solver, miniature solves, ablation statistics (Member 3, Tasks 1-4 and 6)

- **Base:** `main` = `70e2351` (Stage 1 engine freeze). No engine rule was changed.
- **Branch:** `bot3-chaitanya/solver-stats`.
- **Commits that produced results:** `f930e8e` (N = 3 and N = 4, M = 0 solves), `8944f16`
  (M = 1 solves, cross-checks, both match ablations). Every result JSON records its
  `source.commit`, `dirty` (all `false`), the command, Python version (3.12.5), platform and
  UTC date (`run` block). Solver code is identical in both commits; later commits only add
  analysis code and docs.
- **Machine:** Windows 11, Python 3.12.5, numpy 2.2.0, pytest 9.2.0. Heavy jobs ran under the
  shared `HEAVY.lock`; the largest solve cell peaked at about 1.0 GB resident.
- **Rule of this report:** exact small-board results (Sections 1-2) and 5x5 match results
  (Section 4) are kept apart. No cell the solver did not finish is called exact.

## Code changes (all small)

| File | Change | Why |
|---|---|---|
| `evaluation/state_graph.py` | one shared tuple per distinct action | memory on 400k-state graphs; same values |
| `evaluation/ablation.py` | `solver_sweep(keep_solutions=False)` for `solve --abc all` | peak memory = one cell's graph, not eight |
| `evaluation/match_runner.py` | `run_metadata()` (commit, dirty, command, Python, platform, date) | provenance in every result |
| `scripts/run_experiments.py` | provenance taken at program start; full CLI config saved; `rule_draw` response; effects resampled per **distinct opening** (`effects`) next to per seed (`effects_by_seed`); `replicas` check | see Section 4.1 |
| `evaluation/metrics.py` | `draw_rate`, `x_goal_loss_rate` (+ Wilson), `sd_length_plies`; `unit_table(unit='opening')` | separate draw and truncation columns; replica seeds |
| `scripts/crosscheck_solvers.py` (new) | retrograde vs `agents/exact_solver.py`, canonical-merge check | Task 1 |
| `scripts/stage2_analysis.py` (new) | builds `final_eval/analysis.json` from the result files | every table below |
| `scripts/mutation_check_solver_tests.py` (new) | proves the new tests fail on two broken rules | Task 1 |
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

## 3. Task 3: multi-seed experiments: **not run**

`scripts/run_multi_seed_3x3.py` **trains DQN**: for each seed in 42-46 without
`checkpoints/dqn_3x3_s<seed>/final.pt` (none exist) it runs `scripts/train_dqn.py --steps 50000`
(up to 4 in parallel), then checks sign agreement against the exact N3 A1B1C1 K1 M1 solve and
plays 100 matches against Greedy and Random. Per the brief, DQN training belongs to Bot 2, so it
was not run and a `question` went to the conductor. No multi-seed numbers exist yet.

Problems found in the script for whoever runs it:
- it counts `truncated` as a draw (`'draw' in cat or cat == 'truncated'`), against the team decision;
- it caps games at `max_plies=100`, not the runner's 200;
- `std` is `np.std` (population, ddof 0); with 5 seeds the sample SD (ddof 1) is the usual choice;
- `gate_x11_passed` is `ci_low >= 0.90 or iqm >= 0.90`, so the gate passes on the IQM alone.

Planned analysis once 5 checkpoints exist: per seed, win/draw/truncation rates with Wilson 95%
intervals and mean game length; across seeds, mean, sample SD and a t-interval
mean +/- 2.776 * SD / sqrt(5); and a leave-one-seed-out check of whether any single seed moves
the mean outside the other four seeds' interval.

## 4. Task 4: 2^3 factorial match ablation on the final engine

### 4.1 What the pilot design could and could not show

- **All 8 cells tested:** yes, in the pilot and here (`summary.json` `cells`, 8 labels).
- **Final rules:** the pilot ran at `7e2d6da`, before the HOP change, the Greedy changes and the
  Minimax fix. It is stale (`experiments/results/STALE.md`).
- **Enough matches?** No. Both agents are deterministic (Greedy without a seed breaks ties by
  index; Minimax without a time limit is a fixed search). With `opening_plies = 2` on 5x5 there
  are only **9** possible openings in total, and the pilot's seeds 0-9 drew only **8** of them
  (`analysis.json` `opening_counts_N5`). Two pilot seeds were exact replicas, and resampling
  seeds treated them as independent. More seeds at 2 plies cannot add evidence.

The rerun of the pilot design confirms this: `ablation_N5_K3_M10_open2` (30 seeds, 480
logged matches) contains 9 distinct openings and 144 distinct games; every replica group
gave the identical game (`replicas.groups_with_differing_replicas = 0`).

### 4.2 Design (decided with the conductor)

- **Primary:** N = 5, K = 3, M = 10, `minimax:depth=2` vs `greedy`, both seatings, all 8 cells,
  `max_plies 200`, **`opening_plies 6`, seeds 10-49 = 40 distinct openings**, 640 matches.
  `opening_plies 6` is 3 pawn moves each, so no opening can reach a goal row 4 rows away.
  Seeds 10-49 draw 40 distinct openings (`analysis.json` `opening_counts_N5`), so there are no
  replicas (`replicas.distinct_games = 640`).
- **Pilot comparison:** the specified `opening_plies 2` with 30 seeds (480 logged matches,
  144 distinct games).
- The brief asked for `opening_plies 2` and seeds sized to 3 hours of lock time. Because only 9
  openings exist at 2 plies, the conductor set the primary at 30-40 distinct openings, at most
  640 matches. The primary run took 207 s for 640 matches (`ablation_N5_K3_M10_open6/stdout.txt`).
- Effects are resampled by **distinct opening** (paired: one unit holds that opening's games in
  all 8 cells and both seatings), 10,000 bootstrap draws, percentile 95% intervals, two-sided
  bootstrap p-values, Holm over the 7 effects of each response. A p-value printed 0.0000 means
  no bootstrap draw fell on the other side of zero (p < 0.0002).

```
python scripts/run_experiments.py ablation --x minimax:depth=2 --y greedy -N 5 -K 3 -M 10 --seeds 40 --seed-start 10 --opening-plies 6 --max-plies 200 --n-boot 10000 --out-dir experiments/results/final_eval/ablation_N5_K3_M10_open6 --log-dir experiments/results/final_eval/logs
python scripts/run_experiments.py verify-replay experiments/results/final_eval/logs/ablation_N5_K3_M10_open6/matches.jsonl
python scripts/run_experiments.py ablation --x minimax:depth=2 --y greedy -N 5 -K 3 -M 10 --seeds 30 --seed-start 0 --opening-plies 2 --max-plies 200 --n-boot 10000 --out-dir experiments/results/final_eval/ablation_N5_K3_M10_open2 --log-dir experiments/results/final_eval/logs
python scripts/run_experiments.py verify-replay experiments/results/final_eval/logs/ablation_N5_K3_M10_open2/matches.jsonl
```

**Replays:** 640/640 and 480/480 verified (`verify_replay.txt` in each folder, and
`replay_verification` in each `summary.json`). No forfeits, no truncations.

### 4.3 Primary results (`ablation_N5_K3_M10_open6/summary.json`)

Rates are for Minimax (the focus agent). W-L-D-T = goal wins, goal losses, rule draws, truncations.

| Cell | Games | W-L-D-T | Win rate (95% Wilson) | Draw rate (95% Wilson) | Truncation rate | Mean plies (SD) |
|---|---:|---|---|---|---:|---|
| A0B0C0 | 80 | 69-6-5-0 | 0.8625 (0.770-0.921) | 0.0625 (0.027-0.138) | 0 | 27.1 (11.1) |
| A0B0C1 | 80 | 59-16-5-0 | 0.7375 (0.632-0.821) | 0.0625 (0.027-0.138) | 0 | 27.7 (12.0) |
| A0B1C0 | 80 | 29-50-1-0 | 0.3625 (0.266-0.472) | 0.0125 (0.002-0.067) | 0 | 15.1 (5.3) |
| A0B1C1 | 80 | 26-54-0-0 | 0.3250 (0.232-0.434) | 0.0000 (0.000-0.046) | 0 | 17.2 (5.8) |
| A1B0C0 | 80 | 64-10-6-0 | 0.8000 (0.700-0.873) | 0.0750 (0.035-0.154) | 0 | 39.0 (24.9) |
| A1B0C1 | 80 | 51-19-10-0 | 0.6375 (0.528-0.734) | 0.1250 (0.069-0.215) | 0 | 37.8 (21.5) |
| A1B1C0 | 80 | 29-46-5-0 | 0.3625 (0.266-0.472) | 0.0625 (0.027-0.138) | 0 | 18.1 (13.7) |
| A1B1C1 | 80 | 27-47-6-0 | 0.3375 (0.243-0.446) | 0.0750 (0.035-0.154) | 0 | 22.7 (21.8) |

Totals: 354 wins, 248 losses, 38 draws, 0 truncated (`analysis.json`). Draw types per cell are in
`summary.json` `draw_breakdown` (A-off draws are mostly no-legal-action, A-on draws mostly repetition).

Effects on Minimax's goal-win rate (40 openings, `effects.x_goal_win`):

| Effect | Estimate | 95% CI | Raw p | Holm p | Survives Holm |
|---|---:|---|---:|---:|---|
| A | -0.037 | (-0.103, +0.025) | 0.2622 | 0.7866 | no |
| B | -0.413 | (-0.497, -0.328) | 0.0000 | 0.0000 | **yes** |
| C | -0.088 | (-0.150, -0.028) | 0.0020 | 0.0100 | **yes** |
| AB | +0.044 | (-0.000, +0.091) | 0.0656 | 0.2624 | no |
| AC | -0.006 | (-0.028, +0.013) | 0.6616 | 0.7866 | no |
| BC | +0.056 | (+0.016, +0.109) | 0.0012 | 0.0072 | **yes** |
| ABC | +0.013 | (-0.012, +0.037) | 0.3662 | 0.7866 | no |

Effects on rule-draw rate (`effects.rule_draw`): only AC survives Holm (+0.019, CI
(+0.006, +0.034), Holm p 0.0168); A's raw interval excludes zero (+0.050) but its Holm p is 0.1344.

Effects on game length in plies (`effects.length_plies`): A +7.6, B -14.6, C +1.5, AB -3.4,
BC +1.8, ABC +1.1 survive Holm; AC does not (+0.2, Holm p 0.7660).

**Reading.** B lowers Minimax's win rate against Greedy by about 0.41 and shortens games by
about 15 plies. C lowers it by about 0.09. The BC interaction (+0.06) means C hurts Minimax less
when B is on. The **pilot's AB interaction (+0.11) is not supported**: on the final engine it is
+0.044 with a 95% interval touching zero and Holm p 0.26; in the 2-ply rerun it is -0.042.

### 4.4 Pilot comparison (`ablation_N5_K3_M10_open2/summary.json`, 9 openings)

| Effect | Pilot (`7e2d6da`, 10 seeds, seed bootstrap) | Final engine, 2 plies, 9 openings | Final engine, 6 plies, 40 openings |
|---|---|---|---|
| A | -0.04 (-0.26, 0.19) | -0.014 (-0.181, +0.097) | -0.037 (-0.103, +0.025) |
| B | -0.01 (-0.15, 0.13) | **-0.403 (-0.472, -0.306)** | **-0.413 (-0.497, -0.328)** |
| C | -0.06 (-0.19, 0.08) | -0.069 (-0.194, +0.014) | **-0.088 (-0.150, -0.028)** |
| AB | +0.11 (0.03, 0.20) | -0.042 (-0.111, +0.028) | +0.044 (-0.000, +0.091) |
| AC | +0.01 (-0.04, 0.06) | -0.042 (-0.083, +0.000) | -0.006 (-0.028, +0.013) |
| BC | -0.06 (-0.15, 0.03) | +0.014 (-0.028, +0.056) | **+0.056 (+0.016, +0.109)** |
| ABC | -0.04 (-0.21, 0.11) | +0.042 (+0.000, +0.083) | +0.013 (-0.012, +0.037) |

Bold = survives Holm. Pilot numbers are from `person3_pilot/ablation_N5_minimax2_vs_greedy/summary.json`
(no Holm there). In the 2-ply rerun only B survives. Resampling seeds instead of openings on that
run (`effects_by_seed`, 30 "units") would have declared B, C, AB, AC and ABC significant: that
is what counting replicas as evidence does.

### 4.5 Power (`analysis.json` `ablations.*.mde_win_rate_80pct_power`)

An effect is a mean of 4 cells minus a mean of 4 cells, each cell a mean over n independent
openings, so Var(effect) = sd^2 / (2n). With the worst case sd = 0.5 per opening and the
pairing ignored, the smallest effect on win rate found with 80% power is
(z<sub>1-alpha/2</sub> + 0.842) * 0.5 / sqrt(2n):

| Design | n (openings) | MDE at alpha 0.05 | MDE at alpha 0.05/7 (Holm's first step) |
|---|---:|---:|---:|
| primary, 6 plies | 40 | 0.157 | 0.197 |
| pilot design, 2 plies | 9 | 0.330 | 0.416 |

These bounds are conservative: the paired bootstrap standard errors of the primary run give
MDEs of 0.03-0.12 depending on the effect (`mde_80pct_from_bootstrap_se`). C (-0.088) and BC
(+0.056) were detected below the worst-case bound because their real variance is smaller.
The 2-ply design cannot reliably find any win-rate effect under 0.33.

## 5. Task 5 inputs (not done here)

`scripts/make_tables.py` `generate_all_tables()` reads, first match wins where noted:
- ablation (first existing): `experiments/results/person3_pilot/ablation_N5_minimax2_vs_greedy/summary.json`
  (**stale**), then `experiments/results/ablation_5x5_k3/summary.json` (does not exist);
- agreement (each existing): `person3_pilot/agreement_N3_K1_M1.json` (stale),
  `agreement/agreement_3x3.json` (stale), `dqn/agreement_N4.json` (stale, Bot 2);
- benchmark: `dqn/benchmark_N3.json`, `dqn/benchmark_N4.json` (stale, Bot 2);
- solves: every `experiments/results/solves/solve_*.json` (M = 0, still valid; not `final_eval/`);
- greedy: `experiments/results/greedy/summary.json` (partly stale).

So Task 5 must point `make_tables.py` at `final_eval/ablation_N5_K3_M10_open6/summary.json` and
`final_eval/solve_*.json`, and read `effects` (per opening), not `effects_by_seed`.

## 6. Tests run (one file at a time)

| File | Result |
|---|---|
| `tests.test_solver_final_rules` (new) | 18 pass |
| `tests.test_retrograde` | 17 pass |
| `tests.test_exact_solver` | 9 pass |
| `tests.test_match_runner` | 16 pass |
| `tests.test_metrics_holm` | 5 pass (1 new) |
| `tests.test_golden_regression` | pass |
| `tests.test_agents` | pass |
| `pytest tests/test_edge_invariants.py` | 4 pass |

## 7. Not done, and limits

- Task 3 not run (DQN training; waiting on the conductor). Task 5 not started (later item).
- Agent-vs-exact agreement (`agreement` subcommand) was not rerun; the pilot agreement files stay stale.
- N = 4, M = 1 C-on cells are capped at 400k states: no exact claim, no factorial effects there.
- The 5x5 ablation is one opponent pairing, one depth, one board size; its results say nothing
  about optimal play.
