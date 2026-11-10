# Stage 2 Minimax Validation Report (Member 1, Tasks 1-5)

- **Engine and agent tested:** branch `bot1-vedant/minimax-validation`, based on `main` = `70e2351`
  (Stage 1 engine freeze). Code under test is unchanged from `main`: `agents/minimax.py`,
  `agents/evaluation_fn.py`, `wallz_engine.py`.
- **Commits that produced the results** (tree clean each time; recorded as `source.commit` /
  `source.dirty` in every JSON). Later commits only changed `scripts/final_minimax.py` (seed
  selection and reporting), never the agent or engine:

  | Commit | Results |
  |---|---|
  | `4023eac` | `alphabeta_equivalence.json`, `exact_draws_3x3.json`, `ordering.json`, `timing_N9.json`, superseded `vs_greedy_N5*`, `vs_greedy_N5_depth{1,3}_diag*` |
  | `f5aa757` | `vs_greedy_N5_open4_distinct*` (main run), `vs_greedy_N9_timed*` |
  | `d46fc5c` | `vs_greedy_N5_open4_depth{1,3}_diag*` |
  | `8589a55` | `hop_stats.json` |
  | `f755a3a` | `opening_maps.json` (read from the match logs; plays nothing) |
- **Added:** `tests/test_minimax_final_compat.py` (14 tests), `scripts/final_minimax.py`
  (experiment driver), results in `experiments/results/final_minimax/`, this report,
  `docs/report_sections/member1_minimax.md`.
- **Bugs fixed:** none. No bug was found that needed a code change (see the audit table).
- **Machine:** Windows, Python 3.12.5, numpy; pytest 9.2.0 became available during the run.
  Two other bots were running jobs on the same machine, so every timing number carries load noise.

## Commands run (from the repo root)

```
python -m unittest tests.test_minimax_final_compat
python -m unittest tests.test_minimax_phase5
python -m unittest tests.test_agents
python -m unittest tests.test_agent_conformance
python -m unittest tests.test_evaluation_fn
python -m unittest tests.test_exact_solver
python -m unittest tests.test_golden_regression
python -m unittest tests.test_match_runner
python -m pytest tests/test_edge_invariants.py -q
python -m pytest tests/test_greedy_campaign.py -q
python -m pytest tests/test_minimax_final_compat.py -q
python -m pytest tests/test_minimax_phase5.py -q
python -m pytest tests/test_agent_conformance.py -q
python scripts/final_minimax.py equivalence
python scripts/final_minimax.py exact-draws
python scripts/final_minimax.py ordering
python scripts/final_minimax.py timing
python scripts/final_minimax.py vs-greedy -N 5 --seeds 50 --spec minimax:depth=2 --tag N5            # superseded
python scripts/final_minimax.py vs-greedy -N 5 --seeds 10 --spec minimax:depth=1 --tag N5_depth1_diag # superseded
python scripts/final_minimax.py vs-greedy -N 5 --seeds 10 --spec minimax:depth=3 --tag N5_depth3_diag # superseded
python scripts/final_minimax.py vs-greedy -N 5 --seeds 50 --distinct --opening-plies 4 --spec minimax:depth=2 --tag N5_open4_distinct
python scripts/final_minimax.py vs-greedy -N 9 --seeds 10 --opening-plies 2 --spec minimax:depth=2,time_limit=1.0 --tag N9_timed
python scripts/final_minimax.py vs-greedy -N 5 --seeds 50 --distinct --opening-plies 4 --spec minimax:depth=1 --tag N5_open4_depth1_diag
python scripts/final_minimax.py vs-greedy -N 5 --seeds 50 --distinct --opening-plies 4 --spec minimax:depth=3 --tag N5_open4_depth3_diag
python scripts/final_minimax.py hops N5_open4_distinct N5_open4_depth1_diag N5_open4_depth3_diag N9_timed
python scripts/final_minimax.py openings N5 N5_open4_distinct N9_timed N5_depth1_diag N5_depth3_diag N5_open4_depth1_diag N5_open4_depth3_diag
python scripts/run_experiments.py verify-replay experiments/results/final_minimax/vs_greedy_N5.jsonl
python scripts/run_experiments.py verify-replay experiments/results/final_minimax/vs_greedy_N5_open4_distinct.jsonl
python scripts/run_experiments.py verify-replay experiments/results/final_minimax/vs_greedy_N9_timed.jsonl
```

Distinct-opening counts per `opening_plies` (N=5): 2 plies give 9 distinct openings in seeds
0-49 (and still 9 after scanning 2,000 seeds); 4 plies reach 50 distinct after 80 seeds; 6 plies
after 51 seeds. These counts come from a one-off scan in the shell; the chosen 4-ply seed list
and its scan length are recorded in `vs_greedy_N5_open4_distinct_summary.json` (`config.seed_selection`).

`scripts/benchmark_ordering_x8.py` was **not** run directly: it always writes to
`experiments/results/minimax/x8_node_reduction.json` and would overwrite the committed result.
`final_minimax.py ordering` copies its position generator exactly instead (see Task 4).

## Test results

| File | Runner | Result |
|---|---|---|
| `test_minimax_final_compat.py` (new) | unittest and pytest | pass, 14 tests (44 subtests) |
| `test_minimax_phase5.py` | unittest and pytest | pass, 6 |
| `test_agents.py` | unittest | pass, 13 |
| `test_agent_conformance.py` | unittest and pytest | pass, 6 |
| `test_evaluation_fn.py` | unittest | pass, 5 |
| `test_exact_solver.py` | unittest | pass, 9 |
| `test_golden_regression.py` | unittest | pass, 3 |
| `test_match_runner.py` | unittest | pass, 16 |
| `test_edge_invariants.py` | pytest | pass, 4 |
| `test_greedy_campaign.py` | pytest | pass, 4 |

## Task 1: compatibility with the frozen engine

All in `tests/test_minimax_final_compat.py`.

| Requirement | Test | How it checks | Result |
|---|---|---|---|
| Never an illegal action | `test_full_games_only_legal_actions` | 12 games (6 configs: N 3/4/5, K 1-4, M 0-10, A or B/C off once; minimax vs minimax and vs random) to the end or 60 plies; every minimax action is in `get_legal_actions()` | pass |
| Terminal states | `test_returns_none_on_terminal_states` | after a goal win and after a no-legal-move draw, `select_action` returns `None` | pass |
| No move after the game ends | `test_search_never_applies_an_action_to_a_terminal_engine` | `apply_action` wrapped; depth-3 searches on 15 positions; 0 calls on a terminal engine | pass |
| | `test_negamax_on_terminal_does_not_expand` | `_negamax` on a won position returns `-(1000-1)` without calling `get_legal_actions` | pass |
| K-ply blocks | `test_block_positions_match_reference_search` | positions with live blocks for K = 1, 2, 3, 4 (N=4, M=1): root value equals a plain full-width negamax written in the test over the engine API | pass |
| | `test_block_expires_inside_search_tree` | a timer-1 block is gone one ply later; the search then sees the MOVE through that cell as a win (`999`) | pass |
| | `test_full_games_only_legal_actions` | after every ply all timers are in `1..K` | pass |
| Wall budget `rules['M']` | `test_wall_budget_respected` | M = 0 and 1: no WALL with an empty inventory, never more than `2*M` walls | pass |
| No-legal-move draw | `test_no_legal_move_draw_scores_zero` | `evaluate` and `_negamax` give exactly `0.0` on the `no_legal_actions` draw | pass |
| New HOP rule | `test_hop_across_opponent_wall_found_as_win` | opponent wall in front: minimax picks `('HOP', (2, 1))` with value `999` | pass |
| | `test_no_hop_across_own_wall` | same wall owned by the mover: no HOP chosen | pass |

Why minimax follows the frozen rules by construction: it reads legality only from
`engine.get_legal_actions()` (`agents/minimax.py:208,268,345`), simulates with `clone()` +
`apply_action(update_clock=False)` (`:287-288,363-364`) and reads terminal state from
`engine.winner` / `engine.is_draw` (`:330`). Blocks, HOP ownership, the wall budget and both draw
rules therefore come from the engine. Its only own rule-like code is the opponent-path BFS used
for ordering (`:141-147`), which does not affect legality.

## Task 2: audit

| Component | Verdict | Evidence |
|---|---|---|
| Negamax perspective | **correct** | Leaf value is for `engine.current_player` (`minimax.py:90,104`); a finished game is `-(1000-ply)` for the side to move, which is the loser (`:93-97`); every child value is negated (`:290,366`). Full-width search agrees exactly (below). |
| Alpha-beta updates | **correct** (fail-hard) | Child window `(-beta, -alpha)` (`:293-294,369-370`); `alpha = max(alpha, val)` and cut at `beta <= alpha` (`:305-306,380-381`); interior nodes return `alpha` (`:385,387`), the root keeps the exact best value. 118 positions, 5 suites (N 3-5, depth 2-4): root value of AB+ordering and AB unordered equals full-width minimax, max difference `0.0`; unordered AB picked the same move as full width in 118/118 (`alphabeta_equivalence.json`). |
| Move ordering | **works but weak** | Order: root PV, immediate goal moves, pawn moves by row progress, walls on the opponent's shortest path, other walls, blocks (`:149-187`). Pawn "BFS progress" is really row distance (`:166`; comment at `:160-165` says BFS); `cur_d_self` (`:137`) is computed and never used. The history table is shared by both players (`:308,384`). The engine already lists pawn moves first, so the unordered baseline is not random. Measured gain is small (Task 4). |
| Evaluation function | **distance only** | Default `HEvaluator`: `tanh(0.2 * (d_opp - d_self))` (`evaluation_fn.py:35-43`). Walls left, REPLACE, hop and block are **not** scored. Blocks enter only through BFS distance (an expiring block counts as a full obstacle). `LinearEvaluator` (`:46-61`) uses more features but is not the default and is not used in the runs here. Range (-1, 1) stays far below win scores (`1000-ply`). |
| Draw never scored as forced win/loss | **correct** | Engine draws (`no_legal_actions`, repetition) score `0.0` (`minimax.py:99-100`). Against exact retrograde solutions of three 3x3 configs (4,812 / 12,821 / 188,172 states): depth 2 on every non-terminal state with more than one legal action and depth 4 on a random sample of DRAW states. 0 DRAW states claimed as a forced win or loss; 0 wrong forced-win or forced-loss claims among 46,260 win and 9,565 loss claims (`exact_draws_3x3.json`). |
| Time management | **correct, small overshoot** | Deadline checked at every node and before every child (`:237,283,326,359`); `SearchTimeout` aborts the depth (`:251`). N=9, 15 positions, limits 0.1/0.25/0.5/1.0 s: mean overshoot 1.0-1.2 ms, max 2.8 ms (`timing_N9.json`). The match runner times with `perf_counter`, the agent with `time.time`. |
| Iterative deepening | **correct** | Only with `time_limit` (`:234`). A depth's result is kept only if it finished (`:240-246`); a timeout keeps the previous depth's move. Tests: `test_iterative_deepening_keeps_last_completed_depth` (forced timeout at depth 2 returns the depth-1 move and value), `test_iterative_deepening_value_matches_fixed_depth`. If depth 1 itself times out the move is `legal_actions[0]` or a goal move (`:221-225`) and the value is `-inf`. |
| Goal-proximity extension | **works, off by default** | When the horizon is reached and either pawn is at most 2 from goal, search one more ply, at most 2 times per line (`:334-341`). `test_goal_extension_sees_win_beyond_horizon`: depth 1 without it sees a heuristic value; with it, a forced win in 3 plies (`997`). It is a fixed extension, not a quiescence search, and `extend_goal` is `False` in every run here (`match_runner.py` `make_agent`). |

Other notes (no change made):
- With exactly one legal action the agent returns it without search and reports depth 1 and
  0 nodes (`minimax.py:214-218`). These decisions are inside the Task 3 means.
- The search does not know the runner's 200-ply cap, so near the cap it can value a win it cannot reach.

## Task 3: Minimax vs Greedy on the final engine

**Design change (Mr. CEO, from Bot 3's finding).** `minimax:depth=2` without a time limit and
`greedy` are both deterministic, so a match is fixed by its opening. With `opening_plies=2`
on N=5, seeds 0-49 give only **9 distinct openings** (18 distinct games out of 100;
`opening_maps.json`). The main run therefore uses `opening_plies=4`, the smallest tried value
that reaches 50 distinct openings: seeds were scanned from 0 upward and the first seed of each
distinct opening kept (seeds 0-79 scanned; 50 kept). `opening_plies=6` was not needed.
All rates below are over distinct games only.

### Main run: N=5, A1B1C1, K=3, M=10, `minimax:depth=2` vs `greedy`

50 distinct 4-ply pawn-move openings, each played with minimax as P0 and as P1 (100 matches,
100 distinct games), runner cap 200 plies, no time limit, deterministic.
Files: `experiments/results/final_minimax/vs_greedy_N5_open4_distinct_summary.json`
(summary, seed list), `vs_greedy_N5_open4_distinct.jsonl` (matches with replays),
seed -> opening map in `opening_maps.json`. `verify-replay`: 100/100 verified.

| | Minimax as P0 | Minimax as P1 | Overall |
|---|---|---|---|
| Matches (distinct games) | 50 | 50 | 100 |
| Minimax goal wins | 20 | 19 | 39 |
| Greedy goal wins | 29 | 31 | 60 |
| Draws: repetition / no legal action | 1 / 0 | 0 / 0 | 1 / 0 |
| Truncated (200-ply cap) | 0 | 0 | 0 |
| Forfeits | 0 | 0 | 0 |
| Minimax win rate (95% Wilson) | 40% (27.6-53.8%) | 38% (25.9-51.8%) | **39% (30.0-48.8%)** |
| Draw rate / truncation rate | 2% / 0% | 0% / 0% | 1% / 0% |
| Mean game length, plies (incl. 4 opening) | 17.56 | 19.38 | 18.47 |
| Minimax decisions | 349 | 369 | 718 |
| Mean depth completed | 2.000 | 1.995 | 1.997 |
| Mean nodes per decision | 174.9 | 180.5 | 177.8 |
| Mean decision time | 28.7 ms | 29.1 ms | 28.9 ms |

Depth-2 minimax **loses** to Greedy on the final engine (39 vs 60). Mean depth is below 2
because a decision with a single legal action is returned without search and counted as depth 1.

**Same 50 openings, other depths (diagnostic, not the main run;**
`vs_greedy_N5_open4_depth1_diag_summary.json`, `vs_greedy_N5_open4_depth3_diag_summary.json`):

| Spec | Minimax wins | Greedy wins | Draws (all repetition) | Truncated | Win rate (95% Wilson) |
|---|---|---|---|---|---|
| `minimax:depth=1` | 45 | 46 | 9 | 0 | 45% (35.6-54.8%) |
| `minimax:depth=2` (main) | 39 | 60 | 1 | 0 | 39% (30.0-48.8%) |
| `minimax:depth=3` | 45 | 48 | 7 | 0 | 45% (35.6-54.8%) |

No depth beats Greedy, and the intervals overlap, so this does **not** show that deeper search is worse.

**Observed weakness.** HOP may only cross a wall the opponent owns, so every Greedy HOP in these
games crosses a minimax wall. At depth 2, 30 of Greedy's 60 wins contain a Greedy HOP; minimax
placed 207 walls over the 100 games (`hop_stats.json`). The leaf evaluation measures `d_opp` by
BFS and ignores the opponent's unused HOP, so it overrates its own walls while that HOP is
unused. This is an evaluation limit under the new HOP rule, not a search bug (the search is exact
against full width). It was not changed: the evaluation is a team design decision.

**Superseded first run (kept, not used for claims).** Before the design change I ran the brief's
original design (`opening_plies=2`, seeds 0-49, 100 matches; `vs_greedy_N5_summary.json`): 35
minimax wins, 65 Greedy wins. Its 100 matches are only 18 distinct games, so its counts and
interval are not valid samples. The 20-match depth-1/depth-3 checks on seeds 0-9
(`vs_greedy_N5_depth1_diag_summary.json`, `vs_greedy_N5_depth3_diag_summary.json`) are also
only 16 distinct games each, and are superseded by the table above.

Known cosmetic defect: in `vs_greedy_N5_open4_distinct_summary.json` and
`vs_greedy_N9_timed_summary.json` (made at `f5aa757`) the `config.seed_to_opening` field spells
`MOVE` as a list of letters. Fixed in `d46fc5c`; the correct map for every run is in
`opening_maps.json`, and the match logs themselves are correct.

### Larger board: N=9, A1B1C1, K=3, M=10, `minimax:depth=2,time_limit=1.0` vs `greedy`

Opening seeds 0-9, `opening_plies=2`, both seatings, 20 matches, cap 200 plies.
**Timed play is not deterministic**: the depth reached depends on machine load (two other bots
were running), so a rerun can differ. Seeds 0-9 give **8 distinct openings** (seed 3 = seed 1,
seed 8 = seed 4), and those repeats produced the same games here, so the 20 matches are
**16 distinct games** (`opening_maps.json`).
Files: `vs_greedy_N9_timed_summary.json`, `vs_greedy_N9_timed.jsonl`. `verify-replay`: 20/20 verified.

| | Minimax as P0 | Minimax as P1 | Overall |
|---|---|---|---|
| Matches | 10 | 10 | 20 |
| Minimax goal wins | 8 | 6 | 14 |
| Greedy goal wins | 2 | 4 | 6 |
| Draws / truncated / forfeits | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| Minimax win rate, all 20 (95% Wilson) | 80% (49.0-94.3%) | 60% (31.3-83.2%) | 70% (48.1-85.5%) |
| Minimax win rate, 16 distinct games (95% Wilson) | 6/8 (40.9-92.9%) | 5/8 (30.6-86.3%) | **11/16 (44.4-85.8%)** |
| Mean game length, plies (incl. 2 opening) | 18.4 | 25.4 | 21.9 |
| Minimax decisions | 86 | 115 | 201 |
| Mean depth completed | 1.91 | 1.95 | 1.93 |
| Mean nodes per decision | 890.0 | 877.6 | 882.9 |
| Mean decision time | 403 ms | 325 ms | 359 ms |

The distinct-game row is `distinct_game_results` for `N9_timed` in `opening_maps.json`
(repeated games dropped; Greedy won 5 of the 16). With 16 games the interval is wide;
this is a larger-board indication, not an exact finding. All 6 Greedy wins contain a Greedy HOP
(`hop_stats.json`).

## Task 4: search efficiency (move ordering on vs off)

Ordering can be switched off without code changes: `MinimaxAgent(order_moves=False)` uses the
engine's own action order (`minimax.py:272-275,351-354`). No flag was added.

Positions: the exact generator of `scripts/benchmark_ordering_x8.py` (`random.seed(42)`, 2-6
random plies of any action, N=5, default rules A1B1C1 K3 M10), 50 positions. The engine's
HOP rule changed since the old run, so these 50 positions are not the old 50.
Alpha-beta on in both arms; root values were equal in every position.
File: `experiments/results/final_minimax/ordering.json`.

| Depth | Nodes off | Nodes on | Reduction (total) | 95% bootstrap CI | Per-position median (p10, p90) | Positions where on used more | Mean time off / on |
|---|---|---|---|---|---|---|---|
| 1 | 2,892 | 2,892 | 0.0% | 0.0-0.0% | 0.0% | 0 | 5.4 / 5.5 ms |
| 2 | 10,349 | 9,477 | **8.4%** | 3.5-14.2% | 0.9% (0.0, 22.2) | 1 | 34.9 / 38.4 ms |
| 3 | 209,298 | 201,752 | **3.6%** | 2.1-5.2% | 2.1% (-0.2, 9.2) | 7 | 398 / 404 ms |

Depth 1 cannot change: the root never cuts (beta stays infinite), so every root child is searched.

Under a 0.5 s budget (depth cap 6, same 50 positions) both arms reached the same mean depth
(2.56; 2 positions deeper with ordering, 2 deeper without).

**The old 22.66% reduction is not reproduced.** On the frozen engine ordering saves 8.4% of
nodes at depth 2 and 3.6% at depth 3. It does not save time: the ordering BFS costs more than
the pruned nodes (34.9 ms off vs 38.4 ms on at depth 2).

## Exact small-board findings vs larger-board claims

- Exact (3x3, solved graphs): no draw treated as a forced result; no wrong forced claim
  (`exact_draws_3x3.json`).
- Exact equality of alpha-beta and full width holds on N 3-5 positions tested
  (`alphabeta_equivalence.json`). This is a check of the search code, not a claim about play strength.
- Larger boards (N=5, N=9): match results and node counts are empirical; see Tasks 3 and 4.

## Independent break attempt

A fresh subagent saw only the diff, the two markdown files and the result files, and tried to
find one wrong number or one unsound audit claim. It ran short python checks from the worktree root.

| What it tried | Result |
|---|---|
| 24 number groups against the cited JSON: N5 main counts, all Wilson intervals (recomputed with `evaluation/metrics.py` `wilson()`), lengths, decisions, nodes, times; depth-1/3 diagnostics; hop stats; 46,260 / 9,565 exact claims; 118-position equivalence and node counts; overshoot and depth at N9; ordering totals, reductions, CIs, per-position spread, times, timed depth; N9 counts, distinct games (8 openings, 16 games, 11/16) and per-seat intervals; 2-ply and 4-ply distinct counts | all matched |
| 14 audit claims against `agents/minimax.py` lines (fail-hard return, child window, cut test, deadline checks, ID keeps last depth, fallback, draw = 0, terminal loss value, ordering categories, unused `cur_d_self`, row progress, shared history, single-action shortcut, ordering off) | all held |
| Agent and engine unchanged vs `main` | held (`git diff` of `agents/` and `wallz_engine.py` empty) |
| "Deeper is worse" not claimed for the main run | held |
| `python -m unittest tests.test_minimax_final_compat` | 14 tests OK |

**Found and fixed:** (1) the report section said ordering "never changes the result". That holds
for the root value only; ordering can change which of several equal-valued moves is played, and
under a time limit the depth reached. Reworded. (2) Two line ranges in this report were off by
one (`:140-146`, `:155-184`); corrected to `:141-147` and `:149-187`.

Not checked by the reviewer: the other test files, replay verification, and the shell-only scan
counts for 6-ply openings.