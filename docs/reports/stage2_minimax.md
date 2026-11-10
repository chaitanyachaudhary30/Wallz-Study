# Stage 2 Minimax Validation Report (Member 1, Tasks 1-5)

- **Engine and agent tested:** branch `bot1-vedant/minimax-validation`, based on `main` = `70e2351`
  (Stage 1 engine freeze). Code under test is unchanged from `main`: `agents/minimax.py`,
  `agents/evaluation_fn.py`, `wallz_engine.py`.
- **Commit that produced every result:** `4023eac` (tree clean; recorded as `source.commit` /
  `source.dirty` in each JSON). That commit only adds the test file and `scripts/final_minimax.py`.
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
TASK3_COMMANDS
```

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
for ordering (`:140-146`), which does not affect legality.

## Task 2: audit

| Component | Verdict | Evidence |
|---|---|---|
| Negamax perspective | **correct** | Leaf value is for `engine.current_player` (`minimax.py:90,104`); a finished game is `-(1000-ply)` for the side to move, which is the loser (`:93-97`); every child value is negated (`:290,366`). Full-width search agrees exactly (below). |
| Alpha-beta updates | **correct** (fail-hard) | Child window `(-beta, -alpha)` (`:293-294,369-370`); `alpha = max(alpha, val)` and cut at `beta <= alpha` (`:305-306,380-381`); interior nodes return `alpha` (`:385,387`), the root keeps the exact best value. 118 positions, 5 suites (N 3-5, depth 2-4): root value of AB+ordering and AB unordered equals full-width minimax, max difference `0.0`; unordered AB picked the same move as full width in 118/118 (`alphabeta_equivalence.json`). |
| Move ordering | **works but weak** | Order: root PV, immediate goal moves, pawn moves by row progress, walls on the opponent's shortest path, other walls, blocks (`:155-184`). Pawn "BFS progress" is really row distance (`:166`; comment at `:160-165` says BFS); `cur_d_self` (`:137`) is computed and never used. The history table is shared by both players (`:308,384`). The engine already lists pawn moves first, so the unordered baseline is not random. Measured gain is small (Task 4). |
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

### Main run: N=5, A1B1C1, K=3, M=10, `minimax:depth=2` vs `greedy`

Opening seeds 0-49, 2 random pawn-move opening plies, each opening played with minimax as P0
and as P1 (100 matches), runner cap 200 plies, no time limit, deterministic.
Files: `experiments/results/final_minimax/vs_greedy_N5_summary.json` (summary) and
`vs_greedy_N5.jsonl` (100 matches with replays). `verify-replay`: 100/100 verified.

| | Minimax as P0 | Minimax as P1 | Overall |
|---|---|---|---|
| Matches | 50 | 50 | 100 |
| Minimax goal wins | 25 | 10 | 35 |
| Greedy goal wins | 25 | 40 | 65 |
| Draws: repetition / no legal action | 0 / 0 | 0 / 0 | 0 / 0 |
| Truncated (200-ply cap) | 0 | 0 | 0 |
| Forfeits | 0 | 0 | 0 |
| Minimax win rate (95% Wilson) | 50% (36.6-63.4%) | 20% (11.2-33.0%) | **35% (26.4-44.7%)** |
| Draw rate / truncation rate | 0% / 0% | 0% / 0% | 0% / 0% |
| Mean game length, plies (incl. 2 opening) | 11.66 | 11.56 | 11.61 |
| Minimax decisions | 254 | 219 | 473 |
| Mean depth completed | 2.00 | 2.00 | 2.00 |
| Mean nodes per decision | 204.3 | 193.7 | 199.4 |
| Mean decision time | 27.6 ms | 24.2 ms | 26.0 ms |

Depth-2 minimax **loses** to Greedy overall on the final engine. Mean depth is exactly 2.00
because no decision in these games had a single legal action.

**Diagnostic, not part of the main run** (opening seeds 0-9, both seatings, 20 matches each;
`vs_greedy_N5_depth1_diag_summary.json`, `vs_greedy_N5_depth3_diag_summary.json`):

| Spec | Minimax wins | Greedy wins | Draws | Truncated |
|---|---|---|---|---|
| `minimax:depth=1` | 10 | 8 | 2 (repetition) | 0 |
| `minimax:depth=3` | 5 | 15 | 0 | 0 |

Deeper search does worse. Replays show why: the deeper search spends early plies on walls in
front of Greedy's goal row, and Greedy then crosses one of them with its HOP, which is legal
exactly because minimax owns the wall (e.g. `vs_greedy_N5_depth3_diag.jsonl`, seed 0 with
minimax as P0: walls at plies 5 and 7, a block at ply 9, Greedy's HOP to row 0 at ply 12). The leaf
evaluation measures `d_opp` by BFS and ignores the opponent's unused HOP, so each own wall looks
more useful than it is. This is an evaluation weakness under the new HOP rule, not a search bug
(the search is exact against full width). It was not changed here: changing the evaluation is a
design decision for the team.

N9_SECTION

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
