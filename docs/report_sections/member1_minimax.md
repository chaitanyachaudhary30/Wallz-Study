# Member 1: Minimax Search with Alpha-Beta Pruning

All numbers below come from files in `experiments/results/final_minimax/`, produced at commit
`4023eac` on the frozen Stage 1 engine (`main` = `70e2351`). Older minimax numbers (made under
the previous 2-cell HOP rule) are not used.

## 1. Algorithm

`agents/minimax.py` implements depth-limited game-tree search. From the current position it
generates every legal action through the engine, simulates each one on a copy of the engine, and
recurses until a set depth or a finished game. Leaves get a heuristic score. The agent plays the
root action with the best backed-up score. Without a time limit the search runs at one fixed
depth and is deterministic. With a time limit it runs iterative deepening (depth 1, 2, ... up to
the cap) and plays the result of the deepest depth that finished.

## 2. Negamax formulation

The game is zero-sum, so one function serves both players: the value of a position for the side
to move is the maximum over its actions of minus the value of the child for the opponent,

`V(s) = max_a -V(s · a)`.

Leaves are scored for the side to move. A finished game scores `-(1000 - ply)` for the side to
move (it has lost; the opponent reached its goal), so faster wins and slower losses are
preferred. Rule draws (no legal action for the mover, 5-fold repetition) score 0. A full-width
search over 118 test positions (N = 3-5, depth 2-4) gives the same root value as the pruned search
in every position (`alphabeta_equivalence.json`).

## 3. Alpha-beta pruning

Each call carries a window `(alpha, beta)`; the child is searched with `(-beta, -alpha)`. After
each child `alpha = max(alpha, value)`; when `alpha >= beta` the remaining siblings cannot change
the result and are skipped (a cutoff). The implementation is fail-hard: interior nodes return
`alpha`. The root always keeps the exact best value. Root values with pruning equal full-width
minimax exactly (max difference 0.0 over 118 positions), and pruning cuts the node count from
21,339 to 3,094 on 40 N=4 positions at depth 2 and from 65,697 to 3,576 on 20 N=5 positions at
depth 2 (`alphabeta_equivalence.json`; pruned counts include move ordering).

## 4. Evaluation function

The default leaf score is `U = tanh(0.2 · (d_opp - d_self))`, where `d` is the breadth-first
shortest-path length to the goal row, with walls and currently blocked cells as obstacles and
pawns ignored (`agents/evaluation_fn.py`, `HEvaluator`). It lies in (-1, 1), far from the win
scores, so a proven win always beats any heuristic score. It does not score walls left, the
latest-wall REPLACE option, the unused HOP or the unused BLOCK; those matter only through what
the search sees within its depth.

On three exactly solved 3x3 configurations (4,812, 12,821 and 188,172 positions) the search never
called a drawn position a forced win or loss, and none of its 46,260 forced-win and 9,565
forced-loss claims at depth 2 was wrong (`exact_draws_3x3.json`).

## 5. Move ordering

Actions are tried in this order: the previous iteration's best root move, moves that reach the
goal, pawn moves by row progress, walls on the opponent's current shortest path, other walls and
REPLACE, blocks; ties go to a history score raised by `depth²` on each cutoff. Ordering never
changes the result, only the number of nodes.

## 6. Time management

With `time_limit` set, the search checks the clock at every node and before every child. On
timeout it abandons the unfinished depth and returns the last finished depth's move. On 15 N=9
positions with limits from 0.1 to 1.0 s, decisions ran over the limit by 1.0-1.2 ms on average and
2.8 ms at most (`timing_N9.json`). With `depth=8, time_limit=1.0` the mean depth finished at N=9
was 1.87 (`timing_N9.json`).

## 7. Results against Greedy

N=5, all three modifications on, K=3, M=10, `minimax:depth=2` vs `greedy`, 50 openings
(2 random pawn moves each) played with both seatings, 100 matches, no time limit
(`vs_greedy_N5_summary.json`; all 100 replays verified):

| | Minimax as P0 | Minimax as P1 | Overall |
|---|---|---|---|
| Minimax wins / Greedy wins | 25 / 25 | 10 / 40 | 35 / 65 |
| Draws / truncated | 0 / 0 | 0 / 0 | 0 / 0 |
| Minimax win rate (95% Wilson) | 50% (37-63%) | 20% (11-33%) | 35% (26-45%) |
| Mean game length (plies, incl. opening) | 11.7 | 11.6 | 11.6 |
| Mean nodes / mean time per decision | 204 / 27.6 ms | 194 / 24.2 ms | 199 / 26.0 ms |

Every game ended with a goal. Depth-2 minimax is weaker than the one-ply Greedy baseline, most
of all as the second player. A small diagnostic (20 matches each) gave 10-8 with 2 draws at
depth 1 and 5-15 at depth 3 (`vs_greedy_N5_depth1_diag_summary.json`,
`vs_greedy_N5_depth3_diag_summary.json`), so deeper search plays worse. The replays point to
the evaluation: the deeper search builds walls in front of Greedy's goal, and Greedy crosses one
with its one-time HOP, which is allowed precisely because the wall is minimax's. The distance
term ignores the opponent's unused HOP and so overrates own walls.

N9_SECTION7

## 8. Search efficiency

Same 50 N=5 positions (generator of `scripts/benchmark_ordering_x8.py`), alpha-beta on, ordering
on vs off (`ordering.json`):

| Depth | Nodes off | Nodes on | Reduction | 95% bootstrap CI | Mean time off / on |
|---|---|---|---|---|---|
| 2 | 10,349 | 9,477 | 8.4% | 3.5-14.2% | 34.9 / 38.4 ms |
| 3 | 209,298 | 201,752 | 3.6% | 2.1-5.2% | 398 / 404 ms |

Per position the median saving is 0.9% at depth 2 and 2.1% at depth 3, and at depth 3 ordering
used more nodes in 7 of 50 positions. Under a 0.5 s budget both variants reached the same mean
depth, 2.56. The earlier claim of a 22.66% reduction (made before the HOP rule change) is not
reproduced on the final engine. Ordering costs slightly more time than it saves.

## 9. Limitations

- The evaluation is distance only; walls, REPLACE, HOP and BLOCK left in hand are not valued.
- Expiring blocks count as full obstacles in the leaf distance even one ply before they vanish.
- Move ordering is weak (Section 8): pawn progress is measured by row, not by path, and walls
  that cut the opponent's path rank below every pawn move. The history table is shared by both players.
- The search does not see the match runner's 200-ply cap.
- Timed search is not reproducible: the depth reached depends on machine load.
- The goal-proximity extension exists but is off in all runs; there is no quiescence search.
- Exact checks are on 3x3 boards only; larger-board results are empirical.
