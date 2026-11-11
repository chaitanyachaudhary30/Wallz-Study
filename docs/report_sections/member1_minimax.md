# Member 1: Minimax Search with Alpha-Beta Pruning

All numbers below come from files in `experiments/results/final_minimax/`, produced on the
frozen Stage 1 engine (`main` = `70e2351`) with the agent unchanged; each file records its own
commit (see `docs/reports/stage2_minimax.md`). Older minimax numbers (made under
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
REPLACE, blocks; ties go to a history score raised by `depth²` on each cutoff. In a fixed-depth
search ordering never changes the root value (equal in all 50 positions at each depth,
`ordering.json`). It can change which of several equal-valued moves is played, and under a time
limit it can change the depth reached and so the move.

## 6. Time management

With `time_limit` set, the search checks the clock at every node and before every child. On
timeout it abandons the unfinished depth and returns the last finished depth's move. On 15 N=9
positions with limits from 0.1 to 1.0 s, decisions ran over the limit by 1.0-1.2 ms on average and
2.8 ms at most (`timing_N9.json`). With `depth=8, time_limit=1.0` the mean depth finished at N=9
was 1.87 (`timing_N9.json`).

## 7. Results against Greedy

**Main result (exact game count).** N=5, all three modifications on, K=3, M=10,
`minimax:depth=2` vs `greedy`, no time limit. Both agents are deterministic, so a game is fixed
by its opening; 50 distinct 4-ply pawn-move openings were each played with both seatings,
giving 100 distinct games (`vs_greedy_N5_open4_distinct_summary.json`; seed -> opening map in
`opening_maps.json`; all 100 replays verified):

| | Minimax as P0 | Minimax as P1 | Overall |
|---|---|---|---|
| Minimax wins / Greedy wins | 20 / 29 | 19 / 31 | 39 / 60 |
| Draws (repetition) / truncated | 1 / 0 | 0 / 0 | 1 / 0 |
| Minimax win rate (95% Wilson) | 40% (28-54%) | 38% (26-52%) | 39% (30-49%) |
| Mean game length (plies, incl. opening) | 17.6 | 19.4 | 18.5 |
| Mean nodes / mean time per decision | 175 / 28.7 ms | 181 / 29.1 ms | 178 / 28.9 ms |

Depth-2 minimax is weaker than the one-ply Greedy baseline on N=5. On the same 100 games,
depth 1 scored 45-46 with 9 draws and depth 3 scored 45-48 with 7 draws, both 45% (36-55%)
(`vs_greedy_N5_open4_depth1_diag_summary.json`, `vs_greedy_N5_open4_depth3_diag_summary.json`).
No depth tested beats Greedy, and the three intervals overlap.

A likely cause is the evaluation. A HOP may only cross a wall the opponent owns, so every Greedy
HOP crosses a minimax wall. At depth 2, 30 of Greedy's 60 wins contain a Greedy HOP, and minimax
placed 207 walls in the 100 games (`hop_stats.json`). The distance term ignores the opponent's
unused HOP and so overrates its own walls.

**Larger board (empirical, not deterministic).** N=9, same rules, `minimax:depth=2,time_limit=1.0`,
opening seeds 0-9 with 2-ply openings, both seatings: 20 matches but only 8 distinct openings and
16 distinct games (`opening_maps.json`). Minimax won 14 of 20 (8 of 10 as P0, 6 of 10 as P1;
`vs_greedy_N9_timed_summary.json`), 11 of the 16 distinct games (95% Wilson 44-86%;
`distinct_game_results` in `opening_maps.json`). It finished depth 2 in most decisions (mean 1.93)
at 359 ms per decision. Other bots' jobs shared the machine during the run, so these decision
times are load-affected. A rerun beside another bot's DQN training, with the load logged at both
ends, was slower (522 ms, mean depth 1.89) but produced the same 20 games move for move
(`vs_greedy_N9_timed_r2_summary.json`, `n9_r2_load.json`). Timed search depends on machine load, so a rerun can give other games.

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

## New contribution (Wallz-Study): hop-aware evaluation

This is new work added after the audit above. The sections above and their numbers describe the
default scorer and are unchanged. Full details: `docs/reports/stage3_minimax_hop_eval.md`. All
numbers below come from `experiments/results/final_minimax_hop/`.

**Idea.** The default distance ignores that a player with an unused HOP may cross one wall owned
by the other player. The optional `HopAwareEvaluator` (spec `minimax:...,eval=hop`) keeps the same
formula, `tanh(0.2 · (d_opp − d_self))`. While a player's hop is unused, it measures that player's
distance as the shortest path crossing at most one opponent-owned wall. Wall ownership and edges
come from the engine. On 240 random positions this distance equals a search built only from the
engine's own legal MOVE and HOP actions (`tests/test_minimax_hop_eval.py`). The default spec is
unchanged: a full rerun of the 100-game N=5 main run reproduced every game exactly
(`default_repro_diff.json`).

**Results** (N=5, the same 50 distinct openings, both seats, 100 games per row; old-scorer
numbers from Section 7; `paired_d{1,2,3}_default_vs_hop.json`):

| Depth | Old scorer vs Greedy | Hop scorer vs Greedy | Difference (paired bootstrap 95% CI) |
|---|---|---|---|
| 1 | 45 wins | 50 wins (50-50) | +5 points (-1 to +11) |
| 2 | 39 wins | 50 wins (50-48, 2 draws) | +11 points (+1 to +21) |
| 3 | 45 wins | 50 wins (50-46, 4 draws) | +5 points (-5 to +15) |

Head to head at depth 2, the hop scorer beat the default scorer 54-44 with 2 draws (Wilson 44-63%;
`vs_minimax_N5_hop_d2_vs_default_d2_summary.json`). The gain comes entirely from the first seat. As
the second player the hop scorer did the same as the old one at depths 1 and 2, and 6 points worse at depth 3.
It also stops wasting walls: at depth 2, minimax placed 35 walls instead of 207, and Greedy's wins by
hopping a minimax wall fell from 30 to 3 (`hop_stats.json`).

**Larger board (empirical, timed, load-affected).** On N=9 with `time_limit=1.0` the hop scorer won
10 of 20 matches against Greedy (8 of 16 distinct games), against 14 of 20 for the default scorer.
The drop is all in the second seat (60% -> 20%; `paired_N9_timed_default_vs_hop.json`). It placed no
walls at all on N=9. A likely reason, not tested: at depth 2 one new wall can always be hopped while
Greedy's hop is unused, so no wall ever looks useful within the horizon. The hop scorer is therefore
better at N=5 depth 2 but not better everywhere. The default stays the default.
