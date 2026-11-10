# Member 3: Exact Solver, Evaluation Framework and Statistical Analysis

All numbers below come from files in `experiments/results/final_eval/` (named in brackets).
They were produced on the frozen Stage 1 rules (base `main` = `70e2351`) from clean commits
`f930e8e` and `8944f16`; each file records its commit, command, Python version and date.
Exact results on 3x3 and 4x4 boards (Sections 1-3) and match results on 5x5 (Sections 5-7)
answer different questions and are kept apart.

## 1. Exact solver method

The solver answers: with perfect play from the start position, does player 0 win, lose or
draw, and in how many plies?

1. **Graph.** `evaluation/state_graph.py` enumerates every position reachable from the start,
   breadth first. Each successor is made by the engine itself: restore the position, ask
   `get_legal_actions()`, and `apply_action()` each one. The solver contains no rule code.
2. **Positions.** A node is the engine's position key: pawns, walls, wall stocks, each player's
   latest wall, hop and block availability, block timers and side to move. Keys that cannot
   differ in future play are merged (`WallzEngine.canonicalize_key`): the order of fixed walls,
   and wall owners once no hop is left. While a hop remains, owners are kept, because a hop may
   cross only an opponent's wall.
3. **Values.** `evaluation/retrograde.py` labels every node WIN, LOSS or DRAW for the side to
   move, with a rank (plies to the goal under best play). It works backwards from finished games
   and handles cycles (Section 2).
4. **Caps.** A graph larger than the cap (here 400,000 positions or 900 s per cell) is stopped.
   Its unresolved nodes are UNKNOWN, never DRAW, and the cell is reported as capped.

The solved game is the *position game*: reaching the goal row wins, a position with no legal
action is a draw, and endless play is a draw. From a fresh history this equals the engine's
5-fold-repetition rule (a winner following ranks never repeats a position; a defender who stays
in draw positions can keep the game going until repetition). Clocks and the 200-ply runner cap
are not part of it.

## 2. The mathematics of retrograde analysis

Let V(s) in {+1, 0, -1} be the value for the side to move and S(s) its successors. The
values satisfy the negamax equations

- V(s) = -1, rank 0, if the opponent has just reached its goal row (s is terminal);
- V(s) = 0 if s has no legal action;
- otherwise V(s) = max over s' in S(s) of -V(s').

With cycles, these equations have many solutions; the game's value is the one where WIN and
LOSS are the smallest sets that are closed under the rules below (an *attractor*), and every
other node is a draw:

- s is WIN with rank r if some child is LOSS with rank r - 1 (take the fastest such child);
- s is LOSS with rank r if **every** child is WIN, r = 1 + the largest child rank (delay the loss).

The algorithm seeds the queue with terminal nodes (rank 0). It keeps, per node, a count of
distinct children not yet known to be WIN. Popping a LOSS node makes each unlabelled parent WIN.
Popping a WIN node lowers each parent's count; at zero the parent is LOSS. A first-in-first-out
queue handles nodes in rank order, so ranks are exact. When the queue empties on a complete
graph, every unlabelled node is a DRAW: neither side can force the goal from it, and play there
can go on forever. Each edge is touched once, so the cost is linear in the edges.

Two facts make capped graphs safe. A WIN needs only one proven LOSS child, and a LOSS needs all
children of a fully expanded node, so every WIN or LOSS label on a capped graph is still true.
A capped rank is only an upper bound, so rank-based answers there are withheld.

The 2^3 factorial effect of a set of factors F on a measure y is
effect(F) = mean of y over the 4 cells where the product of the +/-1 codes of F is +1, minus the
mean over the other 4. Main effects, two-way and three-way interactions share this scale.

## 3. Solver validation

`tests/test_solver_final_rules.py` (18 tests, all pass, [`test_solver_final_rules.txt`]) checks
the solver on hand-built 3x3 positions and whole small games:

| Case | Expected and observed |
|---|---|
| terminal win / loss | rank 0, LOSS for the side to move, +1 for the player who reached the goal |
| win in 1, forced loss in 2 | WIN rank 1 by the goal move; LOSS rank 2 when every reply allows a win in 1 |
| no legal action | DRAW of type `no_legal_actions` (team rule) |
| cycle | DRAW of type `cycle`; value-keeping play revisits a position |
| block expiry | a timer-1 block still blocks on its last ply and is gone after it (rank changes WIN 1 -> LOSS 2, LOSS 2 -> WIN 3) |
| wall stock | the same position is LOSS 2 with no wall and WIN 5 with one |
| latest-wall REPLACE | LOSS 2 with A off, WIN 5 with A on; older walls never move |
| hop ownership | WIN 1 by HOP across the opponent's wall; WIN 3 across one's own wall or with the hop used |

Three checks show the solver does not simplify the game:
1. every edge is exactly one engine `apply_action` call, every node's edges equal the engine's
   legal list, and replaying each edge on a fresh engine gives the same child (all 5 action types);
2. an independent solver (depth-first enumeration plus value iteration, no shared code) gives the
   same value and rank for every raw position of three small games;
3. two deliberately broken engines (owners dropped from the key; hops over any wall) make 4 and 3
   of the 18 tests fail [`mutation_check_solver_tests.txt`].

The older `agents/exact_solver.py` agrees with `evaluation/retrograde.py` on all 16 cells of
N = 3, K = 1, M = 0 and M = 1: 0 status and 0 rank mismatches, and every group of raw keys merged
into one canonical position has a single value and rank [`crosscheck_N3_K1_M0.json`,
`crosscheck_N3_K1_M1.json`]. For example, N3 A0B0C0 M1 has 1,953 raw keys merged into 1,341
positions. (That adapter now runs on the same graph code, so this check covers key merging; the
independent solver in item 2 covers the algorithm.)

**Results on the final rules** [`solve_N3_K1_M0.json`, `solve_N3_K1_M1.json`,
`solve_N4_K1_M0.json`, `solve_N4_K1_M1.json`, summarised in `analysis.json`]:

| Setting | Complete cells | Root (player 0) | Largest complete cell |
|---|---|---|---|
| N3 K1 M0 | 8 of 8 | WIN in 3 without C, WIN in 5 with C | 642 positions |
| N3 K1 M1 | 8 of 8 | WIN without A (rank 5-9), DRAW with A | A1B1C1: 150,710 positions, 956,644 edges, 50.6 s |
| N4 K1 M0 | 8 of 8 | LOSS in 6 without C, LOSS in 8 with C | 4,512 positions |
| N4 K1 M1 | 4 of 8 (all C-off) | A0: LOSS (rank 10, 8); A1: DRAW; C-on cells **UNKNOWN** (capped) | A1B1C0: 372,160 positions |

Every B-off cell equals the pilot exactly; every B-on change follows from the new HOP rule
[`analysis.json` `pilot_comparison`]. With M = 0 there are no walls, so HOP can never be played
and B-on cells equal B-off cells. Exact factorial effects [`analysis.json` `exact_effects`]: on
N3 M1, A changes player 0's root value by -1 (win -> draw), B and C by 0; C multiplies the state
count by about 12 (log10 effect +1.083), B by about 3.4 (+0.535), A by about 2.7 (+0.433), with
log-scale interactions within 0.021 of zero. N4 M1 has no factorial effects because 4 cells are
capped.

## 4. Evaluation framework

- **Match runner** (`evaluation/match_runner.py`): plays one game between two agent specs
  (`greedy`, `minimax:depth=d`, `random`, `exact`, `dqn:path=...`), checks every action against
  the engine's legal list, and logs the full replay with a state id after each ply. Opening
  pawn moves come from a seed and are the same in all 8 cells, so cells are paired.
- **Outcomes**, one per game: goal win, goal loss, rule draw (repetition or no legal action),
  **truncated** (the runner's 200-ply cap; kept as its own column, never a draw) and forfeits.
- **Replay verification**: `verify-replay` re-simulates each logged game on the current engine
  and checks every action, state id and the outcome.
- **Metrics** (`evaluation/metrics.py`): per cell, win, draw and truncation rates with Wilson 95%
  intervals and mean game length with SD; 2^3 factorial effects; paired bootstrap intervals;
  Holm correction over the 7 effects.

## 5. Experimental method

The match ablation plays depth-2 Minimax against Greedy on 5x5, K = 3, M = 10, in all 8 cells,
both seatings, 200-ply cap.

Both agents are deterministic, so a game is fixed by its opening. With 2 opening plies only 9
openings exist, and the pilot's 10 seeds drew 8 of them [`analysis.json` `opening_counts_N5`]:
replica seeds added no evidence. The primary design therefore uses **6 opening plies and 40
distinct openings** (seeds 10-49): 40 x 2 seatings x 8 cells = 640 matches
[`ablation_N5_K3_M10_open6/summary.json`]. The 2-ply design was rerun for comparison with 30
seeds, which gave 9 openings and 144 distinct games in 480 logged matches; all replicas were
identical [`ablation_N5_K3_M10_open2/summary.json` `replicas`].

The statistical unit is the opening: its games in all 8 cells and both seatings stay together
when resampled. Intervals are percentile intervals from 10,000 paired bootstrap draws;
p-values are two-sided bootstrap p-values; Holm's method controls the family-wise error over the
7 effects at 5%. All 640 and 480 replays verified on the final engine [`verify_replay.txt` in each
folder]. No game was truncated and none was forfeited.

## 6. Multi-seed analysis

Not available yet. The 5-seed study (`scripts/run_multi_seed_3x3.py`) trains a DQN agent per
seed, which is another member's task; it was not run here, so this section has no numbers.
When the five checkpoints exist, the plan is: per seed, win, draw and truncation rates with
Wilson intervals and mean game length; across seeds, mean, sample SD and a t-interval
(mean +/- 2.776 x SD / sqrt(5)); and a leave-one-out check of whether one seed drives the mean.
The script currently counts truncated games as draws and caps games at 100 plies; both need
fixing first (`docs/reports/stage2_solver_stats.md` Section 3).

## 7. Factorial ablation results

Minimax's goal-win rate per cell [`ablation_N5_K3_M10_open6/summary.json` `cells`]:

| Cell | Win rate (95% Wilson) | Draw rate | Truncation rate | Mean plies |
|---|---|---:|---:|---:|
| A0B0C0 | 0.863 (0.770-0.921) | 0.063 | 0 | 27.1 |
| A0B0C1 | 0.738 (0.632-0.821) | 0.063 | 0 | 27.7 |
| A0B1C0 | 0.363 (0.266-0.472) | 0.013 | 0 | 15.1 |
| A0B1C1 | 0.325 (0.232-0.434) | 0 | 0 | 17.2 |
| A1B0C0 | 0.800 (0.700-0.873) | 0.075 | 0 | 39.0 |
| A1B0C1 | 0.638 (0.528-0.734) | 0.125 | 0 | 37.8 |
| A1B1C0 | 0.363 (0.266-0.472) | 0.063 | 0 | 18.1 |
| A1B1C1 | 0.338 (0.243-0.446) | 0.075 | 0 | 22.7 |

(80 games per cell; exact rates such as 0.8625 and 0.0625 are rounded to 3 decimals.)

Factorial effects on the win rate [`summary.json` `effects.x_goal_win`]:

| Effect | Estimate | 95% CI | Holm p |
|---|---:|---|---:|
| A | -0.037 | (-0.103, +0.025) | 0.787 |
| **B** | **-0.413** | (-0.497, -0.328) | < 0.001 |
| **C** | **-0.088** | (-0.150, -0.028) | 0.010 |
| AB | +0.044 | (-0.000, +0.091) | 0.262 |
| AC | -0.006 | (-0.028, +0.013) | 0.787 |
| **BC** | **+0.056** | (+0.016, +0.109) | 0.007 |
| ABC | +0.013 | (-0.012, +0.037) | 0.787 |

B makes depth-2 Minimax much weaker against Greedy (about 41 points of win rate) and shortens
games by about 15 plies. C costs it about 9 points, and less when B is also on (BC). A makes games
about 8 plies longer [`effects.length_plies`] without a clear effect on who wins. The pilot's AB
interaction (+0.11) is not confirmed: +0.044 here, interval touching zero, Holm p 0.26.

## 8. Statistical significance and limitations

- **Which effects are supported.** On the win rate, B, C and BC survive Holm at 5%; A, AB, AC and
  ABC do not. On the draw rate only AC survives. On game length A, B, C, AB, BC and ABC survive.
- **Power.** With 40 independent openings, the smallest win-rate effect found with 80% power is
  0.157 at a 5% level and 0.197 at Holm's strictest step, in the worst case [`analysis.json`
  `mde_win_rate_80pct_power`]. The paired standard errors are smaller (detectable effects of
  0.03-0.12 by effect), which is why C and BC were found. The 2-ply design, with 9 openings,
  cannot reliably detect anything under 0.33.
- **Replicas.** Counting the 2-ply design's 30 seeds as independent would have made five of seven
  win-rate effects "significant"; by opening, only B is [`ablation_N5_K3_M10_open2/summary.json`
  `effects_by_seed` vs `effects`].
- **Scope.** One pairing (depth-2 Minimax vs Greedy), one board (5x5), one K and M. The results
  describe these two agents, not the game's true value; the exact values hold only for the solved
  3x3 and 4x4 cells.
- **Capped cells.** The four C-on cells of N = 4, M = 1 were not solved; no claim is made about them.
- **Bootstrap p-values** have a resolution of 1/5,000 with 10,000 draws; "< 0.001" means no draw
  fell on the other side of zero.
