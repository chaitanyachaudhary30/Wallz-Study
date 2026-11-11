# Member 3: Exact Solver, Evaluation Framework and Statistical Analysis

All numbers below come from files in `experiments/results/final_eval/` (named in brackets).
They were produced on the frozen Stage 1 rules (base `main` = `70e2351`) from clean commits
`f930e8e`, `8944f16` and `4ef8124`; each file records its commit, command, Python version and date.
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

Both agents are deterministic, so a game is fixed by its opening (a few random pawn moves).
With 2 opening plies only 9 openings exist, and the pilot's 10 seeds drew just 8 of them
[`analysis.json` `opening_counts_N5`]: two seeds replayed games already played, so the pilot's
effective sample was 8 openings, not 10 seeds.

The design therefore counts **distinct openings**. The shortest opening with at least 30
distinct versions is 4 plies (99 exist; 2 plies gives 9). Seeds are scanned in order and only
the first seed of each distinct opening is kept; the seed -> opening map is stored with the
results. The primary run uses the first **40 distinct 4-ply openings**, both seatings, all 8
cells: 640 matches [`ablation_N5_K3_M10_open4/summary.json`]. Two more runs are reported: all
9 distinct 2-ply openings (144 matches), the direct comparison with the pilot
[`ablation_N5_K3_M10_open2_pilot_comparison/summary.json`], and 40 distinct 6-ply openings
(640 matches) as a check of how much the opening length matters
[`ablation_N5_K3_M10_open6/summary.json`].

The statistical unit is the opening: its games in all 8 cells and both seatings stay together
when resampled. Intervals are percentile intervals from 10,000 paired bootstrap draws;
p-values are two-sided bootstrap p-values; Holm's method controls the family-wise error over the
7 effects at 5%. All 640, 144 and 640 replays verified on the final engine [`verify_replay.txt`
in each folder]. No game was truncated and none was forfeited.

## 6. Multi-seed analysis

**Method.** Five DQN agents trained on 3x3 (N3 A1B1C1 K1 M1) with seeds 42-46 are each
compared with the exact solve (share of decided positions where the network's value has the
right sign, and mean absolute value error) and play 100 paired games against Greedy and against
Random (`scripts/run_multi_seed_3x3.py`). Per seed: win, draw and truncation rates with Wilson 95%
intervals and mean game length. Across seeds: mean, sample SD, a 95% t-interval
(mean +/- 2.776 x SD / sqrt(5)), and leave-one-seed-out means, which show whether one seed drives
the result. Truncated games (the study's 100-ply cap) are a separate column, never draws.

> **Results: pending Bot 2 checkpoints.** The five checkpoints
> (`checkpoints/final/3x3_s<seed>/final.pt`) do not exist yet. This part will be filled from
> `experiments/results/final_eval/multi_seed_3x3.json` in the later integration item.

## 7. Factorial ablation results

Minimax's goal-win rate per cell, primary run [`ablation_N5_K3_M10_open4/summary.json` `cells`]:

| Cell | Win rate (95% Wilson) | Draw rate | Truncation rate | Mean plies |
|---|---|---:|---:|---:|
| A0B0C0 | 0.713 (0.605-0.800) | 0.075 | 0 | 24.1 |
| A0B0C1 | 0.638 (0.528-0.734) | 0.088 | 0 | 26.2 |
| A0B1C0 | 0.575 (0.466-0.677) | 0.025 | 0 | 15.3 |
| A0B1C1 | 0.500 (0.393-0.607) | 0 | 0 | 16.3 |
| A1B0C0 | 0.625 (0.515-0.723) | 0.213 | 0 | 41.9 |
| A1B0C1 | 0.638 (0.528-0.734) | 0.163 | 0 | 34.4 |
| A1B1C0 | 0.425 (0.323-0.534) | 0.013 | 0 | 15.3 |
| A1B1C1 | 0.413 (0.311-0.522) | 0.013 | 0 | 18.2 |

(80 games per cell; exact rates such as 0.7125 and 0.0875 are rounded half up to 3 decimals.)

Factorial effects on the win rate [`summary.json` `effects.x_goal_win`]:

| Effect | Estimate | 95% CI | Holm p |
|---|---:|---|---:|
| A | -0.081 | (-0.147, -0.019) | 0.070 |
| **B** | **-0.175** | (-0.259, -0.088) | < 0.001 |
| C | -0.037 | (-0.072, -0.006) | 0.111 |
| AB | -0.037 | (-0.078, +0.003) | 0.275 |
| AC | +0.037 | (+0.000, +0.075) | 0.234 |
| BC | -0.006 | (-0.047, +0.034) | 1.000 |
| ABC | -0.006 | (-0.031, +0.019) | 1.000 |

B makes depth-2 Minimax weaker against Greedy (about 18 points of win rate), lowers the draw rate
by about 12 points [`effects.rule_draw`] and shortens games by about 15 plies
[`effects.length_plies`]. A makes games about 7 plies longer. A and C may lower Minimax's win
rate a little (raw intervals exclude zero), but neither survives Holm. The pilot's AB
interaction (+0.11) is not confirmed: -0.037 here (Holm p 0.28) and -0.042 with the pilot's own
2-ply openings.

How the win-rate effects change with the opening length (bold = survives Holm):

| Effect | 2 plies, 9 openings | 4 plies, 40 openings (primary) | 6 plies, 40 openings |
|---|---:|---:|---:|
| A | -0.014 | -0.081 | -0.037 |
| B | **-0.403** | **-0.175** | **-0.413** |
| C | -0.069 | -0.037 | **-0.088** |
| BC | +0.014 | -0.006 | **+0.056** |

B lowers Minimax's win rate in all three, but by very different amounts. C and BC are supported
only with 6-ply openings. So the size of each effect depends on which openings are played.

## 8. Statistical significance and limitations

- **Which effects are supported (primary run).** On the win rate only B survives Holm at 5%;
  A, C, AB, AC, BC and ABC do not. On the loss rate B (+0.297) and AB (+0.091) survive
  [`effects.x_goal_loss`]. On the draw rate only B survives. On game length A, B, AB, AC, BC and
  ABC survive; C does not.
- **Power.** With 40 independent openings, the smallest win-rate effect found with 80% power is
  0.157 at a 5% level and 0.197 at Holm's strictest step, in the worst case [`analysis.json`
  `mde_win_rate_80pct_power`]. The paired standard errors are smaller (detectable effects of
  0.04-0.12 by effect). A (-0.081) and C (-0.037) are below their own detectable sizes (0.092 and
  0.047), so the study could miss effects of that size. For the 2-ply design (9 openings) the
  worst-case bound is 0.33 and the paired estimates 0.06-0.20, so only large effects are found.
- **Effective sample of the pilot.** The pilot's 10 seeds held 8 distinct openings, but its
  bootstrap resampled 10 seeds as if they were independent. Its own match log is gone, so the
  effect on its intervals cannot be checked.
- **Opening length.** Effect sizes differ between 2-, 4- and 6-ply openings (Section 7), so they
  hold for this opening distribution only.
- **Scope.** One pairing (depth-2 Minimax vs Greedy), one board (5x5), one K and M. The results
  describe these two agents, not the game's true value; the exact values hold only for the solved
  3x3 and 4x4 cells.
- **Capped cells.** The four C-on cells of N = 4, M = 1 were not solved; no claim is made about them.
- **Bootstrap p-values** have a resolution of 1/5,000 with 10,000 draws; "< 0.001" means no draw
  fell on the other side of zero.
