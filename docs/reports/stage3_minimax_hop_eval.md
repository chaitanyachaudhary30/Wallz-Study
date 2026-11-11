# Stage 3: Optional Hop-Aware Minimax Evaluator (Member 1, new contribution)

- **Decision:** team option B. This adds a new, optional scorer, and the original scorer stays the
  default and is unchanged. This is a new contribution of the Wallz-Study project, not part of
  the original Stage 2 audit.
- **Base:** `main` = `330153f` (PR #3 merged). Branch `bot1-vedant/hop-aware-eval`.
- **Code commit that produced every result:** `99cd1ef` (tree clean; `source` in every JSON).
  Results committed in `e180d05`. No engine change. `agents/minimax.py` is unchanged.
- **Results folder:** `experiments/results/final_minimax_hop/`.
- **Machine:** Windows, Python 3.12.5. Other bots shared the machine, so every time is load-affected.

## 1. Design

`HopAwareEvaluator` in `agents/evaluation_fn.py:47-96`, next to `HEvaluator`.

| Part | Behaviour | Code |
|---|---|---|
| Plain case | Rule B off, or the player's hop is used: distance = `engine.shortest_path(player)`, the same BFS the default scorer uses | `:63-65` |
| Hop case | BFS over (cell, hop used). An open edge is a normal step. An edge blocked by a wall owned by the other player (`engine.wall_owner_on_edge(...) == 1 - player`) can be crossed once, onto an on-board cell that is not in `blocked_cells` | `:66-86` |
| Pawns | Ignored, the same as the engine's path rule and the default scorer | docstring `:47-56` |
| Formula | `tanh(0.2 * (d_opp - d_self))`, both distances hop-aware; `0.0` if either is infinite | `:89-96` |
| Terminals | Unchanged: handled by `MinimaxAgent.evaluate` before the scorer is called | `agents/minimax.py:93-100` |

All rule facts come from the engine: wall edges (`engine.board.is_edge_blocked`), wall owner
(`engine.wall_owner_on_edge`), blocked cells, board size and goal row. The engine's extra HOP
condition, that the hopper must still reach its goal after landing, needs no code: a BFS
branch that cannot reach the goal never returns a distance. No engine helper was missing.

Selection: `make_agent` in `evaluation/match_runner.py:131-139`. The spec `minimax:...,eval=hop`
builds the agent with a `HopAwareEvaluator` and adds `'eval': 'hop'` to the logged identity.
Without `eval`, it passes `evaluator=None`, which is the constructor's default, so the agent
and its identity are the same as before. Any other `eval` value raises `ValueError`.

## 2. Tests (`tests/test_minimax_hop_eval.py`, 9 tests)

| Test | Checks | Result |
|---|---|---|
| `test_never_longer_than_plain` | 60 random N=5 positions, both players: hop distance <= plain | pass |
| `test_equal_when_hop_used_or_rule_off_or_no_opponent_wall` | hop used: equal; opponent walls removed: equal; 30 positions with Rule B off: equal | pass |
| `test_hand_built_shortcut_over_opponent_wall_only` | walls `(0,1,H)`, `(0,3,H)` in front of P0 at (0,2): plain 6, hop 4 when the walls are P1's; 6 when they are P0's own; 6 once the hop is used | pass |
| `test_blocked_landing_cell_gives_no_shortcut` | landing cell (1,2) blocked: hop goes sideways first, 5; all four landing cells blocked: equals plain | pass |
| `test_equals_engine_brute_force` | 240 player-positions (N=4 K2 M3, N=5 K3 M10, N=5 K1 M4 A off): equal to a BFS over (cell, hop available) whose neighbours are the engine's own `pawn_actions(player)` with the pawn placed on that cell and the other pawn fixed. The other pawn's cell is added to `blocked_cells` on a copy for the scorer side, since the engine's MOVE/HOP cannot enter it. At least one position must get a shorter hop distance. | pass |
| `test_default_spec_unchanged` | `minimax:depth=2` gives an `HEvaluator` and identity `{'depth': 2, 'time_limit': None, 'extend_goal': False}` | pass |
| `test_hop_spec_selects_hop_evaluator` | `eval=hop` gives a `HopAwareEvaluator`; `eval=nope` raises | pass |
| `test_default_agent_matches_plain_minimax` | 15 positions: `make_agent('minimax:depth=2')` picks the same move and value as `MinimaxAgent(p, depth=2)` | pass |
| `test_hop_evaluator_scores_hop_shortcut` | the hand-built position scores higher for P0 under the hop scorer | pass |

Other test files run one at a time, all pass: `test_match_runner` (16), `test_evaluation_fn` (5),
`test_minimax_final_compat` (14), `test_agents` (13), `test_agent_conformance` (6),
pytest `test_minimax_hop_eval.py` (9) and `test_minimax_phase5.py` (6).

### Default unchanged: full rerun

`minimax:depth=2` vs `greedy` was rerun on PR #3's 50 openings with both seats
(`vs_greedy_N5_default_repro*`) and compared with PR #3's `final_minimax/vs_greedy_N5_open4_distinct.jsonl`
(`default_repro_diff.json`). It compares every ply's action, state id and legal-action count, plus the
outcome, length, final state, agent identities, mean nodes and mean depth. Timings are not compared.
**100 of 100 matches are identical**, and none is missing on either side.

## 3. Results (N=5, A1B1C1, K=3, M=10, cap 200, no time limit)

Same 50 distinct 4-ply openings as PR #3 (seed list read from `final_minimax/opening_maps.json`,
each opening regenerated and checked), both seats, 100 distinct games per row
(`final_minimax_hop/opening_maps.json`). All replays verified (`verify-replay`: 100/100 each).
No truncations and no forfeits in any run.

### Hop scorer vs Greedy

| Run (file) | Seat | Minimax wins | Greedy wins | Draws | Win rate (95% Wilson) | Mean length | Mean depth | Mean nodes | Mean time (load-affected) |
|---|---|---|---|---|---|---|---|---|---|
| `minimax:depth=1,eval=hop` (`vs_greedy_N5_hop_d1`) | P0 | 30 | 20 | 0 | 60% (46.2-72.4%) | 14.88 | 1.0 | 47.3 overall | 5.3 ms overall |
| | P1 | 20 | 30 | 0 | 40% (27.6-53.8%) | 10.40 | 1.0 | | |
| | all | 50 | 50 | 0 | **50% (40.4-59.6%)** | 12.64 | 1.0 | | |
| `minimax:depth=2,eval=hop` (`vs_greedy_N5_hop_d2`) | P0 | 31 | 17 | 2 repetition | 62% (48.2-74.1%) | 14.62 | 2.0 | 248.6 | 68.2 ms |
| | P1 | 19 | 31 | 0 | 38% (25.9-51.8%) | 12.10 | 2.0 | 298.4 | 79.5 ms |
| | all | 50 | 48 | 2 repetition | **50% (40.4-59.6%)** | 13.36 | 2.0 | 268.5 | 72.7 ms |
| `minimax:depth=3,eval=hop` (`vs_greedy_N5_hop_d3`) | P0 | 30 | 18 | 2 repetition | 60% (46.2-72.4%) | 14.44 | 3.0 | 4,562 | 455 ms |
| | P1 | 20 | 28 | 2 no legal action | 40% (27.6-53.8%) | 12.20 | 3.0 | 4,268 | 431 ms |
| | all | 50 | 46 | 4 (2 + 2) | **50% (40.4-59.6%)** | 13.32 | 3.0 | 4,442 | 446 ms |

(Lengths include the 4 opening plies. Per-seat nodes and times for depth 1 are in the summary file.)

### Paired: old scorer vs hop scorer, same openings and seats

Old-scorer games are PR #3's logs (`final_minimax/vs_greedy_N5_open4_depth1_diag.jsonl`,
`vs_greedy_N5_open4_distinct.jsonl`, `vs_greedy_N5_open4_depth3_diag.jsonl`). A game is (opening, minimax seat);
a win is a minimax goal win. The bootstrap resamples the 50 openings with both seats together,
10,000 times, rng seed 0 (`paired_d{1,2,3}_default_vs_hop.json`).

| Depth | Old win rate | Hop win rate | Difference | Paired bootstrap 95% CI | Both win / only old / only hop / neither | As P0 old -> hop | As P1 old -> hop |
|---|---|---|---|---|---|---|---|
| 1 | 45% | 50% | +5 points | -1 to +11 | 39 / 6 / 11 / 44 | 50% -> 60% | 40% -> 40% |
| 2 | 39% | 50% | **+11 points** | **+1 to +21** | 33 / 6 / 17 / 44 | 40% -> 62% | 38% -> 38% |
| 3 | 45% | 50% | +5 points | -5 to +15 | 36 / 9 / 14 / 41 | 44% -> 60% | 46% -> 40% |

Only depth 2 has an interval above zero. Every gain is as the first player; as the second player
the hop scorer is equal (depths 1, 2) or 6 points worse (depth 3).

### Wall-then-hop pattern (`hop_stats.json`, `hop_examples.json`)

| Depth | Minimax walls placed, old -> hop | Greedy wins with a Greedy HOP over a minimax wall, old -> hop |
|---|---|---|
| 1 | 181 -> 5 | 16 of 46 -> 1 of 50 |
| 2 | 207 -> 35 | 30 of 60 -> 3 of 48 |
| 3 | 211 -> 39 | 33 of 48 -> 2 of 46 |

Old values are from PR #3 (`final_minimax/hop_stats.json`). The hop scorer mostly stops placing walls
that Greedy can hop.

### Hop scorer vs the default scorer (`vs_minimax_N5_hop_d2_vs_default_d2`)

`minimax:depth=2,eval=hop` vs `minimax:depth=2`, same 50 openings, both seats, 100 games.
In this summary the `greedy_*` fields mean the default minimax.

| Hop scorer seat | Hop wins | Default wins | Draws | Hop win rate (95% Wilson) | Mean length |
|---|---|---|---|---|---|
| P0 | 31 | 17 | 2 no legal action | 62% (48.2-74.1%) | 17.06 |
| P1 | 23 | 27 | 0 | 46% (33.0-59.6%) | 13.66 |
| all | 54 | 44 | 2 | **54% (44.3-63.4%)** | 15.36 |

Hop agent: mean depth 1.993, 228.8 nodes, 37.1 ms per decision.

## 4. Larger board: N=9 timed check (not deterministic, load-affected)

`minimax:depth=2,time_limit=1.0,eval=hop` vs `greedy`, opening seeds 0-9 with 2 opening plies,
both seats (`vs_greedy_N9_hop_timed*`, 20/20 replays verified). Seeds 0-9 give 8 distinct
openings and 16 distinct games (`final_minimax_hop/opening_maps.json`).

| | P0 | P1 | All |
|---|---|---|---|
| Hop wins / Greedy wins, 20 matches | 8 / 2 | 2 / 8 | 10 / 10 |
| Win rate, 20 matches (95% Wilson) | 80% (49.0-94.3%) | 20% (5.7-51.0%) | 50% (29.9-70.1%) |
| Distinct games: hop wins / games (95% Wilson) | 6 / 8 (40.9-92.9%) | 2 / 8 (7.1-59.1%) | **8 / 16 (28.0-72.0%)** |
| Draws / truncated | 0 / 0 | 0 / 0 | 0 / 0 |
| Mean length (incl. 2 opening) | 17.6 | 18.0 | 17.8 |
| Mean depth completed (load-affected) | 1.90 | 1.97 | 1.94 |
| Mean nodes | 762.7 | 774.2 | 768.2 |
| Mean decision time (load-affected) | 330 ms | 293 ms | 312 ms |

Paired against PR #3's default N9 run (`paired_N9_timed_default_vs_hop.json`): 70% -> 50%,
difference -20 points (bootstrap -35 to -5 over the 10 seeds; seeds 1/3 and 4/8 are repeated
openings, so the interval is narrower than 8 truly distinct openings justify). As P0 the rate stays
80%. As P1 it falls from 60% to 20%: of 20 games, 10 both won, 4 only the default won, 0 only the hop
scorer won. On N=9 the hop scorer placed **0 walls** in 20 games, where the default placed 48
(`hop_stats.json`). One likely reason, not tested: at depth 2 a single new wall can always be hopped
while Greedy still holds its hop, so under this scorer no wall ever looks useful within the horizon.

**Load.** `resource_status` was `ample` at the start and end of both the depth-3 and N9 runs (5.4-5.5 GB).
The two runs were started without `HEAVY.lock`, beside other bots, as Mr. CEO allowed. At the start and end
of both runs no Bot 2 or Bot 3 python process was running. `HEAVY.lock` was held by Bot 3 ("ablation 640 +
144 matches") the whole time. Just after the N9 run, Bot 2's wrapper (`bot2-chunk.ps1`, "train dqn 3x3 s45") was
present with no python child. So Bot 2's training was queued but not computing as far as these snapshots show
(`load_record.json`). The two default-run timings in PR #3 were measured under other load, so only
node counts and depth (not ms) are comparable across the PRs.

The default reproduction run and the depth-2 hop run were started while posture read `tight`
(2.3 GB). I noticed this only afterwards. Both were about one-minute, ~100 MB jobs, and every later run
was started at `ample`. Their times (50 ms and 73 ms per decision) are load-affected; their games are
deterministic and not affected.

## 5. Exact vs empirical

All N=5 rows are exact game counts: deterministic agents and distinct openings. The paired
intervals describe these 50 openings only. The N=9 rows are empirical, from one timed run under load.

## 6. Commands (repo root)

```
python -m unittest tests.test_minimax_hop_eval
python -m unittest tests.test_match_runner
python -m unittest tests.test_evaluation_fn
python -m unittest tests.test_minimax_final_compat
python -m unittest tests.test_agents
python -m unittest tests.test_agent_conformance
python -m pytest tests/test_minimax_hop_eval.py -q
python -m pytest tests/test_minimax_phase5.py -q
F="python scripts/final_minimax.py --results-dir final_minimax_hop"
$F vs-greedy -N 5 --opening-plies 4 --openings-from N5_open4_distinct --spec minimax:depth=2 --tag N5_default_repro
$F compare-logs experiments/results/final_minimax/vs_greedy_N5_open4_distinct.jsonl experiments/results/final_minimax_hop/vs_greedy_N5_default_repro.jsonl --name default_repro_diff.json
$F vs-greedy -N 5 --opening-plies 4 --openings-from N5_open4_distinct --spec minimax:depth=2,eval=hop --tag N5_hop_d2
$F vs-greedy -N 5 --opening-plies 4 --openings-from N5_open4_distinct --spec minimax:depth=1,eval=hop --tag N5_hop_d1
$F vs-greedy -N 5 --opening-plies 4 --openings-from N5_open4_distinct --spec minimax:depth=2,eval=hop --opponent minimax:depth=2 --tag N5_hop_d2_vs_default_d2
$F vs-greedy -N 5 --opening-plies 4 --openings-from N5_open4_distinct --spec minimax:depth=3,eval=hop --tag N5_hop_d3
$F vs-greedy -N 9 --seeds 10 --opening-plies 2 --spec minimax:depth=2,time_limit=1.0,eval=hop --tag N9_hop_timed
$F paired experiments/results/final_minimax/vs_greedy_N5_open4_depth1_diag.jsonl minimax:depth=1 experiments/results/final_minimax_hop/vs_greedy_N5_hop_d1.jsonl minimax:depth=1,eval=hop --name paired_d1_default_vs_hop.json
$F paired experiments/results/final_minimax/vs_greedy_N5_open4_distinct.jsonl minimax:depth=2 experiments/results/final_minimax_hop/vs_greedy_N5_hop_d2.jsonl minimax:depth=2,eval=hop --name paired_d2_default_vs_hop.json
$F paired experiments/results/final_minimax/vs_greedy_N5_open4_depth3_diag.jsonl minimax:depth=3 experiments/results/final_minimax_hop/vs_greedy_N5_hop_d3.jsonl minimax:depth=3,eval=hop --name paired_d3_default_vs_hop.json
$F paired experiments/results/final_minimax/vs_greedy_N9_timed.jsonl minimax:depth=2,time_limit=1.0 experiments/results/final_minimax_hop/vs_greedy_N9_hop_timed.jsonl minimax:depth=2,time_limit=1.0,eval=hop --name paired_N9_timed_default_vs_hop.json
$F hops N5_hop_d1 N5_hop_d2 N5_hop_d3 N9_hop_timed
$F hop-examples N5_hop_d1 N5_hop_d2 N5_hop_d3
$F openings N5_hop_d1 N5_hop_d2 N5_hop_d3 N5_default_repro N9_hop_timed
python scripts/run_experiments.py verify-replay experiments/results/final_minimax_hop/<each .jsonl>
```

`load_record.json` was written by a scratch script from the load snapshots taken around the
depth-3 and N9 runs. It is not a committed script.
