# Stage 1 Engine Freeze Report (Member 2, Tasks 1-2, Task 3 compatibility, Task 5 rules doc)

- **Engine tested:** `main` at `ce55e0c` (no engine code changed in this PR).
- **Added:** `tests/test_stage1_engine_freeze.py` (35 unittest tests), this report, updates to `docs/game_rules.md`.
- **Machine:** Windows, Python 3.12, numpy installed; torch, pytest, hypothesis, networkx **not** installed.
- **Commands** (run from the repo root, one file at a time):

```
python -m unittest tests.test_stage1_engine_freeze
python -m unittest tests.<file>            # for every unittest file in the table below
python run_plain.py tests.test_single_source tests.test_greedy_campaign
```

`run_plain.py` is a 20-line scratch helper (not committed) that imports a module and calls each
`test_*` function that takes no pytest fixture. It was used only for the two files that have bare
test functions but do not import pytest.

## Rule tables (new tests, `tests/test_stage1_engine_freeze.py`)

### A. Block duration: active exactly K plies

| Test | Checks | Result |
|---|---|---|
| `TestRuleA_BlockDuration.test_state_right_after_placement` | timer = K right after the BLOCK ply, `block_available` used | pass |
| `test_active_exactly_k_plies_for_several_k` | K = 1, 2, 3, 4: cell blocked at the start of plies t+1..t+K with timer K-j+1, gone after ply t+K | pass |
| `test_move_into_block_rejected_on_kth_ply_allowed_after` | MOVE into the cell rejected on ply t+K (state unchanged), legal after expiry | pass |
| `test_consecutive_blocks_by_both_players` | P0 and P1 blocks active together, independent timers, second block by same player rejected | pass |
| `test_cannot_block_blocked_or_pawn_cell` | block on blocked cell / either pawn rejected | pass |
| `test_expiring_block_still_counts_in_path_checks` | a wall legal only after a timer-1 block expires is rejected (Decision 9) | pass |

Old tests on the same rule: `test_temporary_blocks.py` pass, `test_engine_prerequisites.py` pass,
`test_rules_blocks_hop.py` not run (needs pytest).

### B. Strict action validation

| Test | Checks | Result |
|---|---|---|
| `TestRuleB_StrictValidation.test_illegal_actions_rejected_without_state_change` | 15 illegal or malformed actions (wall cross/overlap/off-board, move across wall/off-board/non-adjacent, hop over own wall, REPLACE of older wall or to same place, block own cell, unknown type, bool coords, bad orientation, wrong shape): each raises `ValueError` and the full `snapshot()` is unchanged | pass |
| `test_invalid_action_does_not_touch_clock` | with `update_clock=True`, a rejected action leaves `time_remaining` and `last_turn_timestamp` unchanged | pass |
| `test_replace_rejected_when_rule_off_or_no_wall_yet` | REPLACE with no wall yet, and with Rule A off | pass |
| `test_actions_after_termination_rejected` | action after a goal win raises, state unchanged | pass |

Old tests: `test_engine_prerequisites.py` pass; `test_strict_actions.py`, `test_action_normalization.py` not run (pytest/hypothesis).

### C. No-legal-move position is a draw

| Test | Checks | Result |
|---|---|---|
| `TestRuleC_NoLegalDraw.test_draw_flags` | pawn's only free neighbour holds the opponent: `is_draw`, `winner None`, `terminal_reason 'no_legal_actions'`, `is_terminal`, empty legal lists, path still open, further action rejected | pass |
| `test_greedy_returns_none_on_draw` | Greedy returns `None` | pass |
| `test_goal_win_flags_agree` | goal win: `winner`, `is_draw False`, `'goal'`, `has_legal_action False`; Greedy returns `None` | pass |

Match runner: maps `engine.is_draw` to category `draw_<terminal_reason>`, so this draw is
`draw_no_legal_actions` (`evaluation/match_runner.py:253-254`). `test_match_runner.py` pass.
Old tests: `test_rules_draw_budget.py` not run (pytest).

### D. Configurable wall budget `rules['M']`

| Test | Checks | Result |
|---|---|---|
| `TestRuleD_WallBudget.test_default_is_10_for_both` | default 10 for both, `WallzConfig().M == 10` | pass |
| `test_configured_budget` | M = 0, 1, 3, 12: both inventories, `max_walls`, WALL actions exist iff M > 0 | pass |
| `test_negative_budget_rejected` | M = -1 raises | pass |
| `test_budget_exhausted_but_replace_allowed` | M = 1: after one wall, WALL rejected ("no walls remaining"), REPLACE still legal and costs nothing | pass |
| `test_greedy_never_walls_with_zero_budget` | Greedy with M = 0 never returns WALL | pass |

Hard-coded 10 search (grep of `agents/`, `web/backend/`, `evaluation/`):

| Place | What it does | Verdict |
|---|---|---|
| `agents/greedy.py` | only uses `engine.get_legal_actions()` / `apply_action` | uses engine |
| `agents/minimax.py` | only uses engine legal actions | uses engine |
| `agents/dqn/encoding.py:89` | inventory planes divide by `engine.rules.get('M', 10)`; engine rules always hold `M` | uses config |
| `agents/dqn/trainer.py:33` | default rules dict `M: 10`, overridable field | default only |
| `agents/exact_solver.py:38,42-43` | default `M: 10`, overridden by `rules` or `max_walls` | default only |
| `web/backend/routes_games.py:45,72,132`, `sessions.py:82,91`, `registry.py:50,67` | `WallzConfig().M` or request value | uses config |

### E. HOP (one step across an opponent's wall, once per game)

| Test | Checks | Result |
|---|---|---|
| `TestRuleE_Hop.test_hop_over_opponent_wall_and_replace_keeps_owner` | P1 moves its wall by REPLACE in front of P0; owner stays 1; MOVE blocked, HOP legal; hop consumed | pass |
| `test_hop_is_one_time` | second hop back across the same opponent wall rejected | pass |
| `test_no_hop_over_own_wall` | own wall: owner 0, no HOP, apply raises | pass |
| `test_no_hop_over_open_edge_or_two_cells` | open edge and the old 2-cell jump both raise | pass |
| `test_no_hop_onto_opponent_pawn_or_blocked_cell` | landing on opponent pawn or a blocked cell rejected; legal once clear | pass |
| `test_no_hop_into_dead_end` | landing cell walled in on all sides: rejected, state unchanged; legal when one side opens | pass |
| `test_no_hop_when_rule_off` | Rule B off: no HOP | pass |
| `test_position_key_keeps_owner_only_while_a_hop_remains` | same walls, different owner: keys differ while a hop remains, equal once both hops are used | pass |

Old tests: `test_hop.py` pass; `test_rules_blocks_hop.py` not run (pytest).

## Task 2 regression list (other items)

| Behaviour | New test | Result |
|---|---|---|
| normal pawn movement | `TestRegression.test_normal_pawn_moves` | pass |
| wall placement | `test_wall_placement_updates_state` | pass |
| wall repositioning + latest-wall restriction | `test_wall_reposition_latest_only` | pass |
| path preservation (WALL and BLOCK) | `test_wall_and_block_path_preservation` | pass |
| repetition detection (5-fold) | `test_fivefold_repetition_draw` | pass |
| repetition limit configurable | `test_repetition_limit_configurable` | pass |
| `max_plies` truncation, off by default | `test_max_plies_truncation` | pass |

## Task 3 Greedy compatibility (no match campaign)

| Test | Checks | Result |
|---|---|---|
| `TestGreedyCompatibility.test_greedy_plays_only_legal_actions` | 6 configs (N = 3, 5, 7; K = 1-4; M = 0-10; all rules off once): every Greedy action is in `get_legal_actions()`, `has_legal_action()` agrees with the list, inventory falls by 1 only on WALL, block timers stay in 1..K and never on a pawn, `None` on terminal | pass |
| `test_greedy_scores_truncation_draw_as_zero` | with `max_plies = 1` every row is a draw with H = 0 | pass |
| `TestRuleC...test_greedy_returns_none_on_draw`, `TestRuleD...test_greedy_never_walls_with_zero_budget` | see above | pass |

Greedy reads block duration only through the engine (it simulates with `clone()` + `apply_action`), so it uses the frozen K behaviour by construction.

## Every test file

| File | Result | Note |
|---|---|---|
| `test_stage1_engine_freeze.py` (new) | pass, 35 tests | |
| `test_action_codec.py` | pass, 4 | |
| `test_agents.py` | pass, 13 | |
| `test_agent_conformance.py` | pass, 6 (1 skipped) | DQN conformance skipped: torch missing |
| `test_dqn_encoding.py` | pass, 4 | |
| `test_engine_prerequisites.py` | pass, 22 | |
| `test_evaluation_fn.py` | pass, 5 | |
| `test_exact_solver.py` | pass, 9 | |
| `test_filter_equivalence.py` | pass, 2 | |
| `test_golden_regression.py` | pass, 3 | |
| `test_greedy.py` | pass, 7 | |
| `test_helpers.py` | pass, 6 | |
| `test_hop.py` | pass, 1 | |
| `test_initial_state.py` | pass, 2 | |
| `test_match_runner.py` | pass, 16 | |
| `test_metrics_holm.py` | pass, 4 | |
| `test_minimax_phase5.py` | pass, 6 | |
| `test_movement.py` | pass, 2 | |
| `test_path_preservation.py` | pass, 2 | |
| `test_replacement.py` | pass, 7 | |
| `test_retrograde.py` | pass, 17 | |
| `test_rl_metrics.py` | pass, 4 | |
| `test_temporary_blocks.py` | pass, 2 | |
| `test_terminal_states.py` | pass, 2 | |
| `test_walls.py` | pass, 2 | |
| `test_wallz_env.py` | pass, 4 | |
| `test_single_source.py` | pass, 3 functions | run with the plain-function helper |
| `test_greedy_campaign.py` | pass, 4 functions | run with the plain-function helper |
| `test_dqn_agent.py` | **not run** (4 skipped) | needs torch |
| `test_dqn_runner_integration.py` | **not run** (2 skipped) | needs torch |
| `test_dqn_trainer.py` | **not run** (3 skipped) | needs torch |
| `test_self_play.py` | **not run** (4 skipped) | needs torch |
| `test_bfs_networkx.py` | **not run** (import error) | needs networkx |
| `test_action_normalization.py` | **not run** | needs pytest |
| `test_edge_invariants.py` | **not run** | needs pytest |
| `test_rules_blocks_hop.py` | **not run** | needs pytest |
| `test_rules_draw_budget.py` | **not run** | needs pytest |
| `test_scenarios.py` | **not run** | needs pytest |
| `test_strict_actions.py` | **not run** | needs pytest + hypothesis |
| `test_greedy_final.py` | **not run** | needs pytest + hypothesis |
| `test_invariants_stateful.py` | **not run** | needs hypothesis |
| `tests/web/test_api_data.py`, `test_api_games.py`, `test_explain.py`, `test_live_training.py` | **not run** | need pytest |

Skipped tests are not counted as passing. A machine with `requirements-dev.txt` installed should run `python -m pytest tests` once before the freeze is final.

## Single source of truth

No separate copy of the game rules was found.

| Component | Finding |
|---|---|
| `src/modified_wallz/` | Not a second engine. `board.py` (wall geometry), `pathfinding.py` (BFS) and `config.py` are the pieces `wallz_engine.py` itself imports (`wallz_engine.py:15-29`). `rules.py`, `state.py`, `transitions.py` only re-export them. `actions.py` encodes actions and builds masks from `engine.get_legal_actions()`. `environment.py` steps through `engine.apply_action`. |
| `agents/greedy.py` | engine only (legal actions, clone, apply, `shortest_path`) |
| `agents/minimax.py:21,141` | imports `bfs_distance_and_path` for **move ordering** only; legality and terminal checks come from the engine. Not a rule copy. |
| `agents/exact_solver.py`, `evaluation/state_graph.py`, `evaluation/retrograde.py` | engine `restore_state_key` / legal actions; `deque` there is graph search, not rules |
| `web/backend/` | `serialize.py:69` takes legal actions from the engine; `tests/test_single_source.py` AST guard passes |
| `web/frontend/src/` | highlights only the `legal` lists the backend sends (`Board.tsx:59,78-82`, `ActionBar.tsx:18`); regex guard in `test_single_source.py` passes. The old "skipped until frontend" stub now runs because the frontend exists. |
| `wallz_engine.py:359` `shortest_path_cells` | a second BFS inside the engine (same rules as `pathfinding.py`: walls and blocked cells, pawns ignored). Used for display/explanations. Small duplication inside the engine itself; left unchanged. |

## Open conventions and decisions for the team

1. **Expiring block in path checks: settled.** `docs/decisions.md` Decision 9 already fixes it as conservative (a timer-1 block still blocks path checks on the ply it expires). `docs/person3_evaluation.md:46-50` still calls it "open for the team"; that line is now out of date. Covered by `test_expiring_block_still_counts_in_path_checks`.
2. **Truncation is two different things.** Engine `max_plies` (off by default) sets `is_draw = True` and `is_truncated = True`. The match runner and `WallzEnv` keep their own 200-ply cap and call it "not a rule draw". If anyone passes `max_plies` inside `rules` to the runner, the outcome would read `draw_truncated` instead of `truncated`. Team decision: keep the cap only in the runner (current practice) or make the engine report truncation with `is_draw = False`.
3. **HOP rule change (4e346a8)** is the fifth change not in the team plan. Results made under the old 2-cell jump (DQN checkpoints, ablation pilots) are stale, as `docs/game_rules.md` says. Whoever owns those results must rerun them later; this PR does not.
4. **pytest-only rule tests.** Four of the main rule files need pytest. Either install `requirements-dev.txt` on every team machine or keep the unittest file added here as the shared check.

## Frozen rule set (code against this)

| Item | Frozen behaviour |
|---|---|
| Board | N x N, N >= 3. P0 starts `(0, N//2)`, goal row N-1. P1 starts `(N-1, N//2)`, goal row 0. P0 moves first. |
| MOVE | one orthogonal step; not across a wall, not off board, not onto the opponent, not onto a blocked cell. No pawn jump. |
| WALL | `(r, c, 'H'/'V')`, `0 <= r, c <= N-2`; no overlap or cross; costs 1 of `M` walls; becomes the player's latest wall. |
| REPLACE (A) | only the player's latest wall; any free legal slot except its own; costs nothing; allowed at 0 inventory; owner unchanged. |
| HOP (B) | once per game; one step to an adjacent cell across a wall **owned by the opponent**; landing not off board, not the opponent's cell, not blocked; the hopper must still reach its goal. |
| BLOCK (C) | once per game; any empty, unblocked cell; impassable for the next K plies (timer K, decremented at the end of each ply, removed at 0); both players' blocks may coexist. |
| Path preservation | after WALL/REPLACE/BLOCK both players must reach their goal row; after HOP the hopper must. Pawns ignored; all current blocks count, including ones expiring this ply. |
| Validation | `apply_action` order: terminal guard, normalise, validate, clock, mutate. Any failure raises `ValueError` before any change. |
| Win | reaching the goal row; clock timeout loses (only with `update_clock=True`). |
| Draws | no legal action for the mover (`'no_legal_actions'`); N-fold repetition of `position_key()` (`repetition_limit`, default 5). |
| Truncation | engine `max_plies` default `None`; runner/`WallzEnv` cap 200, reported as `truncated`, not a draw. |
| Defaults | `A = B = C = True`, `K = 3`, `M = 10`, `repetition_limit = 5`, `time_limit_sec = 600`. |

## Independent break attempt

A fresh reviewer that saw only the diff and `docs/game_rules.md` tried to break one rule claim.
Result: see the section added below after that review.
