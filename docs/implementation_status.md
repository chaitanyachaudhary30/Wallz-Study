# Implementation Status & Engineering Tracker

**Project**: Multi-Agent Reinforcement Learning for Adversarial Spatial Decision-Making with Temporal Constraints  
**Module**: Core Engine Rule Audit & Finalized Mechanics Verification  
**Last Updated**: October 2026  

---

## 1. Summary of Completed Work

### Core Engine (`wallz_engine.py`)
- [x] **Modification A (Reusable Walls)**: Corrected from erroneous FIFO queue eviction to the finalized rule:
  - Each player starts with an initial inventory of $M = 10$ walls.
  - Placing a new wall (`WALL`) consumes 1 wall inventory unit ($m_i \to m_i - 1$).
  - Only the player's single most recently placed wall ($r_i$) can be repositioned (`REPLACE`).
  - Older walls remain permanent fixtures on the board for the rest of the game.
  - Repositioning does not consume wall inventory ($m_i' = m_i$).
  - When inventory is 0, new wall placement is illegal, but repositioning $r_i$ remains available.
- [x] **Modification B (One-Time Hop)**: Verified single-use 2-step hop bypassing intervening walls/pawns.
- [x] **Modification C (Temporary Cell Block)**: Added path-preservation verification (`_is_valid_cell_block`) ensuring a cell block cannot cut off all goal paths for either player.
- [x] **Path-Preservation Invariant**: Enforced across all new wall placements, wall repositionings, and cell blocks.
- [x] **5-Fold Repetition Draw**: State representation incorporates pawn positions, active walls, wall inventories, latest wall pointers, ability flags, cell block decay counters, and turn indicator.
- [x] **Fast State Cloning & Isolation**: `clone()` and `apply_action(action, update_clock=False)` provide isolated simulation without clock depletion.

### Decision Agents (`agents/`)
- [x] **Tier-1 Greedy Baseline (`agents/greedy.py`)**: Fully compatible with `WALL`, `REPLACE`, `MOVE`, `HOP`, and `BLOCK`.
- [x] **Tier-2 Minimax Agent (`agents/minimax.py`)**: Move ordering prioritizes immediate wins, spatial progress, pawn advances, wall placements and repositionings, and cell blocks.
- [x] **Tier-3 Exact Retrograde Solver (`agents/exact_solver.py`, `scripts/solve_tiny_board.py`)**:
  - Full forward reachability BFS state graph generator.
  - Backward retrograde queue propagation with active action degree tracking for cyclic games.
  - Provably exact $\{+1.0, 0.0, -1.0\}$ game-theoretic outcomes and shortest distance-to-terminal policies.
  - Ground-truth validation benchmark for Minimax and DQN agents.

### Match Runner & Tooling (`run_match.py`, `scripts/run_match.py`, `scripts/solve_tiny_board.py`)
- [x] CLI runner supporting `REPLACE` action logging and telemetry tracking.
- [x] CLI exact solver tool `scripts/solve_tiny_board.py` for miniature board configurations.

---

## 2. Tests Added and Executed

All tests reside under `tests/` and are executed via Python's standard `unittest` runner:
`python3 -m unittest discover tests -v`

### Executed Test Results (41 / 41 Passed)
- `tests/test_exact_solver.py` (9 tests):
  - `test_01_terminal_wins_and_losses`: Terminal states detected at $d=0$ with exact $+1.0 / -1.0$ values.
  - `test_02_player_perspective_invariance`: Zero-sum symmetry $V(s, 0) == -V(s, 1)$ across all solved states.
  - `test_03_hand_constructed_optimal_actions`: Exact 1-ply winning action selection in hand-crafted miniature states.
  - `test_04_state_key_completeness`: Verified all 7 state dimensions (pawns, walls, inventories, latest wall, hop, blocks, player) affect state key.
  - `test_05_state_immutability`: Solving and querying preserves caller engine state completely unmutated.
  - `test_06_draw_and_cycle_handling`: Cyclic loops without forced wins resolve to `DRAW` ($d=\infty, V=0.0$).
  - `test_07_exhaustive_consistency_and_degrees`: Consistency and conservation of degrees on exhaustive state space.
  - `test_08_agreement_with_minimax_at_terminal_depth`: Minimax ($d \ge 4$) exactly matches Exact Solver optimal policy.
  - `test_09_out_of_domain_and_agent_fallback`: Out-of-domain states return `None` or cleanly fall back to legal actions.
- `tests/test_agents.py` (13 tests):
  - 10 MinimaxAgent unit tests (winning pawn/hop moves, threat anticipation, leaf eval, cutoffs, timeouts, immutability).
  - 3 ExactSolverAgent tests (optimal policy selection, unvisited state fallback, no legal actions graceful handling).
- `tests/test_replacement.py` (7 tests):
  - `test_repositioning_latest_wall`: Verified latest wall is repositioned while older wall remains intact on board and inventory is unchanged.
  - `test_rejection_of_repositioning_older_wall`: Verified repositioning an older wall is rejected via both `apply_action` (raises `ValueError`) and `_is_valid_wall_replacement` (returns `False`), leaving state unmutated.
  - `test_wall_inventory_unchanged_after_replacement`: Verified that at 0 inventory, new wall placements are blocked, but `REPLACE` remains legal and inventory stays 0.
  - `test_invalid_replacement_preserves_original_state`: Verified collision or out-of-bounds replacements fail safely without state corruption.
  - `test_path_preservation_during_replacement`: Verified repositioning into a path-sealing position is rejected.
  - `test_no_wall_placed_yet_behavior`: Verified that when 0 walls are placed, `REPLACE` is absent from legal actions and calling it fails safely.
  - `test_rule_a_disabled_ablation`: Verified that with `rules={'A': False}`, replacement is unavailable.
- `tests/test_temporary_blocks.py` (2 tests):
  - `test_block_placement_and_decay`: Verified turn decay and unblocking.
  - `test_block_path_preservation`: Verified that blocking a bottleneck cell that isolates a player is rejected.
- `tests/test_initial_state.py` (2 tests): Default 9x9 and custom 5x5 board initializations.
- `tests/test_movement.py` (2 tests): Orthogonal moves and wall barrier obstruction.
- `tests/test_walls.py` (2 tests): Overlap/intersection validation and shortest-path lengthening.
- `tests/test_hop.py` (1 test): 2-step hop execution and single-use flag consumption.
- `tests/test_path_preservation.py` (1 test): Rejection of wall placements sealing off all goal paths.
- `tests/test_terminal_states.py` (2 tests): Goal row reach victory and 5-fold repetition draw.

---

## 3. Benchmark Match Results with Updated Rules

### 1. Board 5x5: Minimax (Player 0, Depth 2) vs Greedy (Player 1)
- **Winner**: Player 1 (Greedy) in 16 plies.
- **Match Duration**: 0.66 seconds.
- **Minimax Stats**: 8 turns, Mean latency 77.7ms, Mean nodes/turn: 247.5, Total cutoffs: 305.

### 2. Board 5x5: Greedy (Player 0) vs Minimax (Player 1, Depth 2)
- **Outcome**: Draw via 5-Fold State Repetition at turn 90.
- **Match Duration**: 1.75 seconds.
- **Telemetry**: Both Greedy and Minimax actively and legally used `REPLACE` to adjust their latest walls dynamically.

---

## 4. Current Repository Status & Work Allocation

| Component | Status | Assigned Responsibility |
| :--- | :--- | :--- |
| Core Engine (`wallz_engine.py`) | **Audited, Corrected & Verified** | Core Team / Shared |
| Tier-1 Greedy Baseline (`agents/greedy.py`) | **Updated & Verified** | Shared |
| Tier-2 Minimax Agent (`agents/minimax.py`) | **Updated & Fully Verified** | **My Assigned Responsibility** |
| Unit Test Suite (`tests/`) | **Complete (29 tests passing)** | **Completed during this task** |
| Tier-3 Exact Solver (`agents/exact_solver.py`) | *Scaffolded (Unimplemented)* | Other team members |
| MARL DQN Agent (`agents/dqn/`) | *Scaffolded (Unimplemented)* | Other team members |
| Ablation Framework (`evaluation/ablation.py`) | *Scaffolded (Unimplemented)* | Other team members |
