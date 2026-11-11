# Modified Wallz — Official Game Rules & Mechanics

## 1. Overview
Modified Wallz is a deterministic, sequential, two-player zero-sum adversarial grid game inspired by Quoridor, enhanced with dynamic spatial constraints, finite reusable resources, and temporal abilities.

The canonical engine implementation in `wallz_engine.py` serves as the **single source of truth** for all rules, validations, legal move generations, and state transitions.

**Status: frozen at Stage 1** (engine as of `main` = `ce55e0c`, verified in `docs/reports/stage1_engine_freeze.md`, tests in `tests/test_stage1_engine_freeze.py`). `wallz_engine.py` and `src/modified_wallz/` have not changed since (checked at `main` = `d91549c`). This file describes the final rules only; Section 3.4 notes the one earlier HOP rule whose results are stale (`experiments/results/STALE.md`).

---

## 2. Board & Pawns
- **Board Grid**: $N \times N$ discrete grid where $N \ge 3$ (standard default $N = 9$; experimental benchmarks use $N \in \{3, 4, 5, 7, 9\}$).
- **Coordinate System**: Rows $0 \dots N-1$, Columns $0 \dots N-1$.
  - Row 0 is at the bottom (Player 0's start row).
  - Row $N-1$ is at the top (Player 1's start row).
- **Player 0**: Starts at `(0, floor(N/2))`. Goal is row $N - 1$.
- **Player 1**: Starts at `(N - 1, floor(N/2))`. Goal is row `0`.
- **Turn Order**: Players strictly alternate turns, executing exactly one legal action per turn. Player 0 moves first.

---

## 3. Action Categories & Syntax

The engine accepts actions either as Python tuples or as JSON lists:

### 1. Pawn Movement (`MOVE`)
- Syntax: `('MOVE', (r, c))` or `["MOVE", [r, c]]`.
- Moves the active player's pawn exactly 1 cell orthogonally (North, South, East, West).
- Constraints:
  - Cannot cross active walls.
  - Cannot land on an opponent pawn (there is no pawn jump).
  - Cannot land on a currently blocked cell.
  - Must remain within the $N \times N$ board boundaries.

### 2. Path-Preservation Invariant
- **Core Invariant**: After every action, each player must still have at least one path from their current cell to their goal row:
  $$\operatorname{PathExists}(s', 0) = 1 \quad \text{and} \quad \operatorname{PathExists}(s', 1) = 1$$
- If an action would isolate either player from their goal row, it is strictly illegal.
- How the engine checks it, per action:
  - `WALL`, `REPLACE`, `BLOCK`: a breadth-first search for **both** players on the board after the action.
  - `HOP`: a search for the **hopping** player from the landing cell (walls do not change, so the opponent's path cannot change).
  - `MOVE`: no search needed. A one-step move can always be undone, so it cannot cut a path.
- Paths ignore pawns: a pawn never counts as an obstacle for path checks (it does block a `MOVE` or `HOP` onto its cell).
- **Expiring blocks still count (team Decision 9, `docs/decisions.md`)**: every block in `blocked_cells` is treated as impassable, including a block whose timer is 1 and which will expire at the end of this same ply. A `WALL`/`REPLACE`/`BLOCK` that would only be legal once that block expires is rejected. This is conservative by at most one ply and is the frozen rule.

### 3. Modification A: Reusable Walls (`WALL`, `REPLACE`)
- Walls are 2-cell barriers placed along grid lines, designated by `(r, c, orient)` where:
  - `r` $\in \{0 \dots N-2\}$
  - `c` $\in \{0 \dots N-2\}$
  - `orient` $\in \{'H', 'V'\}$ ('H' blocks vertical passage between $(r, c)$/$(r, c+1)$ and $(r+1, c)$/$(r+1, c+1)$; 'V' blocks horizontal passage between $(r, c)$/$(r+1, c)$ and $(r, c+1)$/$(r+1, c+1)$).
- Walls cannot overlap with existing walls or cross perpendicular walls through their center.
- **Initial Inventory**: Both players begin with the same finite inventory of $M$ walls, set by `rules['M']` (default $M = 10$; any integer $M \ge 0$; `WallzConfig` rejects $M < 0$). Agents and the web backend read the configured value (`engine.rules['M']`, `engine.max_walls`, `WallzConfig().M`); none hard-codes 10.
- **New Wall Placement (`WALL`)**:
  - Syntax: `('WALL', (r, c, orient))` or `["WALL", [r, c, "H"]]`.
  - Permissible only when $m_i > 0$.
  - Consumes 1 wall from inventory ($m_i' = m_i - 1$).
  - Designates this wall as the player's single most recently placed wall ($r_i$).
  - Older walls placed prior to $r_i$ become permanent fixtures for the remainder of the match.
- **Wall Repositioning / Replacement (`REPLACE`)**:
  - Syntax (both forms accepted and normalized):
    1. Short triple: `('REPLACE', (new_r, new_c, new_orient))`
    2. Explicit pair: `('REPLACE', (old_wall, new_wall))` where `old_wall == latest_wall[player]`.
  - Permissible if and only if Rule A is enabled and the player has placed at least one wall ($r_i \neq \varnothing$).
  - **Recency Constraint**: Only the player's single most recently placed wall ($r_i$) can be repositioned. Older walls remain permanently fixed and cannot be moved or removed.
  - **Budget Neutrality**: Reuses the existing wall on the board; does **not** consume wall inventory ($m_i' = m_i$).
  - **Zero Budget Allowance**: When $m_i = 0$, a player cannot place new `WALL` actions, but **can** continue executing `REPLACE` actions with their latest wall, provided a legal reposition exists that preserves paths.
  - Must satisfy path preservation for both players.

### 4. Modification B: One-Time Hop (`HOP`)
- Syntax: `('HOP', (r, c))` or `["HOP", [r, c]]`.
- A single-use ability per player (`hop_available[player] = True` initially).
- Bypasses **a wall placed by the opponent** (proposal `LegalHop`): the pawn moves into the
  orthogonally adjacent cell on the other side of that wall. Legal only if all hold:
  1. Rule B is on and the player's hop is unused (`hop_available[player]`);
  2. the target cell is orthogonally adjacent and on the board;
  3. the edge between the pawn and the target is blocked by a wall owned by the opponent
     (`engine.wall_owner_on_edge(...) == 1 - player`); an open edge, the player's own wall,
     an adjacent pawn or a blocked cell does **not** allow a hop;
  4. the target is not the opponent's cell and not temporarily blocked;
  5. the hopping pawn still has a path to its goal row afterwards (path preservation).
- Permanently consumed upon execution (`hop_available[player] = False`).
- Wall ownership survives `REPLACE` (the moved wall still belongs to its placer). Because
  ownership now affects legality, `position_key()` keeps fixed walls as `(wall, owner)`
  pairs while either player still has a hop, and merges ownership only once no hop remains.
- History: until 2026-11-08 the engine used team rule R1 (a 2-cell jump needing no
  obstacle). Agents used it as a free sprint on move 1, so the team switched to the
  proposal's definition. Results produced under R1 (DQN checkpoints, ablation pilots)
  do not describe this rule.

### 5. Modification C: Temporary Cell Block (`BLOCK`)
- Syntax: `('BLOCK', (r, c))` or `["BLOCK", [r, c]]`.
- A single-use ability per player (`block_available[player] = True` initially).
- Temporarily blocks a chosen cell for the next $K$ plies (default $K = 3$, configurable via `rules['K']`, $K \ge 1$). See the timeline in Section 4.
- Cannot block an occupied pawn cell or an already blocked cell.
- A blocked cell cannot be entered by `MOVE` or `HOP` and counts as an obstacle in path checks. It does not block walls.
- Both players' blocks may be active at the same time; each timer runs on its own.
- Must preserve at least one valid path to the goal row for both players.
- Permanently consumed upon execution (`block_available[player] = False`).

---

## 4. Temporary Block ($K$-Ply) Timeline

When Player 0 plays `BLOCK` at cell $c^*$ with $K=3$:

| Ply | Active Player | Action | State of Block $c^*$ | Impassable? |
|---|---|---|---|---|
| **$t$** | Player 0 | `BLOCK c*` | Set to $\tau(c^*) = 3$ at end of ply | Impassable immediately |
| **$t+1$** | Player 1 | *any legal action* | Decrements to $\tau(c^*) = 2$ at end of ply | **Yes** (Player 1 cannot enter/cross) |
| **$t+2$** | Player 0 | *any legal action* | Decrements to $\tau(c^*) = 1$ at end of ply | **Yes** (Player 0 cannot enter/cross) |
| **$t+3$** | Player 1 | *any legal action* | Decrements to $\tau(c^*) = 0$ $\to$ removed at end of ply | **Yes** (Player 1 cannot enter/cross) |
| **$t+4$** | Player 0 | *any legal action* | Expired; cell $c^*$ is open | **No** (Cell is fully passable) |

Summary rule: With $K = 3$, the block covers the opponent's next move ($t+1$), the blocker's subsequent move ($t+2$), and the opponent's following move ($t+3$). At the end of ply $t+3$, the timer reaches 0 and the cell becomes fully passable for ply $t+4$ onward.

---

## 5. Strict Validation Order and Error Contract

The canonical method `apply_action(action, update_clock=True)` executes a strict, transactional pipeline. If any check fails, a `ValueError` is raised and the game state remains **completely unmodified**:

1. **Terminal Guard**:
   - If `self.is_terminal()`: raises `ValueError("Game is already over; no further actions may be applied.")`.
2. **Normalization (`_normalize_action`)**:
   - Checks sequence structure and types.
   - Raises `ValueError("Malformed action: ...")` for invalid arities or unknown action types.
3. **Legality Validation (`_validate_action`)**:
   - Evaluates legality without modifying state:
     - `MOVE`: raises `ValueError("Illegal move to (r, c).")` on wall collisions, blocked cells, or bounds.
     - `HOP`: raises `ValueError("Illegal hop to (r, c).")` if Rule B is disabled, the hop is used, or any condition of Section 3.4 fails.
     - `WALL`: raises `ValueError("Player X has no walls remaining in inventory.")` if $m_i \le 0$; raises `ValueError("Illegal wall placement at ...")` if colliding or disconnecting paths.
     - `REPLACE`: raises `ValueError("Reusable wall mechanic (Rule A) is disabled.")`, `ValueError("Player X has not placed any wall yet to reposition.")`, `ValueError("Cannot reposition older wall ...")`, or `ValueError("Illegal wall replacement ...")`.
     - `BLOCK`: raises `ValueError("Player X has no cell block available.")` or `ValueError("Illegal cell block at ...: path preservation violated.")`.
4. **Clock Update**:
   - If `update_clock=True`: deducts elapsed time from active player's clock. If clock $\le 0.0$, flags timeout loss immediately.
5. **Mutation (`_mutate`)**:
   - Updates pawn position, wall list, inventory decrement ($m_i \leftarrow m_i - 1$), latest wall pointer, or ability flags.
6. **Goal Win Condition Check**:
   - If `pawns[player][0] == goals[player]`: sets `winner = player`, `terminal_reason = 'goal'`.
7. **Block Timers Decrement**:
   - Decrements all existing block timers by 1 ply; cleans up expired blocks ($\le 0$); sets new block timer to $K$.
8. **Turn Transition**:
   - Swaps active player (`current_player = 1 - current_player`) and increments `turn_count += 1`.
9. **Step Limit Truncation Check** (only when the engine was built with `max_plies`; the engine default is `None`, i.e. off):
   - If `turn_count >= max_plies`: sets `is_draw = True`, `is_truncated = True`, `terminal_reason = 'truncated'`.
   - `is_draw = True` here is only the engine's flag for its own optional cap. The team counts a truncated game as its own outcome, never as a draw (Section 6, item 3).
10. **No-Legal-Actions Draw Check**:
    - If non-terminal and `has_legal_action(current_player) == False`: sets `is_draw = True`, `terminal_reason = 'no_legal_actions'`.
    - Example: the mover's only open neighbour holds the opponent's pawn (or a blocked cell), and it has no walls, no `REPLACE`, no block and no hop left. Its path still exists (paths ignore pawns), but it cannot act.
11. **N-Fold Repetition Draw Check**:
    - Appends the canonical position key (`position_key()`) to history; if it has now occurred `repetition_limit` times (default 5, configurable): sets `is_draw = True`, `terminal_reason = 'repetition'`.

Steps 1-3 raise before anything changes, so a rejected action leaves the whole state unchanged, including the clock.

---

## 6. Precedence of Termination Conditions

When evaluating the end of a match, precedence is strictly evaluated as:
1. **Clock Timeout**: Instantaneous forfeit upon clock expiry.
2. **Goal Reached**: Immediate win for the player reaching their goal row.
3. **Max Plies Truncation**: not a rule of the game. It is a cap that stops a long game, and **a truncated game is its own outcome, never a draw**. Report draw and truncation rates separately.
   - Match runner (`evaluation/match_runner.py`, `MatchConfig.max_plies = 200`) and `WallzEnv` (`max_plies = 200`): they do **not** pass the cap to the engine. They stop the game themselves and record `outcome.category = 'truncated'` (the engine is not terminal).
   - Engine's own optional cap (`max_plies` in `WallzConfig`, default `None` = off): the engine stops at that ply with `terminal_reason = 'truncated'`, `is_truncated = True` and `is_draw = True`. That `is_draw` flag exists only for this optional cap; read `terminal_reason`, not `is_draw`, to tell it apart from a rule draw.
   - Web demo (`web/backend/sessions.py`): passes its 200-ply cap to the engine, so a capped demo game ends with `terminal.reason = 'truncated'`. The demo shows it as "truncated, not a rule draw", and the Arena counts it in its own `truncated` column (`web/backend/arena.py`).
4. **No Legal Actions**: Draw declared if the player to move has zero legal actions (`terminal_reason = 'no_legal_actions'`; the match runner reports `draw_no_legal_actions`).
5. **N-Fold Repetition**: Draw declared when the same canonical position occurs `repetition_limit` times, default 5 (`terminal_reason = 'repetition'`).

A terminal game rejects every further action with `ValueError` (Section 5, step 1).
