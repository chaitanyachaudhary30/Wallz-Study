# Modified Wallz — Official Game Rules & Mechanics

## 1. Overview
Modified Wallz is a deterministic, sequential, two-player zero-sum adversarial grid game inspired by Quoridor, enhanced with dynamic spatial constraints, finite reusable resources, and temporal abilities.

---

## 2. Board & Pawns
- Played on an $N \times N$ discrete grid (default $N = 9$).
- **Player 0**: Starts at `(0, floor(N/2))`. Goal is row `N - 1`.
- **Player 1**: Starts at `(N - 1, floor(N/2))`. Goal is row `0`.
- Players strictly alternate turns taking exactly one action per turn.

---

## 3. Action Categories

### 1. Pawn Movement (`MOVE`)
- Moves 1 cell orthogonally (North, South, East, West).
- Cannot cross active walls.
- Cannot land on an opponent pawn or a currently blocked cell.

### 2. Path-Preservation Rule
- **Core Invariant**: Every action (wall placement, wall repositioning, move, or cell block) must preserve at least one valid path from each player's current position to their respective goal row:
  $$\operatorname{PathExists}(s', 0) = 1 \quad \text{and} \quad \operatorname{PathExists}(s', 1) = 1$$

### 3. Modification A: Reusable Walls (`WALL`, `REPLACE`)
- Walls are 2-cell barriers placed horizontally (`'H'`) or vertically (`'V'`).
- Walls cannot overlap with existing walls or intersect through another wall's center.
- **Initial Inventory**: Each player starts with a finite inventory of $M = 10$ walls ($m_i = 10$).
- **New Wall Placement (`WALL`)**:
  - Permissible only when $m_i > 0$.
  - Consumes 1 wall from inventory ($m_i' = m_i - 1$).
  - Designates this wall as the player's single most recently placed wall ($r_i$).
  - Older walls placed prior to $r_i$ become permanent fixtures for the remainder of the match.
- **Wall Repositioning / Replacement (`REPLACE`)**:
  - Permissible if and only if the player has placed at least one wall ($r_i \neq \varnothing$).
  - **Recency Constraint**: Only the player's single most recently placed wall ($r_i$) can be repositioned. Older walls remain permanently fixed and cannot be moved or removed.
  - Reuses the existing wall on the board; does **not** consume additional wall inventory ($m_i' = m_i$).
  - When $m_i = 0$, the player cannot place new walls, but can continue repositioning $r_i$.
  - Must satisfy path preservation for both players.

### 4. Modification B: One-Time Hop (`HOP`)
- A single-use ability per player.
- Moves the pawn 2 cells orthogonally (jumping over walls and/or adjacent pawns).
- Cannot land on an opponent pawn or a blocked cell.
- Permanently consumed upon execution (`hop_available[player] = False`).

### 5. Modification C: Temporary Cell Block (`BLOCK`)
- A single-use ability per player.
- Temporarily blocks a chosen cell for $K$ turns (default $K = 3$ plies).
- Cannot block an occupied pawn cell or an already blocked cell.
- Must preserve at least one valid path to the goal for both players.
- The block timer decrements by 1 each turn. Upon reaching 0, the cell is unblocked.
- Permanently consumed upon execution (`block_available[player] = False`).

---

## 4. Termination & Winning Conditions

1. **Goal Reach**: The first pawn to reach its target goal row immediately wins the game.
2. **Clock Depletion**: A player whose clock runs out (`time_remaining <= 0.0s`) immediately loses.
3. **5-Fold Repetition Draw**: If an identical full game state (pawn positions, active walls, wall inventories, latest wall pointers, block timers, ability availability, and active turn indicator) occurs 5 times, the match terminates in a draw.
4. **Step Limit Truncation**: A match truncates as a draw if turn count reaches $T_{\max} = 200$.
