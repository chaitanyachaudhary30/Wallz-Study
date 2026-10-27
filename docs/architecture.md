# System Architecture & Component Integration

This document outlines the architecture, component boundaries, engine interface, state transitions, agent specifications, and data flow of the **Modified Wallz** multi-agent platform.

---

## 1. High-Level Architecture

The platform separates game simulation, state representation, decision agents, and tournament execution into decoupled modules:

```text
┌─────────────────────────────────────────────────────────────┐
│                       Match Runner                          │
│          (run_match.py / scripts/run_match.py)              │
└──────────────┬───────────────────────────────┬──────────────┘
               │                               │
               ▼                               ▼
     ┌──────────────────┐            ┌──────────────────┐
     │   GreedyAgent    │            │   MinimaxAgent   │
     │ (agents/greedy)  │            │ (agents/minimax) │
     └─────────┬────────┘            └─────────┬────────┘
               │                               │
               │ select_action(engine)         │ select_action(engine)
               │                               │
               ▼                               ▼
┌─────────────────────────────────────────────────────────────┐
│                        WallzEngine                          │
│               (wallz_engine.py / src.modified_wallz)        │
│                                                             │
│   • Board state & pawn coordinates                          │
│   • Reusable Walls & Repositioning (Modification A)         │
│   • One-Time Hop Mechanics (Modification B)                 │
│   • Temporary Cell Blocks & Decay (Modification C)          │
│   • BFS Shortest-Path & Path-Preservation Validation        │
│   • 5-Fold Repetition Hashable State History                │
│   • Chess-Clock Per-Player Timers                           │
└─────────────────────────────────────────────────────────────┘
                               ▲
                               │ Ground-Truth Policy & Values
                               │ (scripts/solve_tiny_board.py)
                     ┌──────────────────┐
                     │ ExactSolverAgent │
                     │(exact_solver.py) │
                     └──────────────────┘
```

---

## 2. Engine Interface (`WallzEngine`)

The engine serves as the single source of truth for rules, legal action generation, pathfinding, and state updates.

### Key Methods

| Method | Parameters | Returns | Description |
| :--- | :--- | :--- | :--- |
| `reset()` | None | `dict` | Resets the game to initial positions, wall inventories, and ability states. |
| `clone()` | None | `WallzEngine` | Fast, decoupled state cloning for hypothetical search. |
| `get_legal_actions(player=None)` | `player` (int or None) | `list` of actions | Generates all legally valid actions for the specified or active player. |
| `apply_action(action, update_clock=True)` | `action` (tuple), `update_clock` (bool) | `(state, winner, is_draw)` | Applies action, checks win/draw conditions, decays timers, and advances turn. |
| `reposition_wall(player, new_wall, old_wall=None)` | `player` (int), `new_wall` (tuple), `old_wall` (tuple or None) | `bool` | Repositions the player's most recently placed wall without consuming inventory. |
| `shortest_path(player)` | `player` (int) | `int` or `float('inf')` | Computes BFS unblocked shortest-path distance to goal. |
| `is_path_open(player)` | `player` (int) | `bool` | Verifies whether at least one unblocked path exists to goal. |
| `get_state()` | None | `dict` | Returns an immutable snapshot copy of current game state. |

---

## 3. State Representation & Transitions

### State Tuple
The hashable state key encapsulates:
$$s = (p_0, p_1, W_0, W_1, m_0, m_1, r_0, r_1, h_0, h_1, b_0, b_1, B_{\text{cells}}, z)$$

- $p_0, p_1$: Pawn coordinates $(r, c) \in [0, N-1]^2$.
- $W_0, W_1$: Active wall sets on the board (tuples of $(r, c, \text{orient})$).
- $m_0, m_1$: Remaining unused wall inventory counts ($\in [0, 10]$).
- $r_0, r_1$: Identity of the single most recently placed (repositionable) wall for each player.
- $h_0, h_1$: Binary hop ability availability flags.
- $b_0, b_1$: Binary temporary cell block availability flags.
- $B_{\text{cells}}$: Sorted mapping of active cell blocks to remaining decay plies.
- $z \in \{0, 1\}$: Active player index.

### Action Representation
An action is a tuple `(action_type, action_payload)`:
- `('MOVE', (r, c))`: Orthogonal pawn advance by 1 step.
- `('HOP', (r, c))`: Orthogonal 2-step jump bypassing intervening barriers.
- `('WALL', (r, c, 'H' | 'V'))`: New wall placement (consumes 1 wall from inventory).
- `('REPLACE', (r, c, 'H' | 'V'))`: Wall repositioning of the player's latest wall (does not consume inventory).
- `('BLOCK', (r, c))`: Temporary cell blockage for $K=3$ turns (preserves path to goal).

---

## 4. Agent Architecture & Polymorphic Protocol

All agents conform to a uniform decision protocol:

```python
class BaseAgent:
    def __init__(self, player_id: int):
        self.player_id = player_id

    def select_action(self, engine: WallzEngine) -> tuple | None:
        """Evaluates engine state and returns a legal action tuple or None."""
        raise NotImplementedError
```

### Agent Hierarchy

1. **Tier-1 Greedy Baseline (`GreedyAgent`)**:
   - 1-ply lookahead heuristic.
   - Evaluates immediate winning moves, draw avoidance, and net spatial differential:
     $$\text{Win} \succ \text{Avoid Draw} \succ \max(d_{\text{opp}} - d_{\text{self}}) \succ \max d_{\text{opp}} \succ \min d_{\text{self}} \succ \text{Tie-break}$$

2. **Tier-2 Adversarial Minimax (`MinimaxAgent`)**:
   - Multi-step lookahead up to configurable depth $D$.
   - Full Alpha-Beta pruning with 5-tier deterministic move ordering.
   - Bounded leaf evaluation $U(s) = \tanh(0.2 \cdot (d_{\text{opp}} - d_{\text{self}}))$.
   - Iterative deepening with per-turn deadline fallback.

3. **Tier-3 Exact Retrograde Solver (`ExactSolverAgent`)**:
   - Complete retrograde backward induction with cyclic degree propagation on tractable configurations.
   - Computes provably optimal game-theoretic values $\{+1.0, 0.0, -1.0\}$ and minimum distance-to-terminal policies.
   - Solves $3\times3$ with up to 1 wall / hop ($< 10$s) and $5\times5$ pawn races ($< 0.5$s) as ground truth.

4. **MARL Self-Play Agent (`DQNAgent`)**:
   - *(Planned / In-progress by team)* Value-based deep reinforcement learning with legal-action masking.

