# System Architecture & Component Integration

This document outlines the architecture, component boundaries, engine interface, state transitions, agent specifications, web presentation layer, and data flow of the **Modified Wallz** multi-agent platform.

---

## 1. High-Level Architecture

The platform strictly separates game simulation, state representation, decision agents, evaluation, and tournament execution into decoupled modules, anchored by the canonical engine as the single source of truth:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Web Presentation Layer                           │
│           React 18 + TypeScript (web/frontend) [Pure Drawing / UI]          │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ HTTP / REST (JSON)
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             Web Backend (Flask)                             │
│                  (web/backend: sessions, arena, explain, replays)           │
└──────┬──────────────────────┬───────────────────────┬───────────────────────┘
       │                      │                       │
       ▼                      ▼                       ▼
┌──────────────┐      ┌──────────────┐        ┌──────────────┐
│ GreedyAgent  │      │ MinimaxAgent │        │   DQNAgent   │
│(agents/greedy│      │(agents/minima│        │ (agents/dqn) │
└──────┬───────┘      └───────┬──────┘        └───────┬──────┘
       │                      │                       │
       │ select_action(engine)│ select_action(engine) │ select_action(engine)
       │                      │                       │
       ▼                      ▼                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                                WallzEngine                                  │
│                   (wallz_engine.py / src.modified_wallz)                    │
│                                                                             │
│   • Board state & pawn coordinates (single source of truth)                 │
│   • Reusable Walls & Repositioning (Modification A)                         │
│   • One-Time Hop Mechanics (Modification B)                                 │
│   • Temporary Cell Blocks & Decay (Modification C)                          │
│   • BFS Shortest-Path & Path-Preservation Validation                        │
│   • 5-Fold Repetition Hashable State History (O(1) ChainMap clone)          │
│   • Chess-Clock Per-Player Timers & Rule R3 Stalemate Termination           │
└─────────────────────────────────────────────────────────────────────────────┘
                               ▲
                               │ Ground-Truth Policy & Values
                               │ (evaluation.retrograde)
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
| `has_legal_action(player=None)` | `player` (int or None) | `bool` | Early-exit check for at least one legal action. |
| `apply_action(action, update_clock=True)` | `action` (tuple), `update_clock` (bool) | `(state, winner, is_draw)` | Applies action, checks win/draw conditions, decays timers, and advances turn. |
| `reposition_wall(player, new_wall, old_wall=None)` | `player` (int), `new_wall` (tuple), `old_wall` (tuple or None) | `bool` | Repositions the player's most recently placed wall without consuming inventory. |
| `shortest_path(player)` | `player` (int) | `int` or `float('inf')` | Computes BFS unblocked shortest-path distance to goal. |
| `shortest_path_cells(player)` | `player` (int) | `list` of `(r, c)` | Returns the ordered sequence of cells along the shortest path. |
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
- `('HOP', (r, c))`: one-time step into the adjacent cell across an opponent's wall.
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
   - Exposes `score_actions(engine)` for transparent introspection and web visualization.

2. **Tier-2 Adversarial Minimax (`MinimaxAgent`)**:
   - Multi-step lookahead up to configurable depth $D$.
   - Full Alpha-Beta pruning with 5-tier deterministic move ordering.
   - Bounded leaf evaluation $U(s) = \tanh(0.2 \cdot (d_{\text{opp}} - d_{\text{self}}))$.
   - Iterative deepening with per-turn deadline fallback.
   - Introspection via `get_stats()` and full-window child searches at depth $D-1$.

3. **Tier-3 Exact Retrograde Solver (`ExactSolverAgent`)**:
   - Complete retrograde backward induction with cyclic degree propagation on tractable configurations.
   - Computes provably optimal game-theoretic values $\{+1.0, 0.0, -1.0\}$ and minimum distance-to-terminal policies.
   - Solves $3\times3$ with up to 1 wall / hop ($< 10$s) and $5\times5$ pawn races ($< 0.5$s) as ground truth.

4. **MARL Self-Play Agent (`DQNAgent`)**:
   - Double-DQN with Negamax target formulation and illegal action masking.
   - 14-plane spatial observation tensor ($14 \times N \times N$) under canonical mover perspective.
   - Uniform FIFO replay buffer with horizontal mirror data augmentation.
   - Self-play curriculum with historical snapshot pool and Greedy anchor.
   - Released models in `checkpoints/release/dqn_3x3/` and `checkpoints/release/dqn_4x4/` (validated $>90\%$ sign agreement on Gate X-16).

---

## 5. Web Presentation Layer Architecture

The web application connects a lightweight, modern React frontend to a Flask backend wrapping the canonical engine:

```text
Browser (React 18 + TS) ──[HTTP REST]──> Flask Backend ──> WallzEngine & Agents
        │                                      │
   Pure SVG Grid                          SessionStore
   C-GEOM Drawing                        Agent Registry
   Heatmap Overlays                      Analysis Engine
   No Game Logic                         Replay Verifier
```

### Pure Presentation Frontend (`web/frontend/`)
- **No Rule Duplication**: The React frontend contains **zero game logic**. It never computes legal moves, shortest paths, or collision checks. All legal targets (`state.legal`) and path polylines (`state.paths`) are provided by the backend serialization of the engine.
- **Contract C-GEOM**: Board geometry calculations (`geometry.ts`) strictly translate grid coordinates to SVG canvas space (with Row 0 at the bottom and Row $N-1$ at the top).
- **Five Primary Views**:
  - `Play`: Interactive human vs AI or AI vs AI game session with undo, move hints ("Ask AI"), and error toasts.
  - `Arena`: Synchronous AI vs AI tournaments with speed slider, dual think panels, and Wilson 95% confidence intervals.
  - `Training`: Real-time Recharts dashboards for DQN training curves (win-rate, TD-loss, epsilon), training loop diagram, and benchmark summaries.
  - `Replays`: Replay browser discovering experiment records, interactive scrubber, step-by-step frame inspection, and JSON upload.
  - `Rules`: Interactive explorer running all 9 canonical rule scenarios with live invariant verification badges and markdown viewer for `docs/game_rules.md`.

### Flask Application Backend (`web/backend/`)
- **`serialize.py`**: Converts `WallzEngine` state into Contract C-STATE JSON (positions, inventories, walls with `latest` flag, blocked cell timers, grouped legal actions, and shortest path cell coordinates).
- **`sessions.py`**: In-memory LRU session store managing active games, move history, and transactional undo (which replays history through a fresh engine).
- **`registry.py`**: Dynamic agent factory detecting available checkpoints, gate validation rules, and model caching.
- **`explain.py`**: Decision introspection producing Contract C-EXPL envelopes for Random, Greedy, Minimax, DQN, and Exact agents without diverging from the agent's actual decision.
- **`arena.py`**: Tournament runner executing batches with balanced colors and opening ply seeds.
- **`replays.py`**: Replay discovery and verification, regenerating C-STATE frames ply-by-ply through a fresh `WallzEngine`.
- **`scenarios.py`**: Runner executing the 9 scenario definitions from `tests/scenario_runner.py`.
- **`training_data.py`**: Parser and downsampler for training logs (`train_log.jsonl`) and benchmark summaries.

---

## 6. Single Source of Truth Guarantee

To prevent divergent game rules or stealth bugs across agents and user interfaces, the project enforces four architectural safeguards:

1. **AST & Regex Single-Source Guard (`tests/test_single_source.py`)**:
   - Statically scans all Python files in `web/backend/` and TypeScript files in `web/frontend/src/`.
   - Flags and fails if any banned graph algorithms (`bfs`, `shortest_path`, `is_legal`, `can_place`) or rule computations appear outside `wallz_engine.py` and `src/modified_wallz/`.
2. **Replay Verification Invariant**:
   - Replays are never stored as pre-rendered state blobs. Only action sequences are recorded. When loaded or viewed, every frame is re-executed through `WallzEngine.apply_action`. Any illegal move immediately flags the replay as corrupted.
3. **Scenario Synchronization**:
   - The same 9 canonical rule scenarios defined in `tests/scenario_runner.py` are executed by both automated pytest test suites and the web demo's `/api/scenarios` runner.
4. **Docs Live Endpoint**:
   - `GET /api/rules` reads `docs/game_rules.md` directly from the filesystem, guaranteeing that documentation shown to users in the web app is identical to the repository documentation.

---

## 7. Project Module Map

| Directory / Module | Role | Owner |
|---|---|---|
| `wallz_engine.py` | Canonical game engine (Rules, BFS, Validation, History) | Member 2 |
| `src/modified_wallz/` | Board geometry and core engine package | Member 2 |
| `agents/greedy_agent.py` | Heuristic greedy baseline with `score_actions` | Member 2 |
| `agents/minimax.py` | Alpha-Beta minimax agent with move ordering | Member 1 |
| `agents/evaluation_fn.py` | Minimax heuristic evaluation functions | Member 1 |
| `evaluation/retrograde.py` | Exact retrograde solver and state graph | Member 3 |
| `evaluation/metrics.py` | Tournament statistics and Wilson confidence intervals | Member 3 |
| `agents/dqn/` | Deep Q-Network training, architecture, replay buffer | Member 4 |
| `web/backend/` | Flask REST API, game sessions, and agent explanations | Member 2 |
| `web/frontend/` | React 18 TypeScript interactive presentation app | Member 2 |
| `tests/` | Comprehensive test suite (rules, greedy, web, AST guard) | All Members |
