# Tier-3 Exact Retrograde Solver for Modified Wallz

**Module**: `agents/exact_solver.py`  
**CLI Tool**: `scripts/solve_tiny_board.py`  
**Test Suite**: `tests/test_exact_solver.py` (9 tests), `tests/test_retrograde.py` (17), `tests/test_solver_final_rules.py` (18); counted 2026-11-11, all passing (`docs/implementation_status.md` Section 2)  
**Status**: Implemented and verified under the final rules; final-rule solves in `experiments/results/final_eval/` (`docs/reports/stage2_solver_stats.md`)  

---

## 1. Executive Summary & Purpose

The **Tier-3 Exact Solver** computes the exact, game-theoretic equilibrium outcomes and optimal minimax policies for tractable miniature configurations of Modified Wallz. It serves as an uncompromised, ground-truth benchmark against which heuristic search agents (Tier-2 Minimax) and reinforcement learning policies (DQN / MARL) can be evaluated.

Unlike depth-limited heuristic search agents that approximate state values using evaluation functions, the Exact Solver performs **exhaustive backward induction and retrograde queue propagation on the game graph**, yielding:
1. **Provably Exact Values**: $V^*(s) \in \{+1.0, 0.0, -1.0\}$ indicating whether state $s$ is a forced `WIN`, forced `LOSS`, or optimal `DRAW` under mutual best play.
2. **Optimal Distance-to-Terminal**:
   - For `WIN` states: minimum plies required to force victory ($d^*(s) = \min_{a} d^*(s') + 1$).
   - For `LOSS` states: maximum plies to delay defeat ($d^*(s) = \max_{a} d^*(s') + 1$).
   - For `DRAW` states: $d^*(s) = \infty$, representing non-terminating cyclic play.
3. **Provably Optimal Policies**: $\pi^*(s) = \arg\max_{a} \min_{a'} \dots$ selecting moves that guarantee the theoretical outcome.

---

## 2. State-Space Feasibility & Tractability Analysis

Modified Wallz combines spatial pawn navigation, wall placement, latest-wall repositioning, one-time hops, and temporary cell blocks. The complete Markov state $s$ is represented as a tuple:
\[
s = \left(p, (x_0, y_0), (x_1, y_1), W_0, W_1, m_0, m_1, r_0, r_1, h_0, h_1, B, \tau\right)
\]
where $p \in \{0, 1\}$ is the active player, $(x_i, y_i)$ are pawn coordinates, $W_i$ are wall sets, $m_i$ are wall inventories, $r_i$ are latest-wall pointers, $h_i$ are hop flags, $B$ are blocked cell coordinates, and $\tau$ are block decay timers.

### Combinatorial Complexity Analysis

| Configuration | Board $N$ | Walls $M$ | Hop | Block $K$ | Reachable States $|V|$ | Solve Time (s) | Tractability Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Pawn Race Pure** | 3 | 0 | No | No | 45 | 0.002 s | **Trivial** |
| **Pawn Race + Hop** | 3 | 0 | Yes | No | 186 | 0.025 s | **Trivial** |
| **Pawn Race + 1 Wall** | 3 | 1 | No | No | 1,953 | 0.370 s | **Fast (< 1s)** |
| **Pawn + Hop + 1 Wall** | 3 | 1 | Yes | No | 18,226 | 9.488 s | **Tractable (< 10s)** |
| **Pawn Race Pure (5x5)**| 5 | 0 | No | No | 475 | 0.047 s | **Fast (< 0.1s)** |
| **Pawn Race + Hop (5x5)**| 5 | 0 | Yes | No | 1,910 | 0.389 s | **Fast (< 0.5s)** |
| **Standard 5x5 ($M=10$)**| 5 | 10 | Yes | $K=2$ | $> 10^9$ | Intractable | **Exceeds RAM limits** |
| **Standard 9x9 ($M=10$)**| 9 | 10 | Yes | $K=3$ | $> 10^{18}$ | Intractable | **Intractable (Requires Minimax/DQN)** |

### Rigorous Domain Bounding
The solver strictly bounds its exactness claim to the explored reachable state space. It never approximates large boards or simplifies rules while claiming exactness. The solver includes an explicit `is_solved(state)` predicate to cleanly distinguish states within its solved domain from unvisited or out-of-domain states.

---

## 3. Retrograde Queue Propagation Algorithm

Because Modified Wallz allows pawn movement backwards and forwards, and Modification A allows repositioning the latest wall between valid slots, the directed state graph $G = (V, E)$ contains **cycles**. Standard DAG backward induction (topological sort) fails on graphs with cycles.

To rigorously solve cyclic zero-sum games, `ExactSolver` implements **Retrograde Propagation with Unresolved Action Degree Counters** (an exact game-graph value iteration algorithm):

### Algorithm Formulation

```text
Algorithm: RetrogradeQueuePropagation(G = (V, E))
Input:  Directed reachable state graph where each node u in V represents a legal Markov state,
        adj[u] contains legal forward transitions (action, v), and rev_adj[v] contains incoming transitions (action, u).

1. Initialize:
   status[u]   <- UNKNOWN,  for all u in V
   distance[u] <- inf,      for all u in V
   best_act[u] <- null,     for all u in V
   deg[u]      <- len(adj[u]), count of unresolved legal successor actions
   Q           <- Queue()

2. Seed Terminal Nodes:
   For each u in V:
     If u is terminal (winner is player(u)):
       status[u] <- WIN,  distance[u] <- 0, best_act[u] <- null, Q.push(u)
     Else if u is terminal (winner is 1 - player(u)):
       status[u] <- LOSS, distance[u] <- 0, best_act[u] <- null, Q.push(u)
     Else if u is terminal draw (5-fold repetition or deg[u] == 0):
       status[u] <- DRAW, distance[u] <- 0, best_act[u] <- null, Q.push(u)

3. Backward Queue Propagation:
   While Q is not empty:
     v <- Q.pop()
     For each (action a, predecessor u) in rev_adj[v]:
       If status[u] != UNKNOWN:
         continue

       # If moving to v hands the opponent a LOSS, the active player in u can force a WIN!
       If status[v] == LOSS:
         status[u]   <- WIN
         distance[u] <- distance[v] + 1
         best_act[u] <- a
         Q.push(u)

       # If moving to v hands the opponent a WIN, action a is a losing branch for u.
       Else if status[v] == WIN:
         deg[u] <- deg[u] - 1
         # If ALL legal actions available from u lead to an opponent WIN, u is a forced LOSS!
         If deg[u] == 0:
           status[u]   <- LOSS
           # In a forced loss, optimal defense delays defeat longest (max distance)
           distance[u] <- max_{(act', succ) in adj[u]} distance[succ] + 1
           best_act[u] <- argmax_{(act', succ) in adj[u]} distance[succ]
           Q.push(u)

4. Cycle & Draw Resolution:
   For each u in V with status[u] == UNKNOWN:
     status[u]   <- DRAW
     distance[u] <- inf
     # Select any non-losing transition that preserves the draw
     best_act[u] <- first action a where status[succ] != WIN
```

### Invariants & Guarantees
- **Soundness**: Every state marked `WIN` has at least one transition to a state marked `LOSS` for the next player.
- **Completeness**: Every state marked `LOSS` has all its legal actions transitioning to states marked `WIN` for the next player.
- **Cycle Safety**: Any state participating in a cycle where neither player can force a win is labeled `DRAW` with $V(s) = 0.0$ and $d(s) = \infty$.

---

## 4. Key Game-Theoretic Discoveries

> **Stale (old HOP rule).** Items 2-4 below and the "Hop = Yes" rows of the table in Section 2
> were solved under the pre-`4e346a8` HOP rule (a jump that needed no wall). Under the final
> rule a hop needs an opponent's wall, so with M = 0 it never happens: N3 M0 with hop is the
> same 45-state game as without it, a WIN in 3 for player 0 (`experiments/results/final_eval/solve_N3_K1_M0.json`).
> Final-rule results: `docs/reports/stage2_solver_stats.md` Section 2. Item 1 still holds.

Using the Exact Solver, several non-trivial game-theoretic properties of Modified Wallz were discovered and proven:

### 1. 3×3 Pure Pawn Race ($M=0$, No Hop)
- **Reachable States**: 45
- **Root State Outcome**: Player 0 forces a **WIN in 3 plies**.
- **Winning Path**:
  - Ply 1: P0 moves `(0, 1) -> (1, 1)`.
  - Ply 2: P1 at `(2, 1)` cannot step to `(1, 1)` (occupied) and is forced to move sideways to `(2, 0)` or `(2, 2)`.
  - Ply 3: P0 steps directly to goal row `(2, 1)`, winning the match.
  - Sideways opening moves `(0, 0)` or `(0, 2)` give up the center and yield theoretical `DRAW` (cyclic wandering).

### 2. 3×3 Pawn Race with One-Time Hop ($M=0$, Hop=True)
- **Reachable States**: 186
- **Root State Outcome**: Player 0 is in a **forced LOSS in 2 plies**!
- **Reasoning**: If P0 moves forward to `(1, 1)`, P1 at `(2, 1)` uses their one-time **HOP** to jump directly over P0 into `(0, 1)`, winning in 1 ply! If P0 steps sideways, P1 still reaches row 0 first via hop.

### 3. 3×3 with 1 Wall ($M=1$, Hop=True)
- **Reachable States**: 18,226
- **Root State Outcome**: Player 0 forces a **DRAW**!
- **Reasoning**: If P0 moves the pawn, P1 hops and wins. However, P0 has 1 wall. P0 places a horizontal wall `WALL (0, 0, 'H')`, shutting down P1's immediate hop corridor and holding the equilibrium at **DRAW**.

### 4. 5×5 Pawn Race with Hop ($M=0$, Hop=True)
- **Reachable States**: 1,910
- **Root State Outcome**: Player 0 forces a **WIN in 7 plies**.
- **Optimal First Move**: `('MOVE', (1, 2))`.
- **Finding**: Jumping early with `HOP` is a blunder that forfeits the game; preserving the hop while advancing the center pawn is the unique path to victory.

---

## 5. Empirical Benchmark Comparison with Minimax

To validate both the Exact Solver and the Tier-2 Minimax agent:
1. **At Terminal Search Depth**: On $3\times3$ pawn race (distance to win = 3 plies), `MinimaxAgent(depth=4)` explores all terminal leaves ($+1000$ score) and chooses `('MOVE', (1, 1))`, **in 100% agreement with the exact solver**.
2. **At Shallow Search Depth**: A shallow search agent without sufficient depth may misjudge pawn races; the exact solver serves as the ground truth to quantify the exact suboptimality gap ($V^*(s) - V_{\text{minimax}}(s)$) of heuristic agents.

---

## 6. Python API & CLI Usage

### Python API

```python
from wallz_engine import WallzEngine
from agents.exact_solver import ExactSolver, ExactSolverAgent

# 1. Compile exact solver for a 3x3 board with 1 wall
solver = ExactSolver(board_size=3, rules={'A': True, 'B': True, 'C': False}, max_walls=1)
solver.solve()

# 2. Query game-theoretic value and optimal policy
engine = WallzEngine(board_size=3)
engine.max_walls = 1
engine.wall_inventory = {0: 1, 1: 1}

if solver.is_solved(engine):
    outcome, plies = solver.get_state_outcome(engine)
    val = solver.get_value(engine, player=0)
    best_move = solver.get_optimal_action(engine)
    print(f"Outcome: {outcome}, Plies: {plies}, Value: {val}, Move: {best_move}")

# 3. Instantiate an agent that plays the exact optimal policy
agent = ExactSolverAgent(player_id=0, solver=solver)
action = agent.select_action(engine)
```

### CLI Tool (`scripts/solve_tiny_board.py`)

```bash
# Solve 3x3 pure pawn race
python3 scripts/solve_tiny_board.py --board-size 3 --max-walls 0 --no-hop --show-root-actions

# Solve 3x3 with one-time hop
python3 scripts/solve_tiny_board.py --board-size 3 --max-walls 0 --show-root-actions

# Solve 5x5 pawn race with hop
python3 scripts/solve_tiny_board.py --board-size 5 --max-walls 0 --show-root-actions

# Solve 3x3 with 1 wall and latest-wall repositioning
python3 scripts/solve_tiny_board.py --board-size 3 --max-walls 1 --show-root-actions
```

