# Tier-2 Minimax Adversarial Agent with Alpha-Beta Pruning

This document describes the design, theoretical formulation, search semantics, leaf evaluation, move ordering, and integration of the **Tier-2 Minimax Agent** (`MinimaxAgent`) for Modified Wallz.

---

## 1. Mathematical Formulation

Modified Wallz is formulated as a deterministic, sequential, two-player zero-sum Markov Game:
$$\mathcal{M} = \langle \mathcal{S}, \mathcal{A}_1, \mathcal{A}_2, T, R_1, R_2, \gamma \rangle$$

Because information is complete and transitions are deterministic, minimax search with Alpha-Beta pruning provides a rigorous foundation for evaluating adversarial game trees up to horizon $D$.

### Minimax Recurrence

Let $s \in \mathcal{S}$ denote the environment state, $d \in \mathbb{N}_0$ denote remaining search depth in plies, and $p_{\text{root}} \in \{0, 1\}$ denote the player controlling the agent.

The backed-up minimax value function $V(s, d)$ from the perspective of $p_{\text{root}}$ is defined recursively as:

$$V(s, d) = \begin{cases}
U(s, d), & \text{if } s \in \mathcal{S}_{\text{terminal}} \text{ or } d = 0, \\
\max\limits_{a \in \mathcal{A}(s)} V(T(s, a), d - 1), & \text{if } z(s) = p_{\text{root}}, \\
\min\limits_{a \in \mathcal{A}(s)} V(T(s, a), d - 1), & \text{if } z(s) \neq p_{\text{root}},
\end{cases}$$

where $z(s) \in \{0, 1\}$ is the active player at state $s$, and $T(s, a)$ is the deterministic transition operator.

---

## 2. Evaluation Function

Terminal states take strict precedence over heuristic evaluations.

### Terminal Evaluation

Terminal payoffs are bounded by a large terminal constant $W_{\text{term}} = 1000.0$:

$$U_{\text{term}}(s, \text{ply}) = \begin{cases}
+W_{\text{term}} - \text{ply}, & \text{if } \text{Winner}(s) = p_{\text{root}}, \\
-W_{\text{term}} + \text{ply}, & \text{if } \text{Winner}(s) \neq p_{\text{root}}, \\
0.0, & \text{if } \text{Draw}(s).
\end{cases}$$

Discounting by $\text{ply}$ (distance from the search root) ensures:
1. When winning, the agent prioritizes the most immediate win.
2. When facing unavoidable loss, the agent chooses actions that delay the loss longest.
3. Because $W_{\text{term}} - \text{ply} \gg 1.0$, terminal outcomes always strictly dominate non-terminal heuristic leaf values.

### Non-Terminal Leaf Evaluation

At search horizons ($d = 0$) where neither player has reached a terminal condition, the leaf evaluation function strictly follows the project formalization:

$$H_{p_{\text{root}}}(s) = d_{-p_{\text{root}}}(s) - d_{p_{\text{root}}}(s)$$

$$U(s) = \tanh\big(0.2 \cdot H_{p_{\text{root}}}(s)\big)$$

where:
- $d_{p_{\text{root}}}(s)$ is the shortest unblocked path length from $p_{\text{root}}$'s pawn to its goal row.
- $d_{-p_{\text{root}}}(s)$ is the shortest unblocked path length from the opponent's pawn to its goal row.
- $\tanh(x) \in (-1.0, 1.0)$ provides a smooth, bounded, monotonic scaling of net spatial advantage.

---

## 3. Alpha-Beta Pruning

Alpha-Beta pruning maintains dynamic bounds $[\alpha, \beta]$ along each exploration branch:
- $\alpha$: Maximum score the maximizing player ($p_{\text{root}}$) can guarantee so far ($-\infty$ initially).
- $\beta$: Minimum score the minimizing player (opponent) can hold the maximizer to ($+\infty$ initially).

### Cutoff Condition
Whenever $\beta \le \alpha$, the current sub-branch cannot affect the root decision, enabling immediate pruning:
- At a **MAX** node: after updating $\alpha = \max(\alpha, \text{val})$, if $\beta \le \alpha$, prune remaining sibling actions.
- At a **MIN** node: after updating $\beta = \min(\beta, \text{val})$, if $\beta \le \alpha$, prune remaining sibling actions.

Alpha-Beta pruning guarantees mathematical equivalence to unpruned Minimax while substantially reducing node expansions.

---

## 4. Deterministic Move Ordering

Alpha-Beta efficiency depends heavily on exploring high-quality actions first. `MinimaxAgent` prioritizes candidate actions using a strict 5-tier deterministic hierarchy:

1. **Immediate Winning Moves**: Actions (`MOVE` / `HOP`) that directly land on the target goal row (`act_val[0] == goal`).
2. **Favorable Spatial Progress**: Pawn moves (`MOVE` / `HOP`) sorted by direct progress towards the goal, reducing $d_{\text{self}}$ and maximizing $H$.
3. **Pawn Action Precedence**: pawn actions (`MOVE`, and `HOP` across an opponent's wall) are ordered by distance-to-goal progress.
4. **Wall Placements (`WALL`)**: Defensive and obstruction barriers placed after direct pawn advances.
5. **Temporary Cell Blocks (`BLOCK`)**: Transient blocks explored after wall placements.
6. **Deterministic Tie-Breaking**: Broken deterministically by lower original action index (`-idx`) matching the baseline conventions.

---

## 5. Per-Turn Time Limit & Iterative Deepening

To comply with tournament clocks and prevent timeouts on large boards (e.g. 9×9 with 200+ branching factors):

1. **Guaranteed Fallback Action**: Before launching search, a guaranteed legal action (immediate win if present, or `legal_actions[0]`) is cached.
2. **Iterative Deepening**: Search iterates progressively from depth $d = 1$ to $d = \text{max\_depth}$.
3. **Deadline Guard**: At every node expansion, elapsed time is compared against `start_time + time_limit`. If exceeded, a `SearchTimeout` exception unwinds search.
4. **Graceful Fallback**: If depth $k$ is interrupted by timeout, the agent returns the optimal action from depth $k - 1$. An illegal or null action is never returned.

---

## 6. Implementation & Interface

`MinimaxAgent` is implemented in `agents/minimax.py`:

```python
from agents.minimax import MinimaxAgent

# Initialize agent for Player 0 with max depth 2 and 1.0s time limit
agent = MinimaxAgent(
    player_id=0,
    depth=2,
    time_limit=1.0,
    use_alpha_beta=True,
    order_moves=True
)

action = agent.select_action(engine)
stats = agent.get_stats()
```

### Exported Metrics
- `nodes_expanded`: Total nodes explored during the decision.
- `cutoffs`: Total Alpha-Beta cutoffs triggered.
- `last_search_depth`: Deepest fully completed lookahead depth.
- `last_decision_time`: Wall-clock decision latency (in seconds).
- `last_action_value`: Backed-up minimax score of the selected action.

---

## 7. Negamax Formulation & Search Optimizations (Phase 5)

### Negamax Recurrence
The search tree is refactored into canonical Negamax form:
$$V(s, \alpha, \beta, d) = \max_{a \in \mathcal{A}(s)} \left( -V(T(s, a), -\beta, -\alpha, d - 1) \right)$$
Evaluations at leaf nodes are evaluated from the active side-to-move perspective:
$$U(s) = \tanh\left(0.2 \cdot (d_{\text{opp}}(s) - d_{\text{mover}}(s))\right)$$
This eliminates redundant min/max dual-branching logic while preserving identical search decisions.

### Advanced Move Ordering
1. **PV-First Ordering**: The principal variation move discovered from iteration $d - 1$ is tried first at iteration $d$.
2. **Shortest-Path Wall Prioritization**: Walls and blocks intersecting the opponent's current shortest-path edges are ordered before off-path walls.
3. **History Heuristic**: Non-capturing moves causing beta-cutoffs accrue a history score proportional to $d^2$, dynamically boosting effective candidate moves across sibling branches.
4. **Per-decision state**: `select_action` clears the PV move and history table at the start of every call. Both carry across iterative-deepening depths inside one decision, never across decisions, so a reused agent returns the same move as a fresh one regardless of which positions it saw before (`tests/test_minimax_phase5.py::test_decision_independent_of_previous_calls`).

### Goal-Proximity Search Extension (`extend_goal`)
When `extend_goal=True`, at horizon $d = 0$, if either player is within critical proximity of victory ($\min(d_{\text{self}}, d_{\text{opp}}) \le 2$), the search extends by 1 ply (quiescence extension, capped at 2 extensions per branch) to eliminate horizon effects near goal rows.

### Modular Evaluator Protocol (Contract C7)
`MinimaxAgent` accepts an `evaluator` parameter conforming to the `@runtime_checkable` `EvaluationFunction` protocol:
- `HEvaluator`: Canonical baseline evaluating $\tanh(0.2 \cdot (d_{\text{opp}} - d_{\text{self}}))$.
- `LinearEvaluator(weights, scale=0.5)`: Linear feature combination $\tanh(w \cdot f / 2) = 2\sigma(w \cdot f) - 1$ utilizing 6 spatial and temporal features ($f_1 \dots f_6$).


