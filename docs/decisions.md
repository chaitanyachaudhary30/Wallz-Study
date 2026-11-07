# Engineering & Architecture Decisions

This document records the key architectural and algorithmic decisions made during the implementation of the Tier-2 Minimax Agent and engine integration.

---

### Decision 1: Fast Engine Cloning vs. In-Place Action Reversal
- **Context**: Minimax tree search requires exploring hypothetical successor states across branching trees. Two patterns are common: in-place mutations with an `undo_action()` method, or cloning state instances.
- **Decision**: Implemented `WallzEngine.clone()` and `__deepcopy__()` using fast shallow copies of state dictionaries and coordinate tuples.
- **Rationale**: An `undo_action()` approach in a game with reusable latest-wall repositioning, permanent older walls, temporary cell blocks (with decaying timers), and ability consumption flags introduces high mutation complexity and risk of state corruption. Cloning eliminates state pollution across sibling branches.

---

### Decision 2: Clock Isolation in Simulated Search
- **Context**: The game engine features per-player chess countdown timers (`time_remaining`).
- **Decision**: Added optional parameter `update_clock=True` to `WallzEngine.apply_action()`. In search simulations, `update_clock=False` is passed.
- **Rationale**: Simulated tree nodes represent hypothetical futures, not actual match time spent. Real match time should only be deducted during real turn execution.

---

### Decision 3: Iterative Deepening with Hard Fallback
- **Context**: On 9×9 boards, branch factors exceed 200 actions per ply. Deep searches may exceed per-turn time limits.
- **Decision**: Implemented Iterative Deepening ($d = 1 \dots D$) with a pre-cached legal fallback action and per-node deadline checks.
- **Rationale**: If a deep search iteration times out, the search unwinds cleanly and returns the optimal action from the previous completed depth. The agent is mathematically guaranteed never to return `None` or time out when legal actions exist.

---

### Decision 4: Deterministic 5-Tier Move Ordering
- **Context**: The branch factor of Modified Wallz is high (over 200 actions on open boards). Unordered search causes combinatorial explosion.
- **Decision**: Actions are ordered deterministically:
  1. Immediate winning moves (`MOVE` / `HOP` landing on goal row).
  2. Spatial progress advancing pawn towards goal ($d_{\text{self}}$ reduction).
  3. Pawn moves over wall placements and cell blocks.
  4. Wall placements (`WALL`).
  5. Cell blocks (`BLOCK`).
  6. Original index tie-breaking (`-idx`).
- **Rationale**: Trying promising moves and pawn advances first quickly establishes tight $\alpha$ and $\beta$ bounds, triggering immediate cutoffs on subsequent wall placements.

---

### Decision 5: Strict Adherence to Report Leaf Evaluation
- **Context**: Various heuristic terms could be added to leaf evaluation (e.g., wall count, board center control).
- **Decision**: Implemented strictly $U_i(s) = \tanh(0.2 \cdot (d_{-i}(s) - d_i(s)))$ with terminal state dominance ($W_{\text{term}} - \text{ply}$).
- **Rationale**: Handcrafted auxiliary heuristics introduce bias and violate the formal specification in the project report. Keeping leaf evaluation strictly aligned with the research proposal ensures sound empirical comparisons across baselines and MARL policies.

---

### Decision 6: Retrograde Queue Propagation with Unresolved Degree Counters for Cyclic Game Graphs
- **Context**: Modified Wallz features cycles: pawns can step back and forth, and Rule A allows repositioning the latest wall between valid coordinates indefinitely. Standard backward induction on DAGs (topological ordering) fails or loops indefinitely on state graphs with cycles.
- **Decision**: Implemented game-graph retrograde propagation with active action degree counters ($\text{deg}[u] = |\mathcal{A}(u)|$). Seeding terminal states with $d=0$, when an opponent loss is discovered at $v$, $u$ is immediately labeled a `WIN` ($d = d_v + 1$). When an opponent win occurs at $v$, $\text{deg}[u]$ is decremented. If and only if $\text{deg}[u] == 0$, $u$ is forced into a `LOSS` ($d = \max d_v + 1$). All unreached states after queue exhaustion are proven cycles where neither player can force a win, resolving rigorously to `DRAW` ($d = \infty$, $V = 0.0$).
- **Rationale**: This provides a mathematically complete and sound resolution of cyclic games without relying on depth cutoffs, heuristic approximations, or infinite loops.

---

### Decision 7: Rule R1 — Hop Mechanics and Path Preservation
- **Context**: The proposal's `LegalHop` condition 2 specified "bypasses an opponent's wall".
- **Decision**: Implemented team rule R1: The hop is a 2-cell orthogonal jump over anything. No obstacle is required. The pawn must land on a free, unblocked cell on the board, and must retain an open path to its goal row (a hop into a sealed pocket is illegal).
- **Rationale**: Simplifies spatial geometry, avoids ambiguities in defining what constitutes "bypassing" in complex wall lattices, and acts as a one-time tactical sprint.

---

### Decision 8: Rule R3 — Terminal State on No Legal Actions
- **Context**: What occurs when the side to move has no legal moves available (e.g. trapped without legal hops or wall placements)?
- **Decision**: Implemented team rule R3: If the side to move has no legal actions in a non-terminal position, the match terminates immediately as a Draw (`terminal_reason = 'no_legal_actions'`).
- **Rationale**: Direct precedent in chess stalemate (FIDE Art. 5.2.1), avoids infinite forced-pass loops, and ensures all reachable states have deterministic termination.

---

### Decision 9: Rule R7 — Conservative Block Obstacle Path Checks
- **Context**: Path checks for new walls, replacements, and blocks must ensure reachability to goal rows.
- **Decision**: All active temporary blocks are treated as impassable obstacles during path preservation checks, including blocks with $\tau = 1$ that decay on the transition.
- **Rationale**: Conservative by at most 1 ply, prevents temporary edge-case entrapments, and guarantees that the reachability invariant is never violated.



