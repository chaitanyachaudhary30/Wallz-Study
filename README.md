<<<<<<< HEAD
# Modified Wallz — Core Game Engine & Baseline Heuristic (Task 2)

This repository contains the headless Python game engine (`WallzEngine`) and the Tier-1 Spatial Greedy Baseline (`GreedyAgent`) for the project **"Multi-Agent Reinforcement Learning for Adversarial Spatial Decision-Making with Temporal Constraints"**.

The engine provides a UI-decoupled, deterministic environment interface designed for **Multi-Agent Reinforcement Learning (DQN)**, **Minimax Tree Search with Alpha-Beta Pruning**, and **Exact Retrograde Solving**.

---

## 🚀 Task 2 Deliverables Summary

- **Decoupled Game Engine (`wallz_engine.py`)**: Extracted complete game rules and graph-path validation away from DOM/browser event listeners into a high-performance, headless Python class.
- **Rule Configurations & Ablation Matrix**: Native support for toggling all 3 game mechanics ($2^3 = 8$ ablation configurations).
- **FIFO Rolling Walls Mechanics**: Implemented FIFO queue mechanics for active walls. Placing an 11th wall automatically recycles and removes the player's oldest active wall.
- **5-Fold State Repetition Draw Detection**: Hashable state snapshot tracker preventing infinite looping trajectories.
- **Per-Player Clock Timers**: Integrated chess-style countdown clock with loss-on-time detection.
- **Tier-1 Greedy Baseline Agent (`greedy_agent.py`)**: Implemented deterministic spatial agent operating under strict 5-tier lexicographic action prioritization.
- **Automated Match Script (`run_match.py`)**: Headless match runner verifying state transitions, legal action generation, path preservation, and game truncation at $T_{\max} = 200$.

---

## 🎲 Game Rules & Mechanics

### 1. Core Objective & Board
- Played on an $N \times N$ grid (default $N=9$).
- **Player 0** starts at row `0` (goal: row `N-1`).
- **Player 1** starts at row `N-1` (goal: row `0`).
- Both players alternate turns taking 1 action per turn (`MOVE`, `HOP`, `WALL`, or `BLOCK`).
- **Path-Preservation Rule**: An action is illegal if it blocks all remaining shortest paths to the goal for either player.

### 2. The Three Modified Mechanics
- **Modification A: Rolling Reusable Walls**
  - Maximum of 10 active walls per player.
  - When placing an 11th wall, the player's **oldest wall** (1st placed) is automatically removed.
- **Modification B: One-Time Hop**
  - Single-use ability per player bypassing barriers/opponents (2-step jump) subject to legal landing conditions.
- **Modification C: Temporary Cell Blocking**
  - Single-use ability per player blocking a selected cell for $K$ turns (default $K=3$ plies), decaying turn-by-turn.

### 3. Termination & Draws
- **Goal Reach**: Natural win when a pawn reaches its target goal row.
- **Clock Depletion**: A player loses if their remaining total time drops to `0.0s`.
- **5-Fold Repetition**: Game ends in a draw if the identical state (pawn positions, wall queues, block timers, ability availability, turn indicator) occurs 5 times.
- **Step Limit**: Truncates automatically at $T_{\max} = 200$ turns.

---

## 📂 Repository Structure

```text
wallz_engine/
├── wallz_engine.py       # Core headless game engine and state transition rules
├── greedy_agent.py       # Tier-1 Lexicographic Greedy Baseline Agent
├── run_match.py          # Automated match runner and test logger
├── .gitignore            # Git ignore file
└── README.md             # Project documentation
=======
# FAI_project
>>>>>>> 7fe25fa75f9180633afea26710cb27685560d23b
