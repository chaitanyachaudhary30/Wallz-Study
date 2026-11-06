# Modified Wallz — Multi-Agent System & AI Decision Baselines

[![Tests](https://img.shields.io/badge/tests-96%20passed-brightgreen.svg)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)]()
[![Course](https://img.shields.io/badge/CS329-Foundations%20of%20AI-orange.svg)]()

This repository contains the game environment, decision baselines, and evaluation framework for **"Multi-Agent Reinforcement Learning for Adversarial Spatial Decision-Making with Temporal Constraints"** (CS 329: Foundations of AI / Multi-Agent Systems, IIT Gandhinagar).

The project studies competitive multi-agent planning in **Modified Wallz**—a deterministic, two-player zero-sum adversarial grid game inspired by Quoridor, augmented with dynamic spatial constraints, rolling resource limits, and temporal abilities.

---

## 🎲 Game Rules & Mechanics

The game is played on an $N \times N$ discrete grid (default $N = 9$):
- **Player 0** starts at row `0` (goal: reach row `N - 1`).
- **Player 1** starts at row `N - 1` (goal: reach row `0`).
- Both players alternate turns taking exactly one action per turn (`MOVE`, `HOP`, `WALL`, or `BLOCK`).
- **Path-Preservation Rule**: An action is strictly illegal if it completely blocks all remaining shortest paths from either pawn to its target goal row.

### The Three Novel Modifications
1. **Modification A: Reusable Walls (`WALL`, `REPLACE`)**
   - Each player begins with a finite inventory of 10 walls. Placing a new wall consumes 1 inventory unit.
   - Only the player's single most recently placed wall ($r_i$) can be repositioned (`REPLACE`). Older walls remain permanently fixed.
   - Wall repositioning reuses the active wall on the board and does not consume additional inventory.
2. **Modification B: One-Time Hop (`HOP`)**
   - A single-use 2-step orthogonal jump bypassing intervening walls and pawns under valid landing conditions.
3. **Modification C: Temporary Cell Block (`BLOCK`)**
   - A single-use ability temporarily blocking a selected cell for $K$ turns (default $K = 3$ plies) with turn-by-turn decay. Must preserve at least one valid goal path for both players.

### Termination & Draws
- **Goal Reach**: Natural win when a pawn reaches its designated target goal row.
- **Clock Depletion**: A player immediately loses if their chess clock runs out (`time_remaining <= 0.0s`).
- **5-Fold Repetition**: Game ends in a draw if the identical full state occurs 5 times.
- **Step Limit**: Truncates automatically at $T_{\max} = 200$ turns.

---

## 🚀 Implementation Status Matrix

| Component | Module | Status | Verification |
| :--- | :--- | :--- | :--- |
| **Headless Game Engine** | `wallz_engine.py` / `src.modified_wallz` | **Implemented & Tested** | Reusable walls (latest wall repositioning), 5-fold repetition, path preservation |
| **Tier-1 Greedy Baseline** | `agents/greedy.py` / `greedy_agent.py` | **Implemented & Tested** | 5-tier lexicographical spatial decision-making |
| **Tier-2 Minimax Agent** | `agents/minimax.py` | **Implemented & Tested** | Alpha-Beta pruning, report leaf eval, move ordering, timeout fallback |
| **Match Runner CLI** | `run_match.py` / `scripts/run_match.py` | **Implemented & Tested** | Headless simulation, turn latency, node/cutoff telemetry |
| **Logged Match Runner & Replays** | `evaluation/match_runner.py` / `scripts/run_experiments.py` | **Implemented & Tested** | JSONL logs, outcome taxonomy, paired openings, replay verification, demo export |
| **State Graph & Retrograde Solver** | `evaluation/state_graph.py` / `evaluation/retrograde.py` | **Implemented & Tested** | Engine-driven enumeration with caps, exact ranks, all optimal actions, UNKNOWN on caps |
| **Unit Test Suite** | `tests/` | **Implemented & Tested** | 96/96 unit tests passing |
| **Tier-3 Exact Solver** | `agents/exact_solver.py` | **Implemented & Tested** | Retrograde queue propagation, cycle/draw resolution, tiny board solver |
| **MARL DQN Self-Play** | `agents/dqn/` | *Scaffolded (Unimplemented)* | Assigned to other team members |
| **Ablation Matrix** | `evaluation/ablation.py` / `evaluation/metrics.py` | **Implemented & Piloted** | 2³ factorial effects, paired bootstrap, agent-vs-exact agreement |

---

## 📂 Repository Structure

```text
FAI_project/
├── README.md                     # Project overview and run guides
├── pyproject.toml                # Project packaging configuration
├── wallz_engine.py               # Headless game engine (single source of truth)
├── greedy_agent.py               # Root wrapper for Tier-1 Greedy baseline
├── run_match.py                  # Automated headless match runner
│
├── agents/                       # Decision agents
│   ├── __init__.py               # Agent module exports
│   ├── greedy.py                 # Tier-1 Lexicographic Greedy Baseline
│   ├── minimax.py                # Tier-2 Minimax with Alpha-Beta Pruning
│   ├── exact_solver.py           # Tier-3 Exact Solver (in-progress)
│   └── dqn/                      # Deep Q-Network MARL agents (in-progress)
│
├── src/                          # Modular source package
│   └── modified_wallz/           # Engine exports and modular components
│
├── tests/                        # Comprehensive test suite (unittest)
│   ├── test_agents.py            # Minimax & Greedy agent tests
│   ├── test_movement.py          # Pawn movement & wall blocking tests
│   ├── test_walls.py             # Wall legality & path preservation tests
│   ├── test_hop.py               # Hop ability & consumption tests
│   ├── test_temporary_blocks.py  # Cell block & turn decay tests
│   ├── test_path_preservation.py # Critical path preservation tests
│   ├── test_terminal_states.py   # Goal wins & 5-fold repetition tests
│   ├── test_replacement.py       # Reusable walls & latest wall repositioning tests
│   └── test_initial_state.py     # Default and custom board setup tests
│
├── docs/                         # Technical documentation
│   ├── architecture.md           # Architecture, engine interfaces & data flow
│   ├── decisions.md              # Engineering decisions & design rationale
│   ├── game_rules.md             # Game rules & formal mechanics
│   ├── implementation_status.md  # Detailed implementation and test tracker
│   └── minimax.md                # Minimax mathematical formulation & semantics
│
├── report/                       # Research paper & project proposal
│   ├── project_idea.tex          # LaTeX research proposal (796 lines)
│   ├── project_idea.pdf          # Compiled project proposal PDF
│   └── references.bib            # Bibliography
│
├── scripts/                      # Utility and experiment scripts
│   └── run_match.py              # Script entrypoint for running matches
└── configs/                      # Experiment and ablation configuration files
```

---

## 🛠️ How to Run Tests

The test suite requires no external dependencies and uses Python's standard `unittest` framework:

```bash
python3 -m unittest discover tests
```

To run only the Minimax-specific tests:

```bash
python3 -m unittest tests/test_agents.py
```

---

## ⚔️ How to Run Matches

Run head-to-head matches using the automated match runner:

### 1. Minimax vs. Greedy (Default)
```bash
python3 run_match.py
```

### 2. Configure Minimax Depth and Per-Turn Time Limit
```bash
# Minimax (depth 2, 1.0s time limit) vs Greedy on default 9x9 board
python3 run_match.py --p0 minimax --p1 greedy --depth 2 --time-limit 1.0

# Minimax (depth 3, 2.0s time limit) vs Greedy on a 5x5 board
python3 run_match.py --p0 minimax --p1 greedy --board-size 5 --depth 3 --time-limit 2.0
```

### 3. Greedy vs. Greedy Baseline
```bash
python3 run_match.py --p0 greedy --p1 greedy
```

### Command-Line Arguments
- `--p0`: Player 0 agent (`minimax` or `greedy`, default: `minimax`).
- `--p1`: Player 1 agent (`minimax` or `greedy`, default: `greedy`).
- `--board-size`: Grid dimension $N$ (default: `9`).
- `--depth`: Minimax maximum lookahead depth in plies (default: `2`).
- `--time-limit`: Maximum allowed search duration per turn in seconds (default: `1.0`).
- `--max-steps`: Maximum total match turns before truncation (default: `200`).

---

## 📊 Benchmark Summary: Minimax vs. Greedy

| Board Size | Match-Up | Winner | Plies | Total Match Time | Minimax Mean Latency | Minimax Mean Cutoffs |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **5×5** | Minimax (P0) vs Greedy (P1) | **Player 0 (Minimax)** | 25 | 0.82s | 61.7 ms / turn | 22.0 cutoffs / turn |
| **9×9** | Minimax (P0) vs Greedy (P1) | **Player 0 (Minimax)** | 185 | 93.87s | 987.9 ms / turn | 44.5 cutoffs / turn |
| **9×9** | Greedy (P0) vs Greedy (P1) | *Truncated (Draw)* | 200 | 25.10s | — | — |

Minimax successfully navigates complex spatial trade-offs and dynamic wall recyclings, systematically defeating the Greedy baseline without getting trapped in repetitive draw cycles.

---

## 📏 Evaluation Tools (Person 3)

Logged matches, exact tiny-board solving, A/B/C ablations and replay export are documented in
[docs/person3_evaluation.md](docs/person3_evaluation.md), including the exact game model, result schemas and pilot results.

```bash
python scripts/run_experiments.py match --p0 minimax:depth=2 --p1 greedy -N 5 --abc 111 --out m.json
python scripts/run_experiments.py solve -N 3 -K 1 -M 0 --abc all
python scripts/run_experiments.py ablation --x minimax:depth=2 --y greedy -N 5 --seeds 10 --out-dir experiments/results/my_ablation
```

---

## 🔬 Exact Retrograde Solver (Ground-Truth Benchmark)

Run backward induction and game-theoretic analysis on miniature boards:

```bash
# 1. Solve 3x3 pure pawn race (45 states, 0.002s)
python3 scripts/solve_tiny_board.py --board-size 3 --max-walls 0 --no-hop --show-root-actions

# 2. Solve 3x3 with one-time hop (186 states, 0.025s)
python3 scripts/solve_tiny_board.py --board-size 3 --max-walls 0 --show-root-actions

# 3. Solve 5x5 pawn race with hop (1,910 states, 0.389s)
python3 scripts/solve_tiny_board.py --board-size 5 --max-walls 0 --show-root-actions

# 4. Solve 3x3 with 1 wall and latest-wall repositioning (18,226 states, 9.48s)
python3 scripts/solve_tiny_board.py --board-size 3 --max-walls 1 --show-root-actions
```

See [docs/exact_solver.md](file:///run/media/vedant/DISK/Studies/SEM 5/CS 329/FAI_Project/docs/exact_solver.md) for full algorithmic formulation, complexity analysis, and mathematical proofs.
