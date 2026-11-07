# Modified Wallz — Multi-Agent System & AI Decision Baselines

[![Tests](https://img.shields.io/badge/tests-160%20passed-brightgreen.svg)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)]()
[![Course](https://img.shields.io/badge/CS329-Foundations%20of%20AI-orange.svg)]()

This repository contains the complete game environment, decision baselines, MARL self-play framework, exact retrograde solver, and statistical evaluation pipeline for **"Multi-Agent Reinforcement Learning for Adversarial Spatial Decision-Making with Temporal Constraints"** (CS 329: Foundations of AI / Multi-Agent Systems, IIT Gandhinagar).

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
   - Each player begins with a finite inventory of $M = 10$ walls. Placing a new wall consumes 1 inventory unit.
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
- **No Legal Actions (Stalemate)**: If the side to move has no legal action, the game terminates in an immediate draw (Rule R3).
- **Step Limit**: Truncates automatically at $T_{\max} = 200$ turns.

---

## 🚀 Implementation Status Matrix

| Component | Module | Status | Verification & Gates |
| :--- | :--- | :--- | :--- |
| **Headless Game Engine** | `wallz_engine.py` / `src.modified_wallz` | **Production Ready** | 1.54 ms legal actions, 2.81 $\mu$s O(1) clone, canonical key, Gate X-1 & X-2 passed |
| **Tier-1 Greedy Baseline** | `agents/greedy.py` / `greedy_agent.py` | **Production Ready** | 4-tier lexicographical tie order `(H, cat, -d_self, tie_key)`, draw scoring, seeded tie-break |
| **Tier-2 Minimax Agent** | `agents/minimax.py` | **Production Ready** | Negamax, Alpha-Beta pruning, PV-first ordering, history heuristic, goal extension, Contract C7 |
| **Tier-3 Exact Solver** | `agents/exact_solver.py` / `evaluation.retrograde` | **Production Ready** | Backward induction on cyclic graphs, cycle/stalemate draw separation, $K \in \{1,2,3\}$ solved |
| **MARL Deep Q-Network** | `agents/dqn/` | **Production Ready** | Negamax Double-DQN, 14-plane spatial tensor, ActionCodec, self-play pool, released checkpoints |
| **Evaluation Framework** | `evaluation/` / `scripts/run_experiments.py` | **Production Ready** | Paired 2³ factorial ablations, Holm-Bonferroni correction, Wilson CIs, replay verification |
| **Report Table Generator** | `scripts/make_tables.py` | **Production Ready** | Generates 13 publication-ready Markdown and LaTeX tables across all experimental gates |
| **Test Suite** | `tests/` | **100% Passing** | **160/160 tests passing**, Hypothesis stateful testing, NetworkX BFS validation, conformance suite |

---

## 📂 Repository Structure

```text
FAI_project/
├── pyproject.toml                # Project packaging configuration (setuptools with [dqn,dev] extras)
├── wallz_engine.py               # Canonical headless game engine (single source of truth)
├── greedy_agent.py               # Root wrapper for Tier-1 Greedy baseline
├── run_match.py                  # CLI match runner delegating to evaluation.match_runner
│
├── agents/                       # Decision agents
│   ├── greedy.py                 # Tier-1 Lexicographic Greedy Baseline
│   ├── minimax.py                # Tier-2 Minimax with Alpha-Beta Pruning & Negamax
│   ├── exact_solver.py           # Tier-3 Exact Solver Adapter over evaluation.retrograde
│   ├── evaluation_fn.py          # Contract C7 modular evaluation functions (HEvaluator, LinearEvaluator)
│   └── dqn/                      # MARL Deep Q-Network package
│       ├── agent.py              # DQNAgent implementation & checkpoint loader
│       ├── network.py            # Residual Convolutional Q-Network
│       ├── encoding.py           # 14-plane spatial observation encoder (C4)
│       ├── codec.py              # Discrete action codec and mirror permutations (C3)
│       ├── replay_buffer.py      # Uniform FIFO replay buffer with mirror augmentation
│       ├── trainer.py            # Negamax Double-DQN learner
│       └── self_play.py          # Self-play curriculum & opponent pool
│
├── checkpoints/                  # Trained neural network weights
│   └── release/                  # Production models
│       ├── dqn_3x3/model.pt      # 3x3 release model (Gate X-16 validated: >90% solver agreement)
│       └── dqn_4x4/model.pt      # 4x4 release model
│
├── configs/                      # Experiment and training configurations
│   ├── default.yaml              # Default 9x9 match configuration
│   ├── ablation.yaml             # Factorial ablation configuration
│   ├── training.yaml             # Multi-size DQN training hyperparameter specifications
│   └── experiments/              # Experiment presets (ablation_5x5, agreement_3x3, etc.)
│
├── evaluation/                   # Evaluation and research benchmarking framework
│   ├── state_graph.py            # Canonical state graph generation with resource caps
│   ├── retrograde.py             # Exact backward induction solver for cyclic games
│   ├── match_runner.py           # Canonical match runner, replay verification, agent factory
│   ├── metrics.py                # Wilson CIs, empirical bootstrap, Holm-Bonferroni correction
│   ├── ablation.py               # 2^3 factorial mechanics analysis & agreement studies
│   ├── features.py               # Canonical spatial/temporal feature extraction (f1-f6)
│   └── replay_export.py          # Web demo replay exporter
│
├── scripts/                      # Utility and experiment CLI entrypoints
│   ├── run_experiments.py        # Centralized experiment CLI (matches, ablations, solves, agreement)
│   ├── make_tables.py            # Automated LaTeX and Markdown table generator
│   ├── train_dqn.py              # Self-play DQN training pipeline
│   ├── validate_dqn.py           # Gate X-16 Q-value agreement validator
│   ├── solve_tiny_board.py       # Miniature board retrograde solver CLI
│   ├── feature_study.py          # Gate X-5 feature AUC study
│   └── benchmark_engine_performance.py # Engine throughput & latency benchmark suite
│
├── tests/                        # Comprehensive unit and component test suite
│   ├── test_invariants_stateful.py # Gate X-1: Hypothesis RuleBasedStateMachine invariant testing
│   ├── test_bfs_networkx.py      # Gate X-2: NetworkX independent BFS validation (2,000 layouts)
│   ├── test_agent_conformance.py # Conformance suite for Random, Greedy, Minimax, Exact, DQN
│   ├── test_golden_regression.py # Ground-truth solver root & replay regression tests
│   ├── test_evaluation_fn.py     # EvaluationFunction protocol & feature extractor tests
│   ├── test_metrics_holm.py      # Holm-Bonferroni & stalemate breakdown tests
│   └── ...                       # Engine mechanics, path preservation, abilities, replacements
│
└── docs/                         # Technical documentation & design decisions
    ├── decisions.md              # Engineering decisions D1-D12 and rule formalizations
    ├── game_rules.md             # Formal game mechanics, abilities, and termination rules
    ├── minimax.md                # Minimax / Negamax formulation & search extensions
    ├── dqn.md                    # Deep Q-Network architecture, observation tensor, self-play
    ├── exact_solver.md           # Retrograde propagation theory for cyclic games
    └── architecture.md           # High-level architecture, contracts, and data flow
```

---

## 🛠️ How to Run Tests

Run the full test suite (160 tests):

```bash
python3 -m unittest discover -s tests
```

Run specific test components:
```bash
# Gate X-1: Hypothesis stateful property tests & legality fuzzing
python3 -m unittest tests/test_invariants_stateful.py

# Gate X-2: NetworkX independent BFS equivalence
python3 -m unittest tests/test_bfs_networkx.py

# Agent conformance suite (Random, Greedy, Minimax, Exact, DQN)
python3 -m unittest tests/test_agent_conformance.py

# Golden solver-root regression tests
python3 -m unittest tests/test_golden_regression.py
```

---

## ⚡ Performance Benchmarks

Verify engine throughput and latency targets against engineering thresholds:

```bash
python3 scripts/benchmark_engine_performance.py
```

**Results:**
- $9 \times 9$ start legal-action generation: **1.54 ms** (Target $\le 3.0$ ms) — **PASS**
- $9 \times 9$ midgame legal-action generation: **2.03 ms** (Target $\le 4.0$ ms) — **PASS**
- $5 \times 5$ start legal-action generation: **0.23 ms** (Target $\le 1.0$ ms) — **PASS**
- Engine clone latency: **2.81 $\mu$s** (Target $\le 50.0\ \mu$s) — **PASS**

---

## ⚔️ How to Run Matches

Run head-to-head matches using `run_match.py`:

```bash
# 1. Minimax (depth 2) vs Greedy on default 9x9 board
python3 run_match.py --p0 minimax --p1 greedy --depth 2

# 2. DQN vs Minimax on 3x3 board
python3 run_match.py --p0 dqn:path=checkpoints/release/dqn_3x3/model.pt --p1 minimax:depth=2 --board-size 3

# 3. Greedy vs Greedy on 5x5 board
python3 run_match.py --p0 greedy --p1 greedy --board-size 5
```

---

## 🔬 Running Experiments & Generating Tables

Execute experiments using `scripts/run_experiments.py`:

```bash
# 1. Optimal Action Agreement Benchmark on 3x3 Exact Solutions (Gate X-6)
python3 scripts/run_experiments.py --config configs/experiments/agreement_3x3.yaml

# 2. Paired 2^3 Factorial Mechanic Ablation on 5x5 Board (Gate X-10)
python3 scripts/run_experiments.py --config configs/experiments/ablation_5x5_k3.yaml

# 3. Exact Retrograde Solve of Miniature Boards
python3 scripts/run_experiments.py solve -N 3 -K 1 -M 0 --abc all --out experiments/results/solves/solve_N3_K1_M0.json

# 4. Feature Study & Logistic Regression AUC Evaluation (Gate X-5)
python3 scripts/feature_study.py --solve-file experiments/results/solves/solve_N3_K1_M0.json
```

Generate all 13 publication-ready Markdown and LaTeX tables:

```bash
python3 scripts/make_tables.py
```

Tables are output to `experiments/tables/`:
- `table_x6_agreement_*.md` / `.tex`: Optimal Action Agreement rates (Gate X-6).
- `table_x9_benchmark_*.md` / `.tex`: Match win rates with Wilson 95% CIs (Gate X-9).
- `table_x10a_ablation_cells.md` / `.tex`: 2³ factorial mechanics ablation cells.
- `table_x10b_factorial_effects.md` / `.tex`: Main and interaction effects with Holm-Bonferroni corrected significance.
- `table_x17_draw_breakdown.md` / `.tex`: Rule R3 stalemate vs cycle repetition draw taxonomy.
- `table_x4_solve_*.md` / `.tex`: Exact state space and game-theoretic value distributions.

---

## 🧠 Training DQN Agents

Train self-play Double-DQN agents across board dimensions:

```bash
# Train on 3x3 board
python3 scripts/train_dqn.py --config configs/training.yaml --board 3x3 --seed 42

# Train on 4x4 board
python3 scripts/train_dqn.py --config configs/training.yaml --board 4x4 --seed 42
```
