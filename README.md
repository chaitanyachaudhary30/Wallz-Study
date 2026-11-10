# Modified Wallz — Multi-Agent System & AI Decision Baselines

* [ ] 

This repository contains the complete game environment, decision baselines, MARL self-play framework, exact retrograde solver, and statistical evaluation pipeline for **"Multi-Agent Reinforcement Learning for Adversarial Spatial Decision-Making with Temporal Constraints"** (CS 329: Foundations of AI / Multi-Agent Systems, IIT Gandhinagar).
This repository contains the complete game environment, decision baselines, MARL self-play framework, exact retrograde solver, statistical evaluation pipeline, and full-stack interactive web application for **"Multi-Agent Reinforcement Learning for Adversarial Spatial Decision-Making with Temporal Constraints"** (CS 329: Foundations of AI / Multi-Agent Systems, IIT Gandhinagar).

The project studies competitive multi-agent planning in **Modified Wallz**—a deterministic, two-player zero-sum adversarial grid game inspired by Quoridor, augmented with dynamic spatial constraints, rolling resource limits, and temporal abilities.

---

## 🎲 Game Rules & Mechanics

The game is played on an $N \times N$ discrete grid (default $N = 9$):
The game is played on an $N \times N$ discrete grid (default $N = 9$; experimental benchmarks on $N \in \{3, 4, 5, 7, 9\}$):

- **Player 0** starts at row `0` (goal: reach row `N - 1`).
- **Player 1** starts at row `N - 1` (goal: reach row `0`).
- Both players alternate turns taking exactly one action per turn (`MOVE`, `HOP`, `WALL`, or `BLOCK`).
- **Path-Preservation Rule**: An action is strictly illegal if it completely blocks all remaining shortest paths from either pawn to its target goal row.
- **Player 0** starts at `(0, floor(N/2))` (goal: reach row `N - 1`).
- **Player 1** starts at `(N - 1, floor(N/2))` (goal: reach row `0`).
- Both players alternate turns taking exactly one legal action per turn (`MOVE`, `HOP`, `WALL`, `REPLACE`, or `BLOCK`).
- **Path-Preservation Rule**: Every action is strictly illegal if it completely disconnects either pawn from its target goal row.

### The Three Novel Modifications

1. **Modification A: Reusable Walls (`WALL`, `REPLACE`)**
   - Each player begins with a finite inventory of $M = 10$ walls. Placing a new wall consumes 1 inventory unit.
   - Each player begins with a finite inventory of $M = 10$ walls (configurable via `rules['M']`). Placing a new wall consumes 1 inventory unit.
   - Only the player's single most recently placed wall ($r_i$) can be repositioned (`REPLACE`). Older walls remain permanently fixed.
   - Wall repositioning reuses the active wall on the board and does not consume additional inventory.
   - Wall repositioning reuses the active wall on the board and does not consume additional inventory ($m_i' = m_i$). When $M = 0$, players can continue executing `REPLACE` actions.
   - Accepted syntax: short triple `('REPLACE', (r, c, orient))` or explicit pair `('REPLACE', (old_wall, new_wall))` where `old_wall == latest_wall[player]`.
2. **Modification B: One-Time Hop (`HOP`)**
   - A single-use hop over **a wall placed by the opponent** into the adjacent cell behind it (proposal `LegalHop`). It cannot cross open edges, pawns or the player's own walls, and the landing cell must be free with a path to goal.
3. **Modification C: Temporary Cell Block (`BLOCK`)**
   - A single-use ability temporarily blocking a selected cell for $K$ turns (default $K = 3$ plies) with turn-by-turn decay. Must preserve at least one valid goal path for both players.
   - A single-use ability temporarily blocking a selected cell for $K$ plies (default $K = 3$).
   - **Exact Timeline**: With $K = 3$, the block covers the opponent's next move, the blocker's next move, and the opponent's following move. At the end of the opponent's third move, the timer reaches 0 and expires, leaving the cell open for subsequent turns.

### Strict Validation Order & Atomicity

The canonical engine (`wallz_engine.py`) serves as the single source of truth and executes actions through a strict transactional pipeline:

$$
\text{Terminal check} \longrightarrow \text{Normalise} \longrightarrow \text{Validate} \longrightarrow \text{Clock} \longrightarrow \text{Mutate} \longrightarrow \text{Win/Timers/Turn} \longrightarrow \text{Truncation/Stalemate/Repetition}
$$

Any illegal action raises `ValueError` with descriptive reason text, leaving the board, clocks, inventories, and histories completely unmodified.

### Termination & Draws

- **Goal Reach**: Natural win when a pawn reaches its designated target goal row.
- **Clock Depletion**: A player immediately loses if their chess clock runs out (`time_remaining <= 0.0s`).
- **5-Fold Repetition**: Game ends in a draw if the identical full state occurs 5 times.
- **No Legal Actions (Stalemate)**: If the side to move has no legal action, the game terminates in an immediate draw (Rule R3).
- **Step Limit**: Truncates automatically at $T_{\max} = 200$ turns.
- **No Legal Actions (Stalemate)**: If the player to move has zero legal actions, the game terminates in an immediate draw (Rule R3).
- **Step Limit Truncation**: Matches truncate at $T_{\max} = 200$ plies.

---

## 🚀 Implementation Status Matrix

## 🌐 Interactive Web Application Demo

| Component                                                                                                                                                                      | Module                                                 | Status                     | Verification & Gates                                                                                     |
| :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------- | :------------------------- | :------------------------------------------------------------------------------------------------------- |
| **Headless Game Engine**                                                                                                                                                 | `wallz_engine.py` / `src.modified_wallz`           | **Production Ready** | 1.54 ms legal actions, 2.81 $\mu$s O(1) clone, canonical key, Gate X-1 & X-2 passed                    |
| **Tier-1 Greedy Baseline**                                                                                                                                               | `agents/greedy.py` / `greedy_agent.py`             | **Production Ready** | 4-tier lexicographical tie order `(H, cat, -d_self, tie_key)`, draw scoring, seeded tie-break          |
| **Tier-2 Minimax Agent**                                                                                                                                                 | `agents/minimax.py`                                  | **Production Ready** | Negamax, Alpha-Beta pruning, PV-first ordering, history heuristic, goal extension, Contract C7           |
| **Tier-3 Exact Solver**                                                                                                                                                  | `agents/exact_solver.py` / `evaluation.retrograde` | **Production Ready** | Backward induction on cyclic graphs, cycle/stalemate draw separation, $K \in \{1,2,3\}$ solved         |
| **MARL Deep Q-Network**                                                                                                                                                  | `agents/dqn/`                                        | **Production Ready** | Negamax Double-DQN, 14-plane spatial tensor, ActionCodec, self-play pool, released checkpoints           |
| **Evaluation Framework**                                                                                                                                                 | `evaluation/` / `scripts/run_experiments.py`       | **Production Ready** | Paired 2³ factorial ablations, Holm-Bonferroni correction, Wilson CIs, replay verification              |
| **Report Table Generator**                                                                                                                                               | `scripts/make_tables.py`                             | **Production Ready** | Generates 13 publication-ready Markdown and LaTeX tables across all experimental gates                   |
| **Test Suite**                                                                                                                                                           | `tests/`                                             | **100% Passing**     | **160/160 tests passing**, Hypothesis stateful testing, NetworkX BFS validation, conformance suite |
| The repository features a full-stack interactive web application connecting a pure-drawing React 18 frontend to a Flask API wrapping the canonical engine and decision models: |                                                        |                            |                                                                                                          |

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
│       ├── network.py            # Convolutional Q-Network (plain 4-conv stack + MLP head)
│       ├── encoding.py           # 14-plane spatial observation encoder (C4)
│       ├── (codec)               # Action codec lives in src/modified_wallz/actions.py (C3)
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
    ├── setup.md                  # Comprehensive setup, installation & reproduction guide
    ├── decisions.md              # Engineering decisions D1-D12 and rule formalizations
    ├── game_rules.md             # Formal game mechanics, abilities, and termination rules
    ├── minimax.md                # Minimax / Negamax formulation & search extensions
    ├── dqn.md                    # Deep Q-Network architecture, observation tensor, self-play
    ├── exact_solver.md           # Retrograde propagation theory for cyclic games
    ├── implementation_status.md  # Detailed implementation and test tracker across all 10 phases
    └── architecture.md           # High-level architecture, contracts, and data flow
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Modified Wallz Interactive Demo                       │
│    [Play]        [Arena]        [Training]        [Replays]        [Rules]  │
├───────────────────────────────┬─────────────────────────────────────────────┤
│   Player 0: Human             │   Player 1: Minimax (Depth 2)               │
│   Inventory: 9/10  Hop: ✓  K: ✓│   Inventory: 10/10  Hop: ✓  K: ✓            │
├───────────────────────────────┴─────────────────────────────────────────────┤
│   [ SVG Board Canvas with Legal Highlights, Overlays, and Latest Wall Glow ]│
├─────────────────────────────────────────────────────────────────────────────┤
│   • Think Panel: Heuristic score / Alpha-Beta PV / DQN Q-Values / Solver     │
│   • Actions: [MOVE (4)]  [HOP (0)]  [WALL H (12)]  [WALL V (12)]  [BLOCK (7)]│
│   • Controls: [Undo Ply]  [Show Shortest Paths: ON]  [Ask AI Hint]          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Quick Run (Production Mode)

```bash
# 1. Install frontend dependencies and build production assets
npm --prefix web/frontend ci
npm --prefix web/frontend run build

## 📦 Installation & Quick Setup
# 2. Start the unified web application
.venv/bin/python -m web.backend.app --port 5000
```

Open **`http://localhost:5000`** in your browser.

Follow these steps to set up the environment from scratch (detailed instructions available in [`docs/setup.md`](docs/setup.md)):

### Development Mode (Hot Reloading)



```bash
# 1. Clone repository
git clone https://github.com/vedantb7/FAI_project.git
cd FAI_project
# Terminal 1: Flask backend API
.venv/bin/python -m web.backend.app --port 5000 --debug

# 2. Create and activate virtual environment (Python 3.10+)
python3 -m venv .venv
source .venv/bin/activate  # On Windows (cmd): .venv\Scripts\activate.bat

# 3. Upgrade packaging tools and install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
# Terminal 2: Vite React frontend
npm --prefix web/frontend run dev
```



Open **`http://localhost:5173`** (proxies `/api` to port 5000).

### Web Views Summary

- **Play (`/`)**: Play human vs AI or AI vs AI with live move highlights, shortest-path overlays, undo, and "Ask AI" move hints.
- **Arena (`/arena`)**: Automated AI vs AI matches with speed slider, dual think panels, and batch tournaments with Wilson 95% confidence intervals.
- **Training (`/training`)**: Learning curve dashboards (win-rate, loss, epsilon), interactive Double-DQN loop diagram, and benchmark summaries.
- **Replays (`/replays`)**: Browse and scrub through match recordings, with every frame dynamically verified through the engine.
- **Rules (`/rules`)**: Interactive runner for all 9 canonical rule scenarios with live invariant checks and embedded official rules spec.

---

## 🛠️ How to Run Tests

## 🚀 Implementation Status Matrix

Run the full test suite (160 tests):

| Component                        | Module                                                 |           Status           | Verification & Gates                                                                   |
| :------------------------------- | :----------------------------------------------------- | :------------------------: | :------------------------------------------------------------------------------------- |
| **Headless Game Engine**   | `wallz_engine.py` / `src.modified_wallz`           | **Production Ready** | 1.54 ms legal actions, 2.81 $\mu$s clone, canonical key, single-source AST guard     |
| **Tier-1 Greedy Baseline** | `agents/greedy.py` / `greedy_agent.py`             | **Production Ready** | 4-tier lexicographical tie order `(H, cat, -d_self, tie_key)`, `score_actions` API |
| **Tier-2 Minimax Agent**   | `agents/minimax.py`                                  | **Production Ready** | Negamax, Alpha-Beta pruning, PV-first ordering, goal extension, Contract C7            |
| **Tier-3 Exact Solver**    | `agents/exact_solver.py` / `evaluation.retrograde` | **Production Ready** | Backward induction on cyclic graphs, cycle/stalemate draw separation                   |
| **MARL Deep Q-Network**    | `agents/dqn/`                                        | **Production Ready** | Negamax Double-DQN, 14-plane spatial tensor, ActionCodec, released checkpoints         |
| **Interactive Web Demo**   | `web/backend/` + `web/frontend/`                   | **Production Ready** | React 18 + Flask, zero frontend rules, Contract C-STATE / C-EXPL, 5 views              |
| **Evaluation Framework**   | `evaluation/` / `scripts/run_experiments.py`       | **Production Ready** | Paired $2^3$ factorial ablations, Holm-Bonferroni correction, Wilson CIs             |
| **Greedy Match Campaign**  | `scripts/run_greedy_matches.py`                      | **Production Ready** | 720 matches verified, summary tables generated                                         |
| **Test Suite**             | `tests/` + `web/frontend/`                         |   **100% Passing**   | **301/301 Pytest passed**, 6/6 Vitest passed, AST single-source guard            |

```bash
python3 -m unittest discover -s tests
```

---

Run specific test components:

## 🛠️ How to Run Tests

### Run Full Python Test Suite (301 Tests)

```bash
# Gate X-1: Hypothesis stateful property tests & legality fuzzing
python3 -m unittest tests/test_invariants_stateful.py

# Gate X-2: NetworkX independent BFS equivalence
python3 -m unittest tests/test_bfs_networkx.py

# Agent conformance suite (Random, Greedy, Minimax, Exact, DQN)
python3 -m unittest tests/test_agent_conformance.py

# Golden solver-root regression tests
python3 -m unittest tests/test_golden_regression.py
.venv/bin/python -m pytest -q
```

*Expected Output*: `325 passed in ~62s`. The run refuses to start without torch (DQN tests would all skip); set `WALLZ_SKIP_DQN=1` to skip them on purpose.

---

## ⚡ Performance Benchmarks

Verify engine throughput and latency targets against engineering thresholds:

### Run Frontend Vitest Suite

```bash
python3 scripts/benchmark_engine_performance.py
python3 scripts/benchmark_engine_performance.py --seed 42
npm --prefix web/frontend test -- --run
```

*Expected Output*: `6 passed (2 test files)`.

**Results:**

- $9 \times 9$ start legal-action generation: **1.54 ms** (Target $\le 3.0$ ms) — **PASS**
- $9 \times 9$ midgame legal-action generation: **2.03 ms** (Target $\le 4.0$ ms) — **PASS**
- $5 \times 5$ start legal-action generation: **0.23 ms** (Target $\le 1.0$ ms) — **PASS**
- Engine clone latency: **2.81 $\mu$s** (Target $\le 50.0\ \mu$s) — **PASS**

---

## ⚔️ How to Run Matches

Run head-to-head matches using `run_match.py`:

### Run AST Single-Source Guard

```bash
# 1. Minimax (depth 2) vs Greedy on default 9x9 board
python3 run_match.py --p0 minimax --p1 greedy --depth 2
# 1. Minimax (depth 2) vs Greedy on default 9x9 board with seed 42
python3 run_match.py --p0 minimax --p1 greedy --depth 2 --seed 42

# 2. DQN vs Minimax on 3x3 board
python3 run_match.py --p0 dqn:path=checkpoints/release/dqn_3x3/model.pt --p1 minimax:depth=2 --board-size 3 -K 1 -M 1
# 2. DQN vs Minimax on 3x3 board with seed 42 (-K/-M must match the checkpoint's meta.json)
python3 run_match.py --p0 dqn:path=checkpoints/release/dqn_3x3/model.pt --p1 minimax:depth=2 --board-size 3 -K 1 -M 1 --seed 42

# 3. Greedy vs Greedy on 5x5 board
python3 run_match.py --p0 greedy --p1 greedy --board-size 5
# 3. Greedy vs Greedy on 5x5 board with seed 42
python3 run_match.py --p0 greedy --p1 greedy --board-size 5 --seed 42
.venv/bin/python -m pytest tests/test_single_source.py
```

*Ensures that zero game rules, shortest-path algorithms, or move legality checks exist in the frontend or outside the canonical engine.*

---

## 🔬 Running Experiments & Generating Tables

## 📊 Greedy Match Campaign & Reproducibility

Execute experiments using `scripts/run_experiments.py`:
Run the reproducible greedy match campaign across $N \in \{3, 5, 9\}$ with Random, Minimax, and Self-play:

```bash
# 1. Optimal Action Agreement Benchmark on 3x3 Exact Solutions (Gate X-6)
python3 scripts/run_experiments.py --config configs/experiments/agreement_3x3.yaml
# Smoke test (~1 min)
python scripts/run_greedy_matches.py --quick

# 2. Paired 2^3 Factorial Mechanic Ablation on 5x5 Board (Gate X-10)
python3 scripts/run_experiments.py --config configs/experiments/ablation_5x5_k3.yaml

# 3. Exact Retrograde Solve of Miniature Boards
python3 scripts/run_experiments.py solve -N 3 -K 1 -M 0 --abc all --out experiments/results/solves/solve_N3_K1_M0.json
python3 scripts/run_experiments.py solve -N 3 -K 1 -M 0 --abc all --out experiments/results/solves/solve_N3_K1_M0.json --seed 42

# 4. Feature Study & Logistic Regression AUC Evaluation (Gate X-5)
python3 scripts/feature_study.py --solve-file experiments/results/solves/solve_N3_K1_M0.json
python3 scripts/feature_study.py --board-size 3 --seed 42
# Full campaign (720 matches with verified replays and summary tables)
python scripts/run_greedy_matches.py --seeds 42 43 44 45 46
```

Generate all 13 publication-ready Markdown and LaTeX tables:
Generate all 14 publication-ready Markdown and LaTeX tables:

Generate all publication-ready LaTeX and Markdown tables:

```bash
python3 scripts/make_tables.py
python3 scripts/make_tables.py --seed 42
python scripts/make_tables.py --seed 42
```

Tables are output to `experiments/tables/`.

Tables are output to `experiments/tables/`:

- `table_x6_agreement_*.md` / `.tex`: Optimal Action Agreement rates (Gate X-6).
- `table_x9_benchmark_*.md` / `.tex`: Match win rates with Wilson 95% CIs (Gate X-9).
- `table_x10a_ablation_cells.md` / `.tex`: 2³ factorial mechanics ablation cells.
- `table_x10b_factorial_effects.md` / `.tex`: Main and interaction effects with Holm-Bonferroni corrected significance.
- `table_x17_draw_breakdown.md` / `.tex`: Rule R3 stalemate vs cycle repetition draw taxonomy.
- `table_x4_solve_*.md` / `.tex`: Exact state space and game-theoretic value distributions.

---

## 🧠 Training DQN Agents

## 📂 Repository Structure

Train self-play Double-DQN agents across board dimensions:

```bash
# Train on 3x3 board
python3 -m scripts.train_dqn --config configs/training.yaml --board-size 3 --seed 42 --output-dir checkpoints/final/3x3_s42

# Train on 4x4 board
python3 -m scripts.train_dqn --config configs/training.yaml --board-size 4 --seed 42 --output-dir checkpoints/final/4x4_s42
```text
FAI_project/
├── pyproject.toml                # Project packaging configuration
├── wallz_engine.py               # Canonical headless game engine (single source of truth)
├── greedy_agent.py               # Root wrapper for Tier-1 Greedy baseline
├── run_match.py                  # CLI match runner
│
├── agents/                       # Decision agents
│   ├── greedy.py                 # Tier-1 Lexicographic Greedy Baseline (score_actions)
│   ├── minimax.py                # Tier-2 Minimax with Alpha-Beta Pruning & Negamax
│   ├── exact_solver.py           # Tier-3 Exact Solver Adapter over evaluation.retrograde
│   ├── evaluation_fn.py          # Contract C7 modular evaluation functions
│   └── dqn/                      # MARL Deep Q-Network package
│
├── web/                          # Full-Stack Interactive Demo
│   ├── backend/                  # Flask REST API
│   │   ├── app.py                # Unified Flask server and asset router
│   │   ├── serialize.py          # C-STATE / C-ACTION engine serializers
│   │   ├── sessions.py           # In-memory transactional session store
│   │   ├── registry.py           # Dynamic agent discovery and model cache
│   │   ├── explain.py            # Decision introspection (C-EXPL)
│   │   ├── arena.py              # Synchronous tournament runner
│   │   ├── replays.py            # Replay indexer and engine verifier
│   │   ├── scenarios.py          # 9 canonical rule scenarios runner
│   │   └── training_data.py      # Training curve parser & downsampler
│   └── frontend/                 # React 18 TypeScript Presentation App
│       ├── src/components/       # Board (SVG), PlayerCard, Timeline, ActionBar, ThinkPanel
│       ├── src/pages/            # Play, Arena, Training, Replays, Rules
│       └── src/lib/geometry.ts   # Screen-space coordinate math (zero rule logic)
│
├── evaluation/                   # Evaluation framework (retrograde, metrics, ablation)
├── experiments/                  # Results, tables, and match replays
├── scripts/                      # CLI entrypoints (matches, tables, training, benchmarks)
├── tests/                        # Comprehensive test suite (301 tests)
└── docs/                         # Technical documentation & user guides
    ├── game_rules.md             # Authoritative game rules and mechanics specification
    ├── architecture.md           # System architecture, contracts, and single-source design
    ├── web_demo.md               # Complete user guide and REST API reference for the web demo
    ├── setup.md                  # Installation, configuration, and reproducibility guide
    └── implementation_status.md  # Engineering tracker across all project phases
```
