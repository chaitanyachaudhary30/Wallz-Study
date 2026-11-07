# Modified Wallz — Multi-Agent System & AI Decision Baselines

[![Tests](https://img.shields.io/badge/tests-301%20passed-brightgreen.svg)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)]()
[![React](https://img.shields.io/badge/react-18.3-blue.svg)]()
[![Course](https://img.shields.io/badge/CS329-Foundations%20of%20AI-orange.svg)]()

This repository contains the complete game environment, decision baselines, MARL self-play framework, exact retrograde solver, statistical evaluation pipeline, and full-stack interactive web application for **"Multi-Agent Reinforcement Learning for Adversarial Spatial Decision-Making with Temporal Constraints"** (CS 329: Foundations of AI / Multi-Agent Systems, IIT Gandhinagar).

The project studies competitive multi-agent planning in **Modified Wallz**—a deterministic, two-player zero-sum adversarial grid game inspired by Quoridor, augmented with dynamic spatial constraints, rolling resource limits, and temporal abilities.

---

## 🎲 Game Rules & Mechanics

The game is played on an $N \times N$ discrete grid (default $N = 9$; experimental benchmarks on $N \in \{3, 4, 5, 7, 9\}$):

- **Player 0** starts at `(0, floor(N/2))` (goal: reach row `N - 1`).
- **Player 1** starts at `(N - 1, floor(N/2))` (goal: reach row `0`).
- Both players alternate turns taking exactly one legal action per turn (`MOVE`, `HOP`, `WALL`, `REPLACE`, or `BLOCK`).
- **Path-Preservation Rule**: Every action is strictly illegal if it completely disconnects either pawn from its target goal row.

### The Three Novel Modifications

1. **Modification A: Reusable Walls (`WALL`, `REPLACE`)**
   - Each player begins with a finite inventory of $M = 10$ walls (configurable via `rules['M']`). Placing a new wall consumes 1 inventory unit.
   - Only the player's single most recently placed wall ($r_i$) can be repositioned (`REPLACE`). Older walls remain permanently fixed.
   - Wall repositioning reuses the active wall on the board and does not consume additional inventory ($m_i' = m_i$). When $M = 0$, players can continue executing `REPLACE` actions.
   - Accepted syntax: short triple `('REPLACE', (r, c, orient))` or explicit pair `('REPLACE', (old_wall, new_wall))` where `old_wall == latest_wall[player]`.
2. **Modification B: One-Time Hop (`HOP`)**
   - A single-use 2-step orthogonal jump bypassing intervening walls and pawns under valid landing conditions.
3. **Modification C: Temporary Cell Block (`BLOCK`)**
   - A single-use ability temporarily blocking a selected cell for $K$ plies (default $K = 3$).
   - **Exact Timeline**: With $K = 3$, the block covers the opponent's next move, the blocker's next move, and the opponent's following move. At the end of the opponent's third move, the timer reaches 0 and expires, leaving the cell open for subsequent turns.

### Strict Validation Order & Atomicity

The canonical engine (`wallz_engine.py`) serves as the single source of truth and executes actions through a strict transactional pipeline:
$$\text{Terminal check} \longrightarrow \text{Normalise} \longrightarrow \text{Validate} \longrightarrow \text{Clock} \longrightarrow \text{Mutate} \longrightarrow \text{Win/Timers/Turn} \longrightarrow \text{Truncation/Stalemate/Repetition}$$
Any illegal action raises `ValueError` with descriptive reason text, leaving the board, clocks, inventories, and histories completely unmodified.

### Termination & Draws

- **Goal Reach**: Natural win when a pawn reaches its designated target goal row.
- **Clock Depletion**: A player immediately loses if their chess clock runs out (`time_remaining <= 0.0s`).
- **5-Fold Repetition**: Game ends in a draw if the identical full state occurs 5 times.
- **No Legal Actions (Stalemate)**: If the player to move has zero legal actions, the game terminates in an immediate draw (Rule R3).
- **Step Limit Truncation**: Matches truncate at $T_{\max} = 200$ plies.

---

## 🌐 Interactive Web Application Demo

The repository features a full-stack interactive web application connecting a pure-drawing React 18 frontend to a Flask API wrapping the canonical engine and decision models:

```text
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

### Quick Run (Production Mode)
```bash
# 1. Install frontend dependencies and build production assets
npm --prefix web/frontend ci
npm --prefix web/frontend run build

# 2. Start the unified web application
.venv/bin/python -m web.backend.app --port 5000
```
Open **`http://localhost:5000`** in your browser.

### Development Mode (Hot Reloading)
```bash
# Terminal 1: Flask backend API
.venv/bin/python -m web.backend.app --port 5000 --debug

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

## 🚀 Implementation Status Matrix

| Component | Module | Status | Verification & Gates |
| :--- | :--- | :---: | :--- |
| **Headless Game Engine** | `wallz_engine.py` / `src.modified_wallz` | **Production Ready** | 1.54 ms legal actions, 2.81 $\mu$s clone, canonical key, single-source AST guard |
| **Tier-1 Greedy Baseline** | `agents/greedy.py` / `greedy_agent.py` | **Production Ready** | 4-tier lexicographical tie order `(H, cat, -d_self, tie_key)`, `score_actions` API |
| **Tier-2 Minimax Agent** | `agents/minimax.py` | **Production Ready** | Negamax, Alpha-Beta pruning, PV-first ordering, goal extension, Contract C7 |
| **Tier-3 Exact Solver** | `agents/exact_solver.py` / `evaluation.retrograde` | **Production Ready** | Backward induction on cyclic graphs, cycle/stalemate draw separation |
| **MARL Deep Q-Network** | `agents/dqn/` | **Production Ready** | Negamax Double-DQN, 14-plane spatial tensor, ActionCodec, released checkpoints |
| **Interactive Web Demo** | `web/backend/` + `web/frontend/` | **Production Ready** | React 18 + Flask, zero frontend rules, Contract C-STATE / C-EXPL, 5 views |
| **Evaluation Framework** | `evaluation/` / `scripts/run_experiments.py` | **Production Ready** | Paired $2^3$ factorial ablations, Holm-Bonferroni correction, Wilson CIs |
| **Greedy Match Campaign** | `scripts/run_greedy_matches.py` | **Production Ready** | 720 matches verified, summary tables generated |
| **Test Suite** | `tests/` + `web/frontend/` | **100% Passing** | **301/301 Pytest passed**, 6/6 Vitest passed, AST single-source guard |

---

## 🛠️ How to Run Tests

### Run Full Python Test Suite (301 Tests)
```bash
.venv/bin/python -m pytest -q
```
*Expected Output*: `301 passed in ~72s`.

### Run Frontend Vitest Suite
```bash
npm --prefix web/frontend test -- --run
```
*Expected Output*: `6 passed (2 test files)`.

### Run AST Single-Source Guard
```bash
.venv/bin/python -m pytest tests/test_single_source.py
```
*Ensures that zero game rules, shortest-path algorithms, or move legality checks exist in the frontend or outside the canonical engine.*

---

## 📊 Greedy Match Campaign & Reproducibility

Run the reproducible greedy match campaign across $N \in \{3, 5, 9\}$ with Random, Minimax, and Self-play:

```bash
# Smoke test (~1 min)
python scripts/run_greedy_matches.py --quick

# Full campaign (720 matches with verified replays and summary tables)
python scripts/run_greedy_matches.py --seeds 42 43 44 45 46
```

Generate all publication-ready LaTeX and Markdown tables:
```bash
python scripts/make_tables.py --seed 42
```
Tables are output to `experiments/tables/`.

---

## 📂 Repository Structure

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
