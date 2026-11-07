# Modified Wallz — Web Demo User & API Guide

## 1. Overview
The Modified Wallz Web Application provides a rich, interactive visualization and evaluation interface for human players, classical AI baselines (Greedy, Minimax), exact retrograde solvers, and deep reinforcement learning agents (Double-DQN).

The application is structured into two completely decoupled layers:
- **Backend (Python / Flask)**: Runs the canonical game engine (`WallzEngine`), maintains transactional session histories, introspects agent decision models, and executes verified replay checks.
- **Frontend (React 18 / TypeScript / Vite)**: A responsive SVG board and dashboard interface that contains **zero game logic**, strictly rendering data returned by the backend (Contract C-STATE and Contract C-EXPL).

---

## 2. Quick Start & Setup

### Prerequisites
- Python $\ge 3.10$ with virtual environment created (`.venv`)
- Node.js $\ge 20$ and npm $\ge 10$

### Production Mode (Recommended: Single Process)
Build the frontend production bundle and launch the Flask server:
```bash
# 1. Install frontend dependencies and build production distribution
npm --prefix web/frontend ci
npm --prefix web/frontend run build

# 2. Start the unified Flask server
.venv/bin/python -m web.backend.app --port 5000
```
Open your browser at **`http://localhost:5000`**. The Flask server automatically serves the compiled React single-page app and routes all `/api/*` endpoints.

### Development Mode (Two Processes with Hot Reloading)
```bash
# Terminal 1: Start Flask API backend on port 5000
.venv/bin/python -m web.backend.app --port 5000 --debug

# Terminal 2: Start Vite development server with proxy on port 5173
npm --prefix web/frontend run dev
```
Open **`http://localhost:5173`**. Vite automatically proxies API requests to `http://localhost:5000`.

---

## 3. Web Views & Feature Guide

The application features five primary views accessible via the top navigation bar:

### 1. Play (`/`)
Interactive human vs AI or AI vs AI match environment:
- **Game Configuration**: Customize board dimensions $N \in \{3, 4, 5, 7, 9\}$, wall inventory $M \in [0, 20]$, block duration $K \in [1, 9]$, and toggle mechanics A (Reusable Walls), B (One-Time Hop), and C (Temporary Blocks). Presets are available for trained DQN configurations.
- **Agent Picker**: Select Player 0 and Player 1 independently from available agents: Human, Random, Greedy (unseeded or seeded), Minimax (depths 1–3), Exact Solver, or DQN.
- **Interactive SVG Board**:
  - Pawns for Player 0 (Blue) and Player 1 (Red).
  - Placed walls colored by owner, with a gold glow highlighting the **latest wall** (repositionable under Rule A).
  - Blocked cells hatched with an active countdown badge showing remaining plies ($K \to 0$).
  - Target cell highlights for legal actions matching the selected mode (`MOVE`, `HOP`, `WALL_H`, `WALL_V`, `REPLACE`, `BLOCK`).
  - Clickable wall slots with hover ghosts.
  - Arrow key keyboard shortcuts for orthogonal `MOVE` actions.
- **Side Panels**:
  - **Player Cards**: Display remaining wall inventory, hop/block availability, and shortest-path distance to goal row.
  - **Action Bar**: Filter buttons dynamically enabled based on legal move categories.
  - **Timeline**: Scrollable ply history with formatted action names and decision timestamps.
  - **Undo**: Transactionally rolls back plies (replaying history through a fresh engine to prevent state divergence).
  - **Think Panel**: Real-time explanation panel showing how the selected agent evaluated candidate actions on its turn.
  - **Ask AI (Human Hints)**: When a human player is to move, click "Ask AI" with any agent spec to analyze the current position and receive recommended candidate moves.

### 2. Arena (`/arena`)
Head-to-head automated testing and benchmarking suite:
- **Match Controls**: Configure P0 and P1 AI agents, speed slider (0.25s to 3s per ply), Start, Pause, Step, and Reset.
- **Dual Think Panels**: Side-by-side inspection of both players' internal decision processes simultaneously.
- **Batch Tournaments**: Run synchronous batches ($n \in \{10, 20, 50\}$ games) with balanced colors and alternating opening seeds. Displays win/draw/loss counts, win rates with Wilson 95% confidence interval error bars, and links every match directly into the Replay viewer.
- **Early vs. Final DQN Button**: Automatically sets up an arena match between an early checkpoint and the final trained DQN model to observe policy maturation.

### 3. Training (`/training`)
Visual dashboard for reinforcement learning and model evaluation:
- **Decision Paradigm Cards**: Side-by-side comparison explaining how Greedy (1-ply heuristic), Minimax (tree search + leaf eval), Exact Solver (retrograde induction), and DQN (self-play learning) make choices.
- **Interactive Training Loop Diagram**: SVG visualization of the Double-DQN training architecture (transitions, replay buffer, target network updates, and opponent pool sampling). Ratios and hyperparameters are queried directly from the backend rather than hardcoded.
- **Interactive Learning Curves**: Multi-seed Recharts line graphs showing rolling win rate (100-episode window), episode length, exploration epsilon ($\varepsilon$), and TD-loss curves over environment steps.
- **Benchmark & Gate Tables**: Summary metrics from Gate X-16 (solver sign agreement), Gate X-5 (minimax feature AUC), and Greedy match campaigns.

### 4. Replays (`/replays`)
Inspection and verification tool for recorded matches:
- **Experiment Replay Browser**: Discovers and indexes all experiment runs stored under `experiments/results/`. Filter by configuration, agent type, or outcome.
- **Replay Scrubber**: Interactive timeline slider, play/pause, step forward/backward, and playback speed control.
- **Engine Verification Badge**: Every replay is verified dynamically by executing each action through a fresh `WallzEngine`. If any action is invalid, the player flags the replay as corrupted and pinpoints the exact failing ply.
- **JSON Replay Upload**: Drag-and-drop or upload custom `run_match` JSON output files for instant verification and playback.

### 5. Rules (`/rules`)
Interactive rulebook and verification showcase:
- **Live Rule Scenarios (9 Scenarios)**: Step through canonical scenarios demonstrating every core mechanic: basic moves, wall placement and budget exhaustion, latest-wall repositioning, one-time hops, $K$-ply block decay, path preservation rejections, no-legal-move draws, five-fold repetition, and goal wins.
- **Rule Invariant Badges**: Live assertions (✓/✗) confirm that invariants hold at every step.
- **Play From This State**: Launch an interactive game session starting immediately from any scenario frame.
- **Official Specification Viewer**: Embedded markdown view querying `/api/rules` directly from `docs/game_rules.md`, ensuring zero documentation drift.

---

## 4. Backend REST API Specification

### Game Session Endpoints (`web/backend/routes_games.py`)

| Method | Endpoint | Request Body | Response Body | Description |
|---|---|---|---|---|
| `GET` | `/api/config/defaults` | — | `{defaults, limits, presets}` | Default rules, board limits, and DQN presets |
| `GET` | `/api/agents` | Query params: `N,A,B,C,K,M` | `[AgentInfo]` | List of agents with availability status and reason strings |
| `POST` | `/api/games` | `{config, p0, p1}` | `SessionData` | Create a new game session |
| `GET` | `/api/games/<id>` | — | `SessionData` | Retrieve game session state and history |
| `POST` | `/api/games/<id>/action` | `{"action": [type, value]}` | `SessionData` | Submit a human move (returns 400 with engine error if illegal) |
| `POST` | `/api/games/<id>/step` | `{"explain": true}` | `{action, explanation, session}` | Trigger active AI agent turn and return decision explanation |
| `POST` | `/api/games/<id>/analyze`| `{"agent": "spec"}` | `ExplanationData` | Introspect position using specified agent without advancing state |
| `POST` | `/api/games/<id>/undo` | `{"n": 1, "to_human": true}` | `SessionData` | Undo moves (replays history through fresh engine) |
| `DELETE`| `/api/games/<id>` | — | `204 No Content` | Terminate and clean up session |

### Data & Tournament Endpoints (`web/backend/routes_data.py`)

| Method | Endpoint | Request Body | Response Body | Description |
|---|---|---|---|---|
| `POST` | `/api/arena/batch` | `{config, p0, p1, n, seed}` | `{games, by_spec, mean_length, categories, replays}` | Run synchronous AI tournament with Wilson CIs |
| `GET` | `/api/arena/batch/<bid>` | — | Arena batch summary | Retrieve stored tournament results |
| `GET` | `/api/arena/batch/<bid>/<idx>`| — | Replay data with frames | Retrieve individual match from batch |
| `GET` | `/api/replays` | Query: `offset, limit, q` | `{total, items: [ReplaySummary]}` | List experiment result files |
| `GET` | `/api/replays/item` | Query: `file, index` | `{result, verified, frames}` | Retrieve and verify replay ply-by-ply |
| `POST` | `/api/replays/upload` | Raw replay JSON | `{result, verified, frames}` | Verify and frame uploaded replay JSON |
| `GET` | `/api/scenarios` | — | `[ScenarioSummary]` | List 9 canonical rule scenarios |
| `GET` | `/api/scenarios/<name>` | — | `{scenario, frames: [...]}` | Execute scenario and return verification frames |
| `POST` | `/api/games/from-scenario/<name>`| `{"p0": "spec", "p1": "spec"}` | `SessionData` | Create playable game starting from scenario end |
| `GET` | `/api/training/runs` | — | `[RunSummary]` | List available DQN training runs and checkpoints |
| `GET` | `/api/training/runs/<id>` | — | `{meta, curves, evals, opponents}` | Downsampled training curves (win-rate, loss) |
| `GET` | `/api/training/summary` | — | Aggregated benchmark metrics | Summary of gates and experimental ladders |
| `GET` | `/api/training/config` | — | `DQNConfig` dict | Live hyperparameters from `agents.dqn.trainer` |
| `GET` | `/api/rules` | — | `{"rules": markdown_text}` | Raw markdown text of `docs/game_rules.md` |

---

## 5. Agent Explanation Semantics (Contract C-EXPL)

When `/step` or `/analyze` is called, the backend generates an explanation envelope tailored to the agent family:

| Family | Explanation Semantics | Overlay Data | Source of Truth |
|---|---|---|---|
| **Random** | Uniform distribution over all legal actions. | `value = 1 / len(legal)` for all legal moves. | `RandomAgent` |
| **Greedy** | Distance advantage heuristic $H = d_{\text{opp}} - d_{\text{self}}$. Immediate wins assigned $+\infty$. Rows sorted by score. | `value = H` (wins shown as gold star). | `GreedyAgent.score_actions()` |
| **Minimax** | Root move search. Each candidate move value is evaluated using a full-window Negamax search at depth $D-1$. Features $f_1 \dots f_6$ extracted for chosen move. | Per-move value from mover's perspective. | `MinimaxAgent` stats & full-window child search |
| **DQN** | Action-value distribution $Q(s, a)$ computed by the online neural network. Top-5 candidates and network metadata. | $Q(s, a)$ mapped across board heatmap. | `ActionCodec` + `DQNAgent.model(obs)` |
| **Exact** | Game-theoretic status ($\text{WIN}, \text{DRAW}, \text{LOSS}$), distance-to-terminal rank, and optimal moves. | Categorical values $\{+1, 0, -1\}$ with ply distance labels. | `evaluation.retrograde.Solution` |

---

## 6. Performance & Availability Limits

1. **Minimax Interactive Latency**:
   - On boards with $N \ge 7$, search depth is restricted to $D \le 2$ in interactive play to maintain response times $< 1.5$ seconds per move. Depths $D \ge 3$ are marked unavailable with an explanatory reason.
2. **Exact Solver Tractability**:
   - The retrograde solver is restricted to $N \le 4$ with small wall and block budgets ($M \le 1, K \le 2$). On larger boards, the solver is marked unavailable.
3. **DQN Compatibility Gate**:
   - Checkpoints are strictly validated against board size and rules (`meta.N == config.N` and `meta.rules == config.rules`). Incompatible models are disabled with a message indicating the configuration they were trained for.
4. **Arena Batch Caps**:
   - Batch tournaments are capped at $n \le 50$ games synchronously to prevent server worker timeouts.
