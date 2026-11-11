# Modified Wallz — Web Demo User & API Guide

## 1. Overview
The Modified Wallz Web Application provides a rich, interactive visualization and evaluation interface for human players, classical AI baselines (Greedy, Minimax), exact retrograde solvers, and deep reinforcement learning agents (Double-DQN).

The application is structured into two completely decoupled layers:
- **Backend (Python / Flask)**: Runs the canonical game engine (`WallzEngine`), maintains transactional session histories, introspects agent decision models, and executes verified replay checks.
- **Frontend (React 18 / TypeScript / Vite)**: An SVG board and dashboard that draws what the backend sends (Contract C-STATE and Contract C-EXPL). It holds no copy of the rules: legal targets, block timers, wall counts, hop/block flags, paths and the terminal reason all come from the engine through the API. Section 7 lists every place the frontend touches game data and how the team checks it.

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
- **Match Controls**: Configure P0 and P1 AI agents, speed slider (0.1 s to 1.5 s per ply), Play/Pause, Step, and Reset.
- **Dual Think Panels**: Side-by-side inspection of both players' internal decision processes simultaneously.
- **Batch Tournaments**: Run synchronous batches ($n \in \{6, 10, 20, 50\}$ games) through `evaluation.match_runner.run_match`, alternating colors, opening seeds `seed + i // 2` with 2 opening plies. Shows wins, draws, losses and, separately, truncated games (200-ply runner cap), with the Wilson 95% interval of the win rate. Each batch game can be fetched as a verified replay (`/api/arena/batch/<bid>/<idx>`). The Arena is a demo, not the study's evidence: the study's match numbers are in `experiments/results/final_*`.

### 3. Training (`/training`)
Visual dashboard for reinforcement learning and model evaluation:
- **Decision Paradigm Cards**: Side-by-side comparison explaining how Greedy (1-ply heuristic), Minimax (tree search + leaf eval), Exact Solver (retrograde induction), and DQN (self-play learning) make choices.
- **Interactive Training Loop Diagram**: SVG visualization of the Double-DQN training architecture (transitions, replay buffer, target network updates, and opponent pool sampling). Ratios and hyperparameters are queried directly from the backend rather than hardcoded.
- **Interactive Learning Curves**: Multi-seed Recharts line graphs showing rolling win rate (100-episode window), episode length, exploration epsilon ($\varepsilon$), and TD-loss curves over environment steps.
- **Benchmark & Gate Tables**: Summary metrics from Gate X-16 (solver sign agreement), Gate X-5 (minimax feature AUC), and Greedy match campaigns.

### 4. Replays (`/replays`)
Inspection and verification tool for recorded matches:
- **Experiment Replay Browser**: Discovers and indexes match logs stored under `experiments/results/`, with a filename/ID filter. Older folders may hold games played under earlier rules; `experiments/results/STALE.md` says which.
- **Replay Scrubber**: Timeline slider, play/pause and step forward/backward.
- **Engine Verification Badge**: Every replay is verified by executing each action through a fresh `WallzEngine` and comparing logged state ids. If any action is illegal (or a different legal action was swapped in), the replay is flagged with the failing ply.
- **JSON Replay Upload**: Upload a `run_match` JSON result for verification and playback.

### 5. Rules (`/rules`)
Interactive rulebook and verification showcase:
- **Live Rule Scenarios (9 Scenarios)**: Step through the scenarios in `tests/scenarios/*.json` (the same files `tests/test_scenarios.py` runs): basic moves, wall placement and budget exhaustion, latest-wall repositioning, the one-time hop across an opponent's wall, $K$-ply block expiry, path preservation rejections, no-legal-move draws, five-fold repetition, and goal wins.
- **Rule Invariant Badges**: Each scenario expectation is checked against the engine at its ply and shown as ✓/✗ with its message.
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
   - On boards with $N \ge 7$ only `minimax:depth=1` is offered; `minimax:depth=2` and `minimax:depth=3,time_limit=2.0` are marked unavailable ("too slow for interactive play", `web/backend/registry.py`).
2. **Exact Solver Tractability**:
   - The retrograde solver is offered only for $N \le 4$ with $M \le 1$; otherwise it is marked unavailable with the reason.
   - Solves never block a request. Each runs in a separate process (one at a time) and is written to `experiments/cache/exact/<config>__<source-hash>.pkl`. Until it finishes the agent shows "solving in background". The server pre-solves the preset configs at start-up (disable with `--no-warmup`). On later starts a cached solution is available immediately and loads in about 0.5 s when first used. The source hash covers the engine and solver files, so a rule change automatically invalidates old caches.
   - Before 2026-11-09 the warm-up ran on a thread inside the server. Its pure-Python search held the GIL and made the first page load take about 47 s.
3. **DQN Compatibility Gate**:
   - Checkpoints are strictly validated against board size and rules (`meta.N == config.N` and `meta.rules == config.rules`). Incompatible models are disabled with a message indicating the configuration they were trained for.
   - The demo discovers `checkpoints/release/*` and `checkpoints/dqn_*`. The two release models were trained under the old 2-cell HOP rule (`experiments/results/STALE.md`); they still play only legal moves (the engine's legal mask), but their strength is measured in `experiments/results/final_dqn/` (`release_dqn_*_eval.json`). The final-rule retrained weights (`checkpoints/final/*`) are kept local; only their `manifest.json` is in git, so the demo does not list them.
4. **Arena Batch Caps**:
   - Batch tournaments are capped at $n \le 50$ games synchronously to prevent server worker timeouts.

---

## 7. How the Demo Stays Consistent with the Engine

The browser cannot run the Python engine, so the rules live only on the server:

1. Every game, arena match, scenario and replay runs on `WallzEngine` (`web/backend/sessions.py`, `arena.py` via `evaluation.match_runner.run_match`, `scenarios.py`, `replays.py`).
2. `web/backend/serialize.py::state_json` sends the engine's own data: `legal` is `engine.get_legal_actions()` grouped by type, and `inventory`, `hop`, `block`, `blocked` (with `remaining` timers), `walls` (with `owner` and `latest`), `paths` (`engine.shortest_path_cells`) and `terminal` (`winner`, `terminal_reason`) are read straight from engine fields.
3. The frontend only highlights the `legal` lists, sends the clicked action back, and shows the server's answer. A click on anything else is sent too and rejected by the engine with HTTP 400 and the engine's error text; the state is unchanged.
4. Undo rebuilds a fresh engine and replays the kept history.

Checks:

| Check | File | What it proves |
|---|---|---|
| API vs engine, ply by ply | `tests/web/test_engine_sync.py` | After every scripted ply the served state equals a shadow `WallzEngine` that replays the same actions (legal set, inventory, walls + latest, hop/block flags, block timers, side to move, terminal). Covers MOVE, WALL with inventory running out, REPLACE of the latest wall only, the one-time HOP across an opponent's wall, BLOCK lasting exactly K plies for K = 1, 2, 3, a path-sealing wall being rejected, and the terminal reasons `goal`, `no_legal_actions`, `repetition` and `truncated`. Also checks the Arena counts come from the match outcome. |
| Single-source guard | `tests/test_single_source.py` | No BFS / legality code in `web/backend` (AST) or `web/frontend/src` (regex). |
| Scenarios | `tests/test_scenarios.py`, `tests/web/test_api_data.py` | The 9 scenario files pass in pytest and through `/api/scenarios`. |
| Replays | `tests/web/test_api_data.py` | A tampered or substituted replay fails verification at the right ply. |

The frontend audit (every place `web/frontend/src` touches game data, with file:line) is in `docs/reports/stage3_greedy_demo_docs.md`.

## 8. Docker

```bash
docker build -t wallz-demo .
docker run --rm -p 5000:5000 wallz-demo
```

Inside the container the server binds `0.0.0.0` (`Dockerfile` CMD passes `--host 0.0.0.0`), so the published port is reachable. Run outside Docker, `python -m web.backend.app` keeps its default `--host 127.0.0.1`.
