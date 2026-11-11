# Modified Wallz — Multi-Agent Decision-Making Study

Course project for **"Multi-Agent Reinforcement Learning for Adversarial Spatial Decision-Making with Temporal Constraints"** (CS 329: Foundations of AI / Multi-Agent Systems, IIT Gandhinagar).

The repository holds the game engine, four agents (Greedy, Minimax, an exact retrograde solver, Double-DQN), the match and evaluation pipeline, the final results, and a browser demo.

---

## Game rules (final)

Modified Wallz is a two-player, Quoridor-like pawn race on an $N \times N$ grid. Player 0 starts at `(0, N//2)` and must reach row `N-1`; Player 1 starts at `(N-1, N//2)` and must reach row `0`. Players alternate one legal action per ply: `MOVE`, `WALL`, `REPLACE`, `HOP` or `BLOCK`.

- **A. Reusable walls.** Each player has `M` walls (`rules['M']`, default 10). A `WALL` costs one. `REPLACE` moves only the player's latest wall, costs nothing, and is allowed at zero inventory.
- **B. One-time hop.** Once per game a pawn may step into the adjacent cell across a wall **placed by the opponent**. No hop over open edges, own walls, pawns or blocked cells.
- **C. Temporary block.** Once per game a player may block an empty cell for exactly `K` plies (`rules['K']`, default 3).
- **Path preservation.** Every action must leave both pawns a path to their goal rows.
- **Endings.** Reaching the goal row wins. The player to move having no legal action is a draw; so is the same position occurring `repetition_limit` times (default 5).
- **Truncation is not a draw.** The match runner stops a game at 200 plies and reports it as `truncated`, its own outcome. The engine's optional `max_plies` cap sets `is_draw=True`, but that flag is only the engine's marker for its own optional cap.

`wallz_engine.py` is the only source of the rules. Full rules: [`docs/game_rules.md`](docs/game_rules.md). Freeze report: [`docs/reports/stage1_engine_freeze.md`](docs/reports/stage1_engine_freeze.md).

---

## Status and results

| Area | Status | Results | Report |
|---|---|---|---|
| Engine | Frozen at `ce55e0c` (unchanged since) | — | [`docs/reports/stage1_engine_freeze.md`](docs/reports/stage1_engine_freeze.md) |
| Greedy (`agents/greedy.py`) | Final match campaign done | `experiments/results/final_greedy/` | [`docs/reports/stage3_greedy_demo_docs.md`](docs/reports/stage3_greedy_demo_docs.md) |
| Minimax (`agents/minimax.py`) | Validated; optional hop-aware scorer (`eval=hop`) | `experiments/results/final_minimax/`, `final_minimax_hop/` | [`docs/reports/stage2_minimax.md`](docs/reports/stage2_minimax.md), [`docs/reports/stage3_minimax_hop_eval.md`](docs/reports/stage3_minimax_hop_eval.md) |
| Exact solver, ablations (`evaluation/`) | Final-rule solves and ablations | `experiments/results/final_eval/` | [`docs/reports/stage2_solver_stats.md`](docs/reports/stage2_solver_stats.md) |
| DQN (`agents/dqn/`) | Release models stale (old HOP rule); retrained final-rule models evaluated | `experiments/results/final_dqn/` | [`docs/reports/stage2_dqn.md`](docs/reports/stage2_dqn.md) |
| Web demo (`web/`) | Rules only on the server; API checked against the engine | — | [`docs/web_demo.md`](docs/web_demo.md) |

Older result folders (`experiments/results/greedy/`, `person3_pilot/`, `dqn/`, ...) were made under earlier rules or agents. [`experiments/results/STALE.md`](experiments/results/STALE.md) says which still hold. Earlier headline numbers such as Minimax's "22.66% node reduction" and DQN's "96.47% sign agreement" do not hold under the final rules; the reports above give the final values.

---

## Setup

Python 3.10+ (tested on 3.12). From the repository root:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements-dev.txt   # requirements.txt (numpy, torch, networkx, hypothesis, flask, ...) + pytest
```

More detail: [`docs/setup.md`](docs/setup.md).

## Tests

```bash
python -m pytest tests/test_hop.py -q   # one file at a time, as the team runs them
python -m pytest -q                     # whole suite
```

`tests/conftest.py` stops the run when torch is missing (set `WALLZ_SKIP_DQN=1` to skip DQN tests on purpose). Per-file counts measured for the final status are in [`docs/implementation_status.md`](docs/implementation_status.md) Section 2.

Frontend unit tests (Vitest) need `npm --prefix web/frontend ci` first, then `npm --prefix web/frontend test`.

## Running matches

```bash
python run_match.py --p0 minimax --p1 greedy --depth 2 --seed 42
python scripts/final_greedy.py run --board N5 --pairing minimax2   # writes experiments/results/final_greedy/
```

## Web demo

```bash
npm --prefix web/frontend ci
npm --prefix web/frontend run build
python -m web.backend.app --port 5000      # http://127.0.0.1:5000
```

Docker: `docker build -t wallz-demo .` then `docker run --rm -p 5000:5000 wallz-demo` (the container binds `0.0.0.0`). Views, API and the consistency checks: [`docs/web_demo.md`](docs/web_demo.md).

---

## Repository layout

```text
wallz_engine.py          # the engine: rules, legal actions, path checks, terminal states
src/modified_wallz/      # board geometry, BFS, config (used by the engine), action codec, Gym-style env
agents/                  # greedy.py, minimax.py, evaluation_fn.py, exact_solver.py, dqn/
evaluation/              # match_runner.py, metrics.py, state_graph.py, retrograde.py, ablation.py
scripts/                 # experiment entry points (final_greedy.py, final_minimax.py, eval_dqn_final.py, ...)
experiments/results/     # committed results; final_* folders hold the final-rule evidence
web/backend/             # Flask API over the engine
web/frontend/            # React + TypeScript board and dashboards (no rule logic)
tests/                   # engine, agent, evaluation and web tests
docs/                    # rules, architecture, setup, web demo, stage reports
```

Architecture: [`docs/architecture.md`](docs/architecture.md). Implementation tracker: [`docs/implementation_status.md`](docs/implementation_status.md). Team decisions: [`docs/decisions.md`](docs/decisions.md).
