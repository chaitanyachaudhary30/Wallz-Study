# Stage 3 Report: Greedy Match Campaign, Browser Demo Sync, Final Docs (Member 2, Tasks 3-5)

- **Base:** `main` = `d91549c`. No engine code changed (`wallz_engine.py`, `src/modified_wallz/` untouched).
- **Commits tested:** match sets N3 (random, minimax2) and N5 minimax2 at `edc3d06`; N3 self, N5/N9 random and self at `1edd675`; N9 minimax2 at `69509f5`. Between `edc3d06` and `1edd675`, `scripts/final_greedy.py` changed only how self-play games are counted (once per game, from Player 0) and how `tables.md` is printed; match play is identical. Every summary JSON records its own commit and `dirty: false`.
- **Machine:** Windows 11, Python 3.12.5, pytest 9.2.0, torch 2.15.0+cpu, numpy 2.2.0, Flask 3.1.0, networkx installed, hypothesis **not** installed. Node 22.19 / npm 10.9 (frontend packages installed with `npm ci` from the committed lockfile, approved by the conductor).

## Task 3: Greedy match campaign

### Method

| Item | Choice |
|---|---|
| Script | `scripts/final_greedy.py` (test: `tests/test_final_greedy.py`) |
| Pairings | `greedy` vs `random:seed=R` (R = 0..4); `greedy` vs `minimax:depth=2`; self-play `greedy:seed=S` vs `greedy:seed=S+100` (S = 0..4) |
| Seats | every opening x spec pair is played with Greedy (or seed S) as P0 and as P1 |
| Boards | N3 K1 M1 (exact-solver comparison), N5 K3 M10 (main), N9 K3 M10 (larger-board check); A, B, C all on |
| Openings | shortest `opening_plies` with enough distinct MOVE-only openings; seeds scanned from 0, first seed of each distinct opening action sequence kept. The seed -> opening map is in each summary (`config.seed_to_opening`) |
| Cap | runner cap 200 plies; `truncated` is its own outcome, never a draw |
| Counting | over distinct games (same action list counts once; per Greedy seat for pairings with another agent, once overall for self-play, read from Player 0) |
| CIs | Wilson 95% (`evaluation.metrics.wilson`) |
| Replays | every logged match re-verified with `verify_replay`: 0 failures in all 9 sets |

| Board | Distinct openings possible at 0/1/2/3/4 plies | Chosen plies | Openings | Seeds scanned |
|---|---|---:|---:|---:|
| N3 | 1 / 3 / 8 | 2 | 8 (all) | 0-9 |
| N5 | 1 / 3 / 9 / 30 / 99 | 4 | 50 | 0-79 |
| N9 | 1 / 3 / 9 / 30 | 3 | 10 | 0-9 |

| Set | Matches run | Distinct games | Wall time |
|---|---:|---:|---:|
| N3 random / minimax2 / self | 80 / 16 / 80 | 68 / 16 / 10 | 12 s / 4 s / 14 s |
| N5 random / minimax2 / self | 500 / 100 / 500 | 500 / 100 / 273 | 97 s / 39 s / 119 s |
| N9 random / minimax2 / self | 100 / 20 / 100 | 100 / 20 / 19 | 64 s / 53 s / 79 s |

Seeded Greedy differs from plain Greedy only in breaking exact score ties, so many self-play seed pairs replay the same game (N3: 80 matches, 10 games; N9: 100 matches, 19 games).

### Exact small board: N3 K1 M1 (separate from the match claims below)

The exact solver says the start position is a draw (`final_eval/solve_N3_K1_M1.json`, root `DRAW`, cycle). For each game, the table compares the exact value of the post-opening position for Greedy's seat with what Greedy got. Solution file: `experiments/cache/final_sol_N3_A1B1C1_K1_M1_631031127ae7.pkl` (git-ignored cache, same engine hash as `scripts/eval_dqn_final.py` uses).

| Pairing | Games | Value win: got W / D / L | Value draw: got W / D / L | Value loss: got W / D / L | Games below value |
|---|---:|---|---|---|---:|
| vs random | 68 | 10 / 0 / 0 | 38 / 0 / 0 | 18 / 0 / 2 | 0 |
| vs minimax:depth=2 | 16 | 2 / 2 / 0 | 0 / 2 / 6 | 0 / 0 / 4 | 8 |
| self-play (P0 side) | 10 | 6 / 0 / 0 | 2 / 0 / 2 | — | 2 |

Greedy never fell below the game value against random. Against minimax depth 2 it fell below the value in 8 of 16 games: it lost 6 of 8 drawn positions and drew 2 of 4 won ones.

N3 match outcomes (Greedy's side):

| Pairing | Greedy seat | Games | W | L | Draws by type | Trunc | Win % [95% CI] | Draw % [95% CI] | Mean plies | Greedy ms/move | Opp ms/move |
|---|---|---:|---:|---:|---|---:|---|---|---:|---:|---:|
| random | P0 | 29 | 29 | 0 | 0 | 0 | 100.0 [88.3, 100.0] | 0.0 [0.0, 11.7] | 7.5 | 0.5 | 0.0 |
| random | P1 | 39 | 37 | 2 | 0 | 0 | 94.9 [83.1, 98.6] | 0.0 [0.0, 9.0] | 7.6 | 0.4 | 0.0 |
| random | all | 68 | 66 | 2 | 0 | 0 | 97.1 [89.9, 99.2] | 0.0 [0.0, 5.3] | 7.5 | 0.5 | 0.0 |
| minimax:depth=2 | P0 | 8 | 2 | 4 | repetition 2 | 0 | 25.0 [7.1, 59.1] | 25.0 [7.1, 59.1] | 12.2 | 0.5 | 1.9 |
| minimax:depth=2 | P1 | 8 | 0 | 6 | repetition 2 | 0 | 0.0 [0.0, 32.4] | 25.0 [7.1, 59.1] | 15.0 | 0.6 | 2.5 |
| minimax:depth=2 | all | 16 | 2 | 10 | repetition 4 | 0 | 12.5 [3.5, 36.0] | 25.0 [10.2, 49.5] | 13.6 | 0.6 | 2.3 |
| self-play | W/L = P0 | 10 | 8 | 2 | 0 | 0 | 80.0 [49.0, 94.3] | 0.0 [0.0, 27.8] | 5.6 | 0.5 | 0.5 |

### Larger boards: N5 K3 M10 (main) and N9 K3 M10 (check)

Truncation rate is its own column. No game in any set was a forfeit.

| Board | Pairing | Greedy seat | Games | W | L | Draws by type | Trunc | Win % [95% CI] | Draw % [95% CI] | Trunc % [95% CI] | Mean plies | Greedy ms/move | Opp ms/move |
|---|---|---|---:|---:|---:|---|---:|---|---|---|---:|---:|---:|
| N5 | random | P0 | 250 | 249 | 1 | 0 | 0 | 99.6 [97.8, 99.9] | 0.0 [0.0, 1.5] | 0.0 [0.0, 1.5] | 14.2 | 4.0 | 0.3 |
| N5 | random | P1 | 250 | 249 | 0 | repetition 1 | 0 | 99.6 [97.8, 99.9] | 0.4 [0.1, 2.2] | 0.0 [0.0, 1.5] | 17.1 | 3.8 | 0.3 |
| N5 | random | all | 500 | 498 | 1 | repetition 1 | 0 | 99.6 [98.6, 99.9] | 0.2 [0.0, 1.1] | 0.0 [0.0, 0.8] | 15.7 | 3.9 | 0.3 |
| N5 | minimax:depth=2 | P0 | 50 | 31 | 19 | 0 | 0 | 62.0 [48.2, 74.1] | 0.0 [0.0, 7.1] | 0.0 [0.0, 7.1] | 19.4 | 3.9 | 29.6 |
| N5 | minimax:depth=2 | P1 | 50 | 29 | 20 | repetition 1 | 0 | 58.0 [44.2, 70.6] | 2.0 [0.4, 10.5] | 0.0 [0.0, 7.1] | 17.6 | 4.0 | 29.3 |
| N5 | minimax:depth=2 | all | 100 | 60 | 39 | repetition 1 | 0 | 60.0 [50.2, 69.1] | 1.0 [0.2, 5.4] | 0.0 [0.0, 3.7] | 18.5 | 3.9 | 29.5 |
| N5 | self-play | W/L = P0 | 273 | 132 | 118 | repetition 21, no_legal_actions 1 | 1 | 48.4 [42.5, 54.3] | 8.1 [5.4, 11.9] | 0.4 [0.1, 2.0] | 32.0 | 3.7 | 3.6 |
| N9 | random | P0 | 50 | 50 | 0 | 0 | 0 | 100.0 [92.9, 100.0] | 0.0 [0.0, 7.1] | 0.0 [0.0, 7.1] | 22.5 | 40.8 | 2.3 |
| N9 | random | P1 | 50 | 50 | 0 | 0 | 0 | 100.0 [92.9, 100.0] | 0.0 [0.0, 7.1] | 0.0 [0.0, 7.1] | 22.3 | 34.8 | 1.8 |
| N9 | random | all | 100 | 100 | 0 | 0 | 0 | 100.0 [96.3, 100.0] | 0.0 [0.0, 3.7] | 0.0 [0.0, 3.7] | 22.4 | 37.8 | 2.1 |
| N9 | minimax:depth=2 | P0 | 10 | 2 | 8 | 0 | 0 | 20.0 [5.7, 51.0] | 0.0 [0.0, 27.8] | 0.0 [0.0, 27.8] | 19.8 | 23.6 | 282.6 |
| N9 | minimax:depth=2 | P1 | 10 | 7 | 3 | 0 | 0 | 70.0 [39.7, 89.2] | 0.0 [0.0, 27.8] | 0.0 [0.0, 27.8] | 18.9 | 23.0 | 273.6 |
| N9 | minimax:depth=2 | all | 20 | 9 | 11 | 0 | 0 | 45.0 [25.8, 65.8] | 0.0 [0.0, 16.1] | 0.0 [0.0, 16.1] | 19.4 | 23.3 | 278.4 |
| N9 | self-play | W/L = P0 | 19 | 5 | 11 | repetition 2 | 1 | 26.3 [11.8, 48.8] | 10.5 [2.9, 31.4] | 5.3 [0.9, 24.6] | 46.1 | 30.3 | 29.9 |

Mean plies include the opening plies. The same table, generated from the summaries, is `experiments/results/final_greedy/tables.md`.

What the files support:
- Greedy beats random almost always on every board (N5: 498 of 500 games).
- On N5, Greedy beats `minimax:depth=2` in 60 of 100 games (CI 50.2-69.1%). Member 1's run on the same 50 opening seeds gives the same counts, 60 Greedy wins / 39 minimax wins / 1 draw (`experiments/results/final_minimax/vs_greedy_N5_open4_distinct_summary.json`), as expected for two deterministic agents. On N3, where the solver applies, Greedy is much weaker than minimax (2 wins in 16) and loses drawn positions.
- On N9 the 20-game check is too small to separate the agents (CI 25.8-65.8%); Greedy wins 7 of 10 as P1 and 2 of 10 as P0.
- Self-play draws are mostly repetition draws (N5: 21 of 22). One N5 and one N9 self-play game hit the 200-ply cap; they are counted as truncated, not as draws.

The older `experiments/results/greedy/` files are marked superseded in `experiments/results/STALE.md` (dirty tree, pre-Minimax-fix, repeated openings, and its `summary.json` merges truncated games into draws).

## Task 4: Browser demo sync

### (a) Audit: every place `web/frontend/src` touches game data

Line numbers are for this branch (unchanged lines keep their `d91549c` numbers).

| File:line | What it does | Copies or contradicts the engine? |
|---|---|---|
| `components/Board.tsx:46-70` | Arrow keys pick the neighbouring cell, then send `MOVE` only if it is in `state.legal.MOVE` | No: the engine list decides |
| `components/Board.tsx:74-101` | Highlights MOVE / HOP / BLOCK cells and WALL / REPLACE slots from `state.legal` | No |
| `components/Board.tsx:111-120` | A click sends the action; illegal ones come back as HTTP 400 with the engine's error | No |
| `components/Board.tsx:227` | Block badge shows `blocked[].remaining` from the engine | No |
| `components/Board.tsx:344-355` | Glow on each player's `latest` wall when rule A is on | No (`latest` comes from `engine.latest_wall`) |
| `components/ActionBar.tsx:20-25` | Button counts = lengths of the `legal` lists | No |
| `components/PlayerCard.tsx:12-17` | Inventory, `rules.M`, hop/block flags, path length from `state.paths` | No |
| `components/TerminalBanner.tsx:18-27` (was `:19-25` at `d91549c`) | Text for each terminal reason | **Yes, fixed:** `truncated` was shown as "Draw — Maximum ply cap reached"; also "Rule R3" and "Fivefold" labels |
| `components/ConfigForm.tsx:59, 73, 86` | Board sizes `[3,4,5,7,9]`, M max 20, K max 9 | Not a contradiction: same as the backend `limits`; the backend validates (`routes_games.py::parse_game_config`, `WallzConfig`) |
| `pages/Play.tsx:23-30`, `pages/Arena.tsx:9-16` | Initial config `K: 3, M: 10` | Not a contradiction: engine defaults; Play replaces it with `/api/config/defaults` |
| `pages/Play.tsx:106-121` | Picks a mode whose `legal` list is non-empty | No |
| `pages/Arena.tsx:282-283` | Shows W-D-L from the backend batch | **Yes, fixed in the backend:** `web/backend/arena.py:102-103,116` read `res['winner']`, `res['terminal_reason']`, `res['length']`, which `run_match` does not return, so every Arena game counted as a draw, categories read `unknown` and mean length 0. Truncated games now have their own column |
| `pages/Rules.tsx:66` | "Rule Scenarios (9)" was hard-coded | Fixed to `scenarios.length` |
| `pages/Rules.tsx:168` | Shows `chk.message` | **Yes, fixed in the backend:** `tests/scenario_runner.py:68` names the field `msg`, so every check text was blank; `web/backend/scenarios.py` now adds `message` |
| `pages/Replays.tsx:7-17` | Outcome text from the logged `outcome.category` | No (a forfeit shows as a win, which is what the log says) |
| `components/Timeline.tsx:11-17`, `components/Heatmap.ts:26-35` | Format actions / map overlay values to cells | No |
| `lib/geometry.ts:53-95` | Row 0 drawn at the bottom | No (drawing only) |

No frontend code computes hop distance, block timers, legal moves, wall counts or win/draw conditions.

### (b) Fixes (commit `b47e0d6`, test `69509f5`)

- `web/backend/arena.py`: counts from `res['outcome']`; `truncated` counted apart from draws; `categories` are real outcome categories; mean length from `length_plies`.
- `web/frontend/src/components/TerminalBanner.tsx`: `truncated` -> "Stopped — ply cap reached (truncated, not a rule draw)".
- `web/frontend/src/pages/Arena.tsx`: shows the truncated count.
- `web/backend/scenarios.py`: checks carry `message`. `web/frontend/src/pages/Rules.tsx`: scenario count from the list.

### (c) Consistency and tests

`docs/web_demo.md` Section 7 explains how the demo stays consistent. New `tests/web/test_engine_sync.py` (11 tests) drives games through the API and, after every ply, compares the served state with a shadow `WallzEngine` replaying the same actions:

| Feature | Test |
|---|---|
| pawn movement, wall placement, inventory to 0, REPLACE still legal at 0 | `test_move_wall_and_inventory` |
| REPLACE of the latest wall only (older wall rejected, state unchanged) | `test_replace_moves_only_the_latest_wall` |
| one-time hop across an opponent's wall; second hop rejected | `test_hop_needs_an_opponent_wall_and_is_one_time` |
| block lasts exactly K plies (K = 1, 2, 3) | `test_block_lasts_exactly_k_plies` |
| path preservation rejects a sealing wall | `test_path_preservation_rejects_a_sealing_wall` |
| goal win, then 409 on further actions | `test_goal_win_then_no_more_actions` |
| repetition draw | `test_repetition_draw` |
| demo ply cap -> `truncated` | `test_ply_cap_is_reported_as_truncated` |
| no-legal-action draw | `test_no_legal_action_draw_from_scenario` |
| Rules page check text present | `test_scenario_checks_carry_the_message_the_page_shows` |
| Arena counts match the match outcomes | `test_arena_counts_come_from_the_match_outcome` |

Frontend: Vitest 11 passed in 3 files (one new test: truncated banner is not a draw). `npm run build` (tsc + vite) succeeds.

### (d) Docker

Confirmed: `Dockerfile:39` ran `python -m web.backend.app --port 5000` and `web/backend/app.py:73` defaults `--host` to `127.0.0.1`, so the server listened only inside the container. The CMD now passes `--host 0.0.0.0`; the local default stays `127.0.0.1`. Docker itself was not run on this machine.

## Task 5: Documentation

| File | Change |
|---|---|
| `README.md` | Rewritten. The old file held two interleaved copies (duplicate bullets, two status tables, broken code fences). Now: final rules summary, status table pointing to `experiments/results/final_*` and the stage reports, setup, tests, demo, layout. "Production Ready", "301/301", "160/160", "325 passed", "720 matches", "22.66%" and "96.47%" are no longer stated as current |
| `docs/game_rules.md` | Truncation: a truncated game is its own outcome; the engine's `is_draw=True` belongs only to its own optional `max_plies` cap; the web demo passes its cap to the engine and shows "truncated" |
| `docs/architecture.md` | Repetition limit configurable, inventory range `[0, M]`, block for `K` plies, the real Greedy ranking tuple, final-rule solver coverage, release DQN stale (86.8% under final rules), what the single-source guard really checks, module map `agents/greedy.py` |
| `docs/implementation_status.md` | New "Final Status" table; old phase rows marked superseded or stale; real test counts (Section 2); artifacts point to final folders |
| `docs/web_demo.md` | Arena speed/batch sizes, removed the non-existent "Early vs Final DQN" button, minimax availability on N >= 7, release DQN note, consistency section, Docker |
| `docs/setup.md` | Test-count lines |
| `experiments/results/STALE.md` | `greedy/` rows marked superseded by `final_greedy/` |

`file:///` links and mojibake: a search of every tracked text file (Python scan for `file:///`, `â€`, `Ã`, `Â`, `ðŸ`, U+FFFD) found none at `d91549c`. The only odd byte is a UTF-8 BOM at the start of `experiments/results/final_dqn/tables.md` (Member 4's file, left alone).

### Test counts (each file run once, on its own)

| File | Result | Needs |
|---|---|---|
| `test_action_codec.py` | 4 passed | |
| `test_action_normalization.py` | 21 passed | |
| `test_agent_conformance.py` | 6 passed | torch |
| `test_agents.py` | 13 passed | |
| `test_bfs_networkx.py` | 1 passed | networkx |
| `test_dqn_agent.py` | 4 passed | torch |
| `test_dqn_encoding.py` | 4 passed | torch |
| `test_dqn_final_compat.py` | 19 passed | torch |
| `test_dqn_runner_integration.py` | 2 passed | torch |
| `test_dqn_trainer.py` | 3 passed | torch |
| `test_edge_invariants.py` | 4 passed (55 s) | |
| `test_engine_prerequisites.py` | 22 passed | |
| `test_evaluation_fn.py` | 5 passed | |
| `test_exact_solver.py` | 9 passed | |
| `test_filter_equivalence.py` | 2 passed | |
| `test_final_greedy.py` (new) | 2 passed | |
| `test_golden_regression.py` | 3 passed | |
| `test_greedy.py` | 7 passed | |
| `test_greedy_campaign.py` | 4 passed | |
| `test_greedy_final.py` | **not run** (collection error) | hypothesis |
| `test_helpers.py` | 6 passed | |
| `test_hop.py` | 1 passed | |
| `test_initial_state.py` | 2 passed | |
| `test_invariants_stateful.py` | **not run** (collection error) | hypothesis |
| `test_match_runner.py` | 17 passed | |
| `test_metrics_holm.py` | 5 passed | |
| `test_minimax_final_compat.py` | 14 passed (+44 subtests) | |
| `test_minimax_hop_eval.py` | 9 passed | |
| `test_minimax_phase5.py` | 6 passed | |
| `test_movement.py` | 2 passed | |
| `test_multi_seed_aggregation.py` | 4 passed | torch |
| `test_path_preservation.py` | 2 passed | |
| `test_replacement.py` | 7 passed | |
| `test_retrograde.py` | 17 passed | |
| `test_rl_metrics.py` | 4 passed | |
| `test_rules_blocks_hop.py` | 24 passed | |
| `test_rules_draw_budget.py` | 16 passed | |
| `test_scenarios.py` | 10 passed | |
| `test_self_play.py` | 4 passed | torch |
| `test_single_source.py` | 3 passed | |
| `test_solver_final_rules.py` | 18 passed | |
| `test_stage1_engine_freeze.py` | 35 passed (+34 subtests) | |
| `test_strict_actions.py` | **not run** (collection error) | hypothesis |
| `test_temporary_blocks.py` | 2 passed | |
| `test_terminal_states.py` | 2 passed | |
| `test_walls.py` | 2 passed | |
| `test_wallz_env.py` | 4 passed | |
| `web/test_api_data.py` | 12 passed | |
| `web/test_api_games.py` | 18 passed (61 s) | |
| `web/test_engine_sync.py` (new) | 11 passed | |
| `web/test_explain.py` | 10 passed | torch |
| `web/test_live_training.py` | 3 passed | |
| **52 files** | **405 passed, 0 failed; 3 files not run** | |

All files are pytest files, and `tests/conftest.py` refuses to start without torch unless `WALLZ_SKIP_DQN=1`. Frontend: Vitest 11 passed (3 files).

## Commands

```
git -C "D:\SEM V\Wallz-Study" worktree add "D:\SEM V\Wallz-Study-wt\bot4-demo" -b bot4-abhishek/greedy-demo-docs origin/main
python scripts/final_greedy.py run --board N3 --pairing random      # also minimax2, self
python scripts/final_greedy.py run --board N5 --pairing random      # also minimax2, self
python scripts/final_greedy.py run --board N9 --pairing random      # also minimax2, self
python scripts/final_greedy.py tables
python -m pytest tests/<file>.py -q -p no:cacheprovider             # once per file, stdin closed
npm --prefix web/frontend ci && npm --prefix web/frontend test && npm --prefix web/frontend run build
```

The N3 runs read the cached exact solve (`experiments/cache/final_sol_N3_A1B1C1_K1_M1_631031127ae7.pkl`, copied from Bot 2's worktree; same engine hash). Without the cache the script solves it (~50 s).
