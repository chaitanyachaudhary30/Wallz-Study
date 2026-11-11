# Implementation Status & Engineering Tracker

**Project**: Multi-Agent Reinforcement Learning for Adversarial Spatial Decision-Making with Temporal Constraints  
**Module**: Full System Integration, Baseline Suite, Exact Solver, MARL DQN, and Statistical Evaluation  
**Status**: Final rules frozen (Stage 1); Stage 2-3 validation results merged per member  
**Last Updated**: 2026-11-11 (`main` = `d91549c` plus the Stage 3 Member 2 PR)  

---

## 0. Final Status (read this first)

Numbers in older tables below (Sections 1, 6 and the phase notes) were measured before the final rules and are kept as history. The current evidence is:

| Area | Final status | Evidence |
| :--- | :--- | :--- |
| Rules / engine | Frozen at `ce55e0c`; unchanged since | `docs/game_rules.md`, `docs/reports/stage1_engine_freeze.md` |
| Which old results are stale | Listed per file | `experiments/results/STALE.md` |
| Greedy (Member 2) | Final match campaign: vs random, minimax depth 2, itself; N3, N5, N9 | `experiments/results/final_greedy/`, `docs/reports/stage3_greedy_demo_docs.md` |
| Minimax (Member 1) | Validated; optional hop-aware scorer | `experiments/results/final_minimax/`, `experiments/results/final_minimax_hop/`, `docs/reports/stage2_minimax.md`, `docs/reports/stage3_minimax_hop_eval.md` |
| Exact solver, ablation (Member 3) | Final-rule solves and ablations | `experiments/results/final_eval/`, `docs/reports/stage2_solver_stats.md` |
| DQN (Member 4) | Release models are stale (old HOP); retrained final-rule models evaluated | `experiments/results/final_dqn/`, `docs/reports/stage2_dqn.md` |
| Web demo | Rules only on the server; API checked against the engine | `docs/web_demo.md` Section 7, `tests/web/test_engine_sync.py` |
| Tests | Counted per file on 2026-11-11 | Section 2 below |

## 1. Summary of Completed Phases (history, pre-final-rules)

| Phase | Description | Status | Verification Gate |
| :---: | :--- | :---: | :--- |
| **Phase 1** | DQN & Self-Play Architecture (ActionCodec C3, Observation C4, WallzEnv C5, Negamax Double-DQN) | **Complete** | Checkpoint round-trip, codec involution, legal masks |
| **Phase 2** | DQN Training, Validation & Benchmarks (3×3, 4×4 models, IQM bootstrap metrics) | **Complete, stale** | Gate X-16 was passed under the old HOP rule; the release 3x3 model scores 86.8% under the final rules (`docs/reports/stage2_dqn.md`) |
| **Phase 3** | Engine Performance & Rule Corrections (WallzConfig, canonical position key, $O(1)$ ChainMap clone) | **Complete** | Legal actions $\le 1.54$ ms on 9×9, clone $\le 2.81\ \mu\text{s}$ |
| **Phase 4** | Tier-1 Greedy Baseline (4-tier lexicographical tie order, draw scoring, seeded tie-break) | **Complete** | Seeded deterministic reproducibility |
| **Phase 5** | Tier-2 Minimax / Negamax (PV-first ordering, history heuristic, goal extension, Contract C7) | **Complete** | Search cutoff efficiency, depth-2 latency |
| **Phase 6** | Evaluation Function & Feature Studies (f1–f6 extraction, L2 logistic regression AUC) | **Complete** | Gate X-5 passed (87.29% AUC on 3×3, 95.51% on 4×4) |
| **Phase 7** | Match Runner, Metrics & Logging (Holm-Bonferroni correction, Wilson CIs, table generator) | **Complete** | 14 publication-ready Markdown/LaTeX tables |
| **Phase 8** | State Graph & Exact Solver (Adapter over retrograde engine, cycle/stalemate draw separation) | **Complete** | $K \in \{1, 2, 3\}$ solved for $N=3, 4$ |
| **Phase 9** | Component Testing & Property Verification | **Complete** | Gate X-1 (Hypothesis stateful), Gate X-2 (NetworkX BFS) |
| **Phase 10** | Final System Integration & Acceptance | **Complete** | Test counts at that time are superseded; see Section 2 |

### Member 2 Track: Engine Enhancements, Verification & Web Demo

| Phase | Title | Commit | Status | Verification Gate |
| :---: | :--- | :---: | :---: | :--- |
| **P01** | Tooling, shared test helpers, action normalisation (F2) | `153fe17` | **Complete** | C-SNAP atomicity, JSON lists accepted, 187 tests passing |
| **P02** | Validate-before-mutate (F1), atomicity & post-terminal tests | `bb77405` | **Complete** | Clock isolation on illegal actions, post-flag raises, 197 tests |
| **P03** | Block duration, hop and REPLACE-form rule suite (F3) | `3f7b397` | **Complete** | Frozen fixtures R7 & Hop, exact K timeline, 218 tests |
| **P04** | No-legal-move draw and wall-budget suite (Task 1C & 1D) | `62c1413` | **Complete** | StateGraph first-hit fixture, budget M=0 audit, 234 tests |
| **P05** | Scenario fixtures, single-source guard, coverage map | `42061df` | **Complete** | 9 shared scenarios, AST single-source guard |
| **P06** | Greedy refactor, side-to-move guard (F4), verification | `93be06f` | **Complete** | Golden replay match, score_actions exposure, 247 tests |
| **P07** | Greedy match campaign across factorial rule cells | `5d86602` | **Superseded** | `experiments/results/greedy/` (dirty tree, pre-Minimax fix); replaced by `experiments/results/final_greedy/` |
| **P08** | Web backend core, engine adapter, agent registry | `8bf3589` | **Complete** | Flask endpoints /api/games/*, C-STATE serialization |
| **P09** | Web backend think endpoints (Greedy + Minimax + DQN + Solver) | `9e3fa49` | **Complete** | C-EXPL introspection, /api/games/<id>/analyze |
| **P10** | Web backend arena, replays, scenarios, training data | `d11fc0f` | **Complete** | /api/arena/batch, verified replays, training curves |
| **P11–P13** | Frontend React app, Board, Think panels, Arena, Training, Replays, Rules | `a7ee744` | **Complete** | Interactive board, Vitest tests pass, Vite build succeeds |
| **P14** | Documentation and final acceptance | `cae3f11` | **Superseded** | Rewritten in Stage 3 for the final rules |
| **P15** | (Stretch) Live mini-training stream and Dockerfile | — | **Complete** | Managed subprocess, SSE stream, multi-stage Dockerfile (container now binds `0.0.0.0`) |
| **Stage 1** | Engine freeze, rule tests, Greedy compatibility | PR #1 | **Complete** | `docs/reports/stage1_engine_freeze.md` |
| **Stage 3** | Greedy final campaign, demo sync, final docs | this PR | **Complete** | `docs/reports/stage3_greedy_demo_docs.md` |

---

## 2. Test Suite (counted 2026-11-11, one file at a time)

Each of the 52 files in `tests/` was run on its own with `python -m pytest <file> -q` (Windows, Python 3.12.5, pytest 9.2.0, torch 2.15.0+cpu, networkx installed, **hypothesis not installed**). Per-file results are in `docs/reports/stage3_greedy_demo_docs.md`.

| Group | Files | Result |
| :--- | ---: | :--- |
| Engine, rules, agents, evaluation (no torch use) | 36 | 305 passed (+78 subtests) |
| DQN or DQN checkpoints (need torch) | 9 | 56 passed |
| Web API (`tests/web/`, excluding `test_explain.py`) | 4 | 44 passed |
| Not run: need `hypothesis` (`test_strict_actions.py`, `test_greedy_final.py`, `test_invariants_stateful.py`) | 3 | collection error, 0 run |
| **Total** | **52** | **405 passed, 0 failed; 3 files not run** |

Every file needs pytest. `tests/conftest.py` stops any run when torch is missing unless `WALLZ_SKIP_DQN=1` is set, so a green run includes the DQN tests. `test_bfs_networkx.py` needs networkx.

Frontend (Vitest, after `npm --prefix web/frontend ci`): `npm --prefix web/frontend test` -> 11 passed in 3 files. `npm --prefix web/frontend run build` (tsc + vite) succeeds.

---

## 3. Member 2 Regression Coverage Map

| Rule / Mechanic | Core Unit Tests | Invariant / Property Tests | New Dedicated Suites | Shared Scenarios |
| :--- | :--- | :--- | :--- | :--- |
| **Normal Movement** | `test_movement.py` | `test_invariants_stateful.py` | `test_strict_actions.py` | `move_basics.json` |
| **Wall Placement** | `test_walls.py` | `test_invariants_stateful.py` | `test_strict_actions.py` | `wall_place_and_budget.json` |
| **Wall Repositioning (REPLACE)** | `test_replacement.py` | `test_invariants_stateful.py` | `test_rules_blocks_hop.py` | `replace_latest_only.json` |
| **Latest-Wall Restriction** | `test_replacement.py` | `test_invariants_stateful.py` | `test_rules_blocks_hop.py` | `replace_latest_only.json` |
| **One-Time Hop (Rule B, opponent's wall)** | `test_hop.py` | `test_invariants_stateful.py` | `test_rules_blocks_hop.py` | `hop_over_wall.json` |
| **Block Placement & Timing (Rule C)** | `test_temporary_blocks.py` | `test_invariants_stateful.py` | `test_rules_blocks_hop.py` | `block_expires_after_K.json` |
| **Path Preservation** | `test_path_preservation.py` | `test_invariants_stateful.py` | `test_rules_blocks_hop.py` | `path_preservation_rejects_wall.json` |
| **Five-Fold Repetition** | `test_terminal_states.py` | `test_invariants_stateful.py` | `test_strict_actions.py` | `fivefold_repetition_draw.json` |
| **No-Legal-Move Draw (Task 1C)** | `test_terminal_states.py` | `test_invariants_stateful.py` | `test_rules_draw_budget.py` | `no_legal_move_draw.json` |
| **Wall Inventory Budget (Task 1D)** | `test_walls.py` | `test_invariants_stateful.py` | `test_rules_draw_budget.py` | `wall_place_and_budget.json` |
| **Invalid Action Rejection (Atomicity)** | — | `test_invariants_stateful.py` | `test_action_normalization.py`, `test_strict_actions.py` | All scenarios |
| **Terminal Freezes & Transitions** | `test_terminal_states.py` | `test_invariants_stateful.py` | `test_strict_actions.py` | `goal_win.json` |

> [!NOTE]
> **Golden Solves Consistency**: `test_golden_regression.py` retrogrades match 100% of ground-truth solves under the exact K-ply semantics, confirming complete mathematical consistency.

---

## 4. Wall-Budget Audit (`rules['M']`)

An automated AST and regex scan (`tests/test_rules_draw_budget.py::test_no_hardcoded_wall_budget`) audits the entire codebase outside configuration defaults.
- Zero hardcoded occurrences of `10` exist in `agents/greedy.py`, `evaluation/match_runner.py`, or `evaluation/features.py`.
- Wall inventory is strictly read from `engine.wall_inventory[player]` and initialized from `rules['M']` (defaulting to 10 only at entry points).
- Wall placement ceases when `wall_inventory[player] == 0`, and negative $M$ is rejected at initialization.

---

## 5. Hand-offs and Open Issues

| Issue ID | Target Member | Description & Required Coordination |
| :---: | :---: | :--- |
| **H-1** | **Member 1** | Request public `MinimaxAgent.analyze_root(engine, depth) -> list[(action, value)]` with full-window values; until then `web/backend/explain.py` uses private `_negamax` read-only. |
| **H-2** | **Member 3** | `docs/person3_evaluation.md` §7 and `evaluation/replay_export.py` target the legacy demo; the new web demo replays `run_match` JSON directly. |
| **H-3** | **Member 4** | Verify DQN conformance tests (`tests/test_action_codec.py`, `tests/test_dqn_*`) pass under `_validate_action` / `_mutate` lifecycle ordering (all green). |
| **H-4** | **Member 4** | Hop rule changed (2026-11-08) from R1 (2-cell sprint) to the proposal's opponent-wall hop. `ActionCodec` HOP ids 4–7 now decode to 1-cell deltas (layout and `CODEC_VERSION` unchanged). The release/run checkpoints were trained under R1 and must be retrained. The 14-plane observation has no wall-ownership plane (only the latest walls), so non-latest wall ownership, which now decides hop legality, is invisible to the network. The legal mask keeps play legal, but consider an observation version bump with owner planes. |
| **H-5** | **Member 3** | Hop rule change: `canonical_key` now delegates to `WallzEngine.canonicalize_key` (ownership is kept while a hop is unused). `experiments/results/solves/*` were regenerated. Ablation pilots in `experiments/results/person3_pilot` and the factorial tables were produced under R1 and need re-running. The pilot agreement files (`person3_pilot/agreement_N3_K1_M{0,1}.json`, 16 rows each, 2026-11-06) also predate the intentional greedy tie order and draw scoring (Decision 10, commit `acef2aa`, 2026-11-07) and minimax's per-decision reset (H-7); `experiments/results/agreement/agreement_3x3.json` postdates the greedy change but predates the hop change and H-7. Regenerate all of them together and cite these three causes. |
| **H-6** | **Member 1** | Minimax results/benchmarks produced before the hop rule change (R1) are stale; agents previously opened with a 2-cell hop. |
| **H-7** | **Member 1** | `agents/minimax.py` (Member 1's file) was patched (2026-11-10): `select_action` now resets `pv_move` and `history` at the start of each call. Before, with no time limit, the PV move and history from the previous, unrelated position ordered the root, so ties resolved differently by call order. Measured on 300 sampled N3 K1 M1 states at depth 2: 11 (A0B0C0) and 16 (A1B1C1) choices differed between a reused and a fresh agent, and value-optimal agreement moved 0.983 to 0.980 when only the sampling order was reversed. Values and legality were never affected. Regression test: `tests/test_minimax_phase5.py::test_decision_independent_of_previous_calls`. |

---

## 6. Production Artifacts & Checkpoints

- **Released Neural Network Weights** (trained under the old HOP rule, stale; see `experiments/results/STALE.md`):
  - `checkpoints/release/dqn_3x3/model.pt` + `meta.json`: Double-DQN $3 \times 3$ model, 86.8% Gate X-16 sign agreement under the final rules.
  - `checkpoints/release/dqn_4x4/model.pt` + `meta.json`: Double-DQN $4 \times 4$ model.
  - Final-rule retrained models: `checkpoints/final/*/manifest.json` (weights kept local), evaluated in `experiments/results/final_dqn/`.
- **Exact Retrograde Ground-Truth Solves** (final rules): `experiments/results/final_eval/solve_N3_K1_M0.json`, `solve_N3_K1_M1.json`, `solve_N4_K1_M0.json` (all 8 cells exact), `solve_N4_K1_M1.json` (4 of 8 cells exact). The older `experiments/results/solves/` files: see `STALE.md`.
- **Match results** (final rules): `experiments/results/final_greedy/`, `final_minimax/`, `final_minimax_hop/`, `final_eval/`, `final_dqn/`.
- **Tables**: `experiments/tables/` is git-ignored and regenerated by `scripts/make_tables.py`; the final per-member tables are `experiments/results/final_greedy/tables.md` and `experiments/results/final_dqn/tables.md`.
