# Implementation Status & Engineering Tracker

**Project**: Multi-Agent Reinforcement Learning for Adversarial Spatial Decision-Making with Temporal Constraints  
**Module**: Full System Integration, Baseline Suite, Exact Solver, MARL DQN, and Statistical Evaluation  
**Status**: In Progress (Member 2 Engine Enhancements & Web Application Track)  
**Last Updated**: November 2026  

---

## 1. Summary of Completed Phases

| Phase | Description | Status | Verification Gate |
| :---: | :--- | :---: | :--- |
| **Phase 1** | DQN & Self-Play Architecture (ActionCodec C3, Observation C4, WallzEnv C5, Negamax Double-DQN) | **Complete** | Checkpoint round-trip, codec involution, legal masks |
| **Phase 2** | DQN Training, Validation & Benchmarks (3×3, 4×4 models, IQM bootstrap metrics) | **Complete** | Gate X-16 passed (>90% sign agreement with $V^*$) |
| **Phase 3** | Engine Performance & Rule Corrections (WallzConfig, canonical position key, $O(1)$ ChainMap clone) | **Complete** | Legal actions $\le 1.54$ ms on 9×9, clone $\le 2.81\ \mu\text{s}$ |
| **Phase 4** | Tier-1 Greedy Baseline (4-tier lexicographical tie order, draw scoring, seeded tie-break) | **Complete** | Seeded deterministic reproducibility |
| **Phase 5** | Tier-2 Minimax / Negamax (PV-first ordering, history heuristic, goal extension, Contract C7) | **Complete** | Search cutoff efficiency, depth-2 latency |
| **Phase 6** | Evaluation Function & Feature Studies (f1–f6 extraction, L2 logistic regression AUC) | **Complete** | Gate X-5 passed (87.29% AUC on 3×3, 95.51% on 4×4) |
| **Phase 7** | Match Runner, Metrics & Logging (Holm-Bonferroni correction, Wilson CIs, table generator) | **Complete** | 14 publication-ready Markdown/LaTeX tables |
| **Phase 8** | State Graph & Exact Solver (Adapter over retrograde engine, cycle/stalemate draw separation) | **Complete** | $K \in \{1, 2, 3\}$ solved for $N=3, 4$ |
| **Phase 9** | Component Testing & Property Verification | **Complete** | Gate X-1 (Hypothesis stateful), Gate X-2 (NetworkX BFS) |
| **Phase 10** | Final System Integration & Acceptance | **Complete** | 301/301 unit, component, rule, and web tests passing |

### Member 2 Track: Engine Enhancements, Verification & Web Demo

| Phase | Title | Commit | Status | Verification Gate |
| :---: | :--- | :---: | :---: | :--- |
| **P01** | Tooling, shared test helpers, action normalisation (F2) | `153fe17` | **Complete** | C-SNAP atomicity, JSON lists accepted, 187 tests passing |
| **P02** | Validate-before-mutate (F1), atomicity & post-terminal tests | `bb77405` | **Complete** | Clock isolation on illegal actions, post-flag raises, 197 tests |
| **P03** | Block duration, hop and REPLACE-form rule suite (F3) | `3f7b397` | **Complete** | Frozen fixtures R7 & Hop, exact K timeline, 218 tests |
| **P04** | No-legal-move draw and wall-budget suite (Task 1C & 1D) | `62c1413` | **Complete** | StateGraph first-hit fixture, budget M=0 audit, 234 tests |
| **P05** | Scenario fixtures, single-source guard, coverage map | `42061df` | **Complete** | 9 shared scenarios, AST single-source guard |
| **P06** | Greedy refactor, side-to-move guard (F4), verification | `93be06f` | **Complete** | Golden replay match, score_actions exposure, 247 tests |
| **P07** | Greedy match campaign across factorial rule cells | `5d86602` | **Complete** | 720 matches saved, summary.json, Markdown & LaTeX tables |
| **P08** | Web backend core, engine adapter, agent registry | `8bf3589` | **Complete** | Flask endpoints /api/games/*, C-STATE serialization |
| **P09** | Web backend think endpoints (Greedy + Minimax + DQN + Solver) | `9e3fa49` | **Complete** | C-EXPL introspection, /api/games/<id>/analyze |
| **P10** | Web backend arena, replays, scenarios, training data | `d11fc0f` | **Complete** | /api/arena/batch, verified replays, training curves |
| **P11–P13** | Frontend React app, Board, Think panels, Arena, Training, Replays, Rules | `a7ee744` | **Complete** | Interactive board, Vitest tests pass, Vite build succeeds |
| **P14** | Documentation and final acceptance | `cae3f11` | **Complete** | Final rules, architecture, web demo guide, 301 tests passing |
| **P15** | (Stretch) Live mini-training stream and Dockerfile | Current | **Complete** | Managed subprocess, SSE stream, multi-stage Dockerfile |

---

## 2. Comprehensive Test Suite (304 Tests Passing)

All tests execute cleanly via pytest:
```bash
.venv/bin/python -m pytest -q
```
*Result*: `301 passed in ~72s`.
Frontend Vitest tests:
```bash
npm --prefix web/frontend test -- --run
```
*Result*: `6 passed (2 test files)`.

---

## 3. Member 2 Regression Coverage Map

| Rule / Mechanic | Core Unit Tests | Invariant / Property Tests | New Dedicated Suites | Shared Scenarios |
| :--- | :--- | :--- | :--- | :--- |
| **Normal Movement** | `test_movement.py` | `test_invariants_stateful.py` | `test_strict_actions.py` | `move_basics.json` |
| **Wall Placement** | `test_walls.py` | `test_invariants_stateful.py` | `test_strict_actions.py` | `wall_place_and_budget.json` |
| **Wall Repositioning (REPLACE)** | `test_replacement.py` | `test_invariants_stateful.py` | `test_rules_blocks_hop.py` | `replace_latest_only.json` |
| **Latest-Wall Restriction** | `test_replacement.py` | `test_invariants_stateful.py` | `test_rules_blocks_hop.py` | `replace_latest_only.json` |
| **One-Time Hop (Rule B / R1)** | `test_hop.py` | `test_invariants_stateful.py` | `test_rules_blocks_hop.py` | `hop_over_wall.json` |
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
| **H-5** | **Member 3** | Hop rule change: `canonical_key` now delegates to `WallzEngine.canonicalize_key` (ownership is kept while a hop is unused). `experiments/results/solves/*` were regenerated. Ablation pilots in `experiments/results/person3_pilot` and the factorial tables were produced under R1 and need re-running. |
| **H-6** | **Member 1** | Minimax results/benchmarks produced before the hop rule change (R1) are stale; agents previously opened with a 2-cell hop. |

---

## 6. Production Artifacts & Checkpoints

- **Released Neural Network Weights**:
  - `checkpoints/release/dqn_3x3/model.pt` + `meta.json`: Double-DQN $3 \times 3$ model (Gate X-16 validated: $>90\%$ sign agreement with $V^*$).
  - `checkpoints/release/dqn_4x4/model.pt` + `meta.json`: Double-DQN $4 \times 4$ model.
- **Exact Retrograde Ground-Truth Solves**:
  - `experiments/results/solves/solve_N3_K1_M0.json` (all 8 factorial rule cells)
  - `experiments/results/solves/solve_N3_K2_M0.json`
  - `experiments/results/solves/solve_N3_K3_M0.json`
  - `experiments/results/solves/solve_N4_K1_M0.json`
  - `experiments/results/solves/solve_N4_K2_M0.json`
  - `experiments/results/solves/solve_N4_K3_M0.json`
- **Publication Tables (`experiments/tables/`)**:
  - 14 LaTeX and Markdown tables generated by `scripts/make_tables.py`.
