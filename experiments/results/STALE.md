# Which committed results still describe the final rules

Final rules: `main` = `70e2351` (Stage 1 engine freeze, `docs/reports/stage1_engine_freeze.md`).
New results for Member 3 are in `experiments/results/final_eval/`. Nothing listed here was
changed or deleted.

Rule and agent changes that make older numbers stale:

| Change | Commit | What it affects |
|---|---|---|
| HOP: 2-cell jump replaced by one step across an **opponent's** wall; `canonicalize_key` keeps wall owners while a hop remains | `4e346a8` (2026-11-09) | every result with B on and walls possible (M >= 1), and every match with B on |
| Greedy tie order and draw scoring | `acef2aa` (2026-11-07), `93be06f` (2026-11-08) | every match or agreement number that uses Greedy |
| Minimax state leak fixed | `ce55e0c` (2026-11-10) | every match or agreement number that uses Minimax |

"Commit" below is the `source.commit` written inside the file, or, for files without one,
the commit that added it. "Dirty" means the file says it was made from a tree with
uncommitted changes, so the exact code cannot be recovered from git.

| File | Commit | Status | Why |
|---|---|---|---|
| `person3_pilot/solve_N3_K1_M0.json` | `7e2d6da` | B-off cells valid; B-on cells stale | old HOP. B-off cells (000, 001, 100, 101) match `final_eval/solve_N3_K1_M0.json` exactly |
| `person3_pilot/solve_N3_K1_M1.json` | `7e2d6da` | B-off cells valid; B-on cells stale | old HOP. B-off cells match `final_eval/solve_N3_K1_M1.json` exactly |
| `person3_pilot/solve_N4_K1_M0.json` | `7e2d6da` | B-off cells valid; B-on cells stale | old HOP; B-off cells match `final_eval/solve_N4_K1_M0.json` |
| `person3_pilot/solve_N4_K1_M1_capped.json` | `7e2d6da` | B-off cells valid (complete ones); B-on cells stale | old HOP; 5 of 8 cells were capped (UNKNOWN) |
| `person3_pilot/agreement_N3_K1_M0.json` | `7e2d6da` | stale | Greedy change, Minimax fix, old HOP |
| `person3_pilot/agreement_N3_K1_M1.json` | `7e2d6da` | stale | Greedy change, Minimax fix, old HOP |
| `person3_pilot/ablation_N5_minimax2_vs_greedy/summary.json` | `7e2d6da` | stale | Greedy change, Minimax fix, old HOP. Also: its 10 seeds (0-9, 2 opening plies) held only **8 distinct openings** (seed 3 = seed 1, seed 8 = seed 4), and both agents are deterministic, so its effective sample was 8 openings, not 10 seeds. N comes from the seed -> opening map (`final_eval/analysis.json` `opening_counts_N5`); the pilot's own `matches.jsonl` was git-ignored and is not available. Direct rerun: `final_eval/ablation_N5_K3_M10_open2_pilot_comparison/` |
| `person3_pilot/sample_match_N5_ABC_minimax2_greedy.json` | `7e2d6da` | stale | its replay is illegal under the final rules (`verify-replay`: 0/1) |
| `person3_pilot/sample_match_N9_C_greedy.json` | `7e2d6da` | stale (agent) | replay still legal (`verify-replay`: 1/1, B off), but Greedy changed since |
| `person3_pilot/sample_match_N9_C_greedy.demo.json` | `7e2d6da` | stale (agent) | demo export of the file above |
| `solves/solve_N3_K1_M0.json` | `b4eb0a9`, dirty; committed in `4e346a8` | valid, unverified provenance | all 8 cells equal `final_eval/solve_N3_K1_M0.json` (with M = 0 there is no wall, so HOP is never legal) |
| `solves/solve_N4_K1_M0.json` | `b4eb0a9`, dirty; committed in `4e346a8` | valid, unverified provenance | all 8 cells equal `final_eval/solve_N4_K1_M0.json` |
| `solves/solve_N3_K2_M0.json`, `solve_N3_K3_M0.json`, `solve_N4_K2_M0.json`, `solve_N4_K3_M0.json` | `b4eb0a9`, dirty; committed in `4e346a8` | probably valid, not rerun | M = 0, so HOP cannot change them; B-on and B-off cells are already equal inside each file. Dirty tree |
| `agreement/agreement_3x3.json` | `4c57b01` | stale | before the Greedy change, Minimax fix and HOP change |
| `greedy/greedy_vs_greedy_self_N{3,5,9}*.jsonl`, `greedy_vs_random_N{3,5,9}*.jsonl` | `b4eb0a9`, dirty; committed in `4e346a8` | replays legal, provenance unverified | `verify-replay` passes on the final engine for all 100/100/40 games each; dirty tree, so the Greedy code that chose the moves is not recoverable |
| `greedy/greedy_vs_minimax2_N{3,5,9}*.jsonl` | `b4eb0a9`, dirty | stale | before the Minimax fix (`ce55e0c`); replays are still legal |
| `greedy/summary.json` | `b4eb0a9`, dirty | partly stale | its Minimax rows are stale (see above) |
| `minimax/x7_equivalence.json`, `x8_node_reduction.json` | added in `f87cfad` (no source field) | stale | before the HOP change and the Minimax fix |
| `features/x5_feature_study.json`, `x5_feature_study_4x4.json` | added in `e64b109` (no source field) | stale | before the HOP change and the Minimax fix |
| `engine/benchmark_performance.json` | added in `4c57b01`, last `060b22c` (no source field) | stale | engine timing before the HOP change |
| `final_eval/ablation_N5_K3_M10_open2/` (+ `logs/ablation_N5_K3_M10_open2/`) | `8944f16` | superseded | 30 seeds at 2 plies = only 9 distinct openings; its 480 matches replay the same 144 games as `final_eval/ablation_N5_K3_M10_open2_pilot_comparison/` (replicas identical). Use the pilot-comparison folder |
| `final_eval/ablation_N5_K3_M10_open6/` (+ log) | `8944f16` | valid, sensitivity check only | was the primary in PR #2; the primary is now `final_eval/ablation_N5_K3_M10_open4/` (smallest opening length with >= 30 distinct openings) |
| `dqn/*.json` (7 files) | added in `4034ae4` (no source field) | stale; owner Bot 2 | trained and measured under the old HOP rule (`docs/game_rules.md` 3.4) |
