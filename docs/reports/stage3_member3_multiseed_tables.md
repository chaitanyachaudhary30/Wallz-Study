# Stage 3 report: DQN multi-seed study, agreement rerun, final tables (Member 3)

- **Base:** `main` = `d91549c` (PRs #2, #5, #6 merged). Branch `bot3-chaitanya/multiseed-tables`.
- **No engine, solver or agent code changed.**
- **Result commits (all clean, recorded in each file's `source`):** `d7c8571` (multi-seed),
  `265b261` (agreement), `2b5d9a0` (tables, `final_eval/tables/manifest.json` `run.source`).
- **Checkpoints:** Bot 2's `D:\SEM V\Wallz-Study-wt\bot2-dqn\checkpoints\final\3x3_s<seed>\final.pt`,
  seeds 42-46. Each SHA-256 equals the committed `checkpoints/final/3x3_s<seed>/manifest.json`
  (`multi_seed_3x3.json` `seeds[*].manifest_sha256_match` all `true`; `agreement_N3_K1.json`
  `dqn_checkpoints`). Both scripts stop with an error on a mismatch.
- **Machine:** Windows 11, Python 3.12.5, torch 2.15.0+cpu. Both jobs ran under `HEAVY.lock`.

## Code changes

| File | Change |
|---|---|
| `scripts/run_multi_seed_3x3.py` | `--manifest-dir` SHA-256 check; plays each **distinct** opening once per seating (`--opening-plies 2`, `--distinct-openings`), Random opponent with `--random-seeds`; records `opening_map` and `games_per_seed`. Replaces `--matches`. |
| `scripts/agreement_final.py` (new) | per cell: solve, sample once, score every agent on the same positions; one graph in memory at a time |
| `evaluation/ablation.py` | `sample_nonterminal()`; `agreement(..., nodes=)` to reuse a sample |
| `scripts/make_tables.py` | reads only `final_eval`, `final_minimax`, `final_minimax_hop`, `final_dqn`; stale folders only with `--include-stale`, titled "STALE baseline"; 4-ply ablation titled primary, 2- and 6-ply titled sensitivity; draw-rate column in ablation cell tables; new multi-seed, Minimax-vs-Greedy and DQN tables; writes `manifest.json` (inputs per table); default output `experiments/results/final_eval/tables/` |
| `tests/test_multi_seed_aggregation.py` | +1 test: manifest SHA mismatch exits (5 tests pass) |

Why the multi-seed design changed: the merged script played 100 games per opponent from
the start position (`opening_plies` 0). DQN and Greedy are deterministic, so against Greedy
those 100 games are 2 distinct games, each repeated 50 times. Following the Task 4 rule (count
distinct openings, no replica games), each distinct opening is played once. 3x3 has 8 distinct
2-ply openings (seeds 0, 1, 2, 4, 5, 6, 7, 9); 2 plies is the longest opening that cannot reach
a goal on 3x3. The `--max-plies 100` cap is kept from the merged script and recorded.

## Commands

```
python scripts/run_multi_seed_3x3.py --checkpoint-dir "D:/SEM V/Wallz-Study-wt/bot2-dqn/checkpoints/final" --manifest-dir checkpoints/final --seeds 42 43 44 45 46 --opening-plies 2 --distinct-openings 40 --random-seeds 5 --max-plies 100 --out experiments/results/final_eval/multi_seed_3x3.json
python scripts/agreement_final.py --Ms 0 1 --max-states 300 --seed 42 --dqn-dir "D:/SEM V/Wallz-Study-wt/bot2-dqn/checkpoints/final" --manifest-dir checkpoints/final --out experiments/results/final_eval/agreement_N3_K1.json
python scripts/make_tables.py
```

The agreement job was run twice: the first run started while `scripts/make_tables.py` was
being edited, so it recorded `dirty: true`. It was rerun from a clean commit; all 128 rows and
all 16 sample hashes are identical, and only the clean run is committed.

## 1. Multi-seed study (`final_eval/multi_seed_3x3.json`)

Full per-seed table and reading: `docs/report_sections/member3_eval.md` Section 6.

| Measure | Mean | SD | 95% t-interval | Min | Max | Largest leave-one-out shift |
|---|---:|---:|---|---:|---:|---:|
| sign agreement (1,000 decided states) | 0.9836 | 0.0027 | 0.9802-0.9870 | 0.9800 | 0.9866 | 0.0009 |
| value MAE | 0.1789 | 0.0086 | 0.1683-0.1896 | 0.1655 | 0.1871 | 0.0034 |
| vs Greedy win rate (16 games/seed) | 0.650 | 0.056 | 0.581-0.719 | 0.625 | 0.750 | 0.025 |
| vs Greedy draw rate | 0.225 | 0.056 | 0.156-0.294 | 0.125 | 0.250 | 0.025 |
| vs Greedy loss rate | 0.125 | 0 | - | 0.125 | 0.125 | 0 |
| vs Greedy truncation rate | 0 | 0 | - | 0 | 0 | 0 |
| vs Greedy mean plies | 9.93 | 1.65 | 7.87-11.98 | 7.38 | 11.88 | 0.64 |
| vs Random win rate (80 games/seed) | 0.980 | 0.0068 | 0.9715-0.9885 | 0.975 | 0.9875 | 0.0019 |
| vs Random draw / truncation rate | 0 | 0 | - | 0 | 0 | 0 |
| vs Random mean plies | 10.11 | 0.82 | 9.09-11.12 | 9.43 | 11.46 | 0.34 |

Pooled: Greedy 52-10-18-0 of 80 (win 0.650, Wilson 0.541-0.745; draw 0.225, 0.147-0.328;
truncation 0, 0-0.046). Random 392-8-0-0 of 400 (win 0.980, 0.961-0.990).

**Answer:** DQN performance is consistent across the five seeds. No seed moves any mean by
more than its SD. The limit is the small number of distinct games against Greedy (16 per seed).

**Note for the DQN owner (not changed here):** `final_dqn/final_dqn_3x3_s*_eval.json` reports
200 games against Greedy with `opening_plies 2`. On 3x3 at most 8 distinct 2-ply openings exist,
so those 200 games hold at most 16 distinct games, and their Wilson intervals treat 200 as the
sample size. Its s42 numbers (win 64.0%, draw 21.5%) differ from the 16 distinct games here
(62.5%, 25.0%) because repeated openings weight the same games unequally.

## 2. Agreement rerun (`final_eval/agreement_N3_K1.json`)

Table and reading: `docs/report_sections/member3_eval.md` Section 6. Every agent in a cell is
scored on the same sampled positions (`cells[*].sampled_keys_sha256`): 300 per cell, or all 33
non-terminal positions in the 45-state cells. DQN is scored only in N3 A1B1C1 K1 M1, the only
cell its checkpoints were trained for (other rows: `applicable: false`).

## 3. Tables (`final_eval/tables/`)

17 tables (`.md` and `.tex`) from one run; `manifest.json` lists each table's inputs and the
commit. None reads an old folder. With `--include-stale` three more are made, each titled
"STALE baseline". Run again after the 5x5 DQN and Bot 4's Greedy results land; new final
folders are added next to `FINAL_EVAL` ... `FINAL_DQN` near the top of `scripts/make_tables.py`.

| Table | Input |
|---|---|
| `solve_N3_K1_M0`, `_M1`, `solve_N4_K1_M0`, `_M1` | `final_eval/solve_*.json` |
| `ablation_primary_{cells,effects,draws}` | `final_eval/ablation_N5_K3_M10_open4/` (**primary**) |
| `ablation_sens_open2_*`, `ablation_sens_open6_*` | 2- and 6-ply runs (**sensitivity**) |
| `agreement_N3_K1` | `final_eval/agreement_N3_K1.json` |
| `multi_seed_3x3` | `final_eval/multi_seed_3x3.json` |
| `minimax_vs_greedy` | every `*_summary.json` in `final_minimax/` and `final_minimax_hop/` |
| `dqn_3x3_seeds` | `final_dqn/aggregate_3x3_seeds.json` |

## Tests

| File | Result |
|---|---|
| `tests.test_multi_seed_aggregation` | 5 pass |
| `tests.test_match_runner` | 17 pass |

## Not done

- No new Greedy (Bot 4) or 5x5 DQN tables yet: those results do not exist.
- `minimax_vs_greedy` lists every summary in the two Minimax folders, including diagnostic runs;
  the owner's report says which is primary.
- The `.tex` tables were not compiled with LaTeX.
