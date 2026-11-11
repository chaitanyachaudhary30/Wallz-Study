# Stage 2 DQN Report (Member 4: DQN, MARL and self-play, Tasks 1-7)

- **Engine tested:** `main` = `70e2351` (Stage 1 freeze). No engine code changed.
- **Branch:** `bot2-dhruv/dqn-validation`. Every result JSON in `experiments/results/final_dqn/`
  carries its own commit, dirty flag (all `false`), command, config, seeds, checkpoint SHA-256,
  torch/Python versions and date.
- **Machine:** Windows, Python 3.12.5, numpy 2.2.0. Installed for this work:
  `pytest 9.2.0` (`python -m pip install pytest`) and `torch 2.15.0+cpu`
  (`python -m pip install torch --index-url https://download.pytorch.org/whl/cpu`). Nothing else.
- **Shared machine:** every job over 2 minutes ran under `D:\SEM V\Wallz-Study-wt\HEAVY.lock`,
  one at a time. Training ran in lock chunks of at most 27 minutes with the new
  `--max-minutes` / `--resume`.

Exact small-board ground truth (3x3, 4x4) is in Sections 4-5. There are no larger-board
claims in this report; see Section 6 for why 5x5 and 9x9 were not trained.

## 1. Compatibility with the frozen engine (Task 1)

### Mask equals the engine, two ways

| Check | Size | Result | File |
|---|---|---|---|
| For every action id: `legal_mask[id]` == the engine's validator accepts `decode(id)` (applied to a clone); decoded mask set == `get_legal_actions()`; no two legal actions share an id | 107,525 positions (3x3: 45,827, 4x4: 41,741, 5x5: 19,957), 5,952,580 ids, 7 rule cells, biased and uniform random playouts | 0 mismatches, 0 collisions | `compat_mask_audit.json` |
| Legal action types covered | MOVE 245,720, WALL 262,771, REPLACE 440,371, BLOCK 166,628, HOP 10,197 | | `compat_mask_audit.json` |
| Self-play (online, snapshot and Greedy opponents) on 3x3 K1 M1, 4x4 K1 M0, 5x5 K3 M10: every chosen id is in the mask | unit test | no illegal id | `tests/test_dqn_final_compat.py::TestAgentsPlayLegally` |
| Release checkpoints in `run_match` vs random, both seats | unit test | no `illegal_action`, `no_action`, `agent_error` | same |
| Every evaluation match below (4,800 games) | | `forfeit_by_dqn` = 0 in all files | `*_eval.json` |

Rule cells: `default_K3_M10`, `train3_K1_M1`, `train4_K1_M0`, `K2_M3`, `A0_K1_M2`,
`B0C0_M2`, `all_off_M0` (`scripts/final_dqn_lib.py`).

### Rule-by-rule (new tests in `tests/test_dqn_final_compat.py`)

| Rule | Test | Result |
|---|---|---|
| Block lasts K plies | `test_block_plane_tracks_timer_for_k_plies` (K = 1-4): plane 6 = timer/K each ply, MOVE into the cell masked, plane empty after K plies | pass |
| No-legal-move draw, terminal states | `test_terminal_and_no_legal_draw_give_empty_mask`: goal win and the Stage 1 `no_legal_draw.json` fixture give an all-false mask (codec and `WallzEnv`) | pass |
| Configurable wall budget | `test_wall_budget_planes_and_ids` (M = 0, 1, 3, 12): planes 7/8 = m/M, WALL ids exist iff inventory > 0, REPLACE ids remain at 0 inventory | pass |
| REPLACE latest wall only | `test_replace_moves_latest_wall_only`: plane 4 marks the latest wall only; every REPLACE id moves it and keeps the older wall | pass |
| One-time HOP | `test_one_time_hop_ids`: HOP FWD id 4 legal across the opponent's wall, MOVE FWD masked, hop planes go to 0, no hop ids afterwards, none over an own wall | pass |

## 2. Observation encoding (Task 2)

`test_planes_match_engine_fields` checks all 14 planes against the engine on every grid position
(pawns, H/V walls at canonical slots, latest walls, block timer/K, inventories, hop and block
flags, plane 13 = 1.0). `test_role_swap_equivariance` builds the same position with the players'
roles swapped (`swap_roles_key`) and checks that player 1's view equals player 0's view of the
swap, for the observation, the legal id set and every successor's observation. Both pass: the
flip for the opponent is correct.

### Positions that share an observation (`encoding_alias_audit.json`)

| Finding | Count | Cells | Effect |
|---|---|---|---|
| Different positions, same observation, different legal ids (random play) | 4 pairs out of 65,378 distinct positions | 4x4 `K2_M3`, `A0_K1_M2` | wall owner differs |
| Different positions, same observation, same legal ids, different successors | 0 | | |
| Owner-swapped variants (swap the owners of two fixed walls, rebuilt with `restore_state_key`) | 10,619 of 53,334 change the legal set (3x3: 1,098 / 3,421, 4x4: 5,550 / 22,636, 5x5: 3,971 / 27,277) | only cells with M >= 2 | HOP legality depends on the owner |
| Same position, successors differ between real history and fresh history | 122 positions | several | a 5-fold repetition draw the net cannot see |

Causes:

1. **Fixed-wall ownership is not encoded.** Planes 2-3 merge both players' walls; planes 4-5 mark
   only the latest walls. HOP may cross only an opponent's wall. The test
   `test_known_alias_fixed_wall_owner` pins a 5x5 example. In the trained cells this cannot occur:
   3x3 has M = 1 with Rule A, so every wall is a latest wall; 4x4 has M = 0.
2. **Plane 13 is a constant 1.0**, not a turn clock. The ply count and the repetition count are
   invisible, so the net cannot see an approaching repetition draw or the 200-ply runner cap.

Not fixed: a fix needs a new plane or a changed plane 13, a new `OBS_VERSION`, and makes the
release checkpoints unloadable. It matters only for M >= 2 (5x5, 9x9). Recommended before any
5x5 run with M = 10.

## 3. ActionCodec (Task 3)

| Check | Test | Result |
|---|---|---|
| Every legal action has a unique id in range; decode returns the same action | `test_every_legal_action_has_unique_valid_id_and_round_trips` (all grid positions, HOPs included) | pass |
| Illegal ids stay masked | `test_mask_equals_validator_for_every_id` + the 5.95M-id audit | pass |
| Mirror: `perm[perm[i]] == i`; legal ids of the engine-built mirrored position == `perm` of the legal ids; `mirror_observation(obs)` == observation of the mirrored position; decoded actions mirror cell-for-cell | `test_mirror_against_engine_mirrored_positions` (odd N) | pass |
| Replay mirror augmentation == the mirrored transition (obs, action, next obs, next mask) | `test_mirror_augmentation_equals_mirrored_transition` | pass |

Existing tests (`test_action_codec.py`, `test_dqn_encoding.py`) still pass.

## 4. Release checkpoints under the final rules (Task 4)

The release models were trained under the old 2-cell HOP (meta: commit a665b58, about 50k
steps). SHA-256: `dqn_3x3/model.pt` `63ff1130...d18485`, `dqn_4x4/model.pt` `9980e4c5...81c99d`.

Protocol (`scripts/eval_dqn_final.py`): both seatings, opening_plies 2, runner cap 200;
random and greedy on opening seeds 0..99 (200 games), `minimax:depth=2` and the exact solver
on 0..49 (100 games). Ground truth over **every** non-terminal state of the exact solve:
3x3 `N3_A1B1C1_K1_M1` (150,710 states, 138,646 non-terminal), 4x4 `N4_A1B1C1_K1_M0`
(4,512 states, 4,350 non-terminal; with M = 0 no HOP is ever legal). "3-class value"
counts Q > 0.5 as WIN, Q < -0.5 as LOSS, else DRAW. "Legacy" reruns the earlier protocol
unchanged (3x3: Gate X-16, 1000 states, seed 42; 4x4: `evaluation.ablation.agreement`,
300 states, seed 42).

### Materiality (Task 5 trigger)

| Board | Measure | Old file | Old | Final rules | Change | Material? |
|---|---|---|---:|---:|---:|---|
| 3x3 | value-sign agreement (Gate X-16) | `dqn/gate_x16.json` | 96.47% | 86.80% | -9.7 | yes (> 5) |
| 3x3 | win rate vs greedy | `dqn/benchmark_N3.json` | 62.0% | 24.0% | -38.0 | yes (> 10) |
| 4x4 | value-optimal agreement | `dqn/agreement_N4.json` | 93.3% | 75.7% | -17.6 | yes |
| 4x4 | win rate vs greedy | `dqn/benchmark_N4.json` | 50.0% | 0.0% | -50.0 | yes |

**The 96.47% value-sign agreement does not hold under the final rules: it is 86.80%**
(`release_dqn_3x3_eval.json`, `ground_truth_legacy_protocol`). Over all 3x3 states it is 82.9%.

## 5. Retrained models and before/after tables (Task 5)

Config: `configs/training.yaml` unchanged (3x3: 200k steps, K1 M1; 4x4: 500k steps, K1 M0;
batch 128, lr 1e-4, target sync 1000, epsilon decay 10%). It was feasible, so nothing was
changed. 3x3: five seeds 42-46 (firm deliverable). 4x4: seed 42 only (48 minutes per seed on
the shared machine).

| Checkpoint (git-ignored) | Steps | Wall min | SHA-256 `final.pt` | SHA-256 `model.pt` (same weights) |
|---|---:|---:|---|---|
| `checkpoints/final/3x3_s42/final.pt` | 200,020 | 24.7 | `343ccc5ed083f9d55fc6301b810e05bb94033177797538525e5b39840657c32f` | `6a9beff2ff69698c0aced0f9fe8a36e15cc8a00466f3cd9c0c388e65a5208162` |
| `checkpoints/final/3x3_s43/final.pt` | 200,000 | 20.2 | `524eb9f4c549ef1786c928d1ed7cbb9edba787973a674e22f76a7efb767a40a6` | `89d84e793d773cecc0b0390c94142fc0e79b4bf93425ebb4a4e5a24ff0c7047f` |
| `checkpoints/final/3x3_s44/final.pt` | 200,016 | 15.5 | `facb727a8436debb1be5bd38aac38826a586e1ece3e4f859c4ff289b1998a3ac` | `9bb8841d59c941220cfc57eb57500533a2c5774b3c7cb65a96db3079c0d0c3a3` |
| `checkpoints/final/3x3_s45/final.pt` | 200,022 | 15.9 | `648b7606a205ba689f258dccadd67df20f1e9f10d6d555a69300cd1b1cc9bb94` | `22d9c1def3edb11547aae2acb0dcc4745eb6c97b292c2b11448c002b18f333b1` |
| `checkpoints/final/3x3_s46/final.pt` | 200,032 | 15.3 | `c0117dedd258ed917d0cecfdd19cef5b59ad2c224b7c36c8dc00f6cf5221e12f` | `0dce5961a82873c04e00a0663355d360e230211f9f4e9fc28b33ea84665f7dbe` |
| `checkpoints/final/4x4_s42/final.pt` | 500,006 | 48.1 | `3b1445a4b4e9bd440b27f400a9bc72d3bc5d453d143ee39403bb093b20772132` | `3d44aebafc45b476b50deebb93202db7a8e5f12d951fe112657d2128feb1e440` |

Each run has a committed `manifest.json` next to it (commit, dirty flag, full config, seed,
command, steps, both hashes, wall time, torch/Python). The weights are not committed: they stay
git-ignored, and `.gitignore` now tracks only `checkpoints/final/*/manifest.json`. The training
commit of these runs is reconstructed from the commit log at the run's start (the manifests
say so); `train_dqn.py` now writes it into `run_config.json`. Training code is identical across
the three training commits (d251df3, 9ee24ec, fa79c06): only the `run_config.json` write changed.

### Exact ground truth (all non-terminal states)

| Model | Value-opt % | Rank-opt % | Sign agree % | 3-class value % | MAE | Legacy protocol |
|---|---:|---:|---:|---:|---:|---|
| release 3x3 | 78.0 | 75.5 | 82.9 | 65.4 | 0.446 | sign 86.8% (old 96.5%) |
| final 3x3 s42 | 96.0 | 91.1 | 97.7 | 89.5 | 0.176 | sign 98.1% |
| final 3x3 s43 | 96.2 | 93.3 | 97.9 | 89.7 | 0.183 | sign 97.6% |
| final 3x3 s44 | 96.1 | 92.0 | 97.8 | 89.6 | 0.165 | sign 98.6% |
| final 3x3 s45 | 96.5 | 93.2 | 97.7 | 86.4 | 0.180 | sign 98.4% |
| final 3x3 s46 | 96.0 | 93.0 | 97.8 | 86.7 | 0.188 | sign 98.6% |
| release 4x4 | 79.1 | 65.8 | 84.5 | 67.7 | 0.458 | value-opt 75.7 / rank-opt 64.3 (old 93.3 / 70.7) |
| final 4x4 s42 | 96.3 | 91.0 | 97.5 | 94.1 | 0.095 | value-opt 95.0 / rank-opt 89.0 |

3x3 five-seed spread (`aggregate_3x3_seeds.json`): value-optimal 96.0-96.5% (mean 96.2),
rank-optimal 91.1-93.3% (92.5), sign agreement 97.7-97.9% (97.8).
Per-status rates (WIN / LOSS / DRAW) are in each `*_eval.json` under `ground_truth_population`.

### Matches (win / rule draw / truncated / loss, %)

| Model | vs random | vs greedy | vs minimax d2 | vs exact | Mean plies vs greedy |
|---|---|---|---|---|---:|
| release 3x3 | 78.5 / 0.5 / 0 / 21.0 | 24.0 / 16.5 / 0 / 59.5 | 18 / 10 / 0 / 72 | 18 / 0 / 0 / 82 | 8.4 |
| final 3x3 s42 | 99.0 / 0.5 / 0 / 0.5 | 64.0 / 21.5 / 0 / 14.5 | 70 / 12 / 0 / 18 | 28 / 44 / 0 / 28 | 9.2 |
| final 3x3 s43 | 99.0 / 0 / 0 / 1.0 | 64.0 / 21.5 / 0 / 14.5 | 60 / 22 / 0 / 18 | 28 / 44 / 0 / 28 | 9.6 |
| final 3x3 s44 | 99.5 / 0 / 0 / 0.5 | 64.0 / 21.5 / 0 / 14.5 | 60 / 18 / 4 / 18 | 28 / 44 / 0 / 28 | 9.1 |
| final 3x3 s45 | 99.0 / 0 / 0 / 1.0 | 76.0 / 9.5 / 0 / 14.5 | 60 / 22 / 0 / 18 | 28 / 44 / 0 / 28 | 6.8 |
| final 3x3 s46 | 98.5 / 0.5 / 0 / 1.0 | 64.0 / 21.5 / 0 / 14.5 | 72 / 0 / 0 / 28 | 28 / 44 / 0 / 28 | 10.4 |
| release 4x4 | 53.5 / 21.0 / 0 / 25.5 | 0 / 6.5 / 0 / 93.5 | 0 / 13 / 0 / 87 | 0 / 0 / 0 / 100 | 9.0 |
| final 4x4 s42 | 95.0 / 1.0 / 0 / 4.0 | 50.0 / 22.0 / 0 / 28.0 | 39 / 21 / 0 / 40 | 29 / 27 / 0 / 44 | 12.0 |

Per-seating rates and mean lengths for every opponent are in the files and in
`experiments/results/final_dqn/tables.md`. The only truncations: 4 of 100 games, 3x3 s44 vs
minimax.

**Against the exact solver** (`opening_values_vs_exact.json`): the 100 post-opening positions
are worth 28 wins / 44 draws / 28 losses for the DQN side on 3x3, and 29 / 42 / 29 on 4x4.
All five retrained 3x3 models reach exactly that value in every game. The release 3x3 model
does not. The retrained 4x4 model loses 15 positions that are draws (as player 0: 6 draws,
44 losses where the value is 21 draws, 29 losses).

## 6. 5x5 and 9x9 (Task 5)

| Board | Timing run | Measured | Estimate | Status |
|---|---|---|---|---|
| 5x5 (K3 M10, `board_5`) | 10,052 steps | 74.6 s | 2M steps: about 4.1 h per seed (lower bound; games lengthen as play improves) | waiting for the user's decision; not started |
| 9x9 | 2,056-step smoke (batch 32, buffer 5000) | 34.4 s | 5M steps: at least 23 h | skipped: the plan trains 9x9 only if compute permits |

Source: `training_timing_5x5_9x9.json` (built from the timing runs'
`checkpoints/final/timing5|timing9/train_log.jsonl`).

## 7. Self-play and training machinery (Task 6)

| Check | Evidence | Result |
|---|---|---|
| Sampler mix, measured from `sample_opponent_type()` | `test_opponent_sampler_frequencies` (20,000 draws each) | non-empty pool: 0.50 / 0.40 / 0.10 within 0.01; **empty pool: 0.90 online / 0.10 Greedy**, the snapshot share falls back to online |
| Mix in real training (episodes after the first snapshot) | `training_curves_*.json`, `opponent_mix_episodes` | 3x3: online 49.5-50.6%, snapshot 39.4-41.1%, Greedy 9.4-10.1%; 4x4: 50.3 / 39.7 / 10.0 |
| Share of env steps (what the buffer sees) | `opponent_mix_env_steps` | 3x3: Greedy games are short, so Greedy is only 4.0-4.4% of steps; first 20k steps are 90% online |
| Checkpoint save/load | `test_checkpoint_round_trip`: identical Q-values, meta fields present, rules mismatch refused | pass |
| Replay buffer | `test_replay_fifo_overwrite` (FIFO overwrite, sampling without replacement), mirror test above | pass |
| Target network | `test_target_network_hard_update`: equal at start, differs after 1-2 steps, equal after `target_update_freq` | pass |
| Resume | `test_resume_state_round_trip`; a paused + resumed smoke run gave bit-identical weights to an uninterrupted one | pass |
| Loss and Q curves | `training_curves_*.json` (`loss_curve` per 5k steps, `eval_curve` with Q-probe) | 3x3 loss peaks about 0.02 near 10k steps and ends 0.00025-0.00064; probe mean max-Q stays 0.01-0.38; no divergence |
| Episode endings in training | `terminal_reasons` | 3x3: about half of all episodes end in 5-fold repetition |

## 8. Small fixes in this PR

| File | Fix |
|---|---|
| `scripts/train_dqn.py` | Q-probe and elapsed time logged at each eval; `--threads`; `--max-minutes` / `--resume` with full state (networks, optimizer, buffer, snapshot pool, counters, RNG); `run_config.json` with command, config, git, torch, Python |
| `scripts/benchmark_dqn.py` | `truncated` kept as its own column, not a draw; gate label uses the real board size |
| `scripts/run_multi_seed_3x3.py` | no change in the end: my fix (it imported `validate_cell`, which never existed) conflicted with Bot 3's rewrite merged in PR #2/#3, which already fixes it and loads `checkpoints/final/3x3_s<seed>/final.pt`; the merge takes main's version. Not run here |
| `docs/dqn.md` | rewritten to match the code: plain conv stack, plane 13 constant, batch 128, epsilon 10%, -1e9 masking, no -1 reward, snapshot fallback, working commands |
| `README.md` | DQN status, network description, codec location, `run_match` needs `-K 1 -M 1`, training command uses `--board-size` |
| `.gitignore` | tracks `checkpoints/final/*/manifest.json` only |

New: `tests/test_dqn_final_compat.py` (19 tests), `scripts/final_dqn_lib.py`,
`scripts/audit_dqn_compat.py`, `scripts/eval_dqn_final.py`, `scripts/summarize_dqn_training.py`,
`scripts/make_dqn_manifest.py`, `scripts/make_stage2_dqn_tables.py`, `scripts/opening_values_dqn.py`.

## 9. Tests run (one file at a time, `python -m pytest tests/<file> -q`)

| File | Result |
|---|---|
| `test_dqn_final_compat.py` (new) | 19 passed |
| `test_action_codec.py`, `test_dqn_encoding.py`, `test_dqn_agent.py` | 4, 4, 4 passed |
| `test_dqn_trainer.py`, `test_self_play.py`, `test_dqn_runner_integration.py` | 3, 4, 2 passed |
| `test_agent_conformance.py`, `test_wallz_env.py`, `test_match_runner.py`, `test_stage1_engine_freeze.py` | 6, 4, 16, 35 passed |

With torch installed, none of the DQN tests skip.

## 10. Commands

```
python -m pip install pytest
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python scripts/audit_dqn_compat.py --seeds 3:200 4:100 5:40
python scripts/eval_dqn_final.py --checkpoint checkpoints/release/dqn_4x4/model.pt --out release_dqn_4x4_eval.json
python scripts/eval_dqn_final.py --checkpoint checkpoints/release/dqn_3x3/model.pt --out release_dqn_3x3_eval.json
python -m scripts.train_dqn --board-size 3 --seed <42..46> --threads 2 --resume --max-minutes <chunk> --output-dir checkpoints/final/3x3_s<seed>
python -m scripts.train_dqn --board-size 4 --seed 42 --threads 2 --resume --max-minutes <chunk> --output-dir checkpoints/final/4x4_s42
python scripts/eval_dqn_final.py --checkpoint checkpoints/final/3x3_s<seed>/model.pt --out final_dqn_3x3_s<seed>_eval.json
python scripts/eval_dqn_final.py --checkpoint checkpoints/final/4x4_s42/model.pt --out final_dqn_4x4_s42_eval.json
python scripts/summarize_dqn_training.py --run checkpoints/final/<run>
python scripts/make_dqn_manifest.py --run checkpoints/final/<run> --train-commit <commit> --train-dirty false --note "..."
python scripts/opening_values_dqn.py --evals <eval json names>
python scripts/make_stage2_dqn_tables.py > experiments/results/final_dqn/tables.md
python -m scripts.train_dqn --board-size 5 --steps 10000 --eval-freq 10000 --threads 2 --output-dir checkpoints/final/timing5
python -m scripts.train_dqn --board-size 9 --smoke --threads 2 --output-dir checkpoints/final/timing9
```

`--max-minutes` changes only when a run pauses, never the training itself; the exact value per
run is in its `manifest.json`.

**Branch history.** `merge-tree` against `origin/main` (330153f) reported one conflict, in
`scripts/run_multi_seed_3x3.py`. I resolved it with a merge commit (15e34c8) instead of a rebase,
so that the commits cited by every result JSON and manifest (5f00f4f, 34d0b0e, d251df3, 9ee24ec,
fa79c06, cb83693, c07c394) stay in the branch history. After the merge, the DQN test files above
and a smoke run of `eval_dqn_final.py` give the same numbers (4x4 s42 value-optimal 96.34%).

## 11. Independent break attempt

See the section appended below after the review.
