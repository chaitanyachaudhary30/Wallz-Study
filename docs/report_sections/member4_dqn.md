# Member 4: Deep Q-Network, self-play and MARL

All numbers point to files in `experiments/results/final_dqn/` (abbreviated `final_dqn/`).
Every file records its commit (dirty = false), command, config, seeds, checkpoint SHA-256,
torch 2.15.0+cpu, Python 3.12.5 and date. Validation details: `docs/reports/stage2_dqn.md`.

## 1. Architecture

`QNetwork` (`agents/dqn/network.py`) is a plain convolution stack: a 3x3 convolution from 14
to 64 channels, then three 3x3 convolutions at 64 channels, each followed by ReLU; then a
linear layer to 256 units with ReLU and a linear output with one Q-value per action id. There
is no BatchNorm and no residual connection. The dueling head exists but is off.

Learning is Negamax Double DQN: because the observation is always from the mover's side, the
value of the next state is the opponent's value, so the target is
$y = r - \gamma (1 - \text{term}) Q_{\text{target}}(s', \arg\max_{a' \in \mathcal{L}(s')} Q_{\text{online}}(s', a'))$,
with $\gamma = 0.99$, Huber loss and gradient clipping at 10. One network plays both seats.

## 2. Observation encoding

The input is 14 planes of size N x N from the mover's side (rows flipped for player 1):
own pawn, opponent pawn, H walls, V walls, own latest wall, opponent latest wall, block
timers as $\tau/K$, own and opponent wall inventory $m/M$, own and opponent hop flag, own and
opponent block flag, and a constant plane of ones.

Checked against the engine on every test position, including a role-swap test showing that
player 1's view equals player 0's view of the swapped position (`docs/reports/stage2_dqn.md`, Section 2).
The audit found two blind spots (`final_dqn/encoding_alias_audit.json`):

- owners of walls that are no longer anyone's latest wall are not encoded, yet HOP needs an
  opponent's wall: 10,619 of 53,334 owner-swapped variants change the legal set. This needs
  M >= 2, so it does not affect the trained cells (3x3 M = 1, 4x4 M = 0);
- the ply and repetition counts are not encoded (plane 13 is constant): 122 of 107,525 sampled
  positions have successors that differ only through game history.

## 3. Action encoding

`ActionCodec` (`src/modified_wallz/actions.py`) gives $8 + 4(N-1)^2 + N^2$ ids (33 on 3x3, 60 on
4x4): four MOVE and four HOP directions in the mover's frame, then WALL slots, REPLACE slots
and BLOCK cells. The legal mask comes from `get_legal_actions()`. Over 107,525 random
positions on 3x3-5x5 and seven rule cells, every one of 5,952,580 ids was masked exactly when
the engine's validator accepted it, with no id collisions (`final_dqn/compat_mask_audit.json`).
For odd N a mirror permutation maps each id to its left-right mirror; it is an involution
and matches engine-built mirrored positions.

## 4. Reward design

Sparse and zero-sum: +1 to the player whose move reaches the goal row, 0 for every other move,
rule draw (repetition, no legal action) or truncation. Goal wins and rule draws end the episode;
the 200-ply cap only truncates and still bootstraps. With the negamax target, a win in $r$
plies has value $\gamma^{r-1}$, which is what the exact solver's `discounted_value` returns.

## 5. Replay buffer

A uniform FIFO ring buffer (20,000 transitions on 3x3 and 4x4), float16 observations, sampled
without replacement in batches of 128 once 1,000 transitions exist. Each transition stores the
next state's legal mask for the masked arg-max. On odd boards each sampled transition is
mirrored with probability 0.5; a test checks that a mirrored sample equals the transition of
the engine-built mirrored position.

## 6. Self-play training

Each episode draws its opponent: 50% the online network, 40% a snapshot (pool of 10, one every
20,000 steps), 10% Greedy; the learner's seat is random. Against the online network both
players' transitions are stored, otherwise only the learner's. Measured in the five 3x3 runs
after the first snapshot, the episode mix was 49.5-50.6% online, 39.4-41.1% snapshot and
9.4-10.1% Greedy (`final_dqn/training_curves_3x3_s4*.json`). Two caveats: before the first
snapshot the snapshot share falls back to the online network (90% / 10%), and Greedy games are
short, so Greedy supplies only 4.0-4.4% of the 3x3 training steps.

## 7. Training configuration

From `configs/training.yaml`, unchanged: lr $10^{-4}$, batch 128, a gradient step every 4 env
steps, target copy every 1,000 gradient steps, $\varepsilon$ 1.0 to 0.05 over the first 10% of
steps.

| Board | Rules | Steps | Seeds | Wall time per seed |
|---|---|---:|---|---|
| 3x3 | A B C on, K = 1, M = 1 | 200,000 | 42, 43, 44, 45, 46 | 15-25 min |
| 4x4 | A B C on, K = 1, M = 0 | 500,000 | 42 | 48 min |

Checkpoints and SHA-256: `checkpoints/final/<board>_s<seed>/manifest.json`. Training loss rose
to about 0.02 in the first 10k steps and fell to 0.00025-0.00064 by the end on 3x3; the
probe's mean max-Q stayed between 0.01 and 0.38, with no divergence
(`final_dqn/training_curves_*.json`). 5x5 ran 500k steps on seed 42 (Section 9.1). 9x9 was
skipped: 5M steps would take at least 23 h (`final_dqn/training_timing_5x5_9x9.json`).

## 8. Checkpoint evaluation

### Exact ground truth (small boards only)

Every non-terminal state of the exact solve: 138,646 on 3x3 (K1 M1) and 4,350 on 4x4 (K1 M0).

| Model | Value-optimal | Rank-optimal | Value-sign agreement | File |
|---|---:|---:|---:|---|
| release 3x3 (old HOP rule) | 78.0% | 75.5% | 82.9% | `final_dqn/release_dqn_3x3_eval.json` |
| retrained 3x3, 5 seeds | 96.0-96.5% (mean 96.2) | 91.1-93.3% (92.5) | 97.7-97.9% (97.8) | `final_dqn/aggregate_3x3_seeds.json` |
| release 4x4 (old HOP rule) | 79.1% | 65.8% | 84.5% | `final_dqn/release_dqn_4x4_eval.json` |
| retrained 4x4, seed 42 | 96.3% | 91.0% | 97.5% | `final_dqn/final_dqn_4x4_s42_eval.json` |

The earlier 96.47% sign agreement (Gate X-16 protocol) falls to 86.80% for the release model
under the final rules, and the retrained models reach 97.6-98.6% on the same protocol.

Against the exact solver, the opening positions are worth 28 wins, 44 draws and 28 losses on 3x3.
All five retrained 3x3 models get exactly that result, so they play these openings
optimally. On 4x4 the retrained model loses 15 positions that are draws
(`final_dqn/opening_values_vs_exact.json`).

## 9. Comparison with Minimax and Greedy

Both seatings, opening_plies 2; 200 games vs Greedy (seeds 0-99), 100 vs Minimax depth 2
(seeds 0-49). Columns: win / rule draw / truncated / loss, in %.

| Model | vs Greedy | vs Minimax d2 |
|---|---|---|
| release 3x3 | 24.0 / 16.5 / 0 / 59.5 | 18 / 10 / 0 / 72 |
| retrained 3x3 (5 seeds) | win 64-76 (mean 66.4), loss 14.5 every seed | win 60-72 (mean 64.4), loss 18-28 |
| release 4x4 | 0 / 6.5 / 0 / 93.5 | 0 / 13 / 0 / 87 |
| retrained 4x4 s42 | 50.0 / 22.0 / 0 / 28.0 | 39 / 21 / 0 / 40 |

Sources: `final_dqn/aggregate_3x3_seeds.json` and the `*_eval.json` files; per-seed rows and
per-seating rates are in `final_dqn/tables.md`. Truncation appears only once: 4% of games for
3x3 seed 44 vs Minimax. Seating matters: on 3x3 the retrained models win 81% vs Greedy as player 0
and 47-71% as player 1 (`final_dqn/final_dqn_3x3_s4*_eval.json`).

### 9.1 Larger board: 5x5 (no exact ground truth)

**500k steps, short of the plan's 2M target, by user decision for compute.** Seed 42, rules
K=3, M=10, `board_5` config; 87 minutes of training (`final_dqn/5x5/training_curves_5x5_s42.json`,
checkpoint hashes in `checkpoints/final/5x5_s42/manifest.json`). Openings follow the
distinct-opening rule: 4 plies, the first 50 distinct openings (seeds 0..79), both seats.

| Opponent | Win / draw / truncated / loss, % | Win % as P0 / P1 |
|---|---|---|
| random | 98 / 2 / 0 / 0 | 98 / 98 |
| greedy | 66 / 28 / 0 / 6 | 72 / 60 |
| minimax:depth=2 | 73 / 10 / 0 / 17 | 74 / 72 |

Source: `final_dqn/5x5/final_dqn_5x5_s42_500k_eval.json`. These are match results only; there
is no solver on 5x5, so they make no claim about optimal play.

Known limit: with M=10 the observation cannot show who owns walls that are no longer anyone's
latest wall, yet HOP may only cross an opponent's wall, so the network can misjudge HOP.
14.6% of owner-swapped 5x5 audit positions have a different legal set under the same
observation (`final_dqn/encoding_alias_audit.json`).

## 10. Limitations and generalization

- **Board and rules are fixed per checkpoint.** The network's input and output sizes depend
  on N, and `meta.json` pins A, B, C, K and M; a 3x3 model cannot play 4x4 or another K or M.
- **Exact claims hold only for 3x3 and 4x4.** The one larger board, 5x5, has one seed at
  500k of the planned 2M steps and only match results (Section 9.1); 9x9 was not trained.
- **Encoding blind spots** (Section 2) will matter on boards with M >= 2: HOP legality
  depends on wall owners the network cannot see.
- **4x4 rests on one seed.** The 3x3 spread across five seeds is small (value-optimal
  96.0-96.5%), but the 4x4 number is a single run.
- **Draw handling is the weak point:** the retrained 4x4 model still loses 15 drawn openings
  against the exact solver, and about half of the 3x3 training episodes end in repetition.
- The release checkpoints were trained under the old HOP rule and should not be cited for the
  final game.
