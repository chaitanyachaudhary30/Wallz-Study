# Deep Q-Network (DQN) Self-Play Agent

This file describes the DQN agent as the code implements it (`agents/dqn/`,
`src/modified_wallz/actions.py`, `scripts/train_dqn.py`, `configs/training.yaml`).
Validation against the frozen engine and all measured numbers are in
`docs/reports/stage2_dqn.md` and `experiments/results/final_dqn/`.

---

## 1. Learning target (Negamax Double-DQN)

Modified Wallz is a two-player zero-sum alternating game. Every observation is drawn
from the side to move, so one network plays both seats. A transition
$(s, a, r, s', \text{term}, \text{mask}')$ is trained towards

$$y = r - \gamma (1 - \text{term}) \cdot Q_{\text{target}}\left(s', \arg\max_{a' \in \mathcal{L}(s')} Q_{\text{online}}(s', a')\right)$$

- Reward $r = +1$ to the player whose move reaches the goal row, otherwise $0$
  (draws, truncation and ordinary moves all give 0; the environment never gives $-1$).
- `term` is set for goal wins and rule draws (repetition, no legal action). The runner's
  200-ply cap sets `truncated`, which still bootstraps.
- $\mathcal{L}(s')$ is the legal set from the stored next mask; illegal ids are set to
  $-10^9$ before the arg-max (`agents/dqn/trainer.py`).
- Huber loss ($\delta = 1$), Adam, gradient-norm clip 10, $\gamma = 0.99$.

## 2. Observation (Observation v1, `agents/dqn/encoding.py`)

A float32 tensor of shape $(14, N, N)$, always from the mover's side. For player 1,
rows are flipped ($r \mapsto N-1-r$, wall rows $r \mapsto N-2-r$), so the mover always
starts at canonical row 0 and heads to row $N-1$.

| Plane | Content |
| :---: | :--- |
| 0 | mover pawn (1.0 at its cell) |
| 1 | opponent pawn |
| 2 | every `H` wall, 1.0 at its anchor slot $(r, c)$, both owners |
| 3 | every `V` wall, same layout |
| 4 | mover's latest (replaceable) wall anchor |
| 5 | opponent's latest wall anchor |
| 6 | active blocks: remaining timer $\tau / K \in (0, 1]$ at the cell |
| 7 | mover wall inventory $m / M$ (0 when $M = 0$), constant plane |
| 8 | opponent wall inventory $m / M$, constant plane |
| 9 | mover hop still available (1/0), constant plane |
| 10 | opponent hop still available |
| 11 | mover block still available |
| 12 | opponent block still available |
| 13 | constant 1.0 (board indicator; with zero padding it marks the board edge) |

What the observation does **not** contain (measured in
`experiments/results/final_dqn/encoding_alias_audit.json`):

- the owner of a wall that is no longer anyone's latest wall. HOP legality depends on
  it, so two positions can share an observation and differ in legal actions. This
  cannot happen in the trained cells (3x3 with $M = 1$, 4x4 with $M = 0$), where every
  wall is a latest wall; it does happen with $M \ge 2$;
- the ply count and the repetition count. Channel 13 is constant, so the network cannot
  see an approaching 5-fold repetition draw or the runner's 200-ply cap;
- the rules (A, B, C, K, M). A checkpoint is valid only for the rule cell in its
  `meta.json`; `load_checkpoint(expected_rules=...)` checks it.

## 3. Action codec (ActionCodec v1, `src/modified_wallz/actions.py`)

$|\mathcal{A}| = 8 + 4(N-1)^2 + N^2$: 33 ($N=3$), 60 ($N=4$), 97 ($N=5$), 345 ($N=9$).

| Ids | Action |
| :--- | :--- |
| 0-3 | MOVE `FWD`, `BACK`, `RIGHT`, `LEFT` (mover frame) |
| 4-7 | HOP one cell across an opponent's wall, same four directions |
| $8 \ldots 8 + 2(N-1)^2 - 1$ | WALL at slot $(r, c)$: $8 + 2(r(N-1) + c) + [o = V]$ |
| next $2(N-1)^2$ | REPLACE (move the latest wall) to slot $(r, c, o)$, same layout |
| last $N^2$ | BLOCK cell $(r, c)$: offset $+ rN + c$ |

The legal mask is built from `engine.get_legal_actions()`. For odd $N$,
`mirror_permutation()` maps each id to its left-right mirror (an involution); the
replay buffer uses it for augmentation.

## 4. Network (`agents/dqn/network.py`)

A plain convolution stack, no BatchNorm and no residual connections:

1. Conv $3 \times 3$, 14 $\to$ 64 channels, ReLU.
2. `blocks` = 3 more Conv $3 \times 3$, 64 $\to$ 64, each followed by ReLU.
3. Flatten, Linear ($64 N^2 \to 256$), ReLU, Linear ($256 \to |\mathcal{A}|$).

A dueling head exists (`dueling: true`) but is off in every config.

## 5. Self-play and replay (`agents/dqn/self_play.py`, `agents/dqn/replay_buffer.py`)

- **Opponent per episode:** 50% the online network, 40% a snapshot, 10% `GreedyAgent`.
  Snapshots of the online network are taken every `snapshot_interval` = 20,000 env
  steps into a FIFO pool of 10. While the pool is empty the snapshot share falls back
  to the online network, so the first 20,000 steps run 90% online / 10% Greedy.
- **Seats:** the learner's seat is drawn at random each episode.
- **Storage:** both players' transitions against the online network; only the
  learner's transitions against a snapshot or Greedy.
- **Replay buffer:** uniform FIFO ring buffer, observations stored as float16,
  sampled without replacement; capacity 20,000 (3x3, 4x4) or 100,000 (5x5, 9x9);
  learning starts at 1,000 transitions.
- **Mirror augmentation:** for odd $N$ each sampled transition is mirrored with
  probability 0.5 (observation, action, next observation, next mask).

## 6. Training configuration (`configs/training.yaml`)

Shared: lr $10^{-4}$, batch 128, $\gamma$ 0.99, one gradient step every 4 env steps,
target network hard-copied every 1,000 gradient steps, Double DQN on, dueling off,
$\varepsilon$ linear from 1.0 to 0.05.

| Board | Rules | Steps | $\varepsilon$ decay steps | Buffer | Eval every |
| :--- | :--- | ---: | ---: | ---: | ---: |
| 3x3 | A B C on, K=1, M=1 | 200,000 | 20,000 (10%) | 20,000 | 10,000 |
| 4x4 | A B C on, K=1, M=0 | 500,000 | 50,000 (10%) | 20,000 | 25,000 |
| 5x5 | A B C on, K=3, M=10 | 2,000,000 | 200,000 (10%) | 100,000 | 50,000 |
| 9x9 | A B C on, K=3, M=10 | 5,000,000 | 500,000 (10%) | 100,000 | 100,000 |

Command (run from the repo root; `python -m` puts the root on the import path):

```
python -m scripts.train_dqn --board-size 3 --seed 42 --threads 2 --output-dir checkpoints/final/3x3_s42
```

Each evaluation appends a line to `train_log.jsonl` with win rates vs Greedy and Random
(opening_plies 0, so these in-training numbers are a coarse signal only), the
Q-probe statistics and the elapsed time, and saves `step_<n>.pt`.

## 7. Checkpoints

- `checkpoints/release/dqn_3x3/` and `checkpoints/release/dqn_4x4/` were trained under
  the **old HOP rule** (a 2-cell jump, replaced in 4e346a8). Their `meta.json` says
  commit a665b58, seed 42, about 50,000 steps; the command that produced them is not
  recorded. They are kept unchanged for comparison only.
- Models retrained under the frozen rules are listed, with SHA-256 and evaluation
  files, in `docs/reports/stage2_dqn.md`.

Loading in a match:

```
python run_match.py --p0 dqn:path=<checkpoint>/model.pt --p1 greedy --board-size 3 -K 1 -M 1
```

`-K` and `-M` must match the checkpoint's `meta.json` (the CLI defaults are K=3, M=10).

`load_checkpoint` refuses a checkpoint whose codec or observation version, board size
or rules (A, B, C, K, M) differ from the match.
