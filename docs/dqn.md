# Deep Q-Network (DQN) MARL Self-Play Formulation & Implementation

This document describes the design, network architecture, observation encoding, action codec, learning algorithm, and self-play training curriculum for the **Multi-Agent Deep Q-Network** (`DQNAgent`) baseline in Modified Wallz.

---

## 1. Mathematical Formulation & Negamax Double-DQN

Modified Wallz is a sequential, two-player zero-sum Markov Game. Under the canonical perspective where the side to move is always viewed from a standardized coordinate frame:
- The active player seeks to maximize their expected discounted return.
- The opponent on the subsequent ply seeks to maximize *their* return, which corresponds to the negative of the mover's payoff.

### Negamax Double-DQN Target Recurrence
To prevent overestimation bias in high-branching action spaces while handling the alternating two-player dynamic, transitions $(s, a, r, s', \text{term}, \text{mask}')$ are updated via a **Negamax Double-DQN** target:

$$y = r - \gamma (1 - \text{term}) \cdot Q_{\text{target}}\left(s', \arg\max_{a' \in \mathcal{L}(s')} Q_{\text{online}}(s', a')\right)$$

where:
- $r = +1.0$ if the action immediately reaches the goal row, $-1.0$ if penalized/forfeited, and $0.0$ otherwise.
- $\mathcal{L}(s')$ denotes the set of legal actions at state $s'$, enforced by masking invalid actions with $\text{mask}'$.
- Illegal action values are masked with $-\infty$ (`torch.finfo(float32).min`) prior to the $\arg\max$ selection.
- Huber loss (Smooth L1) with gradient clipping at 10.0 is used for stable backpropagation.

---

## 2. Spatial Observation Representation (Contract C4)

The state is encoded as a float32 tensor of shape $(14, N, N)$ from the perspective of the player to move:

| Plane | Description | Coordinate Semantics |
| :---: | :--- | :--- |
| **0** | Mover pawn position | Binary indicator (1.0 at mover `(r, c)`, 0.0 elsewhere) |
| **1** | Opponent pawn position | Binary indicator (1.0 at opponent `(r, c)`, 0.0 elsewhere) |
| **2** | Horizontal walls (`'H'`) | 1.0 at wall anchor `(r, c)` |
| **3** | Vertical walls (`'V'`) | 1.0 at wall anchor `(r, c)` |
| **4** | Mover latest wall indicator | 1.0 at anchor of mover's reusable wall $r_i$ |
| **5** | Opponent latest wall indicator | 1.0 at anchor of opponent's reusable wall $r_{-i}$ |
| **6** | Active temporary cell blocks | Normalized remaining decay timer: $\tau(r, c) / K \in (0, 1]$ |
| **7** | Mover wall inventory | Uniform scalar fill: $m_{\text{mover}} / M \in [0, 1]$ |
| **8** | Opponent wall inventory | Uniform scalar fill: $m_{\text{opp}} / M \in [0, 1]$ |
| **9** | Mover hop ability available | Uniform binary fill: 1.0 if available, 0.0 if consumed |
| **10** | Opponent hop ability available | Uniform binary fill: 1.0 if available, 0.0 if consumed |
| **11** | Mover block ability available | Uniform binary fill: 1.0 if available, 0.0 if consumed |
| **12** | Opponent block ability available | Uniform binary fill: 1.0 if available, 0.0 if consumed |
| **13** | Elapsed turn clock | Normalized turn progress: $t / T_{\max} \in [0, 1]$ |

### Canonical Perspective Transformation
When Player 1 is to move, the board is transformed via a vertical reflection:
$$r \mapsto N - 1 - r \quad (\text{wall slots: } r \mapsto N - 2 - r)$$
This ensures that the mover always starts at the top and strives to reach the bottom goal row, enabling a single policy network to control both seats symmetrically.

---

## 3. Discrete Action Codec (Contract C3)

The discrete action space is flattened into a compact 1D index space of size:
$$|\mathcal{A}| = 8 + 4(N - 1)^2 + N^2$$
- $N = 3$: 33 actions
- $N = 4$: 60 actions
- $N = 5$: 97 actions
- $N = 9$: 345 actions

### Action ID Allocation
1. **MOVE (4 actions)**: IDs $0 \dots 3$ corresponding to relative directions `FWD`, `BACK`, `RIGHT`, `LEFT`.
2. **HOP (4 actions)**: IDs $4 \dots 7$ corresponding to 2-cell jumps in `FWD`, `BACK`, `RIGHT`, `LEFT`.
3. **WALL ($2(N-1)^2$ actions)**: IDs $8 \dots 8 + 2(N-1)^2 - 1$ mapped as $8 + 2(r(N-1) + c) + [o == \text{'V'}]$.
4. **REPLACE ($2(N-1)^2$ actions)**: Offset by $+2(N-1)^2$ following the WALL block.
5. **BLOCK ($N^2$ actions)**: IDs mapped as $+r \cdot N + c$ across all grid cells.

### Mirror Permutation
For odd $N$, horizontal symmetry is preserved via a closed-form action index involution ($\pi(a)$), allowing $2\times$ data augmentation in the replay buffer without environment re-simulation.

---

## 4. Deep Neural Network Architecture

The Q-network (`QNetwork` in `agents/dqn/network.py`) is structured as:
1. **Input Convolution**: $3 \times 3$ conv (14 channels $\to$ 64 channels), BatchNorm, ReLU.
2. **Residual Backbone**: 3 residual blocks with skip connections and ReLU activations.
3. **Flatten & Dense Head**:
   - Linear layer (flattened features $\to 256$ units) with ReLU.
   - Output projection: Linear layer ($256 \to |\mathcal{A}|$ units) predicting state-action values $Q(s, a)$.

---

## 5. Self-Play Training Curriculum & Replay Buffer

To prevent policy cycling and catastrophic forgetting in self-play:
- **Opponent Mixture**:
  - $50\%$ against the current online network.
  - $40\%$ against a historical snapshot pool (the 10 most recent checkpoints saved every 20k steps).
  - $10\%$ against the Tier-1 `GreedyAgent` baseline to enforce spatial discipline.
- **Randomized Seating**: The learner randomly occupies Player 0 or Player 1 each episode.
- **Replay Buffer**: Uniform FIFO buffer holding $100,000$ transitions stored as float16 observations, sampled with batch size 64.
- **Exploration Schedule**: Linear $\epsilon$-greedy decaying from $1.0 \to 0.05$ over the first $25\%$ of training steps.

---

## 6. Released Checkpoints & Validation

Trained models are versioned and released in [`checkpoints/release/`](file:///home/dhruvv_w/Documents/FAI_project/checkpoints/release/):
- **3×3 Board Model**: `checkpoints/release/dqn_3x3/model.pt` (trained 50,000 steps with Double-DQN).
  - Validated on Gate X-16 against exact retrograde solver ground truth ($>90\%$ sign agreement with $V^*$).
- **4×4 Board Model**: `checkpoints/release/dqn_4x4/model.pt` (trained 50,000 steps).

### Usage Example
```python
from evaluation.match_runner import make_agent, MatchConfig
from wallz_engine import WallzEngine

config = MatchConfig(board_size=3, rules={'A': True, 'B': True, 'C': True, 'K': 1, 'M': 1})
dqn_agent, identity = make_agent("dqn:path=checkpoints/release/dqn_3x3/model.pt", player_id=0, config=config)

engine = WallzEngine(board_size=3, rules=config.rules)
action = dqn_agent.select_action(engine)
print("DQN selected action:", action)
```
