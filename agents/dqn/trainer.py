"""DQN Trainer for Modified Wallz (P1.7).

Implements Negamax Double-DQN learner with masked action selection, Huber loss,
target network synchronization, and epsilon schedule.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    torch = None
    nn = None
    optim = None

from agents.dqn.network import QNetwork
from agents.dqn.replay_buffer import ReplayBuffer
try:
    from modified_wallz.actions import ActionCodec
except ImportError:
    from src.modified_wallz.actions import ActionCodec


@dataclass
class DQNConfig:
    """Hyperparameter and runtime configuration for DQN learner."""
    board_size: int = 3
    rules: dict = field(default_factory=lambda: {'A': True, 'B': True, 'C': True, 'K': 3, 'M': 10})
    lr: float = 1e-4
    gamma: float = 0.99
    batch_size: int = 128
    target_update_freq: int = 1000  # Gradient steps between target network hard updates
    train_freq: int = 4             # Env steps per gradient step
    buffer_capacity: int = 100_000
    min_buffer_size: int = 1_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 10_000
    grad_clip: float = 10.0
    huber_delta: float = 1.0
    double_dqn: bool = True
    dueling: bool = False
    channels: int = 64
    blocks: int = 3
    hidden: int = 256
    mirror_prob: float = 0.5
    seed: int = 42
    device: str = "cpu"


class DQNTrainer:
    """Trainer orchestrating Negamax Double-DQN updates on QNetwork."""

    def __init__(self, config: DQNConfig):
        self.config = config
        self.board_size = config.board_size
        self.num_actions = ActionCodec.get_num_actions(config.board_size)
        self.device = torch.device(config.device) if torch is not None else None

        # Seed initialization
        self.set_seed(config.seed)

        # Networks
        self.online_net = QNetwork(
            board_size=self.board_size,
            num_actions=self.num_actions,
            channels=config.channels,
            blocks=config.blocks,
            hidden=config.hidden,
            dueling=config.dueling,
        ).to(self.device)

        self.target_net = QNetwork(
            board_size=self.board_size,
            num_actions=self.num_actions,
            channels=config.channels,
            blocks=config.blocks,
            hidden=config.hidden,
            dueling=config.dueling,
        ).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=config.lr)
        self.loss_fn = nn.HuberLoss(delta=config.huber_delta)

        # Replay Buffer
        self.buffer = ReplayBuffer(
            capacity=config.buffer_capacity,
            board_size=self.board_size,
            num_actions=self.num_actions,
        )

        # Mirror permutation (if board size is odd)
        if self.board_size % 2 == 1:
            self.mirror_perm = ActionCodec(self.board_size).mirror_permutation()
        else:
            self.mirror_perm = None

        # Tracking counters
        self.total_env_steps = 0
        self.total_grad_steps = 0

    def set_seed(self, seed: int):
        """Sets random seeds across python, numpy and torch."""
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        if torch is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

    def get_epsilon(self, step: int | None = None) -> float:
        """Computes linearly scheduled exploration epsilon."""
        if step is None:
            step = self.total_env_steps

        cfg = self.config
        if step >= cfg.epsilon_decay_steps:
            return cfg.epsilon_end
        frac = float(step) / float(cfg.epsilon_decay_steps)
        return cfg.epsilon_start - frac * (cfg.epsilon_start - cfg.epsilon_end)

    def select_action(
        self,
        obs: np.ndarray,
        legal_mask: list[bool] | np.ndarray,
        epsilon: float = 0.0,
        eval_mode: bool = False,
    ) -> int:
        """Selects an action using masked epsilon-greedy policy.

        Args:
            obs: Observation array of shape (14, N, N).
            legal_mask: Boolean array/list of length num_actions.
            epsilon: Exploration probability (ignored if eval_mode=True).
            eval_mode: If True, executes pure greedy choice over legal actions.

        Returns:
            Chosen discrete action ID.
        """
        legal_indices = [i for i, is_legal in enumerate(legal_mask) if is_legal]
        if not legal_indices:
            raise ValueError("Cannot select action from empty legal_mask.")

        # Exploration branch
        if not eval_mode and random.random() < epsilon:
            return random.choice(legal_indices)

        # Exploitation branch: forward pass through online network
        self.online_net.eval()
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_vals = self.online_net(x).squeeze(0).cpu().numpy()

        # Apply legal action mask: illegal actions get -inf
        masked_q = np.full_like(q_vals, -np.inf)
        masked_q[legal_indices] = q_vals[legal_indices]

        # Greedy choice with deterministic tie-breaking (lowest index)
        best_val = -np.inf
        best_id = legal_indices[0]
        for aid in legal_indices:
            val = masked_q[aid]
            if val > best_val:
                best_val = val
                best_id = aid

        return best_id

    def train_step(self) -> float | None:
        """Executes a single gradient update step using Negamax Double-DQN."""
        if len(self.buffer) < max(self.config.min_buffer_size, self.config.batch_size):
            return None

        cfg = self.config
        batch = self.buffer.sample(
            batch_size=cfg.batch_size,
            mirror_prob=cfg.mirror_prob if self.mirror_perm is not None else 0.0,
            mirror_perm=self.mirror_perm,
        )

        b_obs = torch.as_tensor(batch['obs'], dtype=torch.float32, device=self.device)
        b_act = torch.as_tensor(batch['actions'], dtype=torch.int64, device=self.device).unsqueeze(1)
        b_rew = torch.as_tensor(batch['rewards'], dtype=torch.float32, device=self.device)
        b_next_obs = torch.as_tensor(batch['next_obs'], dtype=torch.float32, device=self.device)
        b_term = torch.as_tensor(batch['terminated'], dtype=torch.float32, device=self.device)
        b_next_mask = torch.as_tensor(batch['next_mask'], dtype=torch.bool, device=self.device)

        # 1. Current Q-values: Q_online(s, a)
        self.online_net.train()
        q_pred = self.online_net(b_obs).gather(1, b_act).squeeze(1)

        # 2. Target Q-values: Negamax Double-DQN target
        with torch.no_grad():
            if cfg.double_dqn:
                # Online network selects best action in next state
                q_next_online = self.online_net(b_next_obs)
                # Mask illegal actions with -1e9
                q_next_online = torch.where(b_next_mask, q_next_online, torch.tensor(-1e9, device=self.device))
                next_actions = q_next_online.argmax(dim=1, keepdim=True)

                # Target network evaluates selected action
                q_next_target = self.target_net(b_next_obs).gather(1, next_actions).squeeze(1)
            else:
                q_next_target = self.target_net(b_next_obs)
                q_next_target = torch.where(b_next_mask, q_next_target, torch.tensor(-1e9, device=self.device))
                q_next_target = q_next_target.max(dim=1)[0]

            # Negamax target formulation:
            # y = r                                  if terminated
            # y = r - gamma * Q_target(s', a*)       otherwise (including truncated)
            y = b_rew - cfg.gamma * (1.0 - b_term) * q_next_target

        # 3. Loss & Gradient step
        loss = self.loss_fn(q_pred, y)

        self.optimizer.zero_grad()
        loss.backward()
        if cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.online_net.parameters(), cfg.grad_clip)
        self.optimizer.step()

        self.total_grad_steps += 1

        # 4. Target network hard synchronization
        if self.total_grad_steps % cfg.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return float(loss.item())
