"""Self-play mechanism and opponent matchmaking pool for DQN (P1.8).

Implements:
- SnapshotPool: FIFO buffer of past network checkpoints (up to 10, saved periodically).
- SelfPlayManager: Alternating self-play trajectory generator with opponent mixture:
  * 0.5 current network
  * 0.4 snapshot pool (with fallback to current if pool is empty)
  * 0.1 GreedyAgent
- Seat randomization and transition storage rules (both sides for current vs current;
  only learner transitions for current vs snapshot/greedy).
"""
from __future__ import annotations

import copy
import random
from typing import Callable, Optional
import numpy as np

try:
    import torch
except ImportError:
    torch = None

from agents.dqn.network import QNetwork
from agents.dqn.trainer import DQNTrainer
from agents.greedy import GreedyAgent


class SnapshotPool:
    """Fixed-capacity FIFO pool of historical model checkpoints for self-play."""

    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self._snapshots: list[dict] = []

    def add(self, state_dict: dict):
        """Adds a copy of model state_dict to the pool (FIFO eviction when full)."""
        cloned_dict = {k: v.cpu().clone() for k, v in state_dict.items()}
        if len(self._snapshots) >= self.max_size:
            self._snapshots.pop(0)
        self._snapshots.append(cloned_dict)

    def sample(self) -> dict | None:
        """Returns a random checkpoint state_dict from the pool, or None if empty."""
        if not self._snapshots:
            return None
        return random.choice(self._snapshots)

    def clear(self):
        """Clears all stored snapshots."""
        self._snapshots.clear()

    def __len__(self) -> int:
        return len(self._snapshots)


class SelfPlayManager:
    """Manages self-play rollouts and opponent sampling for training DQNTrainer."""

    def __init__(
        self,
        trainer: DQNTrainer,
        snapshot_interval: int = 20_000,
        snapshot_pool_size: int = 10,
        mixture: tuple[float, float, float] = (0.5, 0.4, 0.1),
        max_plies: int = 200,
    ):
        """
        Args:
            trainer: DQNTrainer instance to train.
            snapshot_interval: Env steps between taking a snapshot of online_net.
            snapshot_pool_size: Maximum historical checkpoints to keep.
            mixture: (p_current, p_snapshot, p_greedy) probabilities summing to 1.0.
            max_plies: Maximum plies before truncating an environment episode.
        """
        self.trainer = trainer
        self.snapshot_interval = snapshot_interval
        self.mixture = mixture
        self.max_plies = max_plies

        self.pool = SnapshotPool(max_size=snapshot_pool_size)

        # Snapshot network for opponent inference
        self.snapshot_net: Optional[QNetwork] = None
        if torch is not None:
            self.snapshot_net = QNetwork(
                board_size=trainer.board_size,
                num_actions=trainer.num_actions,
                channels=trainer.config.channels,
                blocks=trainer.config.blocks,
                hidden=trainer.config.hidden,
                dueling=trainer.config.dueling,
            ).to(trainer.device)
            self.snapshot_net.eval()

        self.last_snapshot_step = 0

    def sample_opponent_type(self) -> str:
        """Samples opponent type from mixture: 'current', 'snapshot', or 'greedy'."""
        p_cur, p_snap, _ = self.mixture
        r = random.random()
        if r < p_cur:
            return 'current'
        elif r < p_cur + p_snap:
            # Fallback to current if pool has no snapshots yet
            return 'snapshot' if len(self.pool) > 0 else 'current'
        else:
            return 'greedy'

    def run_episode(self, env: Optional[WallzEnv] = None) -> dict:
        """Executes a single self-play training episode.

        Returns:
            Dictionary of episode summary statistics.
        """
        if env is None:
            try:
                from modified_wallz.environment import WallzEnv
            except ImportError:
                from src.modified_wallz.environment import WallzEnv
            env = WallzEnv(
                board_size=self.trainer.board_size,
                rules=self.trainer.config.rules,
                max_plies=self.max_plies,
            )

        obs = env.reset()
        learner_seat = random.choice([0, 1])
        opponent_type = self.sample_opponent_type()

        # Load snapshot weights if playing against a snapshot
        if opponent_type == 'snapshot':
            snap_state = self.pool.sample()
            if snap_state is not None and self.snapshot_net is not None:
                self.snapshot_net.load_state_dict(snap_state)
                self.snapshot_net.eval()

        # Prepare greedy agent if playing against greedy
        opp_seat = 1 - learner_seat
        greedy_opp = GreedyAgent(opp_seat) if opponent_type == 'greedy' else None

        episode_steps = 0
        total_loss = 0.0
        train_steps_count = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            mover = env.to_move
            legal_mask = env.legal_mask()
            epsilon = self.trainer.get_epsilon()

            # Action selection
            if mover == learner_seat:
                action_id = self.trainer.select_action(obs, legal_mask, epsilon=epsilon)
            else:
                if opponent_type == 'current':
                    action_id = self.trainer.select_action(obs, legal_mask, epsilon=epsilon)
                elif opponent_type == 'snapshot':
                    action_id = self._select_snapshot_action(obs, legal_mask)
                elif opponent_type == 'greedy':
                    action_tuple = greedy_opp.select_action(env.engine)
                    if action_tuple is None:
                        # No legal actions available
                        break
                    action_id = env.codec.encode(action_tuple, env.engine, mover=mover)
                else:
                    action_id = self.trainer.select_action(obs, legal_mask, epsilon=epsilon)

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action_id)
            next_mask = env.legal_mask()
            self.trainer.total_env_steps += 1
            episode_steps += 1

            # Storage rule:
            # Store both sides if opponent == current;
            # store only learner transitions if opponent in (snapshot, greedy)
            should_store = (opponent_type == 'current') or (mover == learner_seat)
            if should_store:
                self.trainer.buffer.push(
                    obs=obs,
                    action=action_id,
                    reward=reward,
                    next_obs=next_obs,
                    terminated=terminated,
                    truncated=truncated,
                    next_mask=next_mask,
                )

            # Train step every train_freq env steps
            if self.trainer.total_env_steps % self.trainer.config.train_freq == 0:
                loss = self.trainer.train_step()
                if loss is not None:
                    total_loss += loss
                    train_steps_count += 1

            # Snapshot save check
            if (
                self.trainer.total_env_steps - self.last_snapshot_step >= self.snapshot_interval
                and self.snapshot_interval > 0
            ):
                self.pool.add(self.trainer.online_net.state_dict())
                self.last_snapshot_step = self.trainer.total_env_steps

            obs = next_obs

        # Outcome summary
        winner = env.engine.winner
        is_draw = env.engine.is_draw
        terminal_reason = env.engine.terminal_reason

        avg_loss = (total_loss / train_steps_count) if train_steps_count > 0 else None

        return {
            'episode_length': episode_steps,
            'learner_seat': learner_seat,
            'opponent_type': opponent_type,
            'winner': winner,
            'is_draw': is_draw,
            'terminal_reason': terminal_reason,
            'learner_won': (winner == learner_seat) if winner is not None else False,
            'avg_loss': avg_loss,
            'env_steps': self.trainer.total_env_steps,
            'grad_steps': self.trainer.total_grad_steps,
        }

    def _select_snapshot_action(self, obs: np.ndarray, legal_mask: list[bool] | np.ndarray) -> int:
        """Greedy action selection using the loaded snapshot network."""
        legal_indices = [i for i, is_legal in enumerate(legal_mask) if is_legal]
        if not legal_indices:
            raise ValueError("No legal actions for snapshot.")

        if self.snapshot_net is None or torch is None:
            return random.choice(legal_indices)

        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=self.trainer.device).unsqueeze(0)
            q_vals = self.snapshot_net(x).squeeze(0).cpu().numpy()

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
