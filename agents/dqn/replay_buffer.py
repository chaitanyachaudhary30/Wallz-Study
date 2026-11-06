"""Replay buffer for Modified Wallz DQN (P1.6).

Uniform FIFO ring buffer storing experience transitions with legal next-action masks
and optional horizontal mirror augmentation at sampling time.
"""
from __future__ import annotations

import random
try:
    import numpy as np
except ImportError:
    np = None

from agents.dqn.encoding import mirror_observation


class ReplayBuffer:
    """Experience replay buffer storing transitions in preallocated NumPy arrays."""

    def __init__(self, capacity: int, board_size: int, num_actions: int):
        self.capacity = capacity
        self.board_size = board_size
        self.num_actions = num_actions
        self.ptr = 0
        self.size = 0

        # Memory storage: float16 for observations to reduce RAM footprint
        self.obs = np.zeros((capacity, 14, board_size, board_size), dtype=np.float16)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, 14, board_size, board_size), dtype=np.float16)
        self.terminated = np.zeros(capacity, dtype=bool)
        self.truncated = np.zeros(capacity, dtype=bool)
        self.next_mask = np.zeros((capacity, num_actions), dtype=bool)

    def __len__(self) -> int:
        return self.size

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        terminated: bool,
        truncated: bool,
        next_mask: list[bool] | np.ndarray,
    ):
        """Adds a transition to the FIFO ring buffer."""
        idx = self.ptr
        self.obs[idx] = np.asarray(obs, dtype=np.float16)
        self.actions[idx] = int(action)
        self.rewards[idx] = float(reward)
        self.next_obs[idx] = np.asarray(next_obs, dtype=np.float16)
        self.terminated[idx] = bool(terminated)
        self.truncated[idx] = bool(truncated)
        self.next_mask[idx] = np.asarray(next_mask, dtype=bool)

        self.ptr = (self.ptr + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def sample(
        self,
        batch_size: int,
        rng: random.Random | None = None,
        mirror_prob: float = 0.0,
        mirror_perm: list[int] | None = None,
    ) -> dict[str, np.ndarray]:
        """Samples a batch uniformly without replacement.

        Args:
            batch_size: Number of transitions to sample.
            rng: Optional random.Random instance for reproducible sampling.
            mirror_prob: Probability of horizontal mirror augmentation (p in [0.0, 1.0]).
            mirror_perm: Mirror action permutation mapping action_id -> mirrored_action_id.

        Returns:
            Dictionary containing:
                'obs': np.ndarray of shape (B, 14, N, N), float32
                'actions': np.ndarray of shape (B,), int64
                'rewards': np.ndarray of shape (B,), float32
                'next_obs': np.ndarray of shape (B, 14, N, N), float32
                'terminated': np.ndarray of shape (B,), bool
                'truncated': np.ndarray of shape (B,), bool
                'next_mask': np.ndarray of shape (B, num_actions), bool
        """
        if self.size < batch_size:
            raise ValueError(f"Cannot sample batch_size {batch_size} from buffer of size {self.size}")

        if rng is not None:
            indices = np.array(rng.sample(range(self.size), batch_size), dtype=np.int64)
        else:
            indices = np.random.choice(self.size, size=batch_size, replace=False)

        batch_obs = np.array(self.obs[indices], dtype=np.float32)
        batch_actions = np.array(self.actions[indices], dtype=np.int64)
        batch_rewards = np.array(self.rewards[indices], dtype=np.float32)
        batch_next_obs = np.array(self.next_obs[indices], dtype=np.float32)
        batch_terminated = np.array(self.terminated[indices], dtype=bool)
        batch_truncated = np.array(self.truncated[indices], dtype=bool)
        batch_next_mask = np.array(self.next_mask[indices], dtype=bool)

        # Mirror augmentation (odd N only)
        if mirror_prob > 0.0 and mirror_perm is not None and (self.board_size % 2 == 1):
            perm_arr = np.asarray(mirror_perm, dtype=np.int64)
            for i in range(batch_size):
                coin = rng.random() if rng is not None else random.random()
                if coin < mirror_prob:
                    batch_obs[i] = mirror_observation(batch_obs[i])
                    batch_next_obs[i] = mirror_observation(batch_next_obs[i])
                    batch_actions[i] = perm_arr[batch_actions[i]]
                    batch_next_mask[i] = batch_next_mask[i][perm_arr]

        return {
            'obs': batch_obs,
            'actions': batch_actions,
            'rewards': batch_rewards,
            'next_obs': batch_next_obs,
            'terminated': batch_terminated,
            'truncated': batch_truncated,
            'next_mask': batch_next_mask,
        }
