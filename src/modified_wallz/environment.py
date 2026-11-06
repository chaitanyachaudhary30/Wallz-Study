"""Environment wrapper for Modified Wallz (WallzEnv v1).

Implements Gym/OpenSpiel-style turn-based RL environment interface on top of WallzEngine.
"""
from __future__ import annotations

import random
from wallz_engine import WallzEngine
from .actions import ActionCodec
from agents.dqn.encoding import encode_observation


class WallzEnv:
    """Gym-compatible turn-based environment wrapping WallzEngine."""

    def __init__(self, board_size: int = 3, rules: dict | None = None, max_plies: int = 200):
        self.board_size = board_size
        self.rules = dict(rules) if rules else {}
        self.max_plies = max_plies
        self.codec = ActionCodec(board_size)
        self.engine = WallzEngine(board_size=board_size, rules=self.rules)
        self.rng = random.Random()

    @property
    def to_move(self) -> int:
        """Returns player index currently to move (0 or 1)."""
        return self.engine.current_player

    @property
    def num_actions(self) -> int:
        return self.codec.num_actions

    def reset(self, seed: int | None = None):
        """Resets the environment and returns the canonical initial observation for Player 0."""
        if seed is not None:
            self.rng.seed(seed)
        self.engine = WallzEngine(board_size=self.board_size, rules=self.rules)
        return encode_observation(self.engine, mover=self.engine.current_player)

    def legal_mask(self) -> list[bool]:
        """Returns boolean mask of valid legal action IDs for the side to move."""
        if self.engine.is_terminal():
            return [False] * self.num_actions
        return self.codec.legal_mask(self.engine, mover=self.engine.current_player)

    def legal_ids(self) -> list[int]:
        """Returns list of valid legal action IDs for the side to move."""
        if self.engine.is_terminal():
            return []
        return self.codec.legal_ids(self.engine, mover=self.engine.current_player)

    def step(self, action_id: int):
        """Executes a discrete action ID for the active player.

        Args:
            action_id: Integer action ID in range [0, num_actions - 1].

        Returns:
            (next_obs, reward_for_mover, terminated, truncated, info)
            - next_obs: Canonical observation tensor for the NEXT active player.
            - reward_for_mover: +1.0 if mover reached their goal row on this step, else 0.0.
            - terminated: True if terminal state reached (goal, repetition, or no legal actions).
            - truncated: True if max_plies reached without termination.
            - info: Diagnostic dictionary with terminal_reason, winner, turn_count, mover, action.
        """
        if self.engine.is_terminal():
            raise RuntimeError("Cannot step in terminal environment; call reset() first.")

        mover = self.engine.current_player
        action = self.codec.decode(action_id, self.engine, mover=mover)

        legal = self.engine.get_legal_actions()
        if action not in legal:
            raise ValueError(f"Illegal action {action} (id {action_id}) for player {mover} at {self.engine.pawns[mover]}")

        # Execute transition without chess clock depletion
        self.engine.apply_action(action, update_clock=False)

        # Rewards: canonical sparse reward +1 if the player who just acted won
        reward_for_mover = 1.0 if self.engine.winner == mover else 0.0
        terminated = self.engine.is_terminal()
        truncated = (not terminated) and (self.engine.turn_count >= self.max_plies)

        info = {
            'terminal_reason': self.engine.terminal_reason,
            'winner': self.engine.winner,
            'turn_count': self.engine.turn_count,
            'mover': mover,
            'action': action,
        }

        next_obs = encode_observation(self.engine, mover=self.engine.current_player)
        return next_obs, reward_for_mover, terminated, truncated, info
