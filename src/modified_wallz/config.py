"""Wallz Configuration (P3.1 / C1).

Defines WallzConfig, an immutable configuration for game rules, board geometry,
time controls, ply limits, and repetition thresholds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional


@dataclass(frozen=True)
class WallzConfig:
    board_size: int = 9
    time_limit_sec: float = 600.0
    A: bool = True   # Reusable walls (most recently placed wall can be repositioned)
    B: bool = True   # One-time hop
    C: bool = True   # Temporary cell block
    K: int = 3       # Block duration (K >= 1)
    M: int = 10      # Wall inventory per player (M >= 0)
    max_plies: Optional[int] = None      # Turn/ply cap (leads to truncated if exceeded)
    repetition_limit: int = 5            # N-fold repetition draw threshold (default 5)

    def __post_init__(self):
        if self.board_size < 3:
            raise ValueError(f"board_size must be >= 3, got {self.board_size}")
        if self.K < 1:
            raise ValueError(f"Block duration K must be >= 1, got {self.K}")
        if self.M < 0:
            raise ValueError(f"Wall inventory M must be >= 0, got {self.M}")
        if self.repetition_limit < 1:
            raise ValueError(f"repetition_limit must be >= 1, got {self.repetition_limit}")
        if self.max_plies is not None and self.max_plies < 1:
            raise ValueError(f"max_plies must be >= 1 if specified, got {self.max_plies}")

    @property
    def rules(self) -> Dict[str, Any]:
        """Backwards-compatible rules dictionary."""
        return {
            'A': self.A,
            'B': self.B,
            'C': self.C,
            'K': self.K,
            'M': self.M,
        }

    @classmethod
    def from_rules(
        cls,
        rules: Optional[Mapping[str, Any] | WallzConfig] = None,
        board_size: int = 9,
        time_limit_sec: float = 600.0,
        max_plies: Optional[int] = None,
        repetition_limit: int = 5,
    ) -> WallzConfig:
        """Constructs a validated WallzConfig from a rules dict, an existing config, or keyword args."""
        if isinstance(rules, WallzConfig):
            # If rules is already a WallzConfig, override any explicitly passed parameters
            return WallzConfig(
                board_size=board_size if board_size != 9 else rules.board_size,
                time_limit_sec=time_limit_sec if time_limit_sec != 600.0 else rules.time_limit_sec,
                A=rules.A,
                B=rules.B,
                C=rules.C,
                K=rules.K,
                M=rules.M,
                max_plies=max_plies if max_plies is not None else rules.max_plies,
                repetition_limit=repetition_limit if repetition_limit != 5 else rules.repetition_limit,
            )

        r_dict = dict(rules or {})
        return cls(
            board_size=int(r_dict.get('board_size', board_size)),
            time_limit_sec=float(r_dict.get('time_limit_sec', time_limit_sec)),
            A=bool(r_dict.get('A', True)),
            B=bool(r_dict.get('B', True)),
            C=bool(r_dict.get('C', True)),
            K=int(r_dict.get('K', 3)),
            M=int(r_dict.get('M', 10)),
            max_plies=int(r_dict['max_plies']) if 'max_plies' in r_dict and r_dict['max_plies'] is not None else max_plies,
            repetition_limit=int(r_dict.get('repetition_limit', repetition_limit)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            'board_size': self.board_size,
            'time_limit_sec': self.time_limit_sec,
            'A': self.A,
            'B': self.B,
            'C': self.C,
            'K': self.K,
            'M': self.M,
            'max_plies': self.max_plies,
            'repetition_limit': self.repetition_limit,
        }
