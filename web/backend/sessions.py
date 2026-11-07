"""Game session and LRU store management (P08).

Manages active game instances, player identities, move history, and atomic undo
reconstruction without direct engine state mutation.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional
import uuid

from src.modified_wallz.config import WallzConfig
from web.backend.serialize import action_to_json, parse_action, state_json
from wallz_engine import WallzEngine


@dataclass
class GameSession:
    id: str
    config: Dict[str, Any]
    engine: WallzEngine
    specs: Dict[int, str]
    agents: Dict[int, Optional[Any]]
    history: List[Dict[str, Any]] = field(default_factory=list)
    created: float = field(default_factory=time.time)

    def to_json(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "config": self.config,
            "specs": {str(k): v for k, v in self.specs.items()},
            "state": state_json(self.engine),
            "history": self.history,
            "created": self.created,
        }

    def undo(self, n: int = 1, to_human: bool = False) -> None:
        """Rolls back the game by reconstructing a fresh engine and reapplying history."""
        if not self.history:
            raise ValueError("No actions to undo.")

        plies_to_remove = n
        if to_human:
            # If current to_move is human and last was AI, or if AI just moved,
            # pop until the side to move is the human player
            # By default pop 2 plies if human vs AI
            is_human_p0 = self.specs.get(0) == "human"
            is_human_p1 = self.specs.get(1) == "human"
            if (is_human_p0 and not is_human_p1) or (is_human_p1 and not is_human_p0):
                target_human = 0 if is_human_p0 else 1
                remove_count = 0
                temp_hist = list(self.history)
                while temp_hist:
                    last_ply = temp_hist.pop()
                    remove_count += 1
                    # Next to move after removing last_ply is last_ply['player']
                    if last_ply["player"] == target_human:
                        plies_to_remove = remove_count
                        break

        plies_to_remove = max(1, min(plies_to_remove, len(self.history)))
        kept_history = self.history[:-plies_to_remove]

        # Reconstruct engine from scratch to guarantee single source of truth
        board_size = self.config.get("board_size", 5)
        max_plies = self.config.get("max_plies", 200)
        rules = {
            "A": self.config.get("A", True),
            "B": self.config.get("B", True),
            "C": self.config.get("C", True),
            "K": self.config.get("K", 3),
            "M": self.config.get("M", WallzConfig().M),
        }
        config_obj = WallzConfig(
            board_size=board_size,
            max_plies=max_plies,
            A=rules["A"],
            B=rules["B"],
            C=rules["C"],
            K=rules["K"],
            M=rules["M"],
        )
        fresh_engine = WallzEngine(config=config_obj)

        for step in kept_history:
            act = parse_action(step["action"], fresh_engine)
            fresh_engine.apply_action(act, update_clock=False)

        self.engine = fresh_engine
        self.history = kept_history


class SessionStore:
    """LRU store for active GameSessions with capacity cap."""

    def __init__(self, capacity: int = 64):
        self.capacity = capacity
        self.sessions: OrderedDict[str, GameSession] = OrderedDict()

    def create(
        self,
        config: Dict[str, Any],
        p0_spec: str,
        p1_spec: str,
        agents: Optional[Dict[int, Any]] = None,
        custom_id: Optional[str] = None,
        initial_engine: Optional[WallzEngine] = None,
    ) -> GameSession:
        gid = custom_id or str(uuid.uuid4())[:8]

        board_size = config.get("board_size", 5)
        max_plies = config.get("max_plies", 200)
        rules = {
            "A": config.get("A", True),
            "B": config.get("B", True),
            "C": config.get("C", True),
            "K": config.get("K", 3),
            "M": config.get("M", WallzConfig().M),
        }
        cfg_dict = {"board_size": board_size, "max_plies": max_plies, **rules}

        if initial_engine is not None:
            engine = initial_engine
        else:
            config_obj = WallzConfig(
                board_size=board_size,
                max_plies=max_plies,
                A=rules["A"],
                B=rules["B"],
                C=rules["C"],
                K=rules["K"],
                M=rules["M"],
            )
            engine = WallzEngine(config=config_obj)

        session = GameSession(
            id=gid,
            config=cfg_dict,
            engine=engine,
            specs={0: p0_spec, 1: p1_spec},
            agents=agents or {0: None, 1: None},
        )

        if len(self.sessions) >= self.capacity:
            self.sessions.popitem(last=False)

        self.sessions[gid] = session
        return session

    def get(self, gid: str) -> GameSession:
        if gid not in self.sessions:
            raise KeyError(f"Session '{gid}' not found")
        session = self.sessions[gid]
        self.sessions.move_to_end(gid)
        return session

    def delete(self, gid: str) -> None:
        if gid not in self.sessions:
            raise KeyError(f"Session '{gid}' not found")
        del self.sessions[gid]


SESSION_STORE = SessionStore(capacity=64)
