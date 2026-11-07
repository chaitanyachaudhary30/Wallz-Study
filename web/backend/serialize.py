"""State and action serialization for the web API (P08 / Contract C-STATE & C-ACT).

Converts live WallzEngine state into the canonical frontend contract C-STATE.
Ensures single-source validation for actions parsed from JSON payloads.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from wallz_engine import WallzEngine


def action_to_json(action: Tuple[str, Any]) -> List[Any]:
    """Converts a canonical action tuple to a JSON-serializable list."""
    act_type, act_val = action
    if isinstance(act_val, (tuple, list)):
        if act_type == "REPLACE" and len(act_val) == 2 and isinstance(act_val[0], (tuple, list)):
            val = [list(act_val[0]), list(act_val[1])]
        else:
            val = list(act_val)
    else:
        val = act_val
    return [act_type, val]


def parse_action(obj: Any, engine: Optional[WallzEngine] = None) -> Tuple[str, Any]:
    """Parses and normalizes an action object using WallzEngine._normalize_action."""
    if engine is None:
        engine = WallzEngine()
    return engine._normalize_action(obj)


def state_json(engine: WallzEngine) -> Dict[str, Any]:
    """Serializes engine state to canonical web contract C-STATE."""
    # 1. Walls list with owner and latest flag
    walls_list = []
    for p in (0, 1):
        latest = engine.latest_wall[p]
        for w in engine.walls[p]:
            walls_list.append({
                "r": w[0],
                "c": w[1],
                "o": w[2],
                "owner": p,
                "latest": (w == latest),
            })
    # Stable sort by owner, r, c, o
    walls_list.sort(key=lambda x: (x["owner"], x["r"], x["c"], x["o"]))

    # 2. Blocked cells sorted by (r, c)
    blocked_list = [
        {"r": r, "c": c, "remaining": timer}
        for (r, c), timer in sorted(engine.blocked_cells.items(), key=lambda item: item[0])
    ]

    # 3. Shortest paths for both players
    paths = [
        [list(cell) for cell in engine.shortest_path_cells(0)],
        [list(cell) for cell in engine.shortest_path_cells(1)],
    ]

    # 4. Grouped legal actions
    legal_grouped: Dict[str, List[Any]] = {
        "MOVE": [],
        "HOP": [],
        "WALL": [],
        "REPLACE": [],
        "BLOCK": [],
    }
    legal_actions = engine.get_legal_actions()
    for act_type, act_val in legal_actions:
        if act_type == "REPLACE" and isinstance(act_val, tuple) and len(act_val) == 2 and isinstance(act_val[0], tuple):
            val = [list(act_val[0]), list(act_val[1])]
        elif isinstance(act_val, tuple):
            val = list(act_val)
        else:
            val = act_val
        legal_grouped[act_type].append(val)

    # Sort within each action group for deterministic ordering
    for act_type in legal_grouped:
        legal_grouped[act_type].sort(key=lambda x: str(x))

    return {
        "n": engine.board_size,
        "rules": dict(engine.rules),
        "pawns": [list(engine.pawns[0]), list(engine.pawns[1])],
        "goals": [engine.goals[0], engine.goals[1]],
        "walls": walls_list,
        "inventory": [engine.wall_inventory[0], engine.wall_inventory[1]],
        "hop": [engine.hop_available[0], engine.hop_available[1]],
        "block": [engine.block_available[0], engine.block_available[1]],
        "blocked": blocked_list,
        "to_move": engine.current_player,
        "turn": engine.turn_count,
        "terminal": {
            "over": engine.is_terminal(),
            "winner": engine.winner,
            "reason": engine.terminal_reason,
        },
        "paths": paths,
        "legal": legal_grouped,
        "legal_count": len(legal_actions),
    }
