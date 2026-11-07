"""Shared Test Helpers and Contracts for Engine and Baselines (C-SNAP, C-ACT).

Provides:
- make_engine: constructs test engines with merged rules
- snapshot: exact C-SNAP dictionary representation for engine state immutability / atomicity
- play: plays a sequence of actions with update_clock=False
- random_reachable: generates deterministic reachable states
- illegal_action_pool: generates invalid/illegal and malformed candidate actions
- pass_move: deterministic non-goal MOVE to burn plies
- key_to_json / key_from_json: raw state key JSON serialization helpers
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple, Union

from wallz_engine import WallzEngine


def make_engine(n: int = 5, **rules) -> WallzEngine:
    """Constructs a WallzEngine with default rules merged with overrides."""
    rule_dict = {'A': True, 'B': True, 'C': True, 'K': 3, 'M': 10}
    rule_dict.update(rules)
    return WallzEngine(board_size=n, rules=rule_dict)


def snapshot(engine: WallzEngine) -> dict:
    """Returns an exact C-SNAP engine snapshot dictionary."""
    return {
        'pawns': dict(engine.pawns),
        'walls': {p: list(w) for p, w in engine.walls.items()},
        'wall_inventory': dict(engine.wall_inventory),
        'latest_wall': dict(engine.latest_wall),
        'hop_available': dict(engine.hop_available),
        'block_available': dict(engine.block_available),
        'blocked_cells': dict(engine.blocked_cells),
        'current_player': engine.current_player,
        'turn_count': engine.turn_count,
        'winner': engine.winner,
        'is_draw': engine.is_draw,
        'is_truncated': engine.is_truncated,
        'terminal_reason': engine.terminal_reason,
        'time_remaining': dict(engine.time_remaining),
        'last_turn_timestamp': engine.last_turn_timestamp,
        'history': dict(engine.state_history),
        'board_walls': frozenset(engine.board.walls),
    }


def play(engine: WallzEngine, actions: List[Any]) -> WallzEngine:
    """Applies a sequence of actions with update_clock=False and returns the engine."""
    for action in actions:
        engine.apply_action(action, update_clock=False)
    return engine


def random_reachable(n: int, rules: Dict[str, Any], plies: int, seed: int) -> WallzEngine:
    """Generates a deterministic reachable state by applying uniformly random legal actions."""
    eng = WallzEngine(board_size=n, rules=rules)
    rng = random.Random(seed)
    for _ in range(plies):
        if eng.is_terminal():
            break
        legal = eng.get_legal_actions()
        if not legal:
            break
        action = rng.choice(legal)
        eng.apply_action(action, update_clock=False)
    return eng


def illegal_action_pool(engine: WallzEngine, rng: random.Random, k: int = 20) -> List[Any]:
    """
    Generates a collection of strictly illegal or malformed actions for the current engine state.
    Guarantees no generated action is in engine.get_legal_actions().
    """
    n = engine.board_size
    p = engine.current_player
    opp = 1 - p
    legal = set(engine.get_legal_actions())

    candidates = [
        # (a) Out-of-bounds MOVE / HOP / BLOCK / WALL
        ('MOVE', (-1, 0)),
        ('MOVE', (n, 0)),
        ('MOVE', (0, -1)),
        ('MOVE', (0, n)),
        ('HOP', (-2, 0)),
        ('HOP', (n + 1, 0)),
        ('HOP', (0, -2)),
        ('HOP', (0, n + 1)),
        ('BLOCK', (-1, 0)),
        ('BLOCK', (n, 0)),
        ('WALL', (-1, 0, 'H')),
        ('WALL', (n - 1, 0, 'H')),
        ('WALL', (0, n - 1, 'V')),
        # (b) MOVE onto opponent
        ('MOVE', engine.pawns[opp]),
        # (c) BLOCK on a pawn cell or already-blocked cell
        ('BLOCK', engine.pawns[0]),
        ('BLOCK', engine.pawns[1]),
        # (h) Malformed objects
        None,
        'MOVE',
        ('MOVE',),
        ('MOVE', (1,)),
        ('MOVE', (1.0, 2)),
        ('MOVE', (True, 1)),
        ('FLY', (1, 1)),
        ('WALL', (0, 0, 'X')),
        ('WALL', (0, 0)),
    ]

    # Blocked cells
    for cell in engine.blocked_cells:
        candidates.append(('BLOCK', cell))

    # Overlapping existing walls
    for w in engine.board.walls:
        candidates.append(('WALL', w))

    # REPLACE when target equals latest or onto existing walls
    latest = engine.latest_wall[p]
    if latest is not None:
        candidates.append(('REPLACE', latest))

    # HOP when consumed
    if not engine.hop_available[p]:
        r, c = engine.pawns[p]
        candidates.append(('HOP', (r + 2, c)))

    # BLOCK when consumed
    if not engine.block_available[p]:
        candidates.append(('BLOCK', (0, 0)))

    # Filter out anything that is legally valid
    illegal = [c for c in candidates if c not in legal]

    rng.shuffle(illegal)
    return illegal[:k]


def pass_move(engine: WallzEngine) -> Tuple[str, Tuple[int, int]]:
    """
    Selects a deterministic legal MOVE that does NOT reach the mover's goal row,
    preferring sideways moves (lowest in sorted action order).
    Used to burn plies in timeline tests.
    """
    p = engine.current_player
    goal_row = engine.goals[p]
    moves = [
        a for a in engine.get_legal_actions()
        if a[0] == 'MOVE' and a[1][0] != goal_row
    ]
    if not moves:
        # Fall back to any legal non-goal action if available
        non_goal = [a for a in engine.get_legal_actions() if not (a[0] in ('MOVE', 'HOP') and a[1][0] == goal_row)]
        if non_goal:
            return sorted(non_goal, key=str)[0]
        legal = engine.get_legal_actions()
        if legal:
            return legal[0]
        raise RuntimeError("No moves available to pass.")
    # Sort deterministically
    return sorted(moves, key=lambda a: (abs(a[1][1] - engine.pawns[p][1]) == 0, a[1]))[0]


def key_to_json(key: Tuple) -> List[Any]:
    """Converts a raw engine state hashable key tuple into a JSON-serializable structure."""
    def _conv(item):
        if isinstance(item, (tuple, list)):
            return [_conv(x) for x in item]
        return item
    return _conv(key)


def key_from_json(obj: List[Any]) -> Tuple:
    """Reconstructs a raw engine state hashable key tuple from a JSON-deserialized structure."""
    if len(obj) == 14:
        # Raw key: (p0, p1, walls0, walls1, inv0, inv1, l0, l1, h0, h1, b0, b1, blocks, player)
        (p0, p1, w0, w1, inv0, inv1, l0, l1, h0, h1, b0, b1, blocks, player) = obj
        p0_t = tuple(p0)
        p1_t = tuple(p1)
        w0_t = tuple(tuple(w) for w in w0)
        w1_t = tuple(tuple(w) for w in w1)
        l0_t = tuple(l0) if l0 is not None else None
        l1_t = tuple(l1) if l1 is not None else None
        blocks_t = tuple((tuple(cell), timer) for cell, timer in blocks)
        return (p0_t, p1_t, w0_t, w1_t, inv0, inv1, l0_t, l1_t, h0, h1, b0, b1, blocks_t, player)
    elif len(obj) == 13:
        (p0, p1, fixed, l0, l1, inv0, inv1, h0, h1, b0, b1, blocks, player) = obj
        return (tuple(p0), tuple(p1), frozenset(tuple(w) for w in fixed),
                tuple(l0) if l0 is not None else None, tuple(l1) if l1 is not None else None,
                inv0, inv1, h0, h1, b0, b1,
                tuple((tuple(cell), timer) for cell, timer in blocks), player)
    elif len(obj) == 11:
        (p0, p1, fixed, inv0, inv1, h0, h1, b0, b1, blocks, player) = obj
        return (tuple(p0), tuple(p1), frozenset(tuple(w) for w in fixed),
                inv0, inv1, h0, h1, b0, b1,
                tuple((tuple(cell), timer) for cell, timer in blocks), player)
    return tuple(tuple(x) if isinstance(x, list) else x for x in obj)
