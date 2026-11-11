"""Modular Evaluation Functions for Minimax Search (P6.1 / C7).

Provides the EvaluationFunction interface and concrete implementations:
1. HEvaluator: Canonical spatial difference evaluation U(s) = tanh(0.2 * (d_opp - d_self))
2. LinearEvaluator: Weighted linear combination of hand-crafted spatial and resource features
"""
from __future__ import annotations

import math
from collections import deque
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, runtime_checkable
import numpy as np


@runtime_checkable
class EvaluationFunction(Protocol):
    """Protocol for minimax leaf evaluation."""

    def evaluate(self, engine, player_id: int) -> float:
        """Evaluates engine state from the perspective of player_id."""
        ...


class HEvaluator:
    """
    Canonical spatial heuristic evaluation:
        H_i(s) = d_{-i}(s) - d_i(s)
        U_i(s) = tanh(0.2 * H_i(s))

    Bounded in (-1, 1). Bounded transform preserves exact minimax move ordering.
    """

    def __init__(self, scale: float = 0.2):
        self.scale = scale

    def evaluate(self, engine, player_id: int) -> float:
        d_self = engine.shortest_path(player_id)
        d_opp = engine.shortest_path(1 - player_id)

        if math.isinf(d_self) or math.isinf(d_opp):
            return 0.0

        h_val = d_opp - d_self
        return math.tanh(self.scale * h_val)


class HopAwareEvaluator:
    """
    New contribution (Stage 3, optional; not the default): HEvaluator with hop-aware distances.

    While Rule B is on and a player still holds its hop, that player's distance is the shortest
    path that may cross at most one wall owned by the other player (one orthogonal step onto an
    on-board, unblocked cell). Otherwise it is the engine's plain BFS distance. Like the plain
    distance, pawns are ignored. Formula and range are unchanged: tanh(scale * (d_opp - d_self)).
    Wall ownership and edges come from the engine (wall_owner_on_edge, board.is_edge_blocked).
    """

    def __init__(self, scale: float = 0.2):
        self.scale = scale

    @staticmethod
    def distance(engine, player: int) -> float:
        plain = engine.shortest_path(player)  # also syncs engine.board with the wall lists
        if not (engine.rules['B'] and engine.hop_available[player]):
            return plain
        n, goal, blocked = engine.board_size, engine.goals[player], engine.blocked_cells
        start = (engine.pawns[player], 0)
        seen = {start}
        frontier = deque([(start, 0)])
        while frontier:
            (cell, used), d = frontier.popleft()
            if cell[0] == goal:
                return float(d)
            r, c = cell
            for nxt in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if not (0 <= nxt[0] < n and 0 <= nxt[1] < n) or nxt in blocked:
                    continue
                if not engine.board.is_edge_blocked(cell, nxt):
                    state = (nxt, used)
                elif not used and engine.wall_owner_on_edge(cell, nxt) == 1 - player:
                    state = (nxt, 1)
                else:
                    continue
                if state not in seen:
                    seen.add(state)
                    frontier.append((state, d + 1))
        return plain

    def evaluate(self, engine, player_id: int) -> float:
        d_self = self.distance(engine, player_id)
        d_opp = self.distance(engine, 1 - player_id)

        if math.isinf(d_self) or math.isinf(d_opp):
            return 0.0

        return math.tanh(self.scale * (d_opp - d_self))


class LinearEvaluator:
    """
    Evaluates engine state using a linear weight vector over engineered features:
        U_i(s) = tanh(weights · features(s, i) / 2) = 2 * sigma(weights · features) - 1
    Converts logistic model probability to range (-1, 1) adhering to Contract C7.
    """

    def __init__(self, weights: Sequence[float], scale: float = 0.5):
        self.weights = np.asarray(weights, dtype=np.float64)
        self.scale = scale

    def evaluate(self, engine, player_id: int) -> float:
        from evaluation.features import extract_features
        feats = extract_features(engine, player_id)
        val = float(np.dot(self.weights, feats))
        return math.tanh(self.scale * val)
