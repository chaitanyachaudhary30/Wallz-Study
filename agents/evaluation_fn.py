"""Modular Evaluation Functions for Minimax Search (P6.1 / C7).

Provides the EvaluationFunction interface and concrete implementations:
1. HEvaluator: Canonical spatial difference evaluation U(s) = tanh(0.2 * (d_opp - d_self))
2. LinearEvaluator: Weighted linear combination of hand-crafted spatial and resource features
"""
from __future__ import annotations

import math
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
