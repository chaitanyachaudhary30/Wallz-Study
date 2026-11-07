"""Tier-1 Spatial Greedy Baseline Agent (Phase 4 / C4).

Evaluates all legal actions with 1-ply lookahead using the proposal objective:
    H_i(T(s, a)) = d_{-i}(T(s, a)) - d_i(T(s, a))

Resolves ties through an explicit hierarchy (research §4.3):
    score(a) = (
        H_i(T(s, a)),             # proposal distance-advantage objective
        category_priority(a),     # MOVE=4 > REPLACE=3 > HOP=2 > WALL=1 > BLOCK=0
        -d_i(T(s, a)),            # prefer own spatial progress
        tie_key(a)                # -idx (deterministic) OR rng.random() (seeded)
    )

Immediate win: returned immediately.
Draws (repetition or no legal actions): scored as H = 0.0, preferred over losses (H < 0).
"""
from __future__ import annotations

import copy
import random
from typing import Any, Dict, List, Optional, Tuple


CATEGORY_PRIORITY: Dict[str, int] = {
    'MOVE': 4,
    'REPLACE': 3,
    'HOP': 2,
    'WALL': 1,
    'BLOCK': 0,
}


class GreedyAgent:
    """Tier-1 Spatial Greedy Baseline Agent with explicit tie-breaking and draw scoring."""

    def __init__(self, player_id: int, seed: Optional[int] = None):
        self.player_id = player_id
        self.seed = seed
        self.rng = random.Random(seed) if seed is not None else None

    def select_action(self, engine) -> Optional[Tuple[str, Any]]:
        legal_actions = engine.get_legal_actions(self.player_id)
        if not legal_actions:
            return None

        best_action = None
        best_score = None

        for idx, action in enumerate(legal_actions):
            sim_engine = engine.clone() if hasattr(engine, 'clone') else copy.deepcopy(engine)
            _, winner, is_draw = sim_engine.apply_action(action, update_clock=False)

            # 1. Immediate winning move
            if winner == self.player_id:
                return action

            # Category priority: MOVE=4 > REPLACE=3 > HOP=2 > WALL=1 > BLOCK=0 (P4.1)
            cat_prio = CATEGORY_PRIORITY.get(action[0], 0)

            # Tie key: deterministic -idx or seeded random float (P4.3)
            tie_k = self.rng.random() if self.rng is not None else -idx

            if winner is not None and winner != self.player_id:
                # Immediate loss
                H = -float('inf')
                d_self = float('inf')
            elif is_draw:
                # Rule draw: repetition or no legal action (P4.2)
                # Scored as H = 0.0; strictly preferred over a loss (H < 0)
                H = 0.0
                d_self = sim_engine.shortest_path(self.player_id)
            else:
                d_self = sim_engine.shortest_path(self.player_id)
                d_opp = sim_engine.shortest_path(1 - self.player_id)
                H = d_opp - d_self

            score = (
                H,         # 1. Distance advantage H = d_opp - d_self
                cat_prio,  # 2. Category priority: MOVE > REPLACE > HOP > WALL > BLOCK
                -d_self,   # 3. Prefer own spatial progress
                tie_k,     # 4. Deterministic -idx or seeded random float
            )

            if best_score is None or score > best_score:
                best_score = score
                best_action = action

        if best_action is None and legal_actions:
            best_action = legal_actions[0]

        return best_action
