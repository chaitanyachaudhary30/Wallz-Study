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

    def score_actions(self, engine) -> List[Dict[str, Any]]:
        """Score every legal action exactly as select_action ranks them.

        Each row contains:
            'action': canonical action tuple
            'category': str ('MOVE', 'HOP', etc.)
            'priority': int category priority
            'immediate_win': bool
            'outcome': str in {'win', 'loss', 'draw', 'ongoing'}
            'H': float distance advantage (or inf/-inf/0.0)
            'd_self': float shortest path for player (or inf/0.0)
            'd_opp': Optional[float] shortest path for opponent
            'tie_key': float (-idx or rng draw)
            'score': comparison 4-tuple (H, priority, -d_self, tie_key)

        Note: If an immediate winning move is encountered in legal order,
        further evaluations are short-circuited to match the exact evaluation
        and random draw sequence of select_action.
        """
        legal_actions = engine.get_legal_actions(self.player_id)
        if not legal_actions:
            return []

        rows = []
        for idx, action in enumerate(legal_actions):
            sim_engine = engine.clone() if hasattr(engine, 'clone') else copy.deepcopy(engine)
            _, winner, is_draw = sim_engine.apply_action(action, update_clock=False)
            cat_prio = CATEGORY_PRIORITY.get(action[0], 0)

            # 1. Immediate winning move
            if winner == self.player_id:
                rows.append({
                    'action': action,
                    'category': action[0],
                    'priority': cat_prio,
                    'immediate_win': True,
                    'outcome': 'win',
                    'H': float('inf'),
                    'd_self': 0.0,
                    'd_opp': None,
                    'tie_key': None,
                    'score': (float('inf'), cat_prio, 0.0, 0),
                })
                break

            # Tie key: deterministic -idx or seeded random float (P4.3)
            tie_k = self.rng.random() if self.rng is not None else -idx

            if winner is not None and winner != self.player_id:
                # Immediate loss
                H = -float('inf')
                d_self = float('inf')
                d_opp = None
                outcome = 'loss'
            elif is_draw:
                # Rule draw: repetition or no legal action (P4.2)
                # Scored as H = 0.0; strictly preferred over a loss (H < 0)
                H = 0.0
                d_self = sim_engine.shortest_path(self.player_id)
                d_opp = None
                outcome = 'draw'
            else:
                d_self = sim_engine.shortest_path(self.player_id)
                d_opp = sim_engine.shortest_path(1 - self.player_id)
                H = d_opp - d_self
                outcome = 'ongoing'

            score = (
                H,         # 1. Distance advantage H = d_opp - d_self
                cat_prio,  # 2. Category priority: MOVE > REPLACE > HOP > WALL > BLOCK
                -d_self,   # 3. Prefer own spatial progress
                tie_k,     # 4. Deterministic -idx or seeded random float
            )

            rows.append({
                'action': action,
                'category': action[0],
                'priority': cat_prio,
                'immediate_win': False,
                'outcome': outcome,
                'H': H,
                'd_self': d_self,
                'd_opp': d_opp,
                'tie_key': tie_k,
                'score': score,
            })

        return rows

    def select_action(self, engine) -> Optional[Tuple[str, Any]]:
        # Side-to-move assertion (P5.4 / P06 F4)
        assert engine.current_player == self.player_id, (
            f"Engine side to move ({engine.current_player}) does not match agent player_id ({self.player_id})"
        )
        rows = self.score_actions(engine)
        if not rows:
            return None

        wins = [r for r in rows if r['immediate_win']]
        if wins:
            return wins[0]['action']

        return max(rows, key=lambda r: r['score'])['action']
