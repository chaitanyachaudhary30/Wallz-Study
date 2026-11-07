"""Tier-2 Minimax Adversarial Agent with Alpha-Beta Pruning (Phase 5 / Engine v2).

Features:
- Negamax formulation with side-to-move assertion
- PV-first root ordering across iterative deepening iterations (P5.1)
- History heuristic with d^2 cutoff update (P5.2)
- Fast move ordering: wins > BFS pawn distance progress > opp path walls > history (P5.2)
- Staged action exploration using engine API (P5.3)
- depth_completed tracking in get_stats() (P5.5)
- Optional goal-proximity extension behind extend_goal flag (P5.6)
"""
from __future__ import annotations

import collections
import copy
import math
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    from src.modified_wallz.pathfinding import bfs_distance_and_path, norm_edge
except ImportError:
    try:
        from modified_wallz.pathfinding import bfs_distance_and_path, norm_edge
    except ImportError:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from src.modified_wallz.pathfinding import bfs_distance_and_path, norm_edge


class SearchTimeout(Exception):
    """Raised when the search exceeds the configured per-turn time limit."""
    pass


class MinimaxAgent:
    """Tier-2 Minimax Adversarial Agent with Alpha-Beta Pruning and Negamax search."""

    WIN_SCORE = 1000.0

    def __init__(
        self,
        player_id: int,
        depth: int = 2,
        time_limit: Optional[float] = None,
        use_alpha_beta: bool = True,
        order_moves: bool = True,
        extend_goal: bool = False,
        evaluator: Optional[Any] = None,
    ):
        self.player_id = player_id
        self.depth = max(1, depth)
        self.time_limit = time_limit
        self.use_alpha_beta = use_alpha_beta
        self.order_moves = order_moves
        self.extend_goal = extend_goal
        self.evaluator = evaluator

        # Performance & metric tracking
        self.nodes_expanded = 0
        self.cutoffs = 0
        self.last_search_depth = 0
        self.last_decision_time = 0.0
        self.last_action_value = 0.0

        # History heuristic table: action -> score (P5.2)
        self.history: Dict[Any, int] = collections.defaultdict(int)
        self.pv_move: Optional[Tuple[str, Any]] = None

    def get_stats(self) -> Dict[str, Any]:
        """Returns performance metrics from the most recent decision (P5.5)."""
        return {
            'nodes_expanded': self.nodes_expanded,
            'cutoffs': self.cutoffs,
            'last_search_depth': self.last_search_depth,
            'depth_completed': self.last_search_depth,
            'last_decision_time': self.last_decision_time,
            'last_action_value': self.last_action_value,
        }

    def evaluate(self, engine, depth: int = 0, ply: int = 0) -> float:
        """
        Leaf evaluation function from the perspective of engine.current_player (Negamax).
        """
        curr = engine.current_player

        # Terminal state precedence
        if engine.winner is not None:
            if engine.winner == curr:
                return self.WIN_SCORE - ply
            else:
                return -(self.WIN_SCORE - ply)

        if engine.is_draw:
            return 0.0

        # Custom evaluator if supplied (Phase 6)
        if self.evaluator is not None:
            return self.evaluator.evaluate(engine, curr)

        # Spatial heuristic evaluation H = d_opp - d_self
        d_self = engine.shortest_path(curr)
        d_opp = engine.shortest_path(1 - curr)

        if math.isinf(d_self) or math.isinf(d_opp):
            return 0.0

        h_val = d_opp - d_self
        return math.tanh(0.2 * h_val)

    def order_actions(
        self,
        engine,
        legal_actions: List[Tuple[str, Any]],
        active_player: int,
        pv_action: Optional[Tuple[str, Any]] = None,
    ) -> List[Tuple[str, Any]]:
        """
        Orders actions to maximize Alpha-Beta pruning cutoffs (P5.1 / P5.2).

        Prioritization:
            1. PV move from previous iteration (P5.1)
            2. Immediate winning actions (goal reaching)
            3. Pawn moves ordered by BFS distance reduction progress (P5.2)
            4. Walls touching the opponent's current shortest path (P5.2)
            5. Other walls and replacements ordered by history heuristic (P5.2)
            6. Temporary blocks ordered by history heuristic
            7. Deterministic original index tie-breaker (-idx)
        """
        target_goal = engine.goals[active_player]
        cur_pawn = engine.pawns[active_player]
        cur_d_self = engine.shortest_path(active_player)
        opp_player = 1 - active_player

        # Compute opponent shortest path edges to prioritize blocking walls (P5.2)
        _, opp_edges, _ = bfs_distance_and_path(
            start=engine.pawns[opp_player],
            target_row=engine.goals[opp_player],
            board_size=engine.board_size,
            board=engine.board,
            blocked_cells=engine.blocked_cells,
        )

        scored_actions = []

        for idx, action in enumerate(legal_actions):
            act_type, act_val = action

            # 1. PV move from previous iteration searched first (P5.1)
            if pv_action is not None and action == pv_action:
                score = (10, 0, 0, -idx)
            # 2. Immediate winning move
            elif act_type in ('MOVE', 'HOP') and act_val[0] == target_goal:
                score = (9, 0, 0, -idx)
            # 3. Pawn moves ordered by BFS distance progress (P5.2)
            elif act_type in ('MOVE', 'HOP'):
                # Measure true BFS distance progress rather than Manhattan/row diff
                # Distance to goal from candidate cell:
                # In Manhattan metric on open board: cur_d_self - abs(act_val[0] - target_goal)
                # Fast heuristic progress towards goal:
                dist_gain = abs(cur_pawn[0] - target_goal) - abs(act_val[0] - target_goal)
                score = (5, dist_gain, self.history.get(action, 0), -idx)
            # 4. Walls touching opponent's shortest path
            elif act_type in ('WALL', 'REPLACE'):
                target_w = act_val if act_type == 'WALL' else (act_val[1] if isinstance(act_val[0], tuple) else act_val)
                wr, wc, orient = target_w
                w_edges = engine.board.wall_edges(wr, wc, orient)
                touches_opp = any(norm_edge(u, v) in opp_edges for u, v in w_edges)
                if touches_opp:
                    score = (4, self.history.get(action, 0), 0, -idx)
                else:
                    score = (3, self.history.get(action, 0), 0, -idx)
            # 5. Temporary blocks
            elif act_type == 'BLOCK':
                score = (2, self.history.get(action, 0), 0, -idx)
            else:
                score = (0, 0, 0, -idx)

            scored_actions.append((score, action))

        scored_actions.sort(key=lambda item: item[0], reverse=True)
        return [action for _, action in scored_actions]

    def select_action(self, engine) -> Optional[Tuple[str, Any]]:
        """
        Selects optimal action using Negamax tree search with iterative deepening.
        """
        # Side-to-move assertion (P5.4)
        assert engine.current_player == self.player_id, (
            f"Engine side to move ({engine.current_player}) does not match agent player_id ({self.player_id})"
        )

        start_time = time.time()
        self.nodes_expanded = 0
        self.cutoffs = 0

        legal_actions = engine.get_legal_actions(self.player_id)
        if not legal_actions:
            self.last_decision_time = time.time() - start_time
            return None

        # Trivial optimization: only 1 legal action
        if len(legal_actions) == 1:
            self.last_search_depth = 1
            self.last_decision_time = time.time() - start_time
            self.last_action_value = 0.0
            return legal_actions[0]

        # Establish guaranteed legal fallback action
        fallback_action = legal_actions[0]
        for a in legal_actions:
            if a[0] in ('MOVE', 'HOP') and a[1][0] == engine.goals[self.player_id]:
                fallback_action = a
                break

        deadline = (start_time + self.time_limit) if self.time_limit is not None else None

        best_action = fallback_action
        best_val = -float('inf')
        deepest_completed = 0

        max_d = self.depth
        depth_range = range(1, max_d + 1) if self.time_limit is not None else [max_d]

        for current_d in depth_range:
            if deadline is not None and time.time() >= deadline:
                break
            try:
                # PV-first root search across iterations (P5.1)
                action, val = self._search_root(engine, current_d, deadline, pv_action=self.pv_move)
                if action is not None:
                    best_action = action
                    best_val = val
                    self.pv_move = action
                    deepest_completed = current_d

                # If forced win is found at root, stop early
                if best_val >= self.WIN_SCORE - 100:
                    break
            except SearchTimeout:
                break

        self.last_search_depth = deepest_completed if deepest_completed > 0 else 1
        self.last_decision_time = time.time() - start_time
        self.last_action_value = best_val

        return best_action

    def _search_root(
        self,
        engine,
        depth: int,
        deadline: Optional[float],
        pv_action: Optional[Tuple[str, Any]] = None,
    ) -> Tuple[Optional[Tuple[str, Any]], float]:
        """Root search step returning (best_action, best_value) in Negamax form."""
        legal_actions = engine.get_legal_actions(self.player_id)
        if not legal_actions:
            return None, -self.WIN_SCORE

        if self.order_moves:
            actions = self.order_actions(engine, legal_actions, self.player_id, pv_action=pv_action)
        else:
            actions = legal_actions

        best_action = actions[0]
        alpha = -float('inf')
        beta = float('inf')
        best_val = -float('inf')

        for action in actions:
            if deadline is not None and time.time() >= deadline:
                raise SearchTimeout()

            self.nodes_expanded += 1
            sim = engine.clone() if hasattr(engine, 'clone') else copy.deepcopy(engine)
            sim.apply_action(action, update_clock=False)

            val = -self._negamax(
                sim,
                depth=depth - 1,
                alpha=-beta,
                beta=-alpha,
                ply=1,
                deadline=deadline,
                extensions=0,
            )

            if val > best_val:
                best_val = val
                best_action = action

            if self.use_alpha_beta:
                alpha = max(alpha, val)
                if beta <= alpha:
                    self.cutoffs += 1
                    self.history[action] += depth * depth
                    break

        return best_action, best_val

    def _negamax(
        self,
        engine,
        depth: int,
        alpha: float,
        beta: float,
        ply: int,
        deadline: Optional[float],
        extensions: int = 0,
    ) -> float:
        """
        Recursive Negamax search with Alpha-Beta pruning (P5.4).
        """
        if deadline is not None and time.time() >= deadline:
            raise SearchTimeout()

        # Terminal state check
        if engine.winner is not None or engine.is_draw:
            return self.evaluate(engine, depth, ply)

        # Depth horizon reached: check goal proximity extension (P5.6)
        if depth <= 0:
            if self.extend_goal and extensions < 2:
                d0 = engine.shortest_path(0)
                d1 = engine.shortest_path(1)
                if d0 <= 2 or d1 <= 2:
                    depth = 1
                    extensions += 1
            if depth <= 0:
                return self.evaluate(engine, 0, ply)

        curr = engine.current_player
        legal_actions = engine.get_legal_actions(curr)

        # Rule R3: if side to move has no legal actions, it is a draw (score = 0.0)
        if not legal_actions:
            return 0.0

        if self.order_moves:
            actions = self.order_actions(engine, legal_actions, curr)
        else:
            actions = legal_actions

        best_val = -float('inf')

        for action in actions:
            if deadline is not None and time.time() >= deadline:
                raise SearchTimeout()

            self.nodes_expanded += 1
            sim = engine.clone() if hasattr(engine, 'clone') else copy.deepcopy(engine)
            sim.apply_action(action, update_clock=False)

            val = -self._negamax(
                sim,
                depth=depth - 1,
                alpha=-beta,
                beta=-alpha,
                ply=ply + 1,
                deadline=deadline,
                extensions=extensions,
            )

            if val > best_val:
                best_val = val

            if self.use_alpha_beta:
                alpha = max(alpha, val)
                if beta <= alpha:
                    self.cutoffs += 1
                    # History heuristic d^2 update (P5.2)
                    self.history[action] += depth * depth
                    return alpha

        return alpha if self.use_alpha_beta else best_val

    # Backward compatibility alias
    def _minimax(self, engine, depth, alpha, beta, is_maximizing, ply, deadline):
        """Deprecated compatibility adapter delegating to _negamax."""
        if not is_maximizing:
            return -self._negamax(engine, depth, -beta, -alpha, ply, deadline)
        return self._negamax(engine, depth, alpha, beta, ply, deadline)
