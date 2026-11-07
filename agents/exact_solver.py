"""
Exact Retrograde Solver Adapter over evaluation.state_graph and evaluation.retrograde (P8.1 / C10).

Provides backward-compatible ExactSolver and ExactSolverAgent classes using the
canonical cycle-aware retrograde solver engine with raw keys (canonical=False).
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple, Union

from evaluation.retrograde import solve, Solution, WIN, DRAW, LOSS
from evaluation.state_graph import StateGraph, GraphCaps
from wallz_engine import WallzEngine


class ExactSolver:
    """
    Tier-3 Exact Retrograde Solver for Tractable Miniature Modified Wallz Configurations.
    Adapter over evaluation.state_graph and evaluation.retrograde (P8.1).

    Computes exact game-theoretic values (WIN / LOSS / DRAW) and optimal policies
    via forward reachable state graph generation and backward retrograde propagation.
    """

    def __init__(
        self,
        board_size: int = 3,
        rules: Optional[Dict[str, Any]] = None,
        max_walls: Optional[int] = None,
    ):
        self.board_size = board_size
        self.rules = {
            'A': True,
            'B': True,
            'C': True,
            'K': 3,
            'M': 10,
        }
        if rules:
            self.rules.update(rules)
        if max_walls is not None:
            self.rules['M'] = max_walls
        self.max_walls = max_walls

        # State storage and graph
        # key -> {'status': str, 'distance': int|float, 'best_action': tuple, 'value': float, 'active_player': int}
        self.solutions: Dict[Tuple, Dict[str, Any]] = {}
        self.state_count = 0
        self.win_count = 0
        self.loss_count = 0
        self.draw_count = 0
        self.build_time = 0.0
        self.solve_time = 0.0
        self.is_compiled = False
        self._graph: Optional[StateGraph] = None
        self._solution: Optional[Solution] = None

    def solve(self, initial_engine: Optional[WallzEngine] = None) -> ExactSolver:
        """
        Executes complete forward reachability enumeration and retrograde analysis
        via evaluation.retrograde.
        """
        if initial_engine is not None:
            board_size = initial_engine.board_size
            rules = dict(initial_engine.rules)
            if self.max_walls is not None:
                rules['M'] = self.max_walls
            root_key = initial_engine._get_state_hashable_key()
        else:
            board_size = self.board_size
            rules = dict(self.rules)
            if self.max_walls is not None:
                rules['M'] = self.max_walls
            engine = WallzEngine(board_size=board_size, rules=rules)
            root_key = engine._get_state_hashable_key()

        # Build raw-key state graph (canonical=False keeps raw keys so tests and external tools pass)
        graph = StateGraph(board_size, rules, root_key=root_key, canonical=False).build()
        sol = solve(graph)

        self._graph = graph
        self._solution = sol
        self.build_time = graph.build_seconds
        self.solve_time = sol.solve_seconds

        self.solutions.clear()
        self.win_count = 0
        self.loss_count = 0
        self.draw_count = 0

        for u, key in enumerate(graph.keys):
            st = sol.status(u)
            dist = sol.rank[u]
            if st == 'DRAW' or dist is None:
                dist = float('inf')

            rank_opts = sol.rank_optimal_actions(u)
            best_act = rank_opts[0] if rank_opts else None
            val = float(sol.value[u]) if sol.value[u] is not None else 0.0
            active_p = graph.to_move[u]

            self.solutions[key] = {
                'status': st,
                'distance': dist,
                'best_action': best_act,
                'value': val,
                'active_player': active_p,
            }

            if st == 'WIN':
                self.win_count += 1
            elif st == 'LOSS':
                self.loss_count += 1
            else:
                self.draw_count += 1

        self.state_count = len(self.solutions)
        self.is_compiled = True
        return self

    def is_solved(self, engine_or_key: Union[WallzEngine, Tuple]) -> bool:
        """Checks whether the given engine state is in the solved domain."""
        key = engine_or_key._get_state_hashable_key() if isinstance(engine_or_key, WallzEngine) else engine_or_key
        return key in self.solutions

    def get_state_outcome(self, engine_or_key: Union[WallzEngine, Tuple]) -> Tuple[Optional[str], Optional[Union[int, float]]]:
        """
        Returns (status, distance) from the perspective of the active player.
        status: 'WIN', 'LOSS', or 'DRAW'.
        distance: Plies to terminal outcome (inf for DRAW).
        """
        key = engine_or_key._get_state_hashable_key() if isinstance(engine_or_key, WallzEngine) else engine_or_key
        if key not in self.solutions:
            return None, None
        sol = self.solutions[key]
        return sol['status'], sol['distance']

    def get_value(self, engine_or_key: Union[WallzEngine, Tuple], player: Optional[int] = None) -> Optional[float]:
        """
        Returns exact game-theoretic value (+1.0, 0.0, -1.0).
        If player is specified (0 or 1), evaluates from that player's perspective.
        If player is None, evaluates from active player's perspective.
        """
        key = engine_or_key._get_state_hashable_key() if isinstance(engine_or_key, WallzEngine) else engine_or_key
        if key not in self.solutions:
            return None
        sol = self.solutions[key]
        raw_val = sol['value']
        active_p = sol['active_player']

        if player is None or player == active_p:
            return raw_val
        else:
            return -raw_val

    def get_optimal_action(self, engine: WallzEngine) -> Optional[Tuple[str, Any]]:
        """
        Returns the provably optimal action for the current active player.
        Returns None if state is outside the solved domain or terminal.
        """
        if not isinstance(engine, WallzEngine):
            raise TypeError("Expected WallzEngine instance.")
        key = engine._get_state_hashable_key()
        if key not in self.solutions:
            return None
        return self.solutions[key]['best_action']

    def get_stats(self) -> Dict[str, Any]:
        """Returns solver complexity, metrics, and coverage summary."""
        stats = {
            'board_size': self.board_size,
            'rules': dict(self.rules),
            'max_walls': self.max_walls,
            'state_count': self.state_count,
            'win_count': self.win_count,
            'loss_count': self.loss_count,
            'draw_count': self.draw_count,
            'build_time_sec': self.build_time,
            'solve_time_sec': self.solve_time,
            'total_time_sec': self.build_time + self.solve_time,
            'is_compiled': self.is_compiled,
        }
        if self._solution is not None:
            stats['draw_counts'] = self._solution.draw_counts()
        return stats


class ExactSolverAgent:
    """
    Tier-3 Exact Ground-Truth Baseline Agent.

    Plays provably optimal actions derived from the retrograde table.
    Conforms to standard agent interface: select_action(engine).
    """

    def __init__(
        self,
        player_id: int,
        solver: Optional[ExactSolver] = None,
        board_size: int = 3,
        rules: Optional[Dict[str, Any]] = None,
        max_walls: Optional[int] = None,
    ):
        self.player_id = player_id
        if solver is not None:
            self.solver = solver
        else:
            self.solver = ExactSolver(board_size=board_size, rules=rules, max_walls=max_walls)
            self.solver.solve()

    def select_action(self, engine: WallzEngine) -> Optional[Tuple[str, Any]]:
        """
        Selects the exact optimal action from the solved retrograde solution.
        If the state is outside the solved domain, returns a legal fallback action.
        """
        if not self.solver.is_solved(engine):
            legal = engine.get_legal_actions(self.player_id)
            return legal[0] if legal else None

        opt_action = self.solver.get_optimal_action(engine)
        if opt_action is not None:
            return opt_action

        # Fallback for terminal states or unassigned actions
        legal = engine.get_legal_actions(self.player_id)
        return legal[0] if legal else None

    def stats(self) -> Dict[str, Any]:
        return self.solver.get_stats()

    def get_stats(self) -> Dict[str, Any]:
        return self.solver.get_stats()
