"""Stage 3 (Member 1, new contribution): optional hop-aware minimax evaluator.

Run: python -m unittest tests.test_minimax_hop_eval
"""
import random
import unittest
from collections import deque

from agents.evaluation_fn import HEvaluator, HopAwareEvaluator
from agents.minimax import MinimaxAgent
from evaluation.match_runner import MatchConfig, make_agent
from wallz_engine import WallzEngine

hop_distance = HopAwareEvaluator.distance


def make(n=5, **rules):
    base = {'A': True, 'B': True, 'C': True, 'K': 3, 'M': 10}
    base.update(rules)
    return WallzEngine(board_size=n, rules=base)


def set_position(engine, pawns, walls0=(), walls1=(), blocks=None, to_move=0):
    engine.pawns = {0: pawns[0], 1: pawns[1]}
    engine.walls = {0: list(walls0), 1: list(walls1)}
    engine.latest_wall = {0: None, 1: None}
    engine.blocked_cells = dict(blocks or {})
    engine.current_player = to_move
    engine._sync_board()
    return engine


def random_positions(n, rules, count, seed, max_plies=14):
    rng = random.Random(seed)
    out = []
    while len(out) < count:
        e = make(n, **rules)
        for _ in range(rng.randint(0, max_plies)):
            if e.is_terminal():
                break
            e.apply_action(rng.choice(e.get_legal_actions()), update_clock=False)
        if not e.is_terminal():
            out.append(e)
    return out


def brute_force_distance(engine, player):
    """BFS over (cell, hop still available); neighbours are the engine's own legal MOVE/HOP
    for ``player`` standing on that cell, with the other pawn and all blocks fixed."""
    probe = engine.clone()
    probe.winner, probe.is_draw = None, False
    start = (engine.pawns[player], bool(engine.rules['B'] and engine.hop_available[player]))
    seen = {start}
    frontier = deque([(start, 0)])
    while frontier:
        (cell, hop), d = frontier.popleft()
        if cell[0] == engine.goals[player]:
            return float(d)
        probe.pawns[player] = cell
        probe.hop_available[player] = hop
        for kind, nxt in probe.pawn_actions(player):
            state = (nxt, hop and kind != 'HOP')
            if state not in seen:
                seen.add(state)
                frontier.append((state, d + 1))
    return float('inf')


def other_pawn_as_obstacle(engine, player):
    """Clone in which the other pawn's cell is blocked, matching 'other pawn fixed'."""
    e = engine.clone()
    e.blocked_cells[engine.pawns[1 - player]] = 1
    return e


class TestHopDistance(unittest.TestCase):
    def test_never_longer_than_plain(self):
        for e in random_positions(5, {'K': 3, 'M': 10}, 60, seed=1):
            for p in (0, 1):
                self.assertLessEqual(hop_distance(e, p), e.shortest_path(p))

    def test_equal_when_hop_used_or_rule_off_or_no_opponent_wall(self):
        for e in random_positions(5, {'K': 3, 'M': 10}, 40, seed=2):
            for p in (0, 1):
                used = e.clone()
                used.hop_available[p] = False
                self.assertEqual(hop_distance(used, p), used.shortest_path(p))
                no_opp = e.clone()
                no_opp.walls[1 - p] = []
                no_opp._sync_board()
                self.assertEqual(hop_distance(no_opp, p), no_opp.shortest_path(p))
        for e in random_positions(5, {'B': False, 'K': 3, 'M': 10}, 30, seed=3):
            for p in (0, 1):
                self.assertEqual(hop_distance(e, p), e.shortest_path(p))

    def test_hand_built_shortcut_over_opponent_wall_only(self):
        # P0 at (0,2), goal row 4. Walls block (0,1)-(1,1), (0,2)-(1,2), (0,3)-(1,3), (0,4)-(1,4).
        walls = [(0, 1, 'H'), (0, 3, 'H')]
        opp = set_position(make(5), {0: (0, 2), 1: (4, 2)}, walls1=walls)
        self.assertEqual(opp.shortest_path(0), 6.0)   # around via (0,0)
        self.assertEqual(hop_distance(opp, 0), 4.0)   # hop (0,2)->(1,2), then 3 steps
        own = set_position(make(5), {0: (0, 2), 1: (4, 2)}, walls0=walls)
        self.assertEqual(hop_distance(own, 0), 6.0)   # own walls: no shortcut
        opp.hop_available[0] = False
        self.assertEqual(hop_distance(opp, 0), 6.0)

    def test_blocked_landing_cell_gives_no_shortcut(self):
        walls = [(0, 1, 'H'), (0, 3, 'H')]
        e = set_position(make(5), {0: (0, 2), 1: (4, 2)}, walls1=walls, blocks={(1, 2): 2})
        # (1,2) is blocked, so the best hop is one step sideways first: (0,1)->(1,1), then 3 steps.
        self.assertEqual(hop_distance(e, 0), 5.0)
        e.blocked_cells.update({(1, 1): 2, (1, 3): 2, (1, 4): 2})
        self.assertEqual(hop_distance(e, 0), e.shortest_path(0))

    def test_equals_engine_brute_force(self):
        checked = shorter = 0
        for n, rules, seed in ((4, {'K': 2, 'M': 3}, 10), (5, {'K': 3, 'M': 10}, 11),
                               (5, {'K': 1, 'M': 4, 'A': False}, 12)):
            for e in random_positions(n, rules, 40, seed=seed, max_plies=20):
                for p in (0, 1):
                    mine = hop_distance(other_pawn_as_obstacle(e, p), p)
                    self.assertEqual(mine, brute_force_distance(e, p), (n, rules, p, e.get_state()))
                    checked += 1
                    shorter += mine < other_pawn_as_obstacle(e, p).shortest_path(p)
        self.assertGreater(checked, 200)
        self.assertGreater(shorter, 0)  # the hop actually mattered in some positions


class TestSelection(unittest.TestCase):
    def test_default_spec_unchanged(self):
        cfg = MatchConfig(board_size=5)
        agent, identity = make_agent('minimax:depth=2', 0, cfg)
        self.assertIsInstance(agent.evaluator, HEvaluator)
        self.assertEqual(identity['params'], {'depth': 2, 'time_limit': None, 'extend_goal': False})

    def test_hop_spec_selects_hop_evaluator(self):
        agent, identity = make_agent('minimax:depth=2,eval=hop', 1, MatchConfig(board_size=5))
        self.assertIsInstance(agent.evaluator, HopAwareEvaluator)
        self.assertEqual(identity['params']['eval'], 'hop')
        with self.assertRaises(ValueError):
            make_agent('minimax:depth=2,eval=nope', 0, MatchConfig(board_size=5))

    def test_default_agent_matches_plain_minimax(self):
        for e in random_positions(5, {'K': 3, 'M': 10}, 15, seed=4):
            p = e.current_player
            a, _ = make_agent('minimax:depth=2', p, MatchConfig(board_size=5))
            ref = MinimaxAgent(p, depth=2)
            self.assertEqual(a.select_action(e.clone()), ref.select_action(e.clone()))
            self.assertEqual(a.last_action_value, ref.last_action_value)

    def test_hop_evaluator_scores_hop_shortcut(self):
        walls = [(0, 1, 'H'), (0, 3, 'H')]
        e = set_position(make(5), {0: (0, 2), 1: (4, 2)}, walls1=walls)
        self.assertGreater(HopAwareEvaluator().evaluate(e, 0), HEvaluator().evaluate(e, 0))


if __name__ == '__main__':
    unittest.main()
