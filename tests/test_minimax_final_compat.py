"""Stage 2 (Member 1): MinimaxAgent against the frozen Stage 1 engine.

Covers Task 1 (compatibility) and the testable parts of the Task 2 audit:
legal actions only, terminal handling, K-ply blocks, wall budget M, no-legal-move
draw, the one-step HOP across an opponent wall, alpha-beta == full-width minimax,
iterative deepening keeps the last completed depth, and the goal-proximity extension.

Run: python -m unittest tests.test_minimax_final_compat
"""
import random
import unittest
from unittest import mock

from agents.minimax import MinimaxAgent, SearchTimeout
from wallz_engine import WallzEngine


def make(n=5, **rules):
    base = {'A': True, 'B': True, 'C': True, 'K': 3, 'M': 10}
    base.update(rules)
    return WallzEngine(board_size=n, rules=base)


def play(engine, *actions):
    for a in actions:
        engine.apply_action(a, update_clock=False)
    return engine


def set_position(engine, pawns, walls0=(), walls1=(), blocks=None, to_move=0):
    """Direct state set-up for hand-built positions (tests only)."""
    engine.pawns = {0: pawns[0], 1: pawns[1]}
    engine.walls = {0: list(walls0), 1: list(walls1)}
    engine.latest_wall = {0: None, 1: None}
    engine.blocked_cells = dict(blocks or {})
    engine.current_player = to_move
    engine._sync_board()
    return engine


def random_positions(n, rules, count, seed, max_plies=10):
    """Non-terminal positions reached by uniform random play (any action type)."""
    rng = random.Random(seed)
    out = []
    tries = 0
    while len(out) < count and tries < count * 20:
        tries += 1
        e = make(n, **rules)
        for _ in range(rng.randint(0, max_plies)):
            if e.is_terminal():
                break
            e.apply_action(rng.choice(e.get_legal_actions()), update_clock=False)
        if not e.is_terminal() and len(e.get_legal_actions()) > 1:
            out.append(e)
    return out


def ref_negamax(agent, engine, depth, ply):
    """Plain full-width negamax over the engine API (no pruning, no ordering)."""
    if engine.winner is not None or engine.is_draw or depth <= 0:
        return agent.evaluate(engine, depth, ply)
    best = -float('inf')
    for a in engine.get_legal_actions():
        child = engine.clone()
        child.apply_action(a, update_clock=False)
        best = max(best, -ref_negamax(agent, child, depth - 1, ply + 1))
    return best


def ref_root_value(agent, engine, depth):
    return max(-ref_negamax(agent, play(engine.clone(), a), depth - 1, 1)
               for a in engine.get_legal_actions())


CONFIGS = [
    (3, {'K': 1, 'M': 1}),
    (3, {'K': 3, 'M': 0}),
    (4, {'K': 2, 'M': 2}),
    (4, {'K': 4, 'M': 1, 'A': False}),
    (5, {'K': 3, 'M': 10}),
    (5, {'K': 1, 'M': 3, 'B': False, 'C': False}),
]


class TestLegalityAndTermination(unittest.TestCase):
    def test_full_games_only_legal_actions(self):
        """Minimax vs minimax / random to the end: every chosen action is in the legal list."""
        for n, rules in CONFIGS:
            for seed in range(2):
                with self.subTest(n=n, rules=rules, seed=seed):
                    e = make(n, **rules)
                    rng = random.Random(seed)
                    agents = {p: MinimaxAgent(p, depth=2 if n <= 4 else 1) for p in (0, 1)}
                    plies = 0
                    while not e.is_terminal() and plies < 60:
                        p = e.current_player
                        legal = e.get_legal_actions()
                        if seed == 1 and p == 1:
                            a = rng.choice(legal)
                        else:
                            a = agents[p].select_action(e)
                            self.assertIn(a, legal)
                            if a[0] == 'WALL':
                                self.assertGreater(e.wall_inventory[p], 0)
                        play(e, a)
                        plies += 1
                        for t in e.blocked_cells.values():
                            self.assertTrue(1 <= t <= e.rules['K'])

    def test_returns_none_on_terminal_states(self):
        won = play(make(3), ('MOVE', (1, 1)), ('MOVE', (2, 0)), ('MOVE', (2, 1)))
        self.assertEqual(won.winner, 0)
        self.assertIsNone(MinimaxAgent(1, depth=2).select_action(won))

        trapped = make(3, A=False, B=False, C=False, M=0)
        set_position(trapped, {0: (0, 0), 1: (2, 0)}, walls0=[(0, 0, 'V')], to_move=1)
        play(trapped, ('MOVE', (1, 0)))
        self.assertEqual(trapped.terminal_reason, 'no_legal_actions')
        self.assertIsNone(MinimaxAgent(0, depth=2).select_action(trapped))

    def test_search_never_applies_an_action_to_a_terminal_engine(self):
        calls = {'terminal': 0}
        original = WallzEngine.apply_action

        def guarded(self, action, update_clock=True):
            if self.is_terminal():
                calls['terminal'] += 1
            return original(self, action, update_clock)

        # Positions one or two plies from a goal for either side.
        positions = []
        for e in random_positions(3, {'K': 1, 'M': 1}, 15, seed=7, max_plies=6):
            positions.append(e)
        with mock.patch.object(WallzEngine, 'apply_action', guarded):
            for e in positions:
                MinimaxAgent(e.current_player, depth=3).select_action(e)
        self.assertEqual(calls['terminal'], 0)

    def test_negamax_on_terminal_does_not_expand(self):
        won = play(make(3), ('MOVE', (1, 1)), ('MOVE', (2, 0)), ('MOVE', (2, 1)))
        agent = MinimaxAgent(1, depth=3)
        with mock.patch.object(WallzEngine, 'get_legal_actions', side_effect=AssertionError):
            v = agent._negamax(won, depth=3, alpha=-float('inf'), beta=float('inf'), ply=1, deadline=None)
        # Side to move (P1) has lost.
        self.assertEqual(v, -(MinimaxAgent.WIN_SCORE - 1))


class TestRulesSeenBySearch(unittest.TestCase):
    def test_wall_budget_respected(self):
        for m in (0, 1):
            e = make(4, M=m, K=2)
            agents = {p: MinimaxAgent(p, depth=2) for p in (0, 1)}
            for _ in range(30):
                if e.is_terminal():
                    break
                p = e.current_player
                a = agents[p].select_action(e)
                if e.wall_inventory[p] == 0:
                    self.assertNotEqual(a[0], 'WALL')
                play(e, a)
            self.assertGreaterEqual(min(e.wall_inventory.values()), 0)
            self.assertLessEqual(len(e.walls[0]) + len(e.walls[1]), 2 * m)

    def test_no_legal_move_draw_scores_zero(self):
        trapped = make(3, A=False, B=False, C=False, M=0)
        set_position(trapped, {0: (0, 0), 1: (2, 0)}, walls0=[(0, 0, 'V')], to_move=1)
        play(trapped, ('MOVE', (1, 0)))
        agent = MinimaxAgent(0, depth=2)
        self.assertEqual(agent.evaluate(trapped, 0, 1), 0.0)
        self.assertEqual(agent._negamax(trapped, 2, -float('inf'), float('inf'), 1, None), 0.0)

    def test_hop_across_opponent_wall_found_as_win(self):
        # P1's wall blocks (1,1)-(2,1) and (1,2)-(2,2); only HOP reaches row 2 now.
        e = make(3, M=1)
        set_position(e, {0: (1, 1), 1: (2, 0)}, walls1=[(1, 1, 'H')], to_move=0)
        self.assertIn(('HOP', (2, 1)), e.get_legal_actions())
        agent = MinimaxAgent(0, depth=2)
        self.assertEqual(agent.select_action(e), ('HOP', (2, 1)))
        self.assertEqual(agent.last_action_value, MinimaxAgent.WIN_SCORE - 1)

    def test_no_hop_across_own_wall(self):
        e = make(3, M=1)
        set_position(e, {0: (1, 1), 1: (2, 0)}, walls0=[(1, 1, 'H')], to_move=0)
        a = MinimaxAgent(0, depth=2).select_action(e)
        self.assertNotEqual(a[0], 'HOP')
        self.assertIn(a, e.get_legal_actions())

    def test_block_positions_match_reference_search(self):
        """States with live blocks (timers 1..K): search value == full-width reference."""
        checked = 0
        for k in (1, 2, 3, 4):
            for e in random_positions(4, {'K': k, 'M': 1}, 25, seed=100 + k, max_plies=8):
                if not e.blocked_cells:
                    continue
                agent = MinimaxAgent(e.current_player, depth=2)
                agent.select_action(e.clone())
                self.assertAlmostEqual(agent.last_action_value, ref_root_value(agent, e, 2), places=12)
                checked += 1
        self.assertGreater(checked, 10)

    def test_block_expires_inside_search_tree(self):
        """A block at timer 1 is gone one ply later inside the search (engine-driven)."""
        e = make(3, M=0, A=False, B=False)
        set_position(e, {0: (1, 1), 1: (2, 0)}, blocks={(2, 1): 1}, to_move=1)
        e.block_available = {0: False, 1: False}
        self.assertFalse(e._is_valid_move(0, (2, 1)))
        child = play(e.clone(), ('MOVE', (1, 0)))
        self.assertNotIn((2, 1), child.blocked_cells)
        self.assertIn(('MOVE', (2, 1)), child.get_legal_actions())
        agent = MinimaxAgent(0, depth=1)
        agent.select_action(child)
        self.assertEqual(agent.last_action_value, MinimaxAgent.WIN_SCORE - 1)


class TestSearchCorrectness(unittest.TestCase):
    def test_alpha_beta_equals_full_width(self):
        for n, rules, depth, count in ((3, {'K': 1, 'M': 1}, 3, 12), (4, {'K': 2, 'M': 2}, 2, 12),
                                       (3, {'K': 3, 'M': 0}, 4, 8)):
            for i, e in enumerate(random_positions(n, rules, count, seed=depth * 31 + n)):
                p = e.current_player
                vals = []
                for ab, order in ((True, True), (True, False), (False, True), (False, False)):
                    a = MinimaxAgent(p, depth=depth, use_alpha_beta=ab, order_moves=order)
                    a.select_action(e.clone())
                    vals.append(a.last_action_value)
                with self.subTest(n=n, depth=depth, i=i):
                    for v in vals[1:]:
                        self.assertAlmostEqual(vals[0], v, places=12)

    def test_iterative_deepening_keeps_last_completed_depth(self):
        e = random_positions(4, {'K': 2, 'M': 2}, 1, seed=5)[0]
        p = e.current_player
        d1 = MinimaxAgent(p, depth=1)
        a1 = d1.select_action(e.clone())

        class TimesOutAtDepth2(MinimaxAgent):
            def _search_root(self, engine, depth, deadline, pv_action=None):
                if depth == 2:
                    raise SearchTimeout()
                return super()._search_root(engine, depth, deadline, pv_action)

        idd = TimesOutAtDepth2(p, depth=3, time_limit=100.0)
        self.assertEqual(idd.select_action(e.clone()), a1)
        self.assertEqual(idd.last_search_depth, 1)
        self.assertEqual(idd.last_action_value, d1.last_action_value)

    def test_iterative_deepening_value_matches_fixed_depth(self):
        for e in random_positions(4, {'K': 2, 'M': 2}, 5, seed=9):
            p = e.current_player
            fixed = MinimaxAgent(p, depth=2)
            fixed.select_action(e.clone())
            timed = MinimaxAgent(p, depth=2, time_limit=1000.0)
            timed.select_action(e.clone())
            if timed.last_search_depth == 2:
                self.assertAlmostEqual(timed.last_action_value, fixed.last_action_value, places=12)

    def test_goal_extension_sees_win_beyond_horizon(self):
        # P0 two steps from goal, P1 cannot wall or block: forced win in 3 plies.
        e = make(5, M=0, C=False, B=False)
        set_position(e, {0: (2, 0), 1: (4, 4)}, to_move=0)
        plain = MinimaxAgent(0, depth=1)
        plain.select_action(e.clone())
        ext = MinimaxAgent(0, depth=1, extend_goal=True)
        ext.select_action(e.clone())
        self.assertLess(plain.last_action_value, 1.0)
        self.assertEqual(ext.last_action_value, MinimaxAgent.WIN_SCORE - 3)


if __name__ == '__main__':
    unittest.main()
