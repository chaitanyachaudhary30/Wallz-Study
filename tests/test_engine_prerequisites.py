"""Regression tests for engine fixes needed by the match runner and exact solver."""
import unittest

from wallz_engine import WallzEngine


class TestBlockTimer(unittest.TestCase):
    def test_k1_block_lasts_exactly_one_ply(self):
        e = WallzEngine(board_size=5, rules={'K': 1})
        e.apply_action(('BLOCK', (3, 2)), update_clock=False)
        # Opponent's next ply: (3, 2) is impassable
        self.assertEqual(e.blocked_cells, {(3, 2): 1})
        self.assertNotIn(('MOVE', (3, 2)), e.get_legal_actions())
        e.apply_action(('MOVE', (4, 1)), update_clock=False)
        self.assertEqual(e.blocked_cells, {})

    def test_block_lasts_k_plies(self):
        for K in (1, 2, 3, 4):
            e = WallzEngine(board_size=5, rules={'K': K, 'B': False})
            e.apply_action(('BLOCK', (2, 0)), update_clock=False)
            plies_blocked = 0
            moves = [('MOVE', (4, 1)), ('MOVE', (0, 1)), ('MOVE', (4, 2)), ('MOVE', (0, 2))] * 2
            for a in moves:
                if (2, 0) not in e.blocked_cells:
                    break
                plies_blocked += 1
                e.apply_action(a, update_clock=False)
            self.assertEqual(plies_blocked, K)

    def test_k_must_be_positive(self):
        with self.assertRaises(ValueError):
            WallzEngine(board_size=5, rules={'K': 0})

    def test_block_unavailable_when_rule_off_or_used(self):
        e = WallzEngine(board_size=5, rules={'C': False})
        with self.assertRaises(ValueError):
            e.apply_action(('BLOCK', (2, 2)), update_clock=False)


class TestWallBudget(unittest.TestCase):
    def test_configurable_budget(self):
        e = WallzEngine(board_size=5, rules={'M': 2})
        self.assertEqual(e.wall_inventory, {0: 2, 1: 2})
        self.assertEqual(e.max_walls, 2)

    def test_budget_is_hard_cap_without_rule_a(self):
        e = WallzEngine(board_size=5, rules={'A': False, 'M': 1})
        e.apply_action(('WALL', (2, 0, 'H')), update_clock=False)
        e.apply_action(('MOVE', (4, 1)), update_clock=False)
        kinds = {a[0] for a in e.get_legal_actions()}
        self.assertNotIn('WALL', kinds)
        self.assertNotIn('REPLACE', kinds)

    def test_zero_budget(self):
        e = WallzEngine(board_size=3, rules={'M': 0})
        self.assertFalse(any(a[0] == 'WALL' for a in e.get_legal_actions()))


class TestHopAndMoveValidation(unittest.TestCase):
    def test_hop_requires_opponent_wall(self):
        # Proposal LegalHop (docs/game_rules.md): a hop bypasses an opponent's wall.
        e = WallzEngine(board_size=5)
        self.assertFalse([a for a in e.get_legal_actions() if a[0] == 'HOP'])
        e.walls[1] = [(0, 2, 'H')]          # opponent wall between (0,2) and (1,2)
        self.assertIn(('HOP', (1, 2)), e.get_legal_actions())
        e.walls = {0: [(0, 2, 'H')], 1: []}  # own wall: no hop
        self.assertNotIn(('HOP', (1, 2)), e.get_legal_actions())

    def test_hop_into_sealed_pocket_is_illegal(self):
        e = WallzEngine(board_size=5, rules={'C': False})
        e.pawns[0] = (0, 1)
        # Pocket {(0,0),(1,0)} sealed by opponent walls V(0,0) and H(1,0)
        e.walls[1] = [(0, 0, 'V'), (1, 0, 'H')]
        self.assertTrue(e.is_path_open(0) and e.is_path_open(1))
        self.assertNotIn(('HOP', (0, 0)), e.get_legal_actions())
        with self.assertRaises(ValueError):
            e.apply_action(('HOP', (0, 0)), update_clock=False)

    def test_illegal_move_rejected_without_mutation(self):
        e = WallzEngine(board_size=5)
        key = e._get_state_hashable_key()
        for bad in [('MOVE', (4, 4)), ('MOVE', (4, 2)), ('HOP', (0, 3)), ('JUMP', (1, 2))]:
            with self.assertRaises(ValueError):
                e.apply_action(bad, update_clock=False)
            self.assertEqual(e._get_state_hashable_key(), key)

    def test_move_onto_opponent_rejected(self):
        e = WallzEngine(board_size=3, rules={'B': False})
        e.pawns[0] = (1, 1)
        with self.assertRaises(ValueError):
            e.apply_action(('MOVE', (2, 1)), update_clock=False)


class TestTerminalHandling(unittest.TestCase):
    def test_no_actions_after_goal(self):
        e = WallzEngine(board_size=3, rules={'B': False})
        e.apply_action(('MOVE', (1, 1)), update_clock=False)
        e.apply_action(('MOVE', (2, 0)), update_clock=False)
        e.apply_action(('MOVE', (2, 1)), update_clock=False)
        self.assertEqual(e.winner, 0)
        self.assertEqual(e.terminal_reason, 'goal')
        self.assertEqual(e.get_legal_actions(), [])
        with self.assertRaises(ValueError):
            e.apply_action(('MOVE', (1, 1)), update_clock=False)

    def _stuck_position(self):
        # P0 at (1, 0): up and down sealed by walls, right neighbour (1, 1) will hold P1.
        e = WallzEngine(board_size=3, rules={'A': False, 'B': False, 'C': False, 'M': 0})
        e.walls[1] = [(0, 0, 'H'), (1, 0, 'H')]
        e.pawns[0] = (1, 0)
        e.pawns[1] = (1, 2)
        e.current_player = 1
        e.state_history = {e._get_state_hashable_key(): 1}
        return e

    def test_no_legal_action_is_draw(self):
        e = self._stuck_position()
        self.assertTrue(e.is_path_open(0) and e.is_path_open(1))
        e.apply_action(('MOVE', (1, 1)), update_clock=False)
        self.assertTrue(e.is_draw)
        self.assertIsNone(e.winner)
        self.assertEqual(e.terminal_reason, 'no_legal_actions')
        self.assertEqual(e.get_legal_actions(), [])

    def test_repetition_reason(self):
        e = WallzEngine(board_size=5)
        for _ in range(4):
            for a in [('MOVE', (0, 1)), ('MOVE', (4, 1)), ('MOVE', (0, 2)), ('MOVE', (4, 2))]:
                e.apply_action(a, update_clock=False)
        self.assertTrue(e.is_draw)
        self.assertEqual(e.terminal_reason, 'repetition')

    def test_history_counts_post_transition_state(self):
        e = WallzEngine(board_size=3)
        e.apply_action(('MOVE', (0, 0)), update_clock=False)
        self.assertEqual(e.state_history[e._get_state_hashable_key()], 1)


class TestSnapshots(unittest.TestCase):
    def _played(self):
        e = WallzEngine(board_size=5, rules={'K': 2})
        for a in [('WALL', (1, 0, 'H')), ('WALL', (2, 3, 'V')), ('WALL', (2, 0, 'V')),
                  ('BLOCK', (2, 2)), ('REPLACE', (0, 3, 'H'))]:
            e.apply_action(a, update_clock=False)
        return e

    def test_restore_round_trip(self):
        e = self._played()
        key = e._get_state_hashable_key()
        r = WallzEngine(board_size=5, rules={'K': 2}).restore_state_key(key)
        self.assertEqual(r._get_state_hashable_key(), key)
        self.assertEqual(sorted(r.get_legal_actions()), sorted(e.get_legal_actions()))
        self.assertEqual(r.state_history, {key: 1})
        self.assertEqual(r.blocked_cells, e.blocked_cells)

    def test_restored_transitions_are_deterministic(self):
        key = self._played()._get_state_hashable_key()
        base = WallzEngine(board_size=5, rules={'K': 2})
        for a in base.restore_state_key(key).get_legal_actions():
            k1 = base.restore_state_key(key).clone()
            k1.apply_action(a, update_clock=False)
            k2 = WallzEngine(board_size=5, rules={'K': 2}).restore_state_key(key)
            k2.apply_action(a, update_clock=False)
            self.assertEqual(k1._get_state_hashable_key(), k2._get_state_hashable_key())

    def test_wall_order_and_latest_pointer_in_key(self):
        e = self._played()
        key = e._get_state_hashable_key()
        # P0's REPLACE moved its latest wall (2, 0, 'V') in place; the older wall keeps slot 0
        self.assertEqual(key[2], ((1, 0, 'H'), (0, 3, 'H')))
        self.assertEqual(key[3], ((2, 3, 'V'),))
        self.assertEqual(key[6], (0, 3, 'H'))
        self.assertEqual(key[7], (2, 3, 'V'))
        swapped = list(key)
        swapped[2] = tuple(reversed(key[2]))
        self.assertNotEqual(tuple(swapped), key)

    def test_clock_untouched_without_update(self):
        e = WallzEngine(board_size=5)
        before = dict(e.time_remaining)
        e.apply_action(('MOVE', (1, 2)), update_clock=False)
        self.assertEqual(e.time_remaining, before)

    def test_restore_rejects_wrong_board(self):
        key = WallzEngine(board_size=5)._get_state_hashable_key()
        with self.assertRaises(ValueError):
            WallzEngine(board_size=3).restore_state_key(key)

    def test_has_legal_action_false_when_terminal(self):
        e = WallzEngine(board_size=3, rules={'B': False})
        for a in [('MOVE', (1, 1)), ('MOVE', (2, 0)), ('MOVE', (2, 1))]:
            e.apply_action(a, update_clock=False)
        self.assertFalse(e.has_legal_action())

    def test_restore_detects_terminal(self):
        e = WallzEngine(board_size=3)
        key = list(e._get_state_hashable_key())
        key[0] = (2, 0)
        r = WallzEngine(board_size=3).restore_state_key(tuple(key))
        self.assertEqual(r.winner, 0)


if __name__ == '__main__':
    unittest.main()
