import unittest
from wallz_engine import WallzEngine


class TestHop(unittest.TestCase):
    def test_hop_ability_and_consumption(self):
        """Hop = step across an opponent's wall into the adjacent cell, once per game."""
        engine = WallzEngine(board_size=5)
        self.assertTrue(engine.hop_available[0])
        # No opponent wall next to P0 at the start: no hop
        self.assertFalse([a for a in engine.get_legal_actions(0) if a[0] == 'HOP'])

        engine.apply_action(('MOVE', (1, 2)), update_clock=False)        # P0
        engine.apply_action(('WALL', (1, 2, 'H')), update_clock=False)   # P1 walls P0 in
        self.assertNotIn(('MOVE', (2, 2)), engine.get_legal_actions(0))
        self.assertIn(('HOP', (2, 2)), engine.get_legal_actions(0))

        engine.apply_action(('HOP', (2, 2)), update_clock=False)
        self.assertEqual(engine.pawns[0], (2, 2))
        self.assertFalse(engine.hop_available[0])

        engine.current_player = 0
        self.assertFalse([a for a in engine.get_legal_actions(0) if a[0] == 'HOP'])


if __name__ == '__main__':
    unittest.main()
