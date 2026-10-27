import unittest
from wallz_engine import WallzEngine

class TestHop(unittest.TestCase):
    def test_hop_ability_and_consumption(self):
        engine = WallzEngine(board_size=5)
        self.assertTrue(engine.hop_available[0])
        # Hop is a 2-step jump in orthogonal directions
        legal_actions = engine.get_legal_actions(0)
        hop_actions = [a for a in legal_actions if a[0] == 'HOP']
        self.assertIn(('HOP', (2, 2)), hop_actions)

        # Apply hop
        engine.apply_action(('HOP', (2, 2)), update_clock=False)
        self.assertEqual(engine.pawns[0], (2, 2))
        self.assertFalse(engine.hop_available[0])

        # Hop no longer available for player 0 on their next turn
        engine.current_player = 0
        actions_after = engine.get_legal_actions(0)
        hops_after = [a for a in actions_after if a[0] == 'HOP']
        self.assertEqual(len(hops_after), 0)

if __name__ == '__main__':
    unittest.main()

