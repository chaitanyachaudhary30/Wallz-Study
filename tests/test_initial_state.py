import unittest
from wallz_engine import WallzEngine

class TestInitialState(unittest.TestCase):
    def test_default_initialization(self):
        engine = WallzEngine(board_size=9)
        self.assertEqual(engine.board_size, 9)
        self.assertEqual(engine.pawns[0], (0, 4))
        self.assertEqual(engine.pawns[1], (8, 4))
        self.assertEqual(engine.goals[0], 8)
        self.assertEqual(engine.goals[1], 0)
        self.assertEqual(engine.current_player, 0)
        self.assertEqual(engine.turn_count, 0)
        self.assertIsNone(engine.winner)
        self.assertFalse(engine.is_draw)
        self.assertTrue(engine.hop_available[0])
        self.assertTrue(engine.block_available[0])

    def test_custom_board_size(self):
        engine = WallzEngine(board_size=5)
        self.assertEqual(engine.board_size, 5)
        self.assertEqual(engine.pawns[0], (0, 2))
        self.assertEqual(engine.pawns[1], (4, 2))
        self.assertEqual(engine.goals[0], 4)
        self.assertEqual(engine.goals[1], 0)

if __name__ == '__main__':
    unittest.main()

