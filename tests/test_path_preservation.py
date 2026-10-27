import unittest
from wallz_engine import WallzEngine

class TestPathPreservation(unittest.TestCase):
    def test_cannot_completely_block_player(self):
        engine = WallzEngine(board_size=3)
        # On a 3x3 board, Player 0 starts at (0, 1), goal is row 2
        # Horizontal walls across row 0 would seal Player 0 in row 0
        engine.walls[0].append((0, 0, 'H'))
        # Placing (0, 1, 'H') would completely block row 0 from row 1 & 2
        is_valid = engine._is_valid_wall_placement(0, 0, 1, 'H')
        self.assertFalse(is_valid, "Path preservation rule violated: wall seals off goal")

if __name__ == '__main__':
    unittest.main()

