import unittest
from wallz_engine import WallzEngine

class TestMovement(unittest.TestCase):
    def test_orthogonal_moves(self):
        engine = WallzEngine(board_size=5)
        # Player 0 at (0, 2) can move to (1, 2), (0, 1), (0, 3)
        legal_actions = engine.get_legal_actions(0)
        pawn_moves = [a[1] for a in legal_actions if a[0] == 'MOVE']
        self.assertIn((1, 2), pawn_moves)
        self.assertIn((0, 1), pawn_moves)
        self.assertIn((0, 3), pawn_moves)
        self.assertNotIn((-1, 2), pawn_moves)

    def test_wall_blocks_movement(self):
        engine = WallzEngine(board_size=5)
        # Place horizontal wall between (0, 2) and (1, 2)
        engine.walls[1].append((0, 2, 'H'))
        legal_actions = engine.get_legal_actions(0)
        pawn_moves = [a[1] for a in legal_actions if a[0] == 'MOVE']
        self.assertNotIn((1, 2), pawn_moves)

if __name__ == '__main__':
    unittest.main()

