import unittest
from wallz_engine import WallzEngine

class TestTerminalStates(unittest.TestCase):
    def test_goal_row_victory(self):
        engine = WallzEngine(board_size=5)
        # Player 0 moves directly to goal row 4
        engine.pawns[0] = (3, 2)
        engine.apply_action(('MOVE', (4, 2)), update_clock=False)
        self.assertEqual(engine.winner, 0)

    def test_five_fold_repetition_draw(self):
        engine = WallzEngine(board_size=5)
        # Oscillate between two states repeatedly
        for _ in range(4):
            engine.apply_action(('MOVE', (0, 1)), update_clock=False) # p0 to (0, 1)
            engine.apply_action(('MOVE', (4, 1)), update_clock=False) # p1 to (4, 1)
            engine.apply_action(('MOVE', (0, 2)), update_clock=False) # p0 to (0, 2)
            engine.apply_action(('MOVE', (4, 2)), update_clock=False) # p1 to (4, 2)
        
        # 5th occurrence of initial state
        self.assertTrue(engine.is_draw, "Game should terminate in draw after 5-fold repetition")

if __name__ == '__main__':
    unittest.main()

