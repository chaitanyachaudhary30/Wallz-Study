import unittest
from wallz_engine import WallzEngine

class TestTemporaryBlocks(unittest.TestCase):
    def test_block_placement_and_decay(self):
        engine = WallzEngine(board_size=5, rules={'C': True, 'K': 3})
        self.assertTrue(engine.block_available[0])

        # Player 0 blocks cell (2, 2)
        engine.apply_action(('BLOCK', (2, 2)), update_clock=False)
        self.assertFalse(engine.block_available[0])
        # Remaining turns decrements by 1 immediately after turn
        self.assertEqual(engine.blocked_cells[(2, 2)], 2)

        # Player 1 takes a move
        engine.apply_action(('MOVE', (3, 2)), update_clock=False)
        self.assertEqual(engine.blocked_cells[(2, 2)], 1)

        # Player 0 takes a move -> timer expires and block is removed
        engine.apply_action(('MOVE', (1, 2)), update_clock=False)
        self.assertNotIn((2, 2), engine.blocked_cells)

    def test_block_path_preservation(self):
        """Temporary cell block must preserve at least one valid path to goal for both players."""
        engine3 = WallzEngine(board_size=3, rules={'C': True, 'K': 3})
        # Player 0 at (0, 1), Player 1 at (2, 1). Goal for P0 is row 2, P1 is row 0.
        # Place horizontal walls on sides so only center cell (1, 1) allows passage
        engine3.walls[0].append((0, 0, 'H'))
        engine3.walls[0].append((0, 1, 'H'))
        # If (0, 0, 'H') and (0, 1, 'H') are placed, path is already blocked.
        # Let's place walls so (1, 1) is the bottleneck:
        engine3.walls[0] = [(0, 0, 'V'), (0, 1, 'V')] # walls along vertical columns 0 and 1
        # If center cell (1, 1) is blocked, can player pass?
        # If (1, 1) is the only path, blocking it should be illegal
        engine3.pawns[0] = (0, 1)
        engine3.pawns[1] = (2, 1)
        # Block sides: (1, 0) and (1, 2)
        engine3.blocked_cells[(1, 0)] = 5
        engine3.blocked_cells[(1, 2)] = 5
        # Now (1, 1) is the ONLY passage through row 1
        self.assertFalse(engine3._is_valid_cell_block(1, 1))
        legal_blocks = [a[1] for a in engine3.get_legal_actions(0) if a[0] == 'BLOCK']
        self.assertNotIn((1, 1), legal_blocks)
        with self.assertRaises(ValueError):
            engine3.apply_action(('BLOCK', (1, 1)), update_clock=False)

if __name__ == '__main__':
    unittest.main()

