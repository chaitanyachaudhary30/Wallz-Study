"""Test path preservation invariants (P9.3)."""
import unittest
from wallz_engine import WallzEngine


class TestPathPreservation(unittest.TestCase):
    def test_cannot_completely_block_player(self):
        """
        Genuine sealing test: On a 3x3 board, Player 0 starts at (0, 1) with goal row 2.
        Placing (0, 0, 'H') blocks (0, 0)-(1, 0) and (0, 1)-(1, 1).
        Placing (0, 1, 'V') would block (0, 1)-(0, 2) and (1, 1)-(1, 2).
        The two walls do not overlap, but together they completely trap Player 0 in {(0,0), (0,1)}.
        Therefore, (0, 1, 'V') must be rejected strictly for path violation.
        """
        engine = WallzEngine(board_size=3, rules={'A': False, 'M': 5})
        engine.apply_action(('WALL', (0, 0, 'H')))

        # Ensure board allows the geometry without physical wall overlap
        self.assertTrue(engine.board.can_place_wall(0, 1, 'V'))

        # Switch back to player 0 (or check validity for player 0)
        is_valid = engine._is_valid_wall_placement(0, 0, 1, 'V')
        self.assertFalse(is_valid, "Path preservation rule violated: wall seals off player 0 from goal")

        # Must not appear in legal actions
        engine.current_player = 0
        legal = engine.get_legal_actions()
        self.assertNotIn(('WALL', (0, 1, 'V')), legal)

    def test_wall_preserving_path_is_legal(self):
        """A wall that leaves a viable path to the goal remains legal."""
        engine = WallzEngine(board_size=3, rules={'A': False, 'M': 5})
        # (0, 0, 'H') alone leaves open path via (0, 1) -> (0, 2) -> (1, 2) -> (2, 2)
        is_valid = engine._is_valid_wall_placement(0, 0, 0, 'H')
        self.assertTrue(is_valid)
        self.assertIn(('WALL', (0, 0, 'H')), engine.get_legal_actions())


if __name__ == '__main__':
    unittest.main()
