"""Unit Tests for Phase 5 Minimax Enhancements.

Tests:
1. Side-to-move assertion on wrong player_id (P5.4).
2. PV-first ordering across iterative deepening (P5.1).
3. History heuristic updates on alpha-beta cutoffs (P5.2).
4. Goal-proximity extension (P5.6).
5. get_stats() exposes depth_completed and nodes_expanded (P5.5).
"""
import unittest
from agents.minimax import MinimaxAgent
from wallz_engine import WallzEngine


class TestMinimaxPhase5(unittest.TestCase):
    def test_side_to_move_assertion(self):
        """MinimaxAgent raises AssertionError if called on opponent's turn."""
        e = WallzEngine(board_size=5)
        # It is player 0's turn
        self.assertEqual(e.current_player, 0)
        agent1 = MinimaxAgent(player_id=1, depth=1)
        with self.assertRaises(AssertionError):
            agent1.select_action(e)

    def test_depth_completed_in_stats(self):
        """get_stats() returns depth_completed and nodes_expanded."""
        e = WallzEngine(board_size=5)
        agent = MinimaxAgent(player_id=0, depth=2)
        action = agent.select_action(e)
        self.assertIsNotNone(action)
        stats = agent.get_stats()
        self.assertEqual(stats['depth_completed'], 2)
        self.assertGreater(stats['nodes_expanded'], 0)

    def test_history_heuristic_updated_on_cutoffs(self):
        """Cutoffs during search increment action scores in self.history."""
        e = WallzEngine(board_size=5)
        agent = MinimaxAgent(player_id=0, depth=2, use_alpha_beta=True)
        agent.select_action(e)
        if agent.cutoffs > 0:
            self.assertGreater(len(agent.history), 0)
            self.assertTrue(any(v > 0 for v in agent.history.values()))

    def test_pv_move_searched_first_in_ordering(self):
        """order_actions places pv_action first when provided."""
        e = WallzEngine(board_size=5)
        legal = e.get_legal_actions(0)
        pv_move = legal[-1]  # Pick the last move in legal actions
        agent = MinimaxAgent(player_id=0, depth=1)
        ordered = agent.order_actions(e, legal, active_player=0, pv_action=pv_move)
        self.assertEqual(ordered[0], pv_move)

    def test_goal_proximity_extension_flag(self):
        """Goal proximity extension can be configured and executes without error."""
        e = WallzEngine(board_size=5)
        # Place player 0 at distance 2 from goal row 4
        e.pawns[0] = (2, 2)
        agent = MinimaxAgent(player_id=0, depth=1, extend_goal=True)
        action = agent.select_action(e)
        self.assertIsNotNone(action)


if __name__ == '__main__':
    unittest.main()
