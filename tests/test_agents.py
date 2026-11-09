import math
import unittest
from wallz_engine import WallzEngine
from agents.minimax import MinimaxAgent, SearchTimeout
from agents.greedy import GreedyAgent
from agents.exact_solver import ExactSolver, ExactSolverAgent

class TestMinimaxAgent(unittest.TestCase):

    def setUp(self):
        self.engine5 = WallzEngine(board_size=5)
        self.engine9 = WallzEngine(board_size=9)

    def test_immediate_winning_move_pawn(self):
        """Minimax must select an immediate winning pawn move if available."""
        # Place player 0 one step away from goal row (row 4 in 5x5 grid), opponent out of the way
        self.engine5.pawns[0] = (3, 2)
        self.engine5.pawns[1] = (4, 0)
        agent = MinimaxAgent(player_id=0, depth=1)
        action = agent.select_action(self.engine5)
        self.assertEqual(action, ('MOVE', (4, 2)))

    def test_immediate_winning_move_hop(self):
        """Minimax must select an immediate winning hop if it bridges to the goal row."""
        # P0 one row from goal behind an opponent wall: only the hop wins this ply
        self.engine5.pawns[0] = (3, 2)
        self.engine5.pawns[1] = (4, 0)
        self.engine5.walls[1].append((3, 2, 'H'))
        agent = MinimaxAgent(player_id=0, depth=1)
        action = agent.select_action(self.engine5)
        self.assertEqual(action, ('HOP', (4, 2)))

    def test_avoid_forced_loss(self):
        """Minimax (depth 2) must anticipate opponent's immediate win and block/delay it."""
        # Board 5x5. Player 1 is at (1, 2) with goal row 0.
        # Player 0 is at (0, 0), player 0's turn.
        # If Player 0 does not block Player 1, Player 1 moves to (0, 2) and wins next turn.
        self.engine5.pawns[0] = (0, 0)
        self.engine5.pawns[1] = (1, 2)
        self.engine5.current_player = 0

        agent = MinimaxAgent(player_id=0, depth=2)
        action = agent.select_action(self.engine5)
        
        # Verify Player 0 takes an action that prevents Player 1 from winning next turn
        sim = self.engine5.clone()
        sim.apply_action(action, update_clock=False)
        legal_p1 = sim.get_legal_actions(1)
        for act in legal_p1:
            p1_sim = sim.clone()
            p1_sim.apply_action(act, update_clock=False)
            # Player 1 should not be able to immediately win on their turn
            if act[0] in ('MOVE', 'HOP') and act[1][0] == 0:
                self.assertNotEqual(p1_sim.winner, 1, "Player 0 failed to prevent immediate win")

    def test_terminal_evaluation_values(self):
        """Terminal states must strictly dominate leaf heuristic evaluations."""
        agent = MinimaxAgent(player_id=0, depth=2)

        # 1. Player 0 win
        self.engine5.pawns[0] = (4, 2)
        self.engine5.winner = 0
        score_win = agent.evaluate(self.engine5, depth=0, ply=1)
        self.assertGreater(score_win, 900.0)

        # 2. Player 0 loss
        self.engine5.winner = 1
        score_loss = agent.evaluate(self.engine5, depth=0, ply=1)
        self.assertLess(score_loss, -900.0)

        # 3. Draw
        self.engine5.winner = None
        self.engine5.is_draw = True
        score_draw = agent.evaluate(self.engine5, depth=0, ply=1)
        self.assertEqual(score_draw, 0.0)

        # 4. Non-terminal leaf
        self.engine5.is_draw = False
        self.engine5.pawns[0] = (0, 2)
        self.engine5.pawns[1] = (4, 2)
        score_leaf = agent.evaluate(self.engine5, depth=0, ply=0)
        expected_h = self.engine5.shortest_path(1) - self.engine5.shortest_path(0)
        expected_u = math.tanh(0.2 * expected_h)
        self.assertAlmostEqual(score_leaf, expected_u, places=5)
        self.assertGreaterEqual(score_leaf, -1.0)
        self.assertLessEqual(score_leaf, 1.0)

    def test_alphabeta_equivalence_to_plain_minimax(self):
        """Alpha-Beta pruning must compute the exact same minimax value as unpruned Minimax."""
        # Test on 5x5 board with depth 1 and depth 2
        agent_ab = MinimaxAgent(player_id=0, depth=2, use_alpha_beta=True, order_moves=False)
        agent_plain = MinimaxAgent(player_id=0, depth=2, use_alpha_beta=False, order_moves=False)

        val_ab = agent_ab._search_root(self.engine5, depth=2, deadline=None)[1]
        val_plain = agent_plain._search_root(self.engine5, depth=2, deadline=None)[1]

        self.assertAlmostEqual(val_ab, val_plain, places=6)
        self.assertGreater(agent_ab.cutoffs, 0)

    def test_move_ordering_preserves_minimax_value(self):
        """Move ordering must not alter the true minimax value."""
        agent_ordered = MinimaxAgent(player_id=0, depth=2, use_alpha_beta=True, order_moves=True)
        agent_unordered = MinimaxAgent(player_id=0, depth=2, use_alpha_beta=True, order_moves=False)

        val_ord = agent_ordered._search_root(self.engine5, depth=2, deadline=None)[1]
        val_unord = agent_unordered._search_root(self.engine5, depth=2, deadline=None)[1]

        self.assertAlmostEqual(val_ord, val_unord, places=6)

    def test_agent_never_returns_illegal_action(self):
        """Agent must always select an action strictly present in get_legal_actions()."""
        agents = {0: MinimaxAgent(player_id=0, depth=1), 1: MinimaxAgent(player_id=1, depth=1)}
        for _ in range(5):
            legal_actions = self.engine5.get_legal_actions(self.engine5.current_player)
            action = agents[self.engine5.current_player].select_action(self.engine5)
            self.assertIn(action, legal_actions)
            self.engine5.apply_action(action, update_clock=False)

    def test_search_does_not_mutate_state(self):
        """Search execution must leave the original engine state completely unmutated."""
        state_key_before = self.engine9._get_state_hashable_key()
        pawns_before = dict(self.engine9.pawns)
        walls_before = {0: list(self.engine9.walls[0]), 1: list(self.engine9.walls[1])}

        agent = MinimaxAgent(player_id=0, depth=2, time_limit=1.0)
        _ = agent.select_action(self.engine9)

        state_key_after = self.engine9._get_state_hashable_key()
        self.assertEqual(state_key_before, state_key_after)
        self.assertEqual(pawns_before, self.engine9.pawns)
        self.assertEqual(walls_before, self.engine9.walls)

    def test_timeout_handling_with_fallback(self):
        """Agent must handle tight time limits gracefully, returning a legal fallback action."""
        # Use an impossibly tiny time limit (0.00001s)
        agent = MinimaxAgent(player_id=0, depth=3, time_limit=0.00001)
        action = agent.select_action(self.engine9)
        self.assertIsNotNone(action)
        self.assertIn(action, self.engine9.get_legal_actions(0))

    def test_no_legal_actions_returns_none(self):
        """Agent must return None when no legal actions are available."""
        agent = MinimaxAgent(player_id=0, depth=1)
        # Mock engine with no legal actions
        original_legal = self.engine5.get_legal_actions
        self.engine5.get_legal_actions = lambda p=None: []
        try:
            action = agent.select_action(self.engine5)
            self.assertIsNone(action)
        finally:
            self.engine5.get_legal_actions = original_legal


class TestExactSolverAgent(unittest.TestCase):

    def setUp(self):
        self.solver = ExactSolver(board_size=3, rules={'A': False, 'B': False, 'C': False}, max_walls=0)
        self.solver.solve()
        self.agent = ExactSolverAgent(player_id=0, solver=self.solver)

    def test_agent_selects_provably_optimal_action(self):
        """ExactSolverAgent selects provably optimal action from solved table."""
        engine = WallzEngine(board_size=3, rules={'A': False, 'B': False, 'C': False})
        engine.wall_inventory = {0: 0, 1: 0}
        action = self.agent.select_action(engine)
        self.assertEqual(action, ('MOVE', (1, 1)))

    def test_agent_fallback_on_unsolved_state(self):
        """ExactSolverAgent falls back to legal move on unvisited or out-of-domain state."""
        engine9 = WallzEngine(board_size=9)
        action = self.agent.select_action(engine9)
        self.assertIn(action, engine9.get_legal_actions(0))

    def test_agent_on_terminal_state_fallback(self):
        """When state has no legal actions, returns None."""
        engine = WallzEngine(board_size=3)
        original_legal = engine.get_legal_actions
        engine.get_legal_actions = lambda p=None: []
        try:
            action = self.agent.select_action(engine)
            self.assertIsNone(action)
        finally:
            engine.get_legal_actions = original_legal


if __name__ == '__main__':
    unittest.main()
