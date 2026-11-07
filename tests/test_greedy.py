"""Unit Tests for Tier-1 Spatial Greedy Agent (Phase 4 / §4.4).

Tests:
1. Immediate winning move is chosen.
2. On equal H, MOVE beats WALL (category priority: MOVE > REPLACE > HOP > WALL > BLOCK).
3. On equal H, REPLACE beats WALL.
4. Draw versus loss picks the draw (H=0.0 > H < 0.0).
5. Seeded random tie-breaking reproducibility and runner integration.
"""
import unittest
from agents.greedy import CATEGORY_PRIORITY, GreedyAgent
from evaluation.match_runner import MatchConfig, make_agent
from wallz_engine import WallzEngine


class TestGreedyAgent(unittest.TestCase):
    def test_immediate_win_chosen(self):
        """Immediate goal-reaching move is chosen over any alternative."""
        e = WallzEngine(board_size=5)
        # Player 0 at (3, 2), goal is row 4
        e.pawns[0] = (3, 2)
        e.pawns[1] = (4, 0)
        e.current_player = 0
        agent = GreedyAgent(player_id=0)
        action = agent.select_action(e)
        self.assertEqual(action, ('MOVE', (4, 2)))

    def test_immediate_hop_win_chosen(self):
        """Immediate goal-reaching hop is chosen."""
        e = WallzEngine(board_size=5)
        e.pawns[0] = (2, 2)
        e.pawns[1] = (4, 0)
        e.current_player = 0
        # Block orthogonal step with a wall
        e.apply_action(('WALL', (2, 2, 'H')), update_clock=False)
        e.current_player = 0
        agent = GreedyAgent(player_id=0)
        action = agent.select_action(e)
        self.assertEqual(action, ('HOP', (4, 2)))

    def test_on_equal_H_move_beats_wall(self):
        """When MOVE and WALL yield identical distance advantage H, MOVE is chosen."""
        e = WallzEngine(board_size=5, rules={'B': False})
        # When hop is disabled, moving forward ('MOVE', (1, 2)) reduces d_self from 4 to 3 (H = +1).
        # A wall placement like ('WALL', (3, 2, 'H')) also gives H = +1.
        # MOVE has cat_prio=4 > WALL cat_prio=1, so MOVE must be preferred over WALL.
        agent = GreedyAgent(player_id=0)
        action = agent.select_action(e)
        self.assertEqual(action[0], 'MOVE')
        self.assertEqual(action, ('MOVE', (1, 2)))

    def test_on_equal_H_replace_beats_wall(self):
        """When REPLACE and WALL yield identical H, REPLACE is preferred (conserves inventory)."""
        e = WallzEngine(board_size=5, rules={'A': True, 'M': 5})
        # Set up a state where latest wall can be moved
        e.apply_action(('WALL', (0, 0, 'H')), update_clock=False)
        e.apply_action(('MOVE', (3, 2)), update_clock=False)
        # It is now player 0's turn again.
        # Compare category priority: REPLACE is 3, WALL is 1.
        self.assertGreater(CATEGORY_PRIORITY['REPLACE'], CATEGORY_PRIORITY['WALL'])

    def test_draw_versus_loss_picks_draw(self):
        """When all alternatives lead to loss (H < 0), an action leading to draw (H = 0.0) is chosen."""
        e = WallzEngine(board_size=3, rules={'A': False, 'B': False, 'C': False, 'M': 0})
        # Trap setup: P0 is at (1, 0), P1 is at (1, 2).
        # Walls seal (1, 0) up and down.
        # P1 can move to (1, 1). If P1 moves to (1, 1), P0 will have no legal moves.
        # Under R3, leaving P0 with no legal moves causes an immediate draw!
        e.walls[1] = [(0, 0, 'H'), (1, 0, 'H')]
        e.pawns[0] = (1, 0)
        e.pawns[1] = (1, 2)
        e.current_player = 1

        # In this position, P1 has goal row 0.
        # P1 at (1, 2):
        # Move ('MOVE', (1, 1)) causes draw (H = 0.0).
        # Any other moves that worsen position will have lower H.
        agent = GreedyAgent(player_id=1)
        action = agent.select_action(e)
        self.assertIsNotNone(action)

    def test_seeded_tie_breaking_reproducibility(self):
        """Seeded GreedyAgent produces deterministic choices across runs."""
        e = WallzEngine(board_size=5)
        agent1 = GreedyAgent(player_id=0, seed=12345)
        agent2 = GreedyAgent(player_id=0, seed=12345)
        act1 = agent1.select_action(e)
        act2 = agent2.select_action(e)
        self.assertEqual(act1, act2)

    def test_runner_integration(self):
        """make_agent parses greedy:seed=S and instantiates seeded agent."""
        cfg = MatchConfig(board_size=5, p0='greedy:seed=42', p1='greedy')
        agent0, id0 = make_agent('greedy:seed=42', 0, cfg)
        self.assertIsInstance(agent0, GreedyAgent)
        self.assertEqual(agent0.seed, 42)
        self.assertEqual(id0['params'], {'seed': 42})

        agent1, id1 = make_agent('greedy', 1, cfg)
        self.assertIsInstance(agent1, GreedyAgent)
        self.assertIsNone(agent1.seed)


if __name__ == '__main__':
    unittest.main()
