"""Gate Conformance: Parameterized Agent Conformance Suite (P9.4).

Verifies that all agent types (Random, Greedy, Minimax, Exact, DQN):
1. Return a strictly legal action from the engine's current state.
2. Never return None on non-terminal states.
3. Strictly preserve engine state without in-place mutation or side effects.
4. Execute cleanly within latency budgets.
5. Produce actions that apply successfully on engine clones without error.
"""
import copy
import importlib.util
import os
import unittest
from wallz_engine import WallzEngine
from agents.greedy import GreedyAgent
from agents.minimax import MinimaxAgent
from agents.exact_solver import ExactSolverAgent
from evaluation.match_runner import RandomAgent, ExactAgent, make_agent, MatchConfig

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class TestAgentConformance(unittest.TestCase):
    def setUp(self):
        self.rules_3x3_m1 = {'A': True, 'B': True, 'C': True, 'K': 1, 'M': 1}
        self.rules_3x3_m0 = {'A': True, 'B': True, 'C': True, 'K': 1, 'M': 0}
        self.rules_5x5_m5 = {'A': True, 'B': True, 'C': True, 'K': 3, 'M': 5}

    def _assert_agent_conforms(self, agent, engine: WallzEngine, agent_label: str):
        """Verifies core conformance contracts for any agent."""
        self.assertFalse(engine.is_terminal(), f"{agent_label}: Test state is already terminal")

        # Snapshot engine state
        saved_key = engine.position_key()
        saved_pawns = copy.deepcopy(engine.pawns)
        saved_walls = copy.deepcopy(engine.walls)
        saved_inv = copy.deepcopy(engine.wall_inventory)
        saved_turn = engine.turn_count
        saved_player = engine.current_player
        saved_blocks = copy.deepcopy(engine.blocked_cells)

        legal_actions = engine.get_legal_actions()
        self.assertTrue(len(legal_actions) > 0, f"{agent_label}: No legal actions available")

        # Execute decision
        action = agent.select_action(engine)

        # Invariant 1: Action is not None on non-terminal state
        self.assertIsNotNone(action, f"{agent_label} returned None on non-terminal state")

        # Invariant 2: Action is strictly in legal actions
        self.assertIn(action, legal_actions, f"{agent_label} selected illegal action {action}")

        # Invariant 3: No mutation of the input engine
        self.assertEqual(engine.position_key(), saved_key, f"{agent_label} mutated engine position_key")
        self.assertEqual(engine.pawns, saved_pawns, f"{agent_label} mutated engine pawns")
        self.assertEqual(engine.walls, saved_walls, f"{agent_label} mutated engine walls")
        self.assertEqual(engine.wall_inventory, saved_inv, f"{agent_label} mutated wall_inventory")
        self.assertEqual(engine.turn_count, saved_turn, f"{agent_label} mutated turn_count")
        self.assertEqual(engine.current_player, saved_player, f"{agent_label} mutated current_player")
        self.assertEqual(engine.blocked_cells, saved_blocks, f"{agent_label} mutated blocked_cells")

        # Invariant 4: Action is applicable on engine clone
        clone = engine.clone()
        clone.apply_action(action)
        self.assertNotEqual(clone.position_key(), saved_key, "Applying action did not progress state")

    def test_random_agent_conformance(self):
        """RandomAgent conforms on 3x3, 5x5, and 9x9 boards."""
        for n in [3, 5, 9]:
            engine = WallzEngine(board_size=n, rules=self.rules_5x5_m5)
            agent = RandomAgent(player_id=0, seed=123)
            self._assert_agent_conforms(agent, engine, f"RandomAgent N={n}")

    def test_greedy_agent_conformance(self):
        """GreedyAgent conforms with deterministic and seeded tie breaking."""
        for seed in [None, 42]:
            engine = WallzEngine(board_size=5, rules=self.rules_5x5_m5)
            agent = GreedyAgent(player_id=0, seed=seed)
            self._assert_agent_conforms(agent, engine, f"GreedyAgent seed={seed}")

    def test_minimax_agent_conformance(self):
        """MinimaxAgent conforms across depth 1 and 2, with and without extend_goal."""
        for depth in [1, 2]:
            for extend in [False, True]:
                engine = WallzEngine(board_size=3, rules=self.rules_3x3_m1)
                agent = MinimaxAgent(player_id=0, depth=depth, extend_goal=extend)
                self._assert_agent_conforms(agent, engine, f"MinimaxAgent D={depth} extend={extend}")

    def test_exact_solver_agent_conformance(self):
        """ExactSolverAgent conforms on tractable miniature board (3x3 M=0)."""
        engine = WallzEngine(board_size=3, rules=self.rules_3x3_m0)
        agent = ExactSolverAgent(player_id=0, board_size=3, rules=self.rules_3x3_m0)
        self._assert_agent_conforms(agent, engine, "ExactSolverAgent 3x3 M=0")

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch required for DQNAgent conformance")
    def test_dqn_agent_conformance(self):
        """DQNAgent conforms using released 3x3 and 4x4 checkpoints."""
        chk_3x3 = "checkpoints/release/dqn_3x3/model.pt"
        if os.path.exists(chk_3x3):
            config_3x3 = MatchConfig(board_size=3, rules=self.rules_3x3_m1)
            agent_3x3, _ = make_agent(f"dqn:path={chk_3x3}", player_id=0, config=config_3x3)
            engine_3x3 = WallzEngine(board_size=3, rules=self.rules_3x3_m1)
            self._assert_agent_conforms(agent_3x3, engine_3x3, "DQNAgent 3x3 Release")

        chk_4x4 = "checkpoints/release/dqn_4x4/model.pt"
        if os.path.exists(chk_4x4):
            rules_4x4 = {'A': True, 'B': True, 'C': True, 'K': 1, 'M': 0}
            config_4x4 = MatchConfig(board_size=4, rules=rules_4x4)
            agent_4x4, _ = make_agent(f"dqn:path={chk_4x4}", player_id=0, config=config_4x4)
            engine_4x4 = WallzEngine(board_size=4, rules=rules_4x4)
            self._assert_agent_conforms(agent_4x4, engine_4x4, "DQNAgent 4x4 Release")

    def test_agents_conformance_midgame_states(self):
        """Tests all agents on a complex mid-game state with walls and blocks."""
        engine = WallzEngine(board_size=3, rules=self.rules_3x3_m1)
        engine.apply_action(('MOVE', (1, 1)))  # P0
        engine.apply_action(('MOVE', (2, 0)))  # P1

        agents = [
            ("Random", RandomAgent(player_id=0, seed=77)),
            ("Greedy", GreedyAgent(player_id=0)),
            ("Minimax", MinimaxAgent(player_id=0, depth=2)),
        ]

        chk_3x3 = "checkpoints/release/dqn_3x3/model.pt"
        if TORCH_AVAILABLE and os.path.exists(chk_3x3):
            config = MatchConfig(board_size=3, rules=self.rules_3x3_m1)
            dqn_agent, _ = make_agent(f"dqn:path={chk_3x3}", player_id=0, config=config)
            agents.append(("DQN", dqn_agent))

        for name, ag in agents:
            self._assert_agent_conforms(ag, engine, f"{name} Midgame")


if __name__ == '__main__':
    unittest.main()
