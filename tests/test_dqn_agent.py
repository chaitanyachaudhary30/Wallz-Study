"""Unit tests for DQNAgent and Checkpoint Management (P1.9)."""
import os
import shutil
import tempfile
import unittest

try:
    import numpy as np
    import torch
    from agents.dqn.network import QNetwork
    from agents.dqn.agent import DQNAgent, save_checkpoint, load_checkpoint
    from wallz_engine import WallzEngine
    try:
        from modified_wallz.actions import CODEC_VERSION
    except ImportError:
        from src.modified_wallz.actions import CODEC_VERSION
    from agents.dqn.encoding import OBS_VERSION
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False


@unittest.skipUnless(DEPS_AVAILABLE, "PyTorch and NumPy required for DQNAgent tests")
class TestDQNAgent(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.board_size = 3
        self.rules = {'A': False, 'B': False, 'C': False, 'K': 1, 'M': 0}
        self.model = QNetwork(
            board_size=self.board_size,
            num_actions=33,
            channels=16,
            blocks=1,
            hidden=32,
        )
        self.agent = DQNAgent(
            player_id=0,
            model=self.model,
            board_size=self.board_size,
            rules=self.rules,
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_runner_gotcha_no_get_stats(self):
        """DQNAgent must NOT implement get_stats() to avoid colliding with runner."""
        self.assertFalse(hasattr(self.agent, 'get_stats'))
        self.assertTrue(hasattr(self.agent, 'stats'))
        stats = self.agent.stats()
        self.assertIsInstance(stats, dict)
        self.assertIn('inferences', stats)

    def test_select_action_is_legal(self):
        engine = WallzEngine(board_size=self.board_size, rules=self.rules)
        action = self.agent.select_action(engine)
        legal_actions = engine.get_legal_actions(0)
        self.assertIn(action, legal_actions)
        self.assertEqual(self.agent.stats()['inferences'], 1)

    def test_checkpoint_save_and_load_roundtrip(self):
        meta = {
            'N': self.board_size,
            'rules': self.rules,
            'gamma': 0.99,
            'double': True,
            'dueling': False,
            'channels': 16,
            'blocks': 1,
            'hidden': 32,
            'seed': 42,
            'step': 100,
        }
        weights_path = save_checkpoint(self.temp_dir, self.model, meta, filename="test_model.pt")
        self.assertTrue(os.path.exists(weights_path))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "meta.json")))

        # Load back
        loaded_agent = load_checkpoint(
            checkpoint_path=self.temp_dir,
            player_id=0,
            expected_board_size=self.board_size,
            expected_rules=self.rules,
        )
        self.assertEqual(loaded_agent.board_size, self.board_size)
        self.assertEqual(loaded_agent.codec_version, CODEC_VERSION)
        self.assertEqual(loaded_agent.obs_version, OBS_VERSION)

        # Decision equality test
        engine = WallzEngine(board_size=self.board_size, rules=self.rules)
        act_orig = self.agent.select_action(engine)
        act_loaded = loaded_agent.select_action(engine)
        self.assertEqual(act_orig, act_loaded)

    def test_checkpoint_validation_mismatch(self):
        meta = {
            'N': self.board_size,
            'rules': self.rules,
            'gamma': 0.99,
            'double': True,
            'dueling': False,
            'channels': 16,
            'blocks': 1,
            'hidden': 32,
            'seed': 42,
            'step': 100,
        }
        save_checkpoint(self.temp_dir, self.model, meta)

        # Mismatched board size
        with self.assertRaises(ValueError):
            load_checkpoint(self.temp_dir, expected_board_size=5)

        # Mismatched rules
        with self.assertRaises(ValueError):
            load_checkpoint(self.temp_dir, expected_rules={'A': True, 'B': False, 'C': False, 'K': 1, 'M': 0})


if __name__ == '__main__':
    unittest.main()
