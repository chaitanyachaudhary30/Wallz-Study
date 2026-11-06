"""Unit test for DQN Runner Integration (P1.10)."""
import os
import shutil
import tempfile
import unittest

try:
    import torch
    from agents.dqn.network import QNetwork
    from agents.dqn.agent import save_checkpoint
    from evaluation.match_runner import MatchConfig, run_match, AgentUnavailable
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False


@unittest.skipUnless(DEPS_AVAILABLE, "PyTorch and match runner required")
class TestDQNRunnerIntegration(unittest.TestCase):
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
        self.meta = {
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
        self.ckpt_path = save_checkpoint(self.temp_dir, self.model, self.meta, filename="model.pt")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_run_match_dqn_vs_greedy(self):
        from evaluation.match_runner import verify_replay
        config = MatchConfig(
            board_size=self.board_size,
            rules=self.rules,
            p0=f"dqn:path={self.ckpt_path}",
            p1="greedy",
            max_plies=30,
        )
        match_result = run_match(config)
        self.assertIn('outcome', match_result)
        self.assertFalse(match_result['outcome']['category'].startswith('forfeit'))
        self.assertEqual(match_result['agents']['0']['codec_version'], 1)
        self.assertGreater(match_result['length_plies'], 0)

        ok, msg = verify_replay(match_result)
        self.assertTrue(ok, f"verify_replay failed: {msg}")

    def test_run_match_mismatched_checkpoint_raises_agent_unavailable(self):
        # Match with N=5 but checkpoint is N=3
        config = MatchConfig(
            board_size=5,
            rules=self.rules,
            p0=f"dqn:path={self.ckpt_path}",
            p1="greedy",
        )
        with self.assertRaises(AgentUnavailable):
            run_match(config)


if __name__ == '__main__':
    unittest.main()
