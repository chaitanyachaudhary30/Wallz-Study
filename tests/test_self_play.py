"""Unit tests for Self-Play Mechanism and SnapshotPool (P1.8)."""
import unittest

try:
    import numpy as np
    import torch
    from agents.dqn.trainer import DQNTrainer, DQNConfig
    from agents.dqn.self_play import SnapshotPool, SelfPlayManager
    try:
        from modified_wallz.environment import WallzEnv
    except ImportError:
        from src.modified_wallz.environment import WallzEnv
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False


@unittest.skipUnless(DEPS_AVAILABLE, "PyTorch and NumPy required for self-play tests")
class TestSelfPlay(unittest.TestCase):
    def test_snapshot_pool_fifo_and_cloning(self):
        pool = SnapshotPool(max_size=3)
        self.assertEqual(len(pool), 0)
        self.assertIsNone(pool.sample())

        # Create dummy state dicts
        for i in range(5):
            sd = {'weight': torch.tensor([float(i)])}
            pool.add(sd)

        # Capacity should be capped at 3, holding items 2, 3, 4
        self.assertEqual(len(pool), 3)
        # Sample should return one of tensor([2.]), tensor([3.]), tensor([4.])
        sampled = pool.sample()
        self.assertIsNotNone(sampled)
        val = sampled['weight'].item()
        self.assertIn(val, [2.0, 3.0, 4.0])

    def test_mixture_distribution_and_fallback(self):
        config = DQNConfig(board_size=3, rules={'A': False, 'B': False, 'C': False, 'M': 0})
        trainer = DQNTrainer(config)
        sp = SelfPlayManager(trainer, snapshot_interval=100, mixture=(0.5, 0.4, 0.1))

        # Initially pool is empty, so snapshot falls back to current
        types_empty = [sp.sample_opponent_type() for _ in range(500)]
        self.assertNotIn('snapshot', types_empty)
        self.assertIn('current', types_empty)
        self.assertIn('greedy', types_empty)

        # Now add a snapshot to pool
        sp.pool.add(trainer.online_net.state_dict())
        types_with_pool = [sp.sample_opponent_type() for _ in range(1000)]
        self.assertIn('snapshot', types_with_pool)
        self.assertIn('current', types_with_pool)
        self.assertIn('greedy', types_with_pool)

        # Verify rough proportions: current ~ 50%, snapshot ~ 40%, greedy ~ 10%
        p_cur = types_with_pool.count('current') / 1000.0
        p_snap = types_with_pool.count('snapshot') / 1000.0
        p_greedy = types_with_pool.count('greedy') / 1000.0

        self.assertAlmostEqual(p_cur, 0.5, delta=0.08)
        self.assertAlmostEqual(p_snap, 0.4, delta=0.08)
        self.assertAlmostEqual(p_greedy, 0.1, delta=0.05)

    def test_run_episode_completion(self):
        config = DQNConfig(
            board_size=3,
            rules={'A': False, 'B': False, 'C': False, 'M': 0},
            min_buffer_size=10,
            batch_size=4,
            train_freq=2,
            seed=42,
        )
        trainer = DQNTrainer(config)
        sp = SelfPlayManager(trainer, snapshot_interval=50, mixture=(0.5, 0.4, 0.1))

        # Run multiple episodes to ensure both seats and various opponents run smoothly
        learner_seats = set()
        for _ in range(10):
            res = sp.run_episode()
            self.assertIn('episode_length', res)
            self.assertGreater(res['episode_length'], 0)
            self.assertIn(res['terminal_reason'], ['goal', 'repetition', 'no_legal_actions', 'timeout', 'max_plies'])
            learner_seats.add(res['learner_seat'])

        # Both seat 0 and seat 1 should have been selected
        self.assertIn(0, learner_seats)
        self.assertIn(1, learner_seats)

    def test_transition_storage_rules(self):
        # 1. Opponent == 'current' -> both sides stored
        config_cur = DQNConfig(board_size=3, rules={'A': False, 'B': False, 'C': False, 'M': 0})
        trainer_cur = DQNTrainer(config_cur)
        sp_cur = SelfPlayManager(trainer_cur, mixture=(1.0, 0.0, 0.0))

        initial_buf_size = len(trainer_cur.buffer)
        res = sp_cur.run_episode()
        # In current vs current, every ply is stored
        self.assertEqual(len(trainer_cur.buffer) - initial_buf_size, res['episode_length'])

        # 2. Opponent == 'greedy' -> only learner plies stored
        config_gr = DQNConfig(board_size=3, rules={'A': False, 'B': False, 'C': False, 'M': 0})
        trainer_gr = DQNTrainer(config_gr)
        sp_gr = SelfPlayManager(trainer_gr, mixture=(0.0, 0.0, 1.0))

        initial_buf_size = len(trainer_gr.buffer)
        res = sp_gr.run_episode()
        # Only learner's moves are stored, which is strictly less than or equal to ceil(plies / 2)
        added = len(trainer_gr.buffer) - initial_buf_size
        self.assertLess(added, res['episode_length'])
        self.assertGreater(added, 0)


if __name__ == '__main__':
    unittest.main()
