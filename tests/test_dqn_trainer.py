"""Unit tests for DQNTrainer (P1.7)."""
import unittest

try:
    import numpy as np
    import torch
    from agents.dqn.trainer import DQNTrainer, DQNConfig
    try:
        from modified_wallz.actions import ActionCodec
    except ImportError:
        from src.modified_wallz.actions import ActionCodec
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False


@unittest.skipUnless(DEPS_AVAILABLE, "PyTorch and NumPy required for DQNTrainer tests")
class TestDQNTrainer(unittest.TestCase):
    def setUp(self):
        self.config = DQNConfig(
            board_size=3,
            rules={'A': False, 'B': False, 'C': False, 'M': 0},
            lr=1e-3,
            gamma=0.99,
            batch_size=4,
            min_buffer_size=4,
            target_update_freq=5,
            train_freq=1,
            seed=42,
        )
        self.trainer = DQNTrainer(self.config)

    def test_masked_action_selection(self):
        """Illegal actions are never selected under eval_mode / epsilon=0."""
        obs = np.zeros((14, 3, 3), dtype=np.float32)
        # Create a legal mask where only action ID 5 is legal
        num_actions = self.trainer.num_actions
        legal_mask = np.zeros(num_actions, dtype=bool)
        legal_mask[5] = True

        # Manually alter the network weights or output so that action 0 would have the highest raw value
        with torch.no_grad():
            for p in self.trainer.online_net.parameters():
                p.fill_(1.0)

        # Under eval_mode, only legal action 5 can be selected
        selected = self.trainer.select_action(obs, legal_mask, epsilon=0.0, eval_mode=True)
        self.assertEqual(selected, 5)

        # Even with another subset of legal actions [2, 10]
        legal_mask = np.zeros(num_actions, dtype=bool)
        legal_mask[2] = True
        legal_mask[10] = True
        selected = self.trainer.select_action(obs, legal_mask, epsilon=0.0, eval_mode=True)
        self.assertIn(selected, [2, 10])

    def test_target_network_hard_synchronization(self):
        """Target network updates weights only at target_update_freq."""
        # Fill buffer with dummy transitions
        obs = np.zeros((14, 3, 3), dtype=np.float32)
        mask = np.ones(self.trainer.num_actions, dtype=bool)
        for _ in range(10):
            self.trainer.buffer.push(obs, 0, 1.0, obs, True, False, mask)

        # Modify online net parameters to diverge from target net
        with torch.no_grad():
            for p in self.trainer.online_net.parameters():
                p.add_(10.0)

        # Check weights are different before sync
        diff_found = False
        for p_on, p_tgt in zip(self.trainer.online_net.parameters(), self.trainer.target_net.parameters()):
            if not torch.allclose(p_on, p_tgt):
                diff_found = True
                break
        self.assertTrue(diff_found)

        # Step 4 times (target_update_freq is 5)
        for _ in range(4):
            self.trainer.train_step()

        # Target net should still be distinct
        for p_on, p_tgt in zip(self.trainer.online_net.parameters(), self.trainer.target_net.parameters()):
            self.assertFalse(torch.allclose(p_on, p_tgt))

        # 5th step triggers target update
        self.trainer.train_step()
        self.assertEqual(self.trainer.total_grad_steps, 5)

        # Now target net should match online net exactly
        for p_on, p_tgt in zip(self.trainer.online_net.parameters(), self.trainer.target_net.parameters()):
            self.assertTrue(torch.allclose(p_on, p_tgt))

    def test_one_ply_forced_win_and_loss_sign(self):
        """Negamax Double-DQN target correctly drives win Q > 0 and loss Q < 0."""
        # Train on winning transition: r = +1, terminated = True
        win_trainer = DQNTrainer(DQNConfig(
            board_size=3,
            rules={'A': False, 'B': False, 'C': False, 'M': 0},
            lr=1e-2,
            batch_size=4,
            min_buffer_size=4,
            seed=123,
        ))

        obs = np.ones((14, 3, 3), dtype=np.float32)
        mask = np.ones(win_trainer.num_actions, dtype=bool)

        # Fill with winning transitions for action 0
        for _ in range(10):
            win_trainer.buffer.push(obs, 0, 1.0, obs, True, False, mask)

        for _ in range(50):
            win_trainer.train_step()

        # Check Q-value of action 0 on obs
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            q_win = win_trainer.online_net(x)[0, 0].item()
        self.assertGreater(q_win, 0.5, f"Expected Q-value > 0.5 for winning transition, got {q_win}")

        # Train on losing transition:
        # Action leads to non-terminal s' where opponent has an immediate win (so Q_opp(s') > 0).
        # In negamax, y = r - gamma * Q_tgt(s') = 0 - gamma * (+1.0) = -gamma.
        loss_trainer = DQNTrainer(DQNConfig(
            board_size=3,
            rules={'A': False, 'B': False, 'C': False, 'M': 0},
            lr=1e-2,
            gamma=0.99,
            batch_size=4,
            min_buffer_size=4,
            seed=456,
        ))

        # We first configure target_net to predict +1.0 on next_obs
        next_obs = np.full((14, 3, 3), 2.0, dtype=np.float32)
        # Push opponent-winning transitions into target_net
        with torch.no_grad():
            for p in loss_trainer.target_net.parameters():
                p.zero_()
            # Set bias of last linear layer to +1.0
            if hasattr(loss_trainer.target_net, 'out_head'):
                loss_trainer.target_net.out_head.bias.fill_(1.0)

        # Fill with transitions where r = 0, terminated = False, leading to next_obs
        for _ in range(10):
            loss_trainer.buffer.push(obs, 0, 0.0, next_obs, False, False, mask)

        for _ in range(50):
            loss_trainer.train_step()

        # Now online_net should predict negative value for action 0
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            q_loss = loss_trainer.online_net(x)[0, 0].item()
        self.assertLess(q_loss, -0.5, f"Expected Q-value < -0.5 for losing transition, got {q_loss}")


if __name__ == '__main__':
    unittest.main()
