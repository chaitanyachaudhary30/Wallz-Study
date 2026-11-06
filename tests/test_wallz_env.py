"""Unit tests for WallzEnv (P1.4 / C5)."""
import unittest

try:
    from modified_wallz.environment import WallzEnv
except ImportError:
    from src.modified_wallz.environment import WallzEnv


class TestWallzEnv(unittest.TestCase):
    def test_reset_and_properties(self):
        env = WallzEnv(board_size=3, rules={'A': False, 'B': False, 'C': False, 'M': 0})
        obs = env.reset()
        self.assertEqual(len(obs), 14)
        self.assertEqual(len(obs[0]), 3)
        self.assertEqual(env.to_move, 0)
        self.assertEqual(env.num_actions, 33)

        mask = env.legal_mask()
        self.assertEqual(len(mask), 33)
        self.assertTrue(any(mask))

    def test_step_and_reward_semantics(self):
        # 3x3 pure race: P0 at (0, 1), P1 at (2, 1).
        env = WallzEnv(board_size=3, rules={'A': False, 'B': False, 'C': False, 'M': 0})
        obs = env.reset()

        # P0 moves FWD: (0, 1) -> (1, 1). Canonical ID 0 is FWD.
        next_obs, r, term, trunc, info = env.step(0)
        self.assertEqual(r, 0.0)
        self.assertFalse(term)
        self.assertFalse(trunc)
        self.assertEqual(env.to_move, 1)
        self.assertEqual(info['mover'], 0)

        # P1 moves sideways LEFT (in canonical: world (2, 2) is LEFT for P1, ID 3).
        # Let's check legal IDs for P1
        p1_lids = env.legal_ids()
        self.assertTrue(len(p1_lids) > 0)
        # Execute first legal action for P1
        act_id = p1_lids[0]
        next_obs, r, term, trunc, info = env.step(act_id)
        self.assertEqual(r, 0.0)
        self.assertFalse(term)
        self.assertEqual(env.to_move, 0)

    def test_terminal_win_gives_plus_one_reward(self):
        env = WallzEnv(board_size=3, rules={'A': False, 'B': False, 'C': False, 'M': 0})
        env.reset()
        # P0 at (0, 1) -> (1, 1) (ID 0)
        env.step(0)
        # P1 at (2, 1) -> (2, 0) (RIGHT in canonical: ID 2 or LEFT: ID 3)
        # Let's find action that moves to (2, 0)
        p1_acts = env.legal_ids()
        # Find one that doesn't block (2, 1)
        for aid in p1_acts:
            decoded = env.codec.decode(aid, env.engine, mover=1)
            if decoded[1] == (2, 0):
                env.step(aid)
                break

        # Now P0 is at (1, 1), can step FWD to (2, 1) -> goal line!
        # FWD for P0 is ID 0
        next_obs, r, term, trunc, info = env.step(0)
        self.assertEqual(r, 1.0)
        self.assertTrue(term)
        self.assertEqual(info['winner'], 0)
        self.assertEqual(info['terminal_reason'], 'goal')

        # Stepping again raises RuntimeError
        with self.assertRaises(RuntimeError):
            env.step(0)

    def test_truncation_flag(self):
        env = WallzEnv(board_size=3, max_plies=2)
        env.reset()
        _, _, term, trunc, _ = env.step(0)
        self.assertFalse(trunc)
        # Step 2 reaches max_plies
        p1_id = env.legal_ids()[0]
        _, _, term, trunc, info = env.step(p1_id)
        self.assertTrue(trunc)
        self.assertFalse(term)


if __name__ == '__main__':
    unittest.main()
