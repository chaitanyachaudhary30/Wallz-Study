"""Unit tests for Observation Encoder v1 (P1.3 / C4)."""
import unittest

from wallz_engine import WallzEngine
from agents.dqn.encoding import encode_observation, mirror_observation, OBS_VERSION
from tests.test_action_codec import mirror_engine


class TestObservationEncoding(unittest.TestCase):
    def test_dimensions(self):
        for n in (3, 4, 5, 9):
            engine = WallzEngine(board_size=n)
            obs = encode_observation(engine)
            self.assertEqual(len(obs), 14)
            self.assertEqual(len(obs[0]), n)
            self.assertEqual(len(obs[0][0]), n)

    def test_hand_crafted_planes_player0(self):
        engine = WallzEngine(board_size=5, rules={'A': True, 'B': True, 'C': True, 'K': 3, 'M': 10})
        # P0 at (0, 2), P1 at (4, 2)
        # Apply a WALL by P0 at (1, 1, 'H')
        engine.apply_action(('WALL', (1, 1, 'H')), update_clock=False)
        # Now P1's turn. Let's test observation for mover=0 explicitly
        obs_p0 = encode_observation(engine, mover=0)

        # 0: own pawn at (0, 2)
        self.assertEqual(obs_p0[0][0][2], 1.0)
        self.assertEqual(obs_p0[0][4][2], 0.0)

        # 1: opp pawn at (4, 2)
        self.assertEqual(obs_p0[1][4][2], 1.0)

        # 2: H-wall at (1, 1)
        self.assertEqual(obs_p0[2][1][1], 1.0)
        self.assertEqual(obs_p0[3][1][1], 0.0)

        # 4: own replaceable wall at (1, 1)
        self.assertEqual(obs_p0[4][1][1], 1.0)

        # 7: own inventory 9 / 10 = 0.9
        self.assertAlmostEqual(obs_p0[7][0][0], 0.9)
        # 8: opp inventory 10 / 10 = 1.0
        self.assertAlmostEqual(obs_p0[8][0][0], 1.0)

        # 13: boundary plane
        self.assertEqual(obs_p0[13][0][0], 1.0)
        self.assertEqual(obs_p0[13][4][4], 1.0)

    def test_hand_crafted_planes_player1(self):
        engine = WallzEngine(board_size=5, rules={'A': True, 'B': True, 'C': True, 'K': 3, 'M': 10})
        # P0 advances
        engine.apply_action(('MOVE', (1, 2)), update_clock=False)
        # Now P1 at (4, 2), mover=1
        obs_p1 = encode_observation(engine, mover=1)

        # Canonical coordinates for P1:
        # P1 in world is at row 4 -> canon row 5 - 1 - 4 = 0!
        self.assertEqual(obs_p1[0][0][2], 1.0)
        # P0 in world is at row 1 -> canon row 5 - 1 - 1 = 3!
        self.assertEqual(obs_p1[1][3][2], 1.0)

    def test_mirror_observation_matches_mirrored_engine(self):
        """Verifies mirror_observation(encode(s)) == encode(mirror_engine(s)) across all channels."""
        for n in (3, 5):
            engine = WallzEngine(board_size=n, rules={'A': True, 'B': True, 'C': True, 'K': 3, 'M': 4})
            # Place a wall and block a cell
            engine.apply_action(('WALL', (0, 0, 'H')), update_clock=False)
            if engine.get_legal_actions():
                # Opponent moves
                engine.apply_action(('MOVE', (engine.pawns[1][0] - 1, engine.pawns[1][1])), update_clock=False)

            m_engine = mirror_engine(engine)

            for mover in (0, 1):
                obs = encode_observation(engine, mover=mover)
                mirrored_obs = mirror_observation(obs)
                expected_obs = encode_observation(m_engine, mover=mover)

                for ch in range(14):
                    for r in range(n):
                        for c in range(n):
                            self.assertAlmostEqual(
                                mirrored_obs[ch][r][c], expected_obs[ch][r][c],
                                delta=1e-5,
                                msg=f"Mismatch at N={n}, mover={mover}, ch={ch}, r={r}, c={c}"
                            )


if __name__ == '__main__':
    unittest.main()
