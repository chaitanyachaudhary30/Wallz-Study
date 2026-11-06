"""Unit tests for ActionCodec v1 (P1.2 / X-3)."""
import random
import unittest

from wallz_engine import WallzEngine
try:
    from modified_wallz.actions import ActionCodec, CODEC_VERSION
except ImportError:
    from src.modified_wallz.actions import ActionCodec, CODEC_VERSION


def mirror_engine(engine: WallzEngine) -> WallzEngine:
    """Creates a horizontally mirrored copy of the engine (c -> N - 1 - c)."""
    m = engine.clone()
    n = engine.board_size

    # Pawns
    m.pawns[0] = (engine.pawns[0][0], n - 1 - engine.pawns[0][1])
    m.pawns[1] = (engine.pawns[1][0], n - 1 - engine.pawns[1][1])

    # Walls
    for p in (0, 1):
        m.walls[p] = [(wr, n - 2 - wc, o) for wr, wc, o in engine.walls[p]]
        if engine.latest_wall[p] is not None:
            lwr, lwc, lo = engine.latest_wall[p]
            m.latest_wall[p] = (lwr, n - 2 - lwc, lo)

    # Blocked cells
    m.blocked_cells = {(br, n - 1 - bc): k for (br, bc), k in engine.blocked_cells.items()}
    return m


def mirror_action(action: tuple, board_size: int) -> tuple:
    """Mirrors an engine action tuple horizontally."""
    act_type, act_val = action
    n = board_size
    if act_type in ('MOVE', 'HOP'):
        return (act_type, (act_val[0], n - 1 - act_val[1]))
    elif act_type in ('WALL', 'REPLACE'):
        return (act_type, (act_val[0], n - 2 - act_val[1], act_val[2]))
    elif act_type == 'BLOCK':
        return (act_type, (act_val[0], n - 1 - act_val[1]))
    raise ValueError(f"Unknown action {action}")


class TestActionCodec(unittest.TestCase):
    def test_num_actions(self):
        self.assertEqual(ActionCodec.get_num_actions(3), 33)
        self.assertEqual(ActionCodec.get_num_actions(4), 60)
        self.assertEqual(ActionCodec.get_num_actions(5), 97)
        self.assertEqual(ActionCodec.get_num_actions(9), 345)

    def test_round_trip_and_mask_equality_random_playouts(self):
        """Validates round-trip decode(encode(a)) == a and mask == legal_actions across playouts."""
        for n in (3, 4, 5, 9):
            codec = ActionCodec(n)
            rng = random.Random(42 + n)

            for rules in (
                {'A': False, 'B': False, 'C': False, 'M': 0},
                {'A': True, 'B': True, 'C': True, 'K': 2, 'M': 2},
            ):
                engine = WallzEngine(board_size=n, rules=rules)
                for step in range(30):
                    if engine.is_terminal():
                        break

                    mover = engine.current_player
                    legal = engine.get_legal_actions()
                    if not legal:
                        break

                    mask = codec.legal_mask(engine, mover=mover)
                    self.assertEqual(len(mask), codec.num_actions)

                    # 1. Round trip for every legal action
                    decoded_from_legal = set()
                    for act in legal:
                        aid = codec.encode(act, engine, mover=mover)
                        self.assertTrue(mask[aid], f"Legal action {act} not marked in mask")
                        dec = codec.decode(aid, engine, mover=mover)
                        self.assertEqual(dec, act, f"Round trip mismatch: {act} -> {aid} -> {dec}")
                        decoded_from_legal.add(dec)

                    # 2. Mask equality: decoded actions from mask must match legal actions exactly
                    decoded_from_mask = {
                        codec.decode(aid, engine, mover=mover)
                        for aid, is_legal in enumerate(mask)
                        if is_legal
                    }
                    self.assertEqual(set(legal), decoded_from_mask)

                    # 3. legal_ids matches mask
                    lids = codec.legal_ids(engine, mover=mover)
                    self.assertEqual(set(lids), {aid for aid, is_legal in enumerate(mask) if is_legal})

                    # Advance state
                    action = rng.choice(legal)
                    engine.apply_action(action, update_clock=False)

    def test_mirror_permutation_involution_and_semantics(self):
        """Tests that mirror_permutation is an involution and matches physical board mirroring."""
        for n in (3, 5, 9):
            codec = ActionCodec(n)
            perm = codec.mirror_permutation()
            self.assertEqual(len(perm), codec.num_actions)

            # Involution test: P[P[i]] == i
            for i in range(codec.num_actions):
                self.assertEqual(perm[perm[i]], i)

            # Test against mirror_engine on random playouts
            rng = random.Random(100 + n)
            engine = WallzEngine(board_size=n, rules={'A': True, 'B': True, 'C': True, 'K': 2, 'M': 3})
            for step in range(15):
                if engine.is_terminal():
                    break
                legal = engine.get_legal_actions()
                if not legal:
                    break

                m_engine = mirror_engine(engine)
                mover = engine.current_player

                for act in legal:
                    aid = codec.encode(act, engine, mover=mover)
                    m_act = mirror_action(act, n)
                    m_aid = codec.encode(m_act, m_engine, mover=mover)
                    self.assertEqual(
                        perm[aid], m_aid,
                        f"Mirror mismatch for {act} on N={n}: perm[{aid}]={perm[aid]} != m_aid={m_aid}"
                    )

                action = rng.choice(legal)
                engine.apply_action(action, update_clock=False)

    def test_mirror_even_board_size_raises(self):
        codec = ActionCodec(4)
        with self.assertRaises(ValueError):
            codec.mirror_permutation()


if __name__ == '__main__':
    unittest.main()
