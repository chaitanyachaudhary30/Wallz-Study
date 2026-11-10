"""Stage 2 DQN compatibility with the frozen engine (Member 4, Tasks 1-3).

Covers what tests/test_action_codec.py and tests/test_dqn_encoding.py do not:
* the action mask equals the engine's own validator for EVERY action id
  (decode each id and try it on a clone), not only the ids of get_legal_actions();
* terminal and no-legal-action draws give an empty mask;
* block-duration, wall-budget, REPLACE and one-time HOP behaviour as seen through the
  codec and the 14 observation planes;
* role-swap equivariance (player 1's view equals player 0's view of the swapped
  position) for encoding, mask and successors;
* mirror permutation against engine-built mirrored positions;
* a documented encoding alias (wall ownership of fixed walls is not encoded);
* agent self-play and runner matches never produce an illegal action.

unittest only; torch-dependent tests skip with a reason when torch is missing.
"""
import importlib.util
import json
import os
import random
import unittest

import numpy as np

from wallz_engine import WallzEngine
from agents.dqn.encoding import encode_observation, mirror_observation
from scripts.final_dqn_lib import (RULE_CELLS, engine_accepts, mirror_key, random_positions,
                                   restore, swap_roles_key)
try:
    from modified_wallz.actions import ActionCodec
    from modified_wallz.environment import WallzEnv
except ImportError:
    from src.modified_wallz.actions import ActionCodec
    from src.modified_wallz.environment import WallzEnv

TORCH = importlib.util.find_spec("torch") is not None
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (board, rule cell names, playout seeds, plies) - kept small enough for a unit test;
# scripts/audit_dqn_compat.py runs the same checks at scale.
GRID = [
    (3, list(RULE_CELLS), range(3), 30),
    (4, ["default_K3_M10", "train4_K1_M0", "K2_M3", "A0_K1_M2"], range(2), 30),
    (5, ["default_K3_M10", "K2_M3", "B0C0_M2"], range(2), 25),
]


def positions():
    for n, cells, seeds, plies in GRID:
        for cell in cells:
            for seed in seeds:
                for e in random_positions(n, RULE_CELLS[cell], seed, plies):
                    yield n, cell, e


def make(n=5, **rules):
    r = {'A': True, 'B': True, 'C': True, 'K': 3, 'M': 10}
    r.update(rules)
    return WallzEngine(board_size=n, rules=r)


def play(e, *actions):
    for a in actions:
        e.apply_action(a, update_clock=False)


class TestMaskEqualsEngine(unittest.TestCase):
    def test_mask_equals_validator_for_every_id(self):
        seen_types = set()
        count = 0
        for n, cell, e in positions():
            codec = ActionCodec(n)
            mask = codec.legal_mask(e)
            legal = e.get_legal_actions()
            seen_types |= {a[0] for a in legal}
            # no two legal actions share an id
            self.assertEqual(sum(mask), len(legal), f"id collision N={n} {cell}")
            decoded = set()
            for aid in range(codec.num_actions):
                act = codec.decode(aid, e)
                ok = engine_accepts(e, act)
                self.assertEqual(mask[aid], ok, f"N={n} {cell} id={aid} {act} mask={mask[aid]} engine={ok}")
                if mask[aid]:
                    decoded.add(act)
            self.assertEqual(decoded, set(legal))
            count += 1
        self.assertGreater(count, 300)
        self.assertEqual(seen_types, {'MOVE', 'HOP', 'WALL', 'REPLACE', 'BLOCK'})

    def test_terminal_and_no_legal_draw_give_empty_mask(self):
        # goal win
        e = make(3, K=1, M=0)
        play(e, ('MOVE', (1, 1)), ('MOVE', (2, 0)), ('MOVE', (2, 1)))
        self.assertEqual(e.winner, 0)
        self.assertFalse(any(ActionCodec(3).legal_mask(e)))
        # no-legal-action draw (fixture from the engine freeze)
        with open(os.path.join(ROOT, 'tests', 'fixtures', 'no_legal_draw.json')) as fh:
            fx = json.load(fh)
        def tup(x):
            return tuple(tup(i) for i in x) if isinstance(x, list) else x
        d = restore(fx['board_size'], fx['rules'], tup(fx['child_key']))
        self.assertTrue(d.is_draw)
        self.assertEqual(d.terminal_reason, 'no_legal_actions')
        self.assertFalse(any(ActionCodec(3).legal_mask(d)))
        env = WallzEnv(board_size=3, rules=fx['rules'])
        env.engine = d
        self.assertFalse(any(env.legal_mask()))
        self.assertEqual(env.legal_ids(), [])

    def test_block_plane_tracks_timer_for_k_plies(self):
        for k in (1, 2, 3, 4):
            e = make(5, K=k, M=0)
            play(e, ('BLOCK', (2, 2)))
            codec = ActionCodec(5)
            for j in range(k):
                self.assertEqual(e.blocked_cells.get((2, 2)), k - j)
                for mover in (0, 1):
                    obs = encode_observation(e, mover=mover)
                    r = 2  # centre row is its own flip on N=5
                    self.assertAlmostEqual(float(obs[6][r][2]), (k - j) / k, places=6)
                    self.assertEqual(int(np.count_nonzero(obs[6])), 1)
                self.assertTrue(all(codec.decode(a, e) != ('MOVE', (2, 2)) for a in codec.legal_ids(e)))
                # sideways pawn step, so nobody walks into the block
                r0 = e.pawns[e.current_player][0]
                play(e, sorted(a for a in e.get_legal_actions() if a[0] == 'MOVE' and a[1][0] == r0)[0])
            self.assertNotIn((2, 2), e.blocked_cells)
            self.assertEqual(int(np.count_nonzero(encode_observation(e)[6])), 0)

    def test_wall_budget_planes_and_ids(self):
        for m in (0, 1, 3, 12):
            e = make(5, M=m)
            codec = ActionCodec(5)
            obs = encode_observation(e)
            expect = 1.0 if m > 0 else 0.0
            self.assertTrue(np.allclose(obs[7], expect) and np.allclose(obs[8], expect))
            wall_ids = [a for a in codec.legal_ids(e) if 8 <= a < codec._replace_offset]
            self.assertEqual(bool(wall_ids), m > 0)
        # M=1: after one wall the WALL ids vanish, REPLACE ids appear, inventory plane 0
        e = make(5, M=1)
        play(e, ('WALL', (3, 0, 'H')), ('MOVE', (3, 2)))
        ids = codec.legal_ids(e)
        self.assertFalse([a for a in ids if 8 <= a < codec._replace_offset])
        rep = [a for a in ids if codec._replace_offset <= a < codec._block_offset]
        self.assertTrue(rep)
        self.assertTrue(all(codec.decode(a, e)[0] == 'REPLACE' for a in rep))
        self.assertTrue(np.allclose(encode_observation(e)[7], 0.0))

    def test_replace_moves_latest_wall_only(self):
        e = make(5, M=3)
        codec = ActionCodec(5)
        play(e, ('WALL', (3, 0, 'H')), ('MOVE', (3, 2)), ('WALL', (3, 3, 'H')), ('MOVE', (4, 2)))
        obs = encode_observation(e, mover=0)
        self.assertEqual(float(obs[4][3][3]), 1.0)   # latest wall slot
        self.assertEqual(float(obs[4][3][0]), 0.0)   # older wall is not replaceable
        self.assertEqual(float(obs[2][3][0]), 1.0)
        for a in codec.legal_ids(e):
            act = codec.decode(a, e)
            if act[0] == 'REPLACE':
                trial = e.clone()
                play(trial, act)
                self.assertIn((3, 0, 'H'), trial.walls[0])        # old wall stays
                self.assertNotIn((3, 3, 'H'), trial.walls[0])     # latest one moved

    def test_one_time_hop_ids(self):
        e = make(5)
        codec = ActionCodec(5)
        play(e, ('MOVE', (1, 2)), ('WALL', (3, 0, 'H')), ('WALL', (3, 3, 'H')), ('REPLACE', (1, 2, 'H')))
        ids = codec.legal_ids(e)
        self.assertIn(4, ids)       # HOP FWD across P1's wall
        self.assertNotIn(0, ids)    # MOVE FWD blocked
        self.assertEqual(codec.decode(4, e), ('HOP', (2, 2)))
        self.assertEqual(float(encode_observation(e)[9][0][0]), 1.0)
        play(e, ('HOP', (2, 2)))
        self.assertEqual(float(encode_observation(e, mover=0)[9][0][0]), 0.0)
        self.assertEqual(float(encode_observation(e, mover=1)[10][0][0]), 0.0)
        play(e, ('MOVE', (4, 1)))
        self.assertFalse([a for a in codec.legal_ids(e) if 4 <= a < 8])
        # own wall: no hop id
        e = make(5)
        play(e, ('WALL', (0, 1, 'H')), ('MOVE', (3, 2)))
        self.assertFalse([a for a in codec.legal_ids(e) if 4 <= a < 8])


class TestEncoding(unittest.TestCase):
    def test_planes_match_engine_fields(self):
        for n, cell, e in positions():
            rules = RULE_CELLS[cell]
            for mover in (0, 1):
                obs = encode_observation(e, mover=mover)
                opp = 1 - mover
                fr = (lambda r: r) if mover == 0 else (lambda r: n - 1 - r)
                fw = (lambda r: r) if mover == 0 else (lambda r: n - 2 - r)
                self.assertEqual(obs.shape, (14, n, n))
                self.assertEqual(obs[0][fr(e.pawns[mover][0])][e.pawns[mover][1]], 1.0)
                self.assertEqual(obs[1][fr(e.pawns[opp][0])][e.pawns[opp][1]], 1.0)
                self.assertEqual(obs[0].sum(), 1.0)
                self.assertEqual(obs[1].sum(), 1.0)
                walls = e.walls[0] + e.walls[1]
                self.assertEqual(obs[2].sum() + obs[3].sum(), len(walls))
                for wr, wc, o in walls:
                    self.assertEqual(obs[2 if o == 'H' else 3][fw(wr)][wc], 1.0)
                for ch, p in ((4, mover), (5, opp)):
                    lw = e.latest_wall[p]
                    self.assertEqual(obs[ch].sum(), 0.0 if lw is None else 1.0)
                    if lw is not None:
                        self.assertEqual(obs[ch][fw(lw[0])][lw[1]], 1.0)
                self.assertEqual(int(np.count_nonzero(obs[6])), len(e.blocked_cells))
                for (br, bc), t in e.blocked_cells.items():
                    self.assertAlmostEqual(float(obs[6][fr(br)][bc]), t / rules['K'], places=6)
                m = rules['M']
                self.assertTrue(np.allclose(obs[7], e.wall_inventory[mover] / m if m else 0.0))
                self.assertTrue(np.allclose(obs[8], e.wall_inventory[opp] / m if m else 0.0))
                self.assertTrue(np.allclose(obs[9], float(e.hop_available[mover])))
                self.assertTrue(np.allclose(obs[10], float(e.hop_available[opp])))
                self.assertTrue(np.allclose(obs[11], float(e.block_available[mover])))
                self.assertTrue(np.allclose(obs[12], float(e.block_available[opp])))
                # channel 13 is a constant 1.0 plane (no turn clock, no repetition count)
                self.assertTrue(np.allclose(obs[13], 1.0))

    def test_role_swap_equivariance(self):
        """Player 1's view of s == player 0's view of s with roles swapped, for the
        observation, the legal id set and the observation of every successor."""
        checked = 0
        for n, cell, e in positions():
            rules = RULE_CELLS[cell]
            codec = ActionCodec(n)
            t = restore(n, rules, swap_roles_key(e._get_state_hashable_key(), n))
            self.assertEqual(t.current_player, 1 - e.current_player)
            for m in (0, 1):
                np.testing.assert_array_equal(encode_observation(e, mover=m),
                                              encode_observation(t, mover=1 - m))
            ids_e = sorted(codec.legal_ids(e))
            self.assertEqual(ids_e, sorted(codec.legal_ids(t)))
            for aid in ids_e:
                a, b = e.clone(), t.clone()
                a.apply_action(codec.decode(aid, a), update_clock=False)
                b.apply_action(codec.decode(aid, b), update_clock=False)
                np.testing.assert_array_equal(encode_observation(a), encode_observation(b))
                self.assertEqual(a.winner is None, b.winner is None)
            checked += 1
        self.assertGreater(checked, 300)

    def test_known_alias_fixed_wall_owner(self):
        """Documents a limitation (reported in docs/reports/stage2_dqn.md): ownership of
        walls that are no longer the latest wall is not encoded, but HOP legality depends
        on it. Two positions with swapped owners of one fixed wall encode identically and
        have different legal sets. If the encoding gains an ownership plane, flip this."""
        n, rules = 5, {'A': True, 'B': True, 'C': True, 'K': 3, 'M': 10}
        a_wall, b_wall, c_wall, d_wall = (1, 2, 'H'), (3, 0, 'H'), (0, 0, 'V'), (3, 3, 'V')

        def key(fixed0, fixed1):
            return ((1, 2), (4, 2), (fixed0, b_wall), (fixed1, d_wall), 8, 8, b_wall, d_wall,
                    True, True, True, True, (), 0)
        x = restore(n, rules, key(a_wall, c_wall))   # wall in front of P0 is P0's own
        y = restore(n, rules, key(c_wall, a_wall))   # same wall owned by P1
        np.testing.assert_array_equal(encode_observation(x), encode_observation(y))
        self.assertNotIn(('HOP', (2, 2)), x.get_legal_actions())
        self.assertIn(('HOP', (2, 2)), y.get_legal_actions())


class TestCodec(unittest.TestCase):
    def test_every_legal_action_has_unique_valid_id_and_round_trips(self):
        hops = 0
        for n, cell, e in positions():
            codec = ActionCodec(n)
            ids = []
            for act in e.get_legal_actions():
                aid = codec.encode(act, e)
                self.assertTrue(0 <= aid < codec.num_actions)
                self.assertEqual(codec.decode(aid, e), act)
                ids.append(aid)
                hops += act[0] == 'HOP'
            self.assertEqual(len(ids), len(set(ids)))
        self.assertGreater(hops, 0)

    def test_mirror_against_engine_mirrored_positions(self):
        for n, cell, e in positions():
            if n % 2 == 0:
                continue
            rules = RULE_CELLS[cell]
            codec = ActionCodec(n)
            perm = codec.mirror_permutation()
            self.assertTrue(all(perm[perm[i]] == i for i in range(codec.num_actions)))
            m = restore(n, rules, mirror_key(e._get_state_hashable_key(), n))
            self.assertEqual(sorted(perm[i] for i in codec.legal_ids(e)), sorted(codec.legal_ids(m)))
            np.testing.assert_array_equal(mirror_observation(encode_observation(e)), encode_observation(m))
            for aid in codec.legal_ids(e):
                act = codec.decode(aid, e)
                mact = codec.decode(perm[aid], m)
                self.assertEqual(mact[0], act[0])
                if act[0] in ('MOVE', 'HOP', 'BLOCK'):
                    self.assertEqual(mact[1], (act[1][0], n - 1 - act[1][1]))
                else:
                    self.assertEqual(mact[1], (act[1][0], n - 2 - act[1][1], act[1][2]))


@unittest.skipUnless(TORCH, "torch not installed: self-play legality not checked")
class TestAgentsPlayLegally(unittest.TestCase):
    def test_self_play_never_illegal(self):
        from agents.dqn.trainer import DQNConfig, DQNTrainer
        from agents.dqn.self_play import SelfPlayManager
        for n, cell in ((3, 'train3_K1_M1'), (4, 'train4_K1_M0'), (5, 'default_K3_M10')):
            rules = RULE_CELLS[cell]
            cfg = DQNConfig(board_size=n, rules=rules, batch_size=16, min_buffer_size=32,
                            buffer_capacity=2000, epsilon_decay_steps=200, channels=8,
                            blocks=1, hidden=16, seed=7)
            sp = SelfPlayManager(DQNTrainer(cfg), snapshot_interval=50, max_plies=60)
            sp.pool.add(sp.trainer.online_net.state_dict())
            illegal, steps = [], []
            orig_step = WallzEnv.step

            def checked_step(env, action_id, _orig=orig_step):
                steps.append(action_id)
                if not env.legal_mask()[action_id]:
                    illegal.append((n, action_id))
                return _orig(env, action_id)
            WallzEnv.step = checked_step
            try:
                kinds = set()
                for i in range(12):
                    t = ('current', 'snapshot', 'greedy')[i % 3]
                    sp.sample_opponent_type = lambda t=t: t
                    kinds.add(sp.run_episode()['opponent_type'])
            finally:
                WallzEnv.step = orig_step
            self.assertEqual(illegal, [])
            self.assertGreater(len(steps), 24)
            self.assertEqual(kinds, {'current', 'snapshot', 'greedy'})

    def test_release_checkpoints_never_forfeit_in_runner(self):
        from evaluation.match_runner import MatchConfig, run_match
        for n, rules, path in ((3, RULE_CELLS['train3_K1_M1'], 'checkpoints/release/dqn_3x3/model.pt'),
                               (4, RULE_CELLS['train4_K1_M0'], 'checkpoints/release/dqn_4x4/model.pt')):
            spec = f"dqn:path={os.path.join(ROOT, path)}"
            for seed in range(6):
                for seat in (0, 1):
                    cfg = MatchConfig(board_size=n, rules=rules, p0=spec if seat == 0 else 'random',
                                      p1='random' if seat == 0 else spec, opening_plies=2,
                                      opening_seed=seed)
                    cat = run_match(cfg)['outcome']['category']
                    self.assertNotIn(cat, ('illegal_action', 'no_action', 'agent_error'))


if __name__ == '__main__':
    unittest.main()
