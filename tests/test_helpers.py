"""Unit tests for tests/helpers.py test utilities and C-SNAP snapshot contract."""
import random
import unittest

from tests.helpers import (
    make_engine,
    snapshot,
    play,
    random_reachable,
    illegal_action_pool,
    pass_move,
    key_to_json,
    key_from_json,
)


class TestHelpers(unittest.TestCase):
    def test_snapshot_clone_equality_and_mutation_difference(self):
        eng = make_engine(n=5)
        clone = eng.clone()
        snap_eng = snapshot(eng)
        snap_clone = snapshot(clone)
        self.assertEqual(snap_eng, snap_clone)

        # Apply a legal action on engine
        legal = eng.get_legal_actions()
        eng.apply_action(legal[0], update_clock=False)
        snap_after = snapshot(eng)
        self.assertNotEqual(snap_eng, snap_after)
        self.assertEqual(snap_clone, snapshot(clone))

    def test_play_helper(self):
        eng = make_engine(n=5)
        actions = [('MOVE', (1, 2)), ('MOVE', (3, 2))]
        play(eng, actions)
        self.assertEqual(eng.pawns[0], (1, 2))
        self.assertEqual(eng.pawns[1], (3, 2))
        self.assertEqual(eng.turn_count, 2)

    def test_random_reachable_deterministic(self):
        rules = {'A': True, 'B': True, 'C': True, 'K': 3, 'M': 10}
        e1 = random_reachable(5, rules, plies=10, seed=42)
        e2 = random_reachable(5, rules, plies=10, seed=42)
        s1 = snapshot(e1)
        s2 = snapshot(e2)
        s1.pop('last_turn_timestamp')
        s2.pop('last_turn_timestamp')
        self.assertEqual(s1, s2)
        self.assertEqual(e1.position_key(), e2.position_key())

        e3 = random_reachable(5, rules, plies=10, seed=99)
        s3 = snapshot(e3)
        s3.pop('last_turn_timestamp')
        self.assertNotEqual(s1, s3)

    def test_illegal_action_pool_contains_no_legal_actions(self):
        rules = {'A': True, 'B': True, 'C': True, 'K': 3, 'M': 10}
        rng = random.Random(42)
        for trial in range(10):
            eng = random_reachable(5, rules, plies=trial * 2, seed=trial)
            pool = illegal_action_pool(eng, rng, k=30)
            legal = set(eng.get_legal_actions())
            for ill in pool:
                self.assertNotIn(ill, legal, f"Illegal action {ill} was found in legal actions")

    def test_pass_move(self):
        eng = make_engine(n=5)
        p_move = pass_move(eng)
        self.assertIn(p_move, eng.get_legal_actions())
        self.assertEqual(p_move[0], 'MOVE')
        self.assertNotEqual(p_move[1][0], eng.goals[eng.current_player])

    def test_key_json_roundtrip(self):
        eng = make_engine(n=5)
        raw_key = eng._get_state_hashable_key()
        json_obj = key_to_json(raw_key)
        restored_key = key_from_json(json_obj)
        self.assertEqual(raw_key, restored_key)

        # Restore into engine and check key
        clone = eng.clone()
        clone.restore_state_key(restored_key)
        self.assertEqual(clone._get_state_hashable_key(), raw_key)


if __name__ == '__main__':
    unittest.main()
