import unittest
from wallz_engine import WallzEngine

class TestReplacement(unittest.TestCase):
    def setUp(self):
        self.engine = WallzEngine(board_size=9, rules={'A': True})

    def test_repositioning_latest_wall(self):
        """Player can reposition their most recently placed wall without consuming inventory."""
        # Player 0 places wall 1
        self.engine.apply_action(('WALL', (1, 1, 'H')), update_clock=False)
        self.assertEqual(self.engine.wall_inventory[0], 9)
        self.assertEqual(self.engine.latest_wall[0], (1, 1, 'H'))

        # Player 1 moves pawn
        self.engine.apply_action(('MOVE', (7, 4)), update_clock=False)

        # Player 0 places wall 2
        self.engine.apply_action(('WALL', (3, 3, 'V')), update_clock=False)
        self.assertEqual(self.engine.wall_inventory[0], 8)
        self.assertEqual(self.engine.latest_wall[0], (3, 3, 'V'))

        # Player 1 moves pawn
        self.engine.apply_action(('MOVE', (6, 4)), update_clock=False)

        # Player 0 repositions latest wall from (3, 3, 'V') to (4, 4, 'H')
        self.engine.apply_action(('REPLACE', (4, 4, 'H')), update_clock=False)
        
        # Verify wall 2 was replaced, older wall 1 remains intact, inventory is unchanged
        self.assertIn((1, 1, 'H'), self.engine.walls[0], "Older wall was unexpectedly modified")
        self.assertNotIn((3, 3, 'V'), self.engine.walls[0], "Previous latest wall was not removed")
        self.assertIn((4, 4, 'H'), self.engine.walls[0], "New repositioned wall was not added")
        self.assertEqual(self.engine.latest_wall[0], (4, 4, 'H'))
        self.assertEqual(self.engine.wall_inventory[0], 8, "Repositioning should not consume wall inventory")

    def test_rejection_of_repositioning_older_wall(self):
        """Attempting to reposition an older wall must be strictly rejected."""
        # Player 0 places two walls
        self.engine.apply_action(('WALL', (1, 1, 'H')), update_clock=False)
        self.engine.apply_action(('MOVE', (7, 4)), update_clock=False)
        self.engine.apply_action(('WALL', (3, 3, 'V')), update_clock=False)
        self.engine.apply_action(('MOVE', (6, 4)), update_clock=False)

        # Attempt to reposition older wall (1, 1, 'H') instead of latest wall (3, 3, 'V')
        older_wall = (1, 1, 'H')
        new_loc = (5, 5, 'H')

        # 1. Direct validation check returns False
        self.assertFalse(self.engine._is_valid_wall_replacement(0, new_loc, old_wall=older_wall))
        self.assertFalse(self.engine.reposition_wall(0, new_loc, old_wall=older_wall))

        # 2. apply_action raises ValueError and preserves state
        walls_before = list(self.engine.walls[0])
        latest_before = self.engine.latest_wall[0]
        inv_before = self.engine.wall_inventory[0]

        with self.assertRaises(ValueError):
            self.engine.apply_action(('REPLACE', (older_wall, new_loc)), update_clock=False)

        self.assertEqual(self.engine.walls[0], walls_before)
        self.assertEqual(self.engine.latest_wall[0], latest_before)
        self.assertEqual(self.engine.wall_inventory[0], inv_before)

    def test_wall_inventory_unchanged_after_replacement(self):
        """When wall inventory is exhausted (0), new walls are blocked but latest wall can still be repositioned."""
        # Consume all 10 walls for player 0
        for i in range(10):
            # Place horizontal walls along row i // 2, column (i % 2) * 3
            wr = i // 2
            wc = (i % 2) * 3
            self.engine.apply_action(('WALL', (wr, wc, 'H')), update_clock=False)
            self.engine.apply_action(('MOVE', (8 - (i % 2), 4)), update_clock=False)

        self.assertEqual(self.engine.wall_inventory[0], 0)
        self.assertEqual(len(self.engine.walls[0]), 10)

        # 1. Placing an 11th wall via 'WALL' must be rejected
        legal_types = [a[0] for a in self.engine.get_legal_actions(0)]
        self.assertNotIn('WALL', legal_types, "New wall placement should be illegal when inventory is 0")

        with self.assertRaises(ValueError):
            self.engine.apply_action(('WALL', (7, 0, 'H')), update_clock=False)

        # 2. 'REPLACE' must be available and functional
        self.assertIn('REPLACE', legal_types, "Repositioning should remain legal when inventory is 0")
        
        latest_before = self.engine.latest_wall[0]
        legal_replaces = [a[1] for a in self.engine.get_legal_actions(0) if a[0] == 'REPLACE']
        self.assertTrue(len(legal_replaces) > 0)

        chosen_replace = legal_replaces[0]
        self.engine.apply_action(('REPLACE', chosen_replace), update_clock=False)

        self.assertEqual(self.engine.wall_inventory[0], 0, "Inventory must remain 0 after replacement")
        self.assertEqual(len(self.engine.walls[0]), 10, "Total wall count on board must remain 10")
        self.assertNotIn(latest_before, self.engine.walls[0])
        self.assertIn(chosen_replace, self.engine.walls[0])
        self.assertEqual(self.engine.latest_wall[0], chosen_replace)

    def test_invalid_replacement_preserves_original_state(self):
        """Invalid replacement attempts (collision / out of bounds) must not corrupt state."""
        self.engine.apply_action(('WALL', (2, 2, 'H')), update_clock=False)
        self.engine.apply_action(('WALL', (4, 4, 'H')), update_clock=False) # Opponent wall
        
        walls_p0_before = list(self.engine.walls[0])
        latest_before = self.engine.latest_wall[0]
        inv_before = self.engine.wall_inventory[0]

        # 1. Overlapping opponent wall at (4, 4, 'H')
        self.assertFalse(self.engine._is_valid_wall_replacement(0, (4, 4, 'H')))
        with self.assertRaises(ValueError):
            self.engine.apply_action(('REPLACE', (4, 4, 'H')), update_clock=False)

        # 2. Out of bounds
        self.assertFalse(self.engine._is_valid_wall_replacement(0, (10, 10, 'H')))
        with self.assertRaises(ValueError):
            self.engine.apply_action(('REPLACE', (10, 10, 'H')), update_clock=False)

        self.assertEqual(self.engine.walls[0], walls_p0_before)
        self.assertEqual(self.engine.latest_wall[0], latest_before)
        self.assertEqual(self.engine.wall_inventory[0], inv_before)

    def test_path_preservation_during_replacement(self):
        """Repositioning a wall to a location that seals off all goal paths must be rejected."""
        engine3 = WallzEngine(board_size=3, rules={'A': True})
        # Player 0 at (0, 1), Player 1 at (2, 1)
        engine3.apply_action(('WALL', (1, 0, 'H')), update_clock=False)
        engine3.apply_action(('MOVE', (2, 0)), update_clock=False)

        # Opponent wall at (0, 0, 'H')
        engine3.walls[1].append((0, 0, 'H'))

        # If Player 0 attempts to reposition (1, 0, 'H') to (0, 1, 'H'), it completely traps Player 0 at (0, 1)
        self.assertFalse(engine3._is_valid_wall_replacement(0, (0, 1, 'H')))
        with self.assertRaises(ValueError):
            engine3.apply_action(('REPLACE', (0, 1, 'H')), update_clock=False)

    def test_no_wall_placed_yet_behavior(self):
        """When a player has placed 0 walls, no repositioning action is valid or permissible."""
        self.assertIsNone(self.engine.latest_wall[0])
        legal_actions = self.engine.get_legal_actions(0)
        replace_actions = [a for a in legal_actions if a[0] == 'REPLACE']
        self.assertEqual(len(replace_actions), 0)

        self.assertFalse(self.engine._is_valid_wall_replacement(0, (1, 1, 'H')))
        self.assertFalse(self.engine.reposition_wall(0, (1, 1, 'H')))

        with self.assertRaises(ValueError):
            self.engine.apply_action(('REPLACE', (1, 1, 'H')), update_clock=False)

    def test_rule_a_disabled_ablation(self):
        """When Rule A is disabled, wall replacement is completely unavailable."""
        engine_no_a = WallzEngine(board_size=9, rules={'A': False})
        engine_no_a.apply_action(('WALL', (1, 1, 'H')), update_clock=False)
        engine_no_a.apply_action(('MOVE', (7, 4)), update_clock=False)

        legal_types = [a[0] for a in engine_no_a.get_legal_actions(0)]
        self.assertNotIn('REPLACE', legal_types)
        self.assertFalse(engine_no_a._is_valid_wall_replacement(0, (2, 2, 'H')))

        with self.assertRaises(ValueError):
            engine_no_a.apply_action(('REPLACE', (2, 2, 'H')), update_clock=False)

if __name__ == '__main__':
    unittest.main()
