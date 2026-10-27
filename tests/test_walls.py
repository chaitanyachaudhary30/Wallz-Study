import unittest
from wallz_engine import WallzEngine

class TestWalls(unittest.TestCase):
    def test_wall_legality(self):
        engine = WallzEngine(board_size=5)
        # Place horizontal wall at (1, 1)
        self.assertTrue(engine._is_valid_wall_placement(0, 1, 1, 'H'))
        engine.walls[0].append((1, 1, 'H'))
        # Overlapping wall is illegal
        self.assertFalse(engine._is_valid_wall_placement(1, 1, 1, 'H'))
        # Intersecting vertical wall at same anchor is illegal
        self.assertFalse(engine._is_valid_wall_placement(1, 1, 1, 'V'))

    def test_wall_blocking_shortest_path(self):
        engine = WallzEngine(board_size=5)
        p0_start_path = engine.shortest_path(0)
        # Add horizontal wall in front of player 0
        engine.walls[1].append((0, 2, 'H'))
        p0_new_path = engine.shortest_path(0)
        self.assertGreater(p0_new_path, p0_start_path)

if __name__ == '__main__':
    unittest.main()

