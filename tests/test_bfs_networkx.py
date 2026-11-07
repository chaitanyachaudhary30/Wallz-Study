"""Gate X-2: Validation of BFS Pathfinding against NetworkX Independent Reference (P9.2)."""
import math
import random
import unittest
import networkx as nx

from src.modified_wallz.pathfinding import bfs_distance
from wallz_engine import WallzEngine


def build_networkx_grid(board_size: int, walls, blocked_cells):
    """
    Constructs an independent networkx.Graph representation of the Modified Wallz board.
    Removes edges intersected by walls and nodes occupied by active cell blocks.
    """
    g = nx.Graph()
    n = board_size

    # Add all board cells
    for r in range(n):
        for c in range(n):
            g.add_node((r, c))

    # Add standard 4-neighbor grid edges
    for r in range(n):
        for c in range(n):
            if r + 1 < n:
                g.add_edge((r, c), (r + 1, c))
            if c + 1 < n:
                g.add_edge((r, c), (r, c + 1))

    # Remove edges obstructed by walls
    for wr, wc, orient in walls:
        if orient == 'H':
            # Horizontal wall blocks ((wr, wc), (wr+1, wc)) and ((wr, wc+1), (wr+1, wc+1))
            g.remove_edge((wr, wc), (wr + 1, wc))
            g.remove_edge((wr, wc + 1), (wr + 1, wc + 1))
        elif orient == 'V':
            # Vertical wall blocks ((wr, wc), (wr, wc+1)) and ((wr+1, wc), (wr+1, wc+1))
            g.remove_edge((wr, wc), (wr, wc + 1))
            g.remove_edge((wr + 1, wc), (wr + 1, wc + 1))

    # Remove nodes obstructed by active temporary cell blocks
    for cell in blocked_cells:
        if g.has_node(cell):
            g.remove_node(cell)

    return g


def networkx_shortest_path_to_goal(nx_graph: nx.Graph, pawn: tuple, goal_row: int, board_size: int) -> float:
    """Computes exact shortest path distance from pawn to goal_row via NetworkX BFS."""
    if not nx_graph.has_node(pawn):
        return math.inf

    min_dist = math.inf
    for col in range(board_size):
        target = (goal_row, col)
        if nx_graph.has_node(target):
            try:
                dist = nx.shortest_path_length(nx_graph, source=pawn, target=target)
                if dist < min_dist:
                    min_dist = dist
            except nx.NetworkXNoPath:
                pass

    return min_dist


class TestBFSNetworkXEquivalence(unittest.TestCase):
    def test_x2_bfs_matches_networkx_exhaustive_random_layouts(self):
        """
        Gate X-2: Verifies that engine BFS pathfinding matches an independent NetworkX
        reference 100% across thousands of random board, wall, and block configurations.
        """
        rng = random.Random(42)
        total_trials = 2000

        for trial in range(total_trials):
            # Sample board size N in {3, 5, 9}
            n = rng.choice([3, 5, 9])
            engine = WallzEngine(board_size=n, rules={'A': True, 'B': True, 'C': True, 'M': 10})

            # Randomize pawns on distinct cells
            p0 = (rng.randint(0, n - 1), rng.randint(0, n - 1))
            p1 = (rng.randint(0, n - 1), rng.randint(0, n - 1))
            while p1 == p0:
                p1 = (rng.randint(0, n - 1), rng.randint(0, n - 1))

            engine.pawns[0] = p0
            engine.pawns[1] = p1

            # Place random legal walls
            num_walls = rng.randint(0, min(6, n))
            all_walls = []
            for _ in range(num_walls):
                wr = rng.randint(0, n - 2)
                wc = rng.randint(0, n - 2)
                orient = rng.choice(['H', 'V'])
                if engine.board.can_place_wall(wr, wc, orient):
                    engine.board.add_wall(wr, wc, orient)
                    engine.walls[0].append((wr, wc, orient))
                    all_walls.append((wr, wc, orient))

            # Add random blocked cells (excluding pawns)
            num_blocks = rng.randint(0, min(3, n))
            blocked = {}
            for _ in range(num_blocks):
                br = rng.randint(0, n - 1)
                bc = rng.randint(0, n - 1)
                cell = (br, bc)
                if cell != p0 and cell != p1:
                    blocked[cell] = rng.randint(1, 3)

            engine.blocked_cells = blocked

            # Compute via NetworkX reference
            nx_graph = build_networkx_grid(n, all_walls, blocked)
            nx_dist_p0 = networkx_shortest_path_to_goal(nx_graph, p0, engine.goals[0], n)
            nx_dist_p1 = networkx_shortest_path_to_goal(nx_graph, p1, engine.goals[1], n)

            # Compute via Engine / Pathfinding BFS
            eng_dist_p0 = engine.shortest_path(0)
            eng_dist_p1 = engine.shortest_path(1)
            bfs_p0 = bfs_distance(p0, engine.goals[0], n, engine.board, blocked)
            bfs_p1 = bfs_distance(p1, engine.goals[1], n, engine.board, blocked)

            # Assert 100% equivalence
            self.assertEqual(eng_dist_p0, bfs_p0)
            self.assertEqual(eng_dist_p1, bfs_p1)
            self.assertEqual(eng_dist_p0, nx_dist_p0, f"Mismatch P0 at trial {trial} N={n}: eng={eng_dist_p0} vs nx={nx_dist_p0}")
            self.assertEqual(eng_dist_p1, nx_dist_p1, f"Mismatch P1 at trial {trial} N={n}: eng={eng_dist_p1} vs nx={nx_dist_p1}")


if __name__ == '__main__':
    unittest.main()
