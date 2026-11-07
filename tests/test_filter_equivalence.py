"""Test Filter Equivalence (P3.5 / P3.6 Checkpoint).

Verifies that the exact shortest-path filter in WallzEngine.get_legal_actions()
produces identical legal action sets to exhaustive brute force checking.
"""
import random
import unittest
from wallz_engine import WallzEngine


def brute_force_legal_actions(engine: WallzEngine, player: int = None):
    """Exhaustively generates legal actions without shortest-path filter shortcuts."""
    if player is None:
        player = engine.current_player
    if engine.is_terminal():
        return []
    actions = []
    r, c = engine.pawns[player]

    # Moves
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        if engine._is_valid_move(player, (r + dr, c + dc)):
            actions.append(('MOVE', (r + dr, c + dc)))

    # Hops
    for dr, dc in ((-2, 0), (2, 0), (0, -2), (0, 2)):
        if engine._is_valid_hop(player, (r + dr, c + dc)):
            actions.append(('HOP', (r + dr, c + dc)))

    n = engine.board_size
    # Walls
    if engine.wall_inventory[player] > 0:
        for wr in range(n - 1):
            for wc in range(n - 1):
                for orient in ('H', 'V'):
                    if engine._is_valid_wall_placement(player, wr, wc, orient):
                        actions.append(('WALL', (wr, wc, orient)))

    # Replacements
    if engine.rules['A'] and engine.latest_wall[player] is not None:
        for wr in range(n - 1):
            for wc in range(n - 1):
                for orient in ('H', 'V'):
                    if engine._is_valid_wall_replacement(player, (wr, wc, orient)):
                        actions.append(('REPLACE', (wr, wc, orient)))

    # Blocks
    if engine.rules['C'] and engine.block_available[player]:
        for br in range(n):
            for bc in range(n):
                if engine._is_valid_cell_block(br, bc):
                    actions.append(('BLOCK', (br, bc)))

    return actions


class TestFilterEquivalence(unittest.TestCase):
    def test_filter_matches_brute_force_start_positions(self):
        for n in (3, 4, 5, 9):
            for a in (True, False):
                for b in (True, False):
                    for c in (True, False):
                        e = WallzEngine(board_size=n, rules={'A': a, 'B': b, 'C': c, 'K': 2, 'M': 5})
                        filtered = set(e.get_legal_actions())
                        brute = set(brute_force_legal_actions(e))
                        self.assertEqual(filtered, brute, f"Mismatch on start N={n}, A={a}, B={b}, C={c}")

    def test_filter_matches_brute_force_midgame_states(self):
        random.seed(42)
        for trial in range(50):
            n = random.choice([3, 4, 5])
            rules = {
                'A': random.choice([True, False]),
                'B': random.choice([True, False]),
                'C': random.choice([True, False]),
                'K': random.randint(1, 3),
                'M': random.randint(1, 5),
            }
            e = WallzEngine(board_size=n, rules=rules)
            # Play a random sequence of plies
            for _ in range(random.randint(2, 12)):
                if e.is_terminal():
                    break
                legal = e.get_legal_actions()
                if not legal:
                    break
                a = random.choice(legal)
                e.apply_action(a, update_clock=False)

            if not e.is_terminal():
                filtered = set(e.get_legal_actions())
                brute = set(brute_force_legal_actions(e))
                self.assertEqual(filtered, brute, f"Mismatch at trial {trial} state: {e.get_state()}")


if __name__ == '__main__':
    unittest.main()
