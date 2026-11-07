"""Golden Regression Tests against Solved Ground-Truth Retrograde Sweeps (P9.5).

Asserts that:
1. Re-solving 3x3 miniature boards reproduces the exact root values, ranks, state counts,
   and draw counts recorded in committed golden pilot solve JSONs.
2. Verified match replays remain 100% valid under current engine execution rules.
"""
import json
import os
import unittest

from evaluation.match_runner import MatchConfig, run_match, verify_replay
from evaluation.retrograde import solve_config, Solution
from evaluation.state_graph import GraphCaps


class TestGoldenSolverRegression(unittest.TestCase):
    SOLVE_DIR = os.path.join(os.path.dirname(__file__), "..", "experiments", "results", "solves")

    def test_golden_3x3_k1_m0_reproduction(self):
        """Validates all 8 factorial rule cells against committed solve_N3_K1_M0.json."""
        path = os.path.join(self.SOLVE_DIR, "solve_N3_K1_M0.json")
        self.assertTrue(os.path.exists(path), f"Golden solve file missing: {path}")

        with open(path, "r", encoding="utf-8") as f:
            golden = json.load(f)

        caps = GraphCaps(max_states=50_000, max_seconds=30.0)

        for cell in golden["cells"]:
            rules = cell["rules"]
            board_size = cell["board_size"]
            sol = solve_config(board_size, rules, caps)
            r = sol.root()

            # Assert complete exact solve
            self.assertTrue(sol.exact, f"Solve was not exact for rules {rules}")
            self.assertEqual(len(sol.graph.keys), cell["states"], f"State count mismatch for rules {rules}")
            self.assertEqual(sol.graph.edge_count, cell["edges"], f"Edge count mismatch for rules {rules}")

            # Assert game-theoretic root value and rank
            self.assertEqual(sol.status(r), cell["root_status"], f"Root status mismatch for rules {rules}")
            self.assertEqual(sol.rank[r], cell["root_rank"], f"Root rank mismatch for rules {rules}")

            # Assert draw counts
            draws = sol.draw_counts()
            self.assertEqual(draws["no_legal_actions"], cell["draw_counts"]["no_legal_actions"])
            self.assertEqual(draws["cycle"], cell["draw_counts"]["cycle"])
            self.assertEqual(draws["total"], cell["draw_counts"]["total"])

    def test_golden_3x3_k2_k3_subset_reproduction(self):
        """Validates representative cells for K=2 and K=3 against golden JSONs."""
        for filename in ["solve_N3_K2_M0.json", "solve_N3_K3_M0.json"]:
            path = os.path.join(self.SOLVE_DIR, filename)
            if not os.path.exists(path):
                continue

            with open(path, "r", encoding="utf-8") as f:
                golden = json.load(f)

            # Test first cell (pure pawn race) and last cell (full rules)
            test_cells = [golden["cells"][0], golden["cells"][2]]
            caps = GraphCaps(max_states=50_000, max_seconds=30.0)

            for cell in test_cells:
                rules = cell["rules"]
                sol = solve_config(cell["board_size"], rules, caps)
                r = sol.root()

                self.assertEqual(sol.status(r), cell["root_status"])
                self.assertEqual(sol.rank[r], cell["root_rank"])
                self.assertEqual(len(sol.graph.keys), cell["states"])

    def test_match_replay_verification(self):
        """Verifies that match runner replays execute with 100% hash and action integrity."""
        cfg = MatchConfig(
            board_size=3,
            rules={'A': True, 'B': True, 'C': True, 'K': 1, 'M': 1},
            p0='greedy',
            p1='minimax:depth=2',
            max_plies=40,
            opening_seed=101,
            record_replay=True,
        )
        result = run_match(cfg)
        self.assertIn('replay', result)
        ok, msg = verify_replay(result)
        self.assertTrue(ok, f"verify_replay failed: {msg}")


if __name__ == '__main__':
    unittest.main()
