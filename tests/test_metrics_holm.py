"""Unit tests for Phase 7: Match runner, logging, Holm-Bonferroni correction, and table generation (P7.1-P7.5)."""
import unittest
from evaluation.match_runner import MatchConfig, run_match as execute_match
from evaluation.metrics import holm_bonferroni, summarize_cells, effects_with_ci
from run_match import run_match


class TestMetricsAndHolm(unittest.TestCase):
    def test_holm_bonferroni_empty_and_single(self):
        """Holm-Bonferroni handles empty and single p-value inputs."""
        self.assertEqual(holm_bonferroni({}), {})
        self.assertEqual(holm_bonferroni([]), [])

        single = holm_bonferroni([0.03], alpha=0.05)
        self.assertEqual(len(single), 1)
        self.assertAlmostEqual(single[0]['adjusted_p'], 0.03)
        self.assertTrue(single[0]['significant'])

    def test_holm_bonferroni_step_down_logic(self):
        """Holm-Bonferroni applies step-down multipliers m - k + 1 and preserves monotonicity."""
        # 3 hypotheses with raw p-values:
        # p1 = 0.01 (rank 1, multiplier 3 -> 0.03 <= 0.05 -> SIG)
        # p2 = 0.03 (rank 2, multiplier 2 -> 0.06 > 0.05 -> NOT SIG)
        # p3 = 0.04 (rank 3, multiplier 1 -> 0.04 -> max(0.04, 0.06) = 0.06 -> NOT SIG)
        p_dict = {'testA': 0.01, 'testB': 0.04, 'testC': 0.03}
        res = holm_bonferroni(p_dict, alpha=0.05)

        self.assertEqual(res['testA']['rank'], 1)
        self.assertAlmostEqual(res['testA']['adjusted_p'], 0.03)
        self.assertTrue(res['testA']['significant'])

        self.assertEqual(res['testC']['rank'], 2)
        self.assertAlmostEqual(res['testC']['adjusted_p'], 0.06)
        self.assertFalse(res['testC']['significant'])

        self.assertEqual(res['testB']['rank'], 3)
        self.assertAlmostEqual(res['testB']['adjusted_p'], 0.06)
        self.assertFalse(res['testB']['significant'])

    def test_summarize_cells_stalemate_tracking(self):
        """summarize_cells accurately tracks R3 stalemates and repetition draws separately."""
        mock_results = [
            # Game 1: Win
            {
                'match_id': 'm1',
                'config_label': 'cell1',
                'config': {'opening_seed': 0},
                'agents': {'0': {'spec': 'x'}, '1': {'spec': 'y'}},
                'outcome': {'category': 'goal', 'winner': 0, 'detail': 'goal'},
                'length_plies': 10,
                'mean_legal_actions': {'0': 15.0, '1': 14.0},
                'decision_seconds': {'0': {'mean': 0.01}, '1': {'mean': 0.01}},
            },
            # Game 2: R3 stalemate (no legal actions)
            {
                'match_id': 'm2',
                'config_label': 'cell1',
                'config': {'opening_seed': 1},
                'agents': {'0': {'spec': 'x'}, '1': {'spec': 'y'}},
                'outcome': {'category': 'draw_no_legal_actions', 'winner': None, 'detail': 'no_legal_actions'},
                'length_plies': 8,
                'mean_legal_actions': {'0': 10.0, '1': 10.0},
                'decision_seconds': {'0': {'mean': 0.01}, '1': {'mean': 0.01}},
            },
            # Game 3: 5-fold repetition draw
            {
                'match_id': 'm3',
                'config_label': 'cell1',
                'config': {'opening_seed': 2},
                'agents': {'0': {'spec': 'x'}, '1': {'spec': 'y'}},
                'outcome': {'category': 'draw_repetition', 'winner': None, 'detail': 'repetition'},
                'length_plies': 20,
                'mean_legal_actions': {'0': 12.0, '1': 12.0},
                'decision_seconds': {'0': {'mean': 0.01}, '1': {'mean': 0.01}},
            },
        ]

        summary = summarize_cells(mock_results, 'x')
        self.assertIn('cell1', summary)
        c = summary['cell1']
        self.assertEqual(c['n_games'], 3)
        self.assertEqual(c['counts']['win'], 1)
        self.assertEqual(c['counts']['draw'], 2)  # Backward compatible total draws
        self.assertEqual(c['draw_breakdown']['stalemate'], 1)
        self.assertEqual(c['draw_breakdown']['repetition'], 1)
        self.assertAlmostEqual(c['stalemate_rate'], 1 / 3)
        self.assertAlmostEqual(c['repetition_rate'], 1 / 3)
        self.assertIsNotNone(c['stalemate_rate_ci95'])

    def test_replica_seeds_count_once_by_opening(self):
        """unit='opening' merges seeds that drew the same opening; unit='seed' does not."""
        def game(seed, opening, label, win):
            return {'match_id': f'{label}-{seed}', 'config_label': label,
                    'config': {'opening_seed': seed}, 'opening': {'actions': opening},
                    'agents': {'0': {'spec': 'x'}, '1': {'spec': 'y'}},
                    'outcome': {'category': 'goal', 'winner': 0 if win else 1},
                    'length_plies': 5, 'mean_legal_actions': {'0': 1.0, '1': 1.0},
                    'decision_seconds': {'0': {'mean': 0.01}, '1': {'mean': 0.01}}}
        labels = [f'N3_A{a}B{b}C{c}_K1_M0' for a in (0, 1) for b in (0, 1) for c in (0, 1)]
        # seeds 0-2 share opening 'o1', seed 3 has 'o2'; X wins only with A on and opening o1
        results = [game(s, ['o1'] if s < 3 else ['o2'], lab, lab[4] == '1' and s < 3)
                   for s in range(4) for lab in labels]
        by_seed = effects_with_ci(results, 'x', 'x_goal_win', n_boot=200, unit='seed')
        by_open = effects_with_ci(results, 'x', 'x_goal_win', n_boot=200, unit='opening')
        self.assertEqual((by_seed['n_units'], by_open['n_units']), (4, 2))
        self.assertAlmostEqual(by_seed['effects']['A']['estimate'], 0.75)
        self.assertAlmostEqual(by_open['effects']['A']['estimate'], 0.5)

    def test_run_match_wrapper(self):
        """run_match.py acts as a thin wrapper over match_runner and returns expected types."""
        winner, plies, metrics = run_match(
            p0_type='greedy',
            p1_type='greedy',
            board_size=3,
            max_steps=10,
            verbose=False,
        )
        self.assertIn(winner, [0, 1, None])
        self.assertGreater(plies, 0)
        self.assertIn('result', metrics)
        self.assertEqual(metrics['result']['schema_version'], 1)


if __name__ == '__main__':
    unittest.main()
