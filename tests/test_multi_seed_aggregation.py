"""Multi-seed aggregation in scripts/run_multi_seed_3x3.py, on synthetic results (no torch)."""
import math
import unittest

from scripts.run_multi_seed_3x3 import (aggregate, across_seeds, checkpoint_path, main,
                                        outcome_row, t975)


def row(win, loss, draw, trunc, mean_len):
    cats = ['win'] * win + ['loss'] * loss + ['draw'] * draw + ['truncated'] * trunc
    return outcome_row(cats, [mean_len] * len(cats))


class TestMultiSeedAggregation(unittest.TestCase):
    def test_truncated_is_its_own_column(self):
        r = row(5, 3, 1, 1, 10)
        self.assertEqual(r['counts']['truncated'], 1)
        self.assertEqual(r['counts']['draw'], 1)
        self.assertAlmostEqual(r['draw_rate'], 0.1)
        self.assertAlmostEqual(r['truncation_rate'], 0.1)
        self.assertAlmostEqual(r['win_rate'] + r['loss_rate'] + r['draw_rate'] + r['truncation_rate'], 1.0)
        with self.assertRaises(ValueError):
            outcome_row(['draw_truncated'], [1])

    def test_across_seeds_t_interval(self):
        xs = [0.5, 0.6, 0.7, 0.8, 0.9]
        s = across_seeds(xs)
        sd = math.sqrt(sum((x - 0.7) ** 2 for x in xs) / 4)  # sample SD, n - 1
        self.assertAlmostEqual(s['mean'], 0.7)
        self.assertAlmostEqual(s['sd'], sd)
        half = 2.776 * sd / math.sqrt(5)
        self.assertAlmostEqual(s['ci95_t'][0], 0.7 - half)
        self.assertAlmostEqual(s['ci95_t'][1], 0.7 + half)
        self.assertAlmostEqual(s['leave_one_out_means'][0], (0.6 + 0.7 + 0.8 + 0.9) / 4)
        self.assertAlmostEqual(s['max_leave_one_out_shift'], 0.05)
        self.assertEqual(t975(4), 2.776)
        self.assertIsNone(across_seeds([0.3])['ci95_t'])

    def test_aggregate_five_seeds(self):
        seeds = []
        for i, seed in enumerate((42, 43, 44, 45, 46)):
            seeds.append({'seed': seed, 'sign_agreement': 0.9 + 0.01 * i, 'mae': 0.1,
                          'vs': {'greedy': row(6 + i, 4 - i, 0, 0, 12 + i),
                                 'random': row(9, 0, 0, 1, 8)}})
        agg = aggregate(seeds)
        self.assertEqual(agg['seeds'], [42, 43, 44, 45, 46])
        g = agg['vs']['greedy']
        self.assertAlmostEqual(g['win_rate']['mean'], 0.8)
        self.assertAlmostEqual(g['mean_length_plies']['mean'], 14.0)
        self.assertEqual(g['pooled']['n_games'], 50)
        self.assertEqual(g['pooled']['counts']['win'], 40)
        r = agg['vs']['random']
        self.assertAlmostEqual(r['truncation_rate']['mean'], 0.1)
        self.assertEqual(r['draw_rate']['mean'], 0.0)
        self.assertEqual(r['pooled']['counts']['truncated'], 5)
        self.assertEqual(r['truncation_rate']['sd'], 0.0)       # identical seeds: zero-width interval
        self.assertAlmostEqual(agg['sign_agreement']['mean'], 0.92)
        lo, hi = g['pooled']['win_rate_ci95']
        self.assertTrue(lo < 0.8 < hi)

    def test_manifest_sha_mismatch_exits(self):
        import hashlib
        import json
        import tempfile
        from pathlib import Path
        from scripts.run_multi_seed_3x3 import check_manifests
        with tempfile.TemporaryDirectory() as d:
            ck, man = Path(d, 'ck', '3x3_s42'), Path(d, 'man', '3x3_s42')
            ck.mkdir(parents=True)
            man.mkdir(parents=True)
            (ck / 'final.pt').write_bytes(b'weights')
            good = hashlib.sha256(b'weights').hexdigest()
            (man / 'manifest.json').write_text(json.dumps({'sha256': good}))
            self.assertEqual(check_manifests([42], Path(d, 'ck'), Path(d, 'man')), {42: good})
            (man / 'manifest.json').write_text(json.dumps({'sha256': '0' * 64}))
            with self.assertRaises(SystemExit) as cm:
                check_manifests([42], Path(d, 'ck'), Path(d, 'man'))
            self.assertIn('!= manifest', str(cm.exception.code))

    def test_missing_checkpoint_exits_without_training(self):
        self.assertEqual(checkpoint_path('checkpoints/final', 42).as_posix(), 'checkpoints/final/3x3_s42/final.pt')
        with self.assertRaises(SystemExit) as cm:
            main(['--checkpoint-dir', 'no/such/dir', '--seeds', '42'])
        self.assertIn('missing checkpoints', str(cm.exception.code))


if __name__ == '__main__':
    unittest.main()
