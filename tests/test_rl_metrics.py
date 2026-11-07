"""Unit tests for RL Evaluation Metrics (P2.1)."""
import unittest
import numpy as np

from evaluation.rl_metrics import interquartile_mean, bootstrap_ci, iqm_with_ci


class TestRLMetrics(unittest.TestCase):
    def test_iqm_basic(self):
        # Symmetric uniform data: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
        # 25% trim removes bottom 2.5 and top 2.5 items, middle is centered at 5.5
        data = np.arange(1, 11)
        iqm = interquartile_mean(data)
        self.assertAlmostEqual(iqm, 5.5, places=5)

    def test_iqm_outlier_robustness(self):
        # Data with severe outliers
        # Normal data centered around 10
        data = np.array([10.0, 10.2, 9.8, 10.1, 9.9, 1000.0, -500.0, 10.0])
        iqm = interquartile_mean(data)
        # Mean is heavily skewed (~80)
        self.assertGreater(np.mean(data), 60.0)
        # IQM should be near 10
        self.assertAlmostEqual(iqm, 10.0, delta=0.5)

    def test_bootstrap_ci_coverage(self):
        # Generate Gaussian sample
        rng = np.random.default_rng(12345)
        sample = rng.normal(loc=50.0, scale=5.0, size=100)

        ci_low, ci_high = bootstrap_ci(
            sample,
            statistic_fn=np.mean,
            num_bootstraps=1000,
            confidence_level=0.95,
            random_state=42,
        )
        self.assertLess(ci_low, np.mean(sample))
        self.assertGreater(ci_high, np.mean(sample))
        # 50.0 should be well within the interval
        self.assertLess(ci_low, 50.0)
        self.assertGreater(ci_high, 50.0)

    def test_iqm_with_ci_aggregator(self):
        scores = [0.65, 0.70, 0.68, 0.72, 0.69]
        res = iqm_with_ci(scores, num_bootstraps=500, random_state=42)
        self.assertIn('iqm', res)
        self.assertIn('ci_lower', res)
        self.assertIn('ci_upper', res)
        self.assertIn('mean', res)
        self.assertEqual(res['n'], 5)
        self.assertLessEqual(res['ci_lower'], res['iqm'])
        self.assertGreaterEqual(res['ci_upper'], res['iqm'])


if __name__ == '__main__':
    unittest.main()
