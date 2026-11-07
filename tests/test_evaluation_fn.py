"""Unit tests for Phase 6: Evaluation Function and Feature Extraction (P6.1, P6.2, C7)."""
import math
import unittest
import numpy as np

from agents.evaluation_fn import EvaluationFunction, HEvaluator, LinearEvaluator
from agents.minimax import MinimaxAgent
from evaluation.features import extract_features, FEATURE_NAMES
from wallz_engine import WallzEngine


class TestEvaluationFunction(unittest.TestCase):
    def setUp(self):
        self.rules = {'A': True, 'B': True, 'C': True, 'K': 1, 'M': 2}
        self.engine = WallzEngine(board_size=3, rules=self.rules)

    def test_protocol_conformance(self):
        """HEvaluator and LinearEvaluator conform to EvaluationFunction protocol."""
        h_eval = HEvaluator()
        self.assertIsInstance(h_eval, EvaluationFunction)

        lin_eval = LinearEvaluator(weights=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.assertIsInstance(lin_eval, EvaluationFunction)

    def test_h_evaluator_symmetry_and_values(self):
        """HEvaluator computes tanh(0.2 * H) correctly and exhibits anti-symmetry."""
        h_eval = HEvaluator(scale=0.2)

        # Initial state on 3x3: P0 at (0, 1) goal row 2 (d=2), P1 at (2, 1) goal row 0 (d=2)
        # H = d_opp - d_self = 2 - 2 = 0
        v0 = h_eval.evaluate(self.engine, 0)
        v1 = h_eval.evaluate(self.engine, 1)
        self.assertAlmostEqual(v0, 0.0, places=5)
        self.assertAlmostEqual(v1, 0.0, places=5)

        # Move P0 forward: (0, 1) -> (1, 1). P0 goal row 2 (d=1), P1 goal row 0 (d=2)
        # For P0: H = 2 - 1 = +1 => tanh(0.2 * 1) = tanh(0.2)
        self.engine.apply_action(('MOVE', (1, 1)))
        v0 = h_eval.evaluate(self.engine, 0)
        v1 = h_eval.evaluate(self.engine, 1)
        expected_v0 = math.tanh(0.2)
        self.assertAlmostEqual(v0, expected_v0, places=5)
        self.assertAlmostEqual(v1, -expected_v0, places=5)
        self.assertAlmostEqual(v0, -v1, places=5)

    def test_extract_features_structure_and_values(self):
        """extract_features produces 6 features matching specifications."""
        feats_p0 = extract_features(self.engine, 0)
        self.assertEqual(len(feats_p0), 6)
        self.assertEqual(len(FEATURE_NAMES), 6)

        # At start:
        # f1: H = 0
        self.assertEqual(feats_p0[0], 0.0)
        # f2: Tempo = +1 for P0 (turn 0)
        self.assertEqual(feats_p0[1], 1.0)
        # f3: Wall advantage = M0 - M1 = 2 - 2 = 0
        self.assertEqual(feats_p0[2], 0.0)
        # f4: Hop advantage = 1 - 1 = 0
        self.assertEqual(feats_p0[3], 0.0)
        # f5: Block advantage = 1 - 1 = 0
        self.assertEqual(feats_p0[4], 0.0)
        # f6: Replaceable wall advantage = 0 - 0 = 0
        self.assertEqual(feats_p0[5], 0.0)

        # Features relative to P1 at start
        feats_p1 = extract_features(self.engine, 1)
        self.assertEqual(feats_p1[0], 0.0)
        self.assertEqual(feats_p1[1], -1.0)  # P1 not to move
        self.assertEqual(feats_p1[2], 0.0)

    def test_linear_evaluator_identity(self):
        """LinearEvaluator computes tanh(w·f/2) matching 2*sigma(w·f) - 1."""
        weights = [1.0, 0.5, 0.2, 0.1, 0.1, 0.1]
        lin_eval = LinearEvaluator(weights=weights, scale=0.5)

        val0 = lin_eval.evaluate(self.engine, 0)
        feats = extract_features(self.engine, 0)
        raw_dot = float(np.dot(weights, feats))

        # Check tanh(0.5 * dot)
        expected_tanh = math.tanh(0.5 * raw_dot)
        # Check 2 * sigmoid(dot) - 1
        expected_sig = 2.0 / (1.0 + math.exp(-raw_dot)) - 1.0
        self.assertAlmostEqual(val0, expected_tanh, places=5)
        self.assertAlmostEqual(val0, expected_sig, places=5)
        self.assertTrue(-1.0 < val0 < 1.0)

    def test_minimax_agent_with_custom_evaluator(self):
        """MinimaxAgent accepts and searches with custom evaluator."""
        # Default agent has HEvaluator
        default_agent = MinimaxAgent(player_id=0, depth=1)
        self.assertIsInstance(default_agent.evaluator, HEvaluator)

        # Custom agent with LinearEvaluator
        weights = [2.0, 0.0, 0.5, 1.0, 0.5, 0.0]
        custom_eval = LinearEvaluator(weights=weights)
        custom_agent = MinimaxAgent(player_id=0, depth=1, evaluator=custom_eval)
        self.assertIs(custom_agent.evaluator, custom_eval)

        # Both agents select valid actions
        act_def = default_agent.select_action(self.engine)
        act_custom = custom_agent.select_action(self.engine)
        legal_actions = self.engine.get_legal_actions()
        self.assertIn(act_def, legal_actions)
        self.assertIn(act_custom, legal_actions)


if __name__ == '__main__':
    unittest.main()
