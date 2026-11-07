"""Feature Study and Logistic Regression AUC Benchmark (P6.2 / Gate X-5).

Evaluates the predictive fidelity of:
1. Model 1: H alone (f1 = d_opp - d_self)
2. Model 2: All engineered features (f1..f6: distance, tempo, wall count, abilities, replaceable wall)

Fits logistic regression models using pure NumPy on complete retrograde-solved states
(WIN vs LOSS positions) and computes AUC via the Mann-Whitney-Wilcoxon formulation.

Saves report to experiments/results/features/x5_feature_study.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys
import time
from typing import Any, Dict, List, Tuple
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.features import extract_features, FEATURE_NAMES
from evaluation.retrograde import solve_config, WIN, LOSS
from wallz_engine import WallzEngine


def compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Computes Area Under the ROC Curve using Mann-Whitney U rank statistic."""
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    n_pos = len(pos)
    n_neg = len(neg)
    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Sort scores to compute rank sum efficiently
    all_scores = np.concatenate([pos, neg])
    all_labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
    order = np.argsort(all_scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(all_scores) + 1)

    # Handle ties by averaging ranks
    unique_scores, inverse, counts = np.unique(all_scores, return_inverse=True, return_counts=True)
    if len(unique_scores) < len(all_scores):
        tie_ranks = np.bincount(inverse, weights=ranks) / counts
        ranks = tie_ranks[inverse]

    pos_rank_sum = np.sum(ranks[:n_pos])
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def fit_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.05,
    epochs: int = 1000,
    reg: float = 1e-4,
) -> Tuple[np.ndarray, float]:
    """Fits L2-regularized logistic regression via gradient descent."""
    n_samples, n_features = X.shape
    weights = np.zeros(n_features, dtype=np.float64)
    bias = 0.0

    # Feature standardization
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1.0
    X_norm = (X - mean) / std

    for _ in range(epochs):
        logits = np.dot(X_norm, weights) + bias
        # Stable sigmoid
        preds = 1.0 / (1.0 + np.exp(-np.clip(logits, -25.0, 25.0)))
        error = preds - y
        grad_w = (np.dot(X_norm.T, error) / n_samples) + reg * weights
        grad_b = np.mean(error)
        weights -= lr * grad_w
        bias -= lr * grad_b

    # Convert back to unstandardized weights
    raw_weights = weights / std
    raw_bias = bias - np.sum(weights * mean / std)
    return raw_weights, raw_bias


def run_feature_study(
    board_size: int = 3,
    max_states: int = 5000,
    seed: int = 42,
    output_path: str = "experiments/results/features/x5_feature_study.json",
) -> Dict[str, Any]:
    print("============================================================")
    print(f"Starting X-5 Feature Study on N={board_size} (max_states={max_states})")
    print("============================================================")

    # Load or solve retrograde solution
    rules = {'A': True, 'B': True, 'C': True, 'K': 1, 'M': 1 if board_size == 3 else 0}
    cache_path = Path("experiments/cache") / f"sol_N{board_size}_A1B1C1_K1_M{rules['M']}.pkl"

    if cache_path.exists():
        print(f"Loading cached solve from {cache_path}...")
        with open(cache_path, 'rb') as f:
            sol = pickle.load(f)
    else:
        print("Solving retrograde graph...")
        sol = solve_config(board_size, rules)

    # Collect decided states (WIN or LOSS)
    decided_nodes = [
        node for node, val in enumerate(sol.value)
        if val in (WIN, LOSS)
    ]
    print(f"Found {len(decided_nodes)} decided positions in retrograde solution.")

    # Sample states if dataset is large
    rng = np.random.RandomState(seed)
    if len(decided_nodes) > max_states:
        sampled_nodes = rng.choice(decided_nodes, size=max_states, replace=False)
    else:
        sampled_nodes = decided_nodes

    print(f"Extracting features for {len(sampled_nodes)} sampled states...")
    t0 = time.time()
    dummy_engine = WallzEngine(board_size=board_size, rules=rules)

    X_list = []
    y_list = []

    for node in sampled_nodes:
        engine = dummy_engine.restore_state_key(sol.graph.keys[node])
        curr = engine.current_player
        feats = extract_features(engine, curr)
        label = 1.0 if sol.value[node] == WIN else 0.0
        X_list.append(feats)
        y_list.append(label)

    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.float64)
    extract_time = time.time() - t0
    print(f"Feature extraction completed in {extract_time:.2f}s")

    # Model 1: H alone (feature 0)
    X_h = X[:, [0]]
    w_h, b_h = fit_logistic_regression(X_h, y)
    scores_h = 1.0 / (1.0 + np.exp(-np.clip(np.dot(X_h, w_h) + b_h, -25.0, 25.0)))
    auc_h = compute_auc(y, scores_h)

    # Model 2: All features (f1..f6)
    w_all, b_all = fit_logistic_regression(X, y)
    scores_all = 1.0 / (1.0 + np.exp(-np.clip(np.dot(X, w_all) + b_all, -25.0, 25.0)))
    auc_all = compute_auc(y, scores_all)

    # Feature correlation with winning
    correlations = {}
    for i, name in enumerate(FEATURE_NAMES):
        std_i = np.std(X[:, i])
        if std_i == 0 or np.std(y) == 0:
            correlations[name] = 0.0
        else:
            correlations[name] = round(float(np.corrcoef(X[:, i], y)[0, 1]), 4)

    report = {
        'board_size': board_size,
        'rules': rules,
        'num_samples': len(sampled_nodes),
        'positive_class_ratio': round(float(np.mean(y)), 4),
        'model_h_alone': {
            'features': ['f1_distance_advantage'],
            'weights': [round(float(w_h[0]), 4)],
            'bias': round(float(b_h), 4),
            'auc': round(auc_h, 4),
        },
        'model_all_features': {
            'features': FEATURE_NAMES,
            'weights': [round(float(w), 4) for w in w_all],
            'bias': round(float(b_all), 4),
            'auc': round(auc_all, 4),
        },
        'feature_correlations_with_win': correlations,
        'auc_improvement': round(auc_all - auc_h, 4),
        'gate_x5_passed': bool(auc_h > 0.70 and auc_all > 0.70),
    }

    print("\n============================================================")
    print("GATE X-5 FEATURE STUDY RESULTS:")
    print(f"  Model 1 (H alone) AUC:     {auc_h * 100:.2f}%")
    print(f"  Model 2 (All Features) AUC: {auc_all * 100:.2f}% (delta = +{(auc_all - auc_h) * 100:.2f}%)")
    print("  Feature Correlations:")
    for k, v in correlations.items():
        print(f"    {k:28s}: {v:+.4f}")
    print(f"  Gate X-5 Passed:           {report['gate_x5_passed']}")
    print("============================================================")

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print(f"Saved feature study report to {out_file}")
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run X-5 Feature Study.")
    parser.add_argument("--board-size", type=int, default=3)
    parser.add_argument("--max-states", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="experiments/results/features/x5_feature_study.json")
    args = parser.parse_args()

    run_feature_study(
        board_size=args.board_size,
        max_states=args.max_states,
        seed=args.seed,
        output_path=args.output,
    )
