#!/usr/bin/env python3
"""Run 3x3 Multi-Seed Stability Study (P2.3 / X-11).

Trains and evaluates DQN across 5 seeds (42, 43, 44, 45, 46) on 3x3 board.
For each seed:
1. Runs training if final.pt doesn't exist.
2. Validates against ground-truth retrograde solver (Gate X-16 sign agreement & MAE).
3. Evaluates tournament performance vs Greedy and Random (100 matches each).
4. Computes IQM and 95% bootstrap confidence intervals across the seeds.
5. Saves results to experiments/results/dqn/multi_seed_3x3_x11.json.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List

import numpy as np

from evaluation.match_runner import MatchConfig, run_match
from evaluation.retrograde import solve_config
from evaluation.rl_metrics import bootstrap_ci, interquartile_mean, iqm_with_ci
from evaluation.metrics import wilson


def evaluate_seed_vs_baselines(
    dqn_checkpoint: str,
    board_size: int = 3,
    rules: dict | None = None,
    num_matches: int = 100,
) -> Dict[str, Any]:
    """Evaluates a DQN checkpoint against greedy and random over paired matches."""
    if rules is None:
        rules = {'A': True, 'B': True, 'C': True, 'K': 1, 'M': 1}

    dqn_spec = f"dqn:path={dqn_checkpoint}"
    results = {}

    for opp in ['greedy', 'random']:
        wins = 0
        draws = 0
        losses = 0
        plies_list = []

        for i in range(num_matches // 2):
            for dqn_seat in (0, 1):
                p0 = dqn_spec if dqn_seat == 0 else opp
                p1 = opp if dqn_seat == 0 else dqn_spec

                cfg = MatchConfig(
                    board_size=board_size,
                    rules=rules,
                    p0=p0,
                    p1=p1,
                    max_plies=100,
                    opening_seed=i * 1001 + dqn_seat,
                )

                res = run_match(cfg)
                plies_list.append(res['length_plies'])
                outcome = res.get('outcome', {})
                cat = outcome.get('category', '')
                winner = outcome.get('winner')

                if cat.startswith('forfeit'):
                    if winner == dqn_seat:
                        wins += 1
                    else:
                        losses += 1
                elif cat == 'goal':
                    if winner == dqn_seat:
                        wins += 1
                    else:
                        losses += 1
                elif 'draw' in cat or cat == 'truncated':
                    draws += 1
                else:
                    losses += 1

        total = wins + draws + losses
        win_rate = wins / total if total > 0 else 0.0
        results[opp] = {
            'total_matches': total,
            'wins': wins,
            'draws': draws,
            'losses': losses,
            'win_rate': round(win_rate, 4),
            'ci95': [round(x, 4) for x in wilson(wins, total)],
            'avg_plies': round(float(np.mean(plies_list)), 2) if plies_list else 0.0,
        }

    return results


def run_multi_seed_study(
    seeds: List[int] = [42, 43, 44, 45, 46],
    steps: int = 50000,
    output_path: str = "experiments/results/dqn/multi_seed_3x3_x11.json",
):
    print("============================================================")
    print(f"Starting X-11 Multi-Seed Stability Study on 3x3")
    print(f"Seeds: {seeds} | Steps per seed: {steps}")
    print("============================================================")

    # Pre-load or solve retrograde solution for N3_A1B1C1_K1_M1
    cache_path = Path("experiments/cache/sol_N3_A1B1C1_K1_M1.pkl")
    rules_3x3 = {'A': True, 'B': True, 'C': True, 'K': 1, 'M': 1}
    import pickle
    if cache_path.exists():
        print(f"Loading cached 3x3 retrograde solution from {cache_path}...")
        with open(cache_path, 'rb') as f:
            sol = pickle.load(f)
    else:
        print("Solving 3x3 retrograde graph...")
        sol = solve_config(3, rules_3x3)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(sol, f)

    # Train any missing seeds in parallel
    missing_seeds = [s for s in seeds if not (Path(f"checkpoints/dqn_3x3_s{s}") / "final.pt").exists()]
    if missing_seeds:
        print(f"Training missing seeds {missing_seeds} in parallel...")
        import concurrent.futures

        def train_one_seed(s):
            print(f"Starting training for seed {s}...")
            ckpt_dir = Path(f"checkpoints/dqn_3x3_s{s}")
            cmd = [
                sys.executable,
                "scripts/train_dqn.py",
                "--board-size", "3",
                "--steps", str(steps),
                "--seed", str(s),
                "--eval-freq", "10000",
                "--output-dir", str(ckpt_dir),
            ]
            subprocess.run(cmd, check=True)
            print(f"Finished training for seed {s}.")
            return s

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(missing_seeds))) as executor:
            list(executor.map(train_one_seed, missing_seeds))

    seed_results = []

    for seed in seeds:
        checkpoint_dir = Path(f"checkpoints/dqn_3x3_s{seed}")
        final_pt = checkpoint_dir / "final.pt"

        # Validate Gate X-16 sign agreement
        print(f"Validating Gate X-16 for Seed {seed}...")
        from scripts.validate_dqn import validate_cell
        cell_res = validate_cell(final_pt, 3, rules_3x3, sol=sol)

        sign_agreement = cell_res['sign_agreement']
        mae = cell_res['mae']
        print(f"  Seed {seed} Sign Agreement: {sign_agreement * 100:.2f}% | MAE: {mae:.4f}")

        # Evaluate vs Baselines
        print(f"Evaluating vs Baselines for Seed {seed} (100 matches)...")
        bench_res = evaluate_seed_vs_baselines(str(final_pt), board_size=3, rules=rules_3x3, num_matches=100)
        greedy_wr = bench_res['greedy']['win_rate']
        random_wr = bench_res['random']['win_rate']
        print(f"  Seed {seed} Win Rate vs Greedy: {greedy_wr * 100:.1f}% | vs Random: {random_wr * 100:.1f}%")

        seed_results.append({
            'seed': seed,
            'checkpoint': str(final_pt),
            'sign_agreement': sign_agreement,
            'mae': mae,
            'win_rate_vs_greedy': greedy_wr,
            'win_rate_vs_random': random_wr,
            'details_vs_baselines': bench_res,
            'x16_cell_metrics': cell_res,
        })

    # Aggregate metrics across seeds with IQM and bootstrap 95% CIs
    sign_agreements = [r['sign_agreement'] for r in seed_results]
    maes = [r['mae'] for r in seed_results]
    greedy_wrs = [r['win_rate_vs_greedy'] for r in seed_results]
    random_wrs = [r['win_rate_vs_random'] for r in seed_results]

    sign_iqm, sign_ci = iqm_with_ci(sign_agreements)
    mae_iqm, mae_ci = iqm_with_ci(maes)
    greedy_iqm, greedy_ci = iqm_with_ci(greedy_wrs)
    random_iqm, random_ci = iqm_with_ci(random_wrs)

    aggregate_summary = {
        'num_seeds': len(seeds),
        'seeds': seeds,
        'sign_agreement': {
            'mean': round(float(np.mean(sign_agreements)), 4),
            'std': round(float(np.std(sign_agreements)), 4),
            'iqm': round(float(sign_iqm), 4),
            'ci95': [round(float(sign_ci[0]), 4), round(float(sign_ci[1]), 4)],
        },
        'mae': {
            'mean': round(float(np.mean(maes)), 4),
            'std': round(float(np.std(maes)), 4),
            'iqm': round(float(mae_iqm), 4),
            'ci95': [round(float(mae_ci[0]), 4), round(float(mae_ci[1]), 4)],
        },
        'win_rate_vs_greedy': {
            'mean': round(float(np.mean(greedy_wrs)), 4),
            'std': round(float(np.std(greedy_wrs)), 4),
            'iqm': round(float(greedy_iqm), 4),
            'ci95': [round(float(greedy_ci[0]), 4), round(float(greedy_ci[1]), 4)],
        },
        'win_rate_vs_random': {
            'mean': round(float(np.mean(random_wrs)), 4),
            'std': round(float(np.std(random_wrs)), 4),
            'iqm': round(float(random_iqm), 4),
            'ci95': [round(float(random_ci[0]), 4), round(float(random_ci[1]), 4)],
        },
        'gate_x11_passed': bool(float(sign_ci[0]) >= 0.90 or float(sign_iqm) >= 0.90),
    }

    print("\n============================================================")
    print("X-11 MULTI-SEED STABILITY SUMMARY (5 SEEDS):")
    print(f"Sign Agreement IQM:    {sign_iqm * 100:.2f}% (95% CI: [{sign_ci[0] * 100:.2f}%, {sign_ci[1] * 100:.2f}%])")
    print(f"Value MAE IQM:         {mae_iqm:.4f} (95% CI: [{mae_ci[0]:.4f}, {mae_ci[1]:.4f}])")
    print(f"Win Rate vs Greedy IQM: {greedy_iqm * 100:.2f}% (95% CI: [{greedy_ci[0] * 100:.2f}%, {greedy_ci[1] * 100:.2f}%])")
    print(f"Win Rate vs Random IQM: {random_iqm * 100:.2f}% (95% CI: [{random_ci[0] * 100:.2f}%, {random_ci[1] * 100:.2f}%])")
    print(f"Gate X-11 Passed:       {aggregate_summary['gate_x11_passed']}")
    print("============================================================")

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    report = {
        'aggregate': aggregate_summary,
        'seeds': seed_results,
    }
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print(f"Report saved to {out_file}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 3x3 multi-seed stability study.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--output", type=str, default="experiments/results/dqn/multi_seed_3x3_x11.json")
    args = parser.parse_args()

    run_multi_seed_study(seeds=args.seeds, steps=args.steps, output_path=args.output)
