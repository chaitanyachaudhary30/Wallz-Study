#!/usr/bin/env python3
"""Evaluate Ground Truth Optimal-Action Agreement on 4x4 (P2.4 / X-6).

Computes agreement rate between trained DQN agent and exact retrograde solution
for N4_A1B1C1_K1_M0 using evaluation.ablation.agreement.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import time

from evaluation.ablation import agreement
from evaluation.retrograde import solve_config


def main():
    parser = argparse.ArgumentParser(description="Evaluate 4x4 DQN optimal action agreement (P2.4).")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to 4x4 checkpoint .pt")
    parser.add_argument("--states", type=int, default=300, help="Number of states to sample (default 300)")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed (default 42)")
    parser.add_argument("--output", type=str, default="experiments/results/dqn/agreement_N4.json", help="Output path")
    args = parser.parse_args()

    cache_file = Path("experiments/cache/sol_N4_A1B1C1_K1_M0.pkl")
    rules = {'A': True, 'B': True, 'C': True, 'K': 1, 'M': 0}

    if cache_file.exists():
        print(f"Loading cached 4x4 retrograde solution from {cache_file}...")
        with open(cache_file, 'rb') as f:
            sol = pickle.load(f)
    else:
        print("Solving 4x4 retrograde graph...")
        sol = solve_config(4, rules)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(sol, f)

    dqn_spec = f"dqn:path={args.checkpoint}"
    print(f"Evaluating agreement for {dqn_spec} on {args.states} sampled non-terminal states...")
    t0 = time.time()
    res = agreement(sol, dqn_spec, max_states=args.states, seed=args.seed)
    elapsed = time.time() - t0

    res['elapsed_seconds'] = round(elapsed, 2)
    rates_all = res.get('rates', {}).get('all', {})
    val_opt = rates_all.get('value_optimal_rate', 0.0)
    rank_opt = rates_all.get('rank_optimal_rate', 0.0)

    print("============================================================")
    print(f"4x4 GROUND TRUTH AGREEMENT (N4_A1B1C1_K1_M0):")
    print(f"Value-Optimal Action Agreement: {val_opt * 100:.2f}% (95% CI: {rates_all.get('value_optimal_ci95')})")
    print(f"Rank-Optimal Action Agreement:  {rank_opt * 100:.2f}%")
    print("============================================================")

    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2)

    print(f"Saved agreement report to {out_file}")


if __name__ == "__main__":
    main()
