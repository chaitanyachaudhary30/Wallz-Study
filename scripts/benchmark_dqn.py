#!/usr/bin/env python3
"""DQN Tournament Benchmark against Greedy, Random and Minimax (P2.7 / X-9).

Runs tournament matches across >=100 seeds with seat swapping (200 matches per pair).
Calculates win rates, draw rates, loss rates, and Wilson 95% confidence intervals.
Evaluates gate criterion: 5x5 win rate against Greedy > 50% with 95% CI lower bound > 50%.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List

from evaluation.match_runner import MatchConfig, run_match, verify_replay
from evaluation.metrics import wilson
from evaluation.rl_metrics import bootstrap_ci


def run_tournament(
    dqn_spec: str,
    opponents: List[str],
    board_size: int = 5,
    rules: dict | None = None,
    num_seeds: int = 100,
    max_plies: int = 200,
    move_time_limit: float | None = None,
) -> Dict[str, Any]:
    """Executes paired tournament across seeds with seat swaps."""
    if rules is None:
        rules = {'A': True, 'B': True, 'C': True, 'K': 3, 'M': 10}

    results = {}
    total_matches = num_seeds * 2 * len(opponents)
    completed_matches = 0
    t0 = time.time()

    print(f"Starting X-9 Tournament for {dqn_spec} on N={board_size}")
    print(f"Opponents: {opponents} | Seeds: {num_seeds} (x2 seatings = {num_seeds * 2} per opponent)")
    print("============================================================")

    for opp_spec in opponents:
        print(f"--- Running series: DQN vs {opp_spec} ---")
        wins = 0
        draws = 0
        truncated = 0
        losses = 0
        forfeits = 0
        plies_list = []

        for seed in range(num_seeds):
            # Two seatings per seed: (DQN as P0, Opp as P1) and (Opp as P0, DQN as P1)
            for dqn_seat in (0, 1):
                p0 = dqn_spec if dqn_seat == 0 else opp_spec
                p1 = opp_spec if dqn_seat == 0 else dqn_spec

                cfg = MatchConfig(
                    board_size=board_size,
                    rules=rules,
                    p0=p0,
                    p1=p1,
                    max_plies=max_plies,
                    opening_plies=2,
                    opening_seed=seed,
                    move_time_limit=move_time_limit,
                )

                res = run_match(cfg)
                completed_matches += 1
                plies_list.append(res['length_plies'])

                outcome = res.get('outcome', {})
                cat = outcome.get('category', '')
                winner = outcome.get('winner')

                if cat.startswith('forfeit'):
                    # Check who forfeited
                    if winner == dqn_seat:
                        wins += 1
                    else:
                        forfeits += 1
                        losses += 1
                elif cat == 'goal':
                    if winner == dqn_seat:
                        wins += 1
                    else:
                        losses += 1
                elif cat == 'truncated':
                    truncated += 1  # runner cap: own column, never a draw (team decision)
                elif cat.startswith('draw'):
                    draws += 1
                else:
                    losses += 1

            if (seed + 1) % 25 == 0 or (seed + 1) == num_seeds:
                n_played = (seed + 1) * 2
                print(
                    f"  Seed {seed + 1:3d}/{num_seeds} -> "
                    f"Win: {wins}/{n_played} ({wins / n_played * 100:.1f}%), "
                    f"Draw: {draws}/{n_played} ({draws / n_played * 100:.1f}%), "
                    f"Loss: {losses}/{n_played} ({losses / n_played * 100:.1f}%)"
                )

        n_total = num_seeds * 2
        w_low, w_high = wilson(wins, n_total)

        results[f"vs_{opp_spec}"] = {
            'opponent': opp_spec,
            'total_matches': n_total,
            'wins': wins,
            'draws': draws,
            'truncated': truncated,
            'losses': losses,
            'forfeits': forfeits,
            'win_rate': round(wins / n_total, 4),
            'draw_rate': round(draws / n_total, 4),
            'truncation_rate': round(truncated / n_total, 4),
            'loss_rate': round(losses / n_total, 4),
            'win_rate_wilson_ci95': [round(w_low, 4), round(w_high, 4)],
            'mean_plies': round(sum(plies_list) / len(plies_list), 1) if plies_list else 0.0,
        }

    elapsed = time.time() - t0
    print("============================================================")
    print(f"Tournament completed in {elapsed:.1f}s ({completed_matches} matches)")
    print("============================================================")

    # Gate check (defined for 5x5 in the plan; applied to whatever board is run): win rate vs greedy > 50% with Wilson lower bound > 50%
    vs_greedy = results.get('vs_greedy')
    gate_passed = False
    if vs_greedy is not None:
        ci_low = vs_greedy['win_rate_wilson_ci95'][0]
        gate_passed = (ci_low > 0.50)
        print(f"{board_size}x{board_size} Gate vs Greedy: Win Rate = {vs_greedy['win_rate'] * 100:.1f}%, 95% CI Lower Bound = {ci_low * 100:.1f}%")
        print(f"Gate Status: {'PASSED' if gate_passed else 'FAILED'} (Requires lower bound > 50.0%)")

    return {
        'dqn_spec': dqn_spec,
        'board_size': board_size,
        'rules': rules,
        'num_seeds': num_seeds,
        'elapsed_seconds': round(elapsed, 2),
        'results': results,
        'gate_vs_greedy_passed': gate_passed,
        'timestamp': time.time(),
    }


def main():
    parser = argparse.ArgumentParser(description="Run X-9 Tournament Benchmark for DQN agent.")
    parser.add_argument("--dqn-path", type=str, required=True, help="Path to DQN checkpoint file")
    parser.add_argument("--board-size", type=int, default=5, help="Board size (default 5)")
    parser.add_argument("--seeds", type=int, default=100, help="Number of seeds (default 100)")
    parser.add_argument("--opponents", nargs="+", default=["greedy", "random"], help="Opponents list")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    dqn_spec = f"dqn:path={args.dqn_path}"
    out_dir = Path("experiments/results/dqn")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = args.output if args.output else f"experiments/results/dqn/benchmark_N{args.board_size}.json"

    # Default rules based on board size
    if args.board_size == 3:
        rules = {'A': True, 'B': True, 'C': True, 'K': 1, 'M': 1}
    elif args.board_size == 4:
        rules = {'A': True, 'B': True, 'C': True, 'K': 1, 'M': 0}
    else:
        rules = {'A': True, 'B': True, 'C': True, 'K': 3, 'M': 10}

    res = run_tournament(
        dqn_spec=dqn_spec,
        opponents=args.opponents,
        board_size=args.board_size,
        rules=rules,
        num_seeds=args.seeds,
    )

    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2)

    print(f"Results saved to {out_file}")


if __name__ == "__main__":
    main()
