#!/usr/bin/env python3
"""Checkpoint Ladder Evaluation across Training Progress (P2.6 / X-12).

Evaluates a sequence of checkpoints from a training run against fixed anchors
(Greedy baseline, Random) and exact solver optimal-action agreement.
Generates progress curves tracking self-play learning vs step count.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Dict, List

from evaluation.ablation import agreement
from evaluation.match_runner import MatchConfig, run_match
from evaluation.retrograde import solve_config


def evaluate_checkpoint_ladder(
    checkpoint_dir: str,
    board_size: int = 3,
    rules: dict | None = None,
    num_matches: int = 20,
    max_solver_states: int = 150,
) -> Dict[str, Any]:
    """Evaluates all intermediate and final checkpoints in checkpoint_dir."""
    ckpt_path = Path(checkpoint_dir)
    if rules is None:
        if board_size == 3:
            rules = {'A': True, 'B': True, 'C': True, 'K': 1, 'M': 1}
        elif board_size == 4:
            rules = {'A': True, 'B': True, 'C': True, 'K': 1, 'M': 0}
        else:
            rules = {'A': True, 'B': True, 'C': True, 'K': 3, 'M': 10}

    # Discover and sort checkpoints by step count
    pt_files = list(ckpt_path.glob("step_*.pt"))
    final_file = ckpt_path / "final.pt"
    if final_file.exists() and final_file not in pt_files:
        pt_files.append(final_file)

    def get_step(p: Path) -> int:
        match = re.search(r"step_(\d+)", p.name)
        if match:
            return int(match.group(1))
        return 999_999_999  # final

    sorted_ckpts = sorted(pt_files, key=get_step)
    if not sorted_ckpts:
        raise FileNotFoundError(f"No checkpoint .pt files found in {checkpoint_dir}")

    print(f"Discovered {len(sorted_ckpts)} checkpoints for ladder evaluation in {checkpoint_dir}")

    # Load or solve retrograde solution if small board
    solution = None
    if board_size in (3, 4):
        try:
            import pickle
            cache_file = Path("experiments/cache") / f"sol_N{board_size}_A1B1C1_K{rules.get('K', 1)}_M{rules.get('M', 0)}.pkl"
            if cache_file.exists():
                print(f"Loading cached solve from {cache_file}...")
                with open(cache_file, 'rb') as f:
                    solution = pickle.load(f)
            else:
                print(f"Solving ground truth for N={board_size}...")
                solution = solve_config(board_size=board_size, rules=rules)
                if solution.exact:
                    cache_file.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        with open(cache_file, 'wb') as f:
                            pickle.dump(solution, f)
                    except Exception:
                        pass
            if not solution.exact:
                solution = None
        except Exception as e:
            print(f"Warning: could not solve configuration: {e}")
            solution = None

    ladder_records = []
    t0 = time.time()

    for idx, ckpt in enumerate(sorted_ckpts):
        step_val = get_step(ckpt)
        step_label = str(step_val) if step_val != 999_999_999 else "final"
        dqn_spec = f"dqn:path={ckpt}"
        print(f"[{idx + 1}/{len(sorted_ckpts)}] Evaluating step {step_label} ({ckpt.name})...")

        # 1. Matches against Greedy
        greedy_wins = 0
        for m in range(num_matches):
            p0 = dqn_spec if m % 2 == 0 else "greedy"
            p1 = "greedy" if m % 2 == 0 else dqn_spec
            cfg = MatchConfig(board_size=board_size, rules=rules, p0=p0, p1=p1, max_plies=150, opening_seed=m * 101)
            res = run_match(cfg)
            winner = res.get('outcome', {}).get('winner')
            if winner == (0 if m % 2 == 0 else 1):
                greedy_wins += 1

        win_rate_greedy = round(greedy_wins / num_matches, 4)

        # 2. Matches against Random
        random_wins = 0
        for m in range(num_matches):
            p0 = dqn_spec if m % 2 == 0 else "random"
            p1 = "random" if m % 2 == 0 else dqn_spec
            cfg = MatchConfig(board_size=board_size, rules=rules, p0=p0, p1=p1, max_plies=150, opening_seed=m * 101)
            res = run_match(cfg)
            winner = res.get('outcome', {}).get('winner')
            if winner == (0 if m % 2 == 0 else 1):
                random_wins += 1

        win_rate_random = round(random_wins / num_matches, 4)

        # 3. Solver agreement (if available)
        val_opt_rate = None
        rank_opt_rate = None
        if solution is not None:
            try:
                ag_res = agreement(solution, dqn_spec, max_states=max_solver_states, seed=42)
                rates_all = ag_res.get('rates', {}).get('all', {})
                val_opt_rate = round(rates_all.get('value_optimal_rate', 0.0), 4)
                rank_opt_rate = round(rates_all.get('rank_optimal_rate', 0.0), 4)
            except Exception as e:
                print(f"  Solver agreement evaluation failed: {e}")

        rec = {
            'checkpoint': str(ckpt),
            'filename': ckpt.name,
            'step': step_label,
            'win_rate_vs_greedy': win_rate_greedy,
            'win_rate_vs_random': win_rate_random,
            'value_optimal_agreement': val_opt_rate,
            'rank_optimal_agreement': rank_opt_rate,
        }
        ladder_records.append(rec)
        print(f"   -> Win vs Greedy: {win_rate_greedy * 100:.1f}%, Win vs Random: {win_rate_random * 100:.1f}%, Value-Opt: {val_opt_rate}")

    return {
        'checkpoint_dir': checkpoint_dir,
        'board_size': board_size,
        'rules': rules,
        'total_checkpoints': len(ladder_records),
        'elapsed_seconds': round(time.time() - t0, 2),
        'ladder': ladder_records,
        'timestamp': time.time(),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoint progression ladder (X-12).")
    parser.add_argument("--dir", type=str, required=True, help="Path to checkpoint directory")
    parser.add_argument("--board-size", type=int, default=3, help="Board size (default 3)")
    parser.add_argument("--matches", type=int, default=20, help="Matches per anchor (default 20)")
    parser.add_argument("--output", type=str, default="experiments/results/dqn/ladder_x12.json", help="Output path")
    args = parser.parse_args()

    res = evaluate_checkpoint_ladder(
        checkpoint_dir=args.dir,
        board_size=args.board_size,
        num_matches=args.matches,
    )

    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2)

    print(f"Ladder results written to {out_file}")


if __name__ == "__main__":
    main()
