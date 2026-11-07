#!/usr/bin/env python3
"""DQN Validation Script against Ground Truth Exact Solvers (P2.2 / Gate X-16).

Compares max Q(s, a) against Solution.discounted_value(s, gamma) on complete 3x3 cells:
- N3_A0B0C0_K1_M0
- N3_A1B1C1_K1_M1

Pass Criterion:
- Sign agreement on decided non-terminal nodes (WIN/LOSS) must be above 90% (Gate X-16).
- Reports Mean Absolute Error (MAE) and optimal action agreement rates.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Dict, List
import numpy as np
import torch

from agents.dqn.agent import load_checkpoint, DQNAgent
from agents.dqn.encoding import encode_observation
from evaluation.match_runner import MatchConfig
from evaluation.metrics import wilson
from evaluation.retrograde import solve_config
from evaluation.state_graph import NONTERMINAL, GraphCaps


def evaluate_cell_x16(
    solution,
    agent: DQNAgent,
    gamma: float = 0.99,
    max_states: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Evaluates sign agreement and MAE on a solved retrograde cell."""
    g = solution.graph
    nonterminal_nodes = [u for u in range(len(g.keys)) if g.terminal[u] is NONTERMINAL]
    rng = random.Random(seed)

    if len(nonterminal_nodes) > max_states:
        sampled_nodes = sorted(rng.sample(nonterminal_nodes, max_states))
    else:
        sampled_nodes = nonterminal_nodes

    decided_count = 0
    sign_agreements = 0
    abs_errors = []
    status_errors = {'WIN': [], 'LOSS': [], 'DRAW': []}
    value_optimal_matches = 0
    rank_optimal_matches = 0

    device = agent.device
    codec = agent.codec
    model = agent.model
    model.eval()

    for u in sampled_nodes:
        engine = g.restore(g.keys[u])
        mover = g.to_move[u]
        status = solution.status(u)
        v_star = solution.discounted_value(u, gamma)

        # Get legal actions and mask
        legal_mask = codec.legal_mask(engine, mover=mover)
        legal_indices = [i for i, ok in enumerate(legal_mask) if ok]

        # Compute Q-values
        obs = encode_observation(engine, mover=mover)
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            q_vals = model(x).squeeze(0).cpu().numpy()

        # Legal max Q
        legal_q = q_vals[legal_indices]
        v_dqn = float(np.max(legal_q))

        # Best action selected by DQN
        best_id = legal_indices[int(np.argmax(legal_q))]
        best_action = codec.decode(best_id, engine, mover=mover)

        # Check action optimality
        val_opt_set = solution.value_optimal_actions(u)
        rank_opt_set = solution.rank_optimal_actions(u)
        if best_action in val_opt_set:
            value_optimal_matches += 1
        if best_action in rank_opt_set:
            rank_optimal_matches += 1

        err = abs(v_dqn - v_star)
        abs_errors.append(err)
        if status in status_errors:
            status_errors[status].append(err)

        # Sign agreement check on decided nodes (WIN or LOSS)
        if status in ('WIN', 'LOSS'):
            decided_count += 1
            # For WIN: v_star > 0; for LOSS: v_star < 0
            if (v_star > 0 and v_dqn > 0) or (v_star < 0 and v_dqn < 0):
                sign_agreements += 1

    sign_agreement_rate = (sign_agreements / decided_count) if decided_count > 0 else float('nan')
    ci_low, ci_high = wilson(sign_agreements, decided_count) if decided_count > 0 else (None, None)

    return {
        'total_evaluated': len(sampled_nodes),
        'population_nonterminal': len(nonterminal_nodes),
        'decided_states': decided_count,
        'sign_agreement_count': sign_agreements,
        'sign_agreement_rate': round(sign_agreement_rate, 4),
        'sign_agreement_ci95': [round(ci_low, 4), round(ci_high, 4)] if ci_low is not None else None,
        'mae_all': round(float(np.mean(abs_errors)), 4) if abs_errors else None,
        'mae_win': round(float(np.mean(status_errors['WIN'])), 4) if status_errors['WIN'] else None,
        'mae_loss': round(float(np.mean(status_errors['LOSS'])), 4) if status_errors['LOSS'] else None,
        'mae_draw': round(float(np.mean(status_errors['DRAW'])), 4) if status_errors['DRAW'] else None,
        'value_optimal_agreement': round(value_optimal_matches / len(sampled_nodes), 4),
        'rank_optimal_agreement': round(rank_optimal_matches / len(sampled_nodes), 4),
    }


def validate_checkpoint(
    checkpoint_path: str,
    gamma: float = 0.99,
    max_states_m1: int = 1000,
    seed: int = 42,
    cache_dir: str = "experiments/cache",
) -> Dict[str, Any]:
    """Runs Validation Gate X-16 for a given checkpoint."""
    import pickle
    print(f"Loading checkpoint: {checkpoint_path}")
    # Load agent for seat 0
    agent = load_checkpoint(checkpoint_path, player_id=0, expected_board_size=3)

    cells = [
        ('N3_A0B0C0_K1_M0', {'A': False, 'B': False, 'C': False, 'K': 1, 'M': 0}, 1000),
        ('N3_A1B1C1_K1_M1', {'A': True, 'B': True, 'C': True, 'K': 1, 'M': 1}, max_states_m1),
    ]

    cell_results = {}
    total_decided = 0
    total_agreements = 0

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    for label, rules, max_states in cells:
        pkl_file = cache_path / f"sol_{label}.pkl"
        if pkl_file.exists():
            print(f"Loading cached solve for {label} from {pkl_file}...")
            t0 = time.time()
            with open(pkl_file, 'rb') as f:
                sol = pickle.load(f)
            solve_sec = time.time() - t0
            print(f"  Loaded {len(sol.graph.keys)} states in {solve_sec:.2f}s")
        else:
            print(f"Solving / analyzing {label}...")
            t0 = time.time()
            sol = solve_config(board_size=3, rules=rules)
            solve_sec = time.time() - t0
            print(f"  Solved {len(sol.graph.keys)} states in {solve_sec:.2f}s")
            try:
                with open(pkl_file, 'wb') as f:
                    pickle.dump(sol, f)
            except Exception as e:
                print(f"  Warning: could not cache solve: {e}")

        eval_res = evaluate_cell_x16(sol, agent, gamma=gamma, max_states=max_states, seed=seed)
        eval_res['solve_seconds'] = round(solve_sec, 2)
        cell_results[label] = eval_res

        total_decided += eval_res['decided_states']
        total_agreements += eval_res['sign_agreement_count']

        print(
            f"  {label} -> Sign Agreement: {eval_res['sign_agreement_rate'] * 100:.1f}% "
            f"({eval_res['sign_agreement_count']}/{eval_res['decided_states']}), "
            f"MAE: {eval_res['mae_all']}, "
            f"Value-Opt Agree: {eval_res['value_optimal_agreement'] * 100:.1f}%"
        )

    overall_rate = (total_agreements / total_decided) if total_decided > 0 else 0.0
    gate_passed = (overall_rate >= 0.90)

    print("============================================================")
    print(f"GATE X-16 RESULT: {'PASSED' if gate_passed else 'FAILED'}")
    print(f"Overall Sign Agreement on Decided States: {overall_rate * 100:.2f}% (Threshold: 90.0%)")
    print("============================================================")

    return {
        'checkpoint': checkpoint_path,
        'gamma': gamma,
        'overall_sign_agreement': round(overall_rate, 4),
        'gate_threshold': 0.90,
        'gate_passed': gate_passed,
        'cells': cell_results,
        'timestamp': time.time(),
    }


def main():
    parser = argparse.ArgumentParser(description="Validate DQN checkpoint against ground truth (Gate X-16).")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt or directory")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor gamma (default 0.99)")
    parser.add_argument("--output", type=str, default="experiments/results/dqn/gate_x16.json", help="Output JSON path")
    parser.add_argument("--max-states", type=int, default=1000, help="Max states to sample from large cells")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed")
    args = parser.parse_args()

    results = validate_checkpoint(
        checkpoint_path=args.checkpoint,
        gamma=args.gamma,
        max_states_m1=args.max_states,
        seed=args.seed,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f"Results written to {out_path}")
    if not results['gate_passed']:
        exit(1)


if __name__ == "__main__":
    main()
