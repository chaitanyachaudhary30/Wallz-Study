"""Benchmark Move Ordering Node Reductions (Gate X-8).

Measures node expansions and alpha-beta cutoff efficiency across configurations:
1. No move ordering (unordered alpha-beta)
2. Static heuristic ordering
3. Static ordering + History heuristic (d^2 update)

Saves report to experiments/results/minimax/x8_node_reduction.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.minimax import MinimaxAgent
from wallz_engine import WallzEngine


def benchmark_ordering(num_positions: int = 50, board_size: int = 5, depth: int = 2) -> Dict[str, Any]:
    print("============================================================")
    print(f"Starting Gate X-8 Benchmark: N={board_size}, depth={depth}, positions={num_positions}")
    print("============================================================")
    random.seed(42)

    positions = []
    while len(positions) < num_positions:
        e = WallzEngine(board_size=board_size)
        plies = random.randint(2, 6)
        for _ in range(plies):
            if e.is_terminal():
                break
            legal = e.get_legal_actions()
            if not legal:
                break
            e.apply_action(random.choice(legal), update_clock=False)
        if not e.is_terminal() and e.get_legal_actions():
            positions.append(e)

    nodes_unordered = 0
    nodes_static = 0
    nodes_history = 0

    cutoffs_unordered = 0
    cutoffs_static = 0
    cutoffs_history = 0

    t0 = time.time()

    # Pre-train history agent over a few positions to populate history table
    history_agent = MinimaxAgent(player_id=0, depth=depth, use_alpha_beta=True, order_moves=True)

    for idx, pos in enumerate(positions):
        curr = pos.current_player

        # 1. Unordered
        a_unord = MinimaxAgent(player_id=curr, depth=depth, use_alpha_beta=True, order_moves=False)
        a_unord.select_action(pos.clone())
        nodes_unordered += a_unord.nodes_expanded
        cutoffs_unordered += a_unord.cutoffs

        # 2. Static ordering
        a_static = MinimaxAgent(player_id=curr, depth=depth, use_alpha_beta=True, order_moves=True)
        a_static.select_action(pos.clone())
        nodes_static += a_static.nodes_expanded
        cutoffs_static += a_static.cutoffs

        # 3. Static + History
        history_agent.player_id = curr
        history_agent.select_action(pos.clone())
        nodes_history += history_agent.nodes_expanded
        cutoffs_history += history_agent.cutoffs

        if (idx + 1) % 10 == 0 or (idx + 1) == num_positions:
            print(f"  Processed {idx + 1}/{num_positions} positions...")

    elapsed = time.time() - t0

    reduction_static = (1.0 - nodes_static / nodes_unordered) * 100 if nodes_unordered > 0 else 0.0
    reduction_history = (1.0 - nodes_history / nodes_unordered) * 100 if nodes_unordered > 0 else 0.0

    report = {
        'board_size': board_size,
        'search_depth': depth,
        'num_positions': num_positions,
        'elapsed_seconds': round(elapsed, 2),
        'unordered': {
            'total_nodes': nodes_unordered,
            'mean_nodes': round(nodes_unordered / num_positions, 1),
            'cutoffs': cutoffs_unordered,
        },
        'static_ordering': {
            'total_nodes': nodes_static,
            'mean_nodes': round(nodes_static / num_positions, 1),
            'cutoffs': cutoffs_static,
            'node_reduction_pct': round(reduction_static, 2),
        },
        'static_plus_history': {
            'total_nodes': nodes_history,
            'mean_nodes': round(nodes_history / num_positions, 1),
            'cutoffs': cutoffs_history,
            'node_reduction_pct': round(reduction_history, 2),
        },
        'gate_x8_passed': bool(reduction_static > 0.0 and nodes_static < nodes_unordered),
    }

    print("\n============================================================")
    print("GATE X-8 BENCHMARK RESULTS:")
    print(f"  Unordered Mean Nodes:       {report['unordered']['mean_nodes']}")
    print(f"  Static Ordering Mean Nodes: {report['static_ordering']['mean_nodes']} ({reduction_static:.1f}% reduction)")
    print(f"  + History Heuristic Nodes:  {report['static_plus_history']['mean_nodes']} ({reduction_history:.1f}% reduction)")
    print(f"  Gate X-8 Passed:            {report['gate_x8_passed']}")
    print("============================================================")

    out_file = Path("experiments/results/minimax/x8_node_reduction.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmark X-8 move ordering node reduction.")
    parser.add_argument("--positions", type=int, default=50)
    parser.add_argument("--board-size", type=int, default=5)
    parser.add_argument("--depth", type=int, default=2)
    args = parser.parse_args()
    benchmark_ordering(num_positions=args.positions, board_size=args.board_size, depth=args.depth)
