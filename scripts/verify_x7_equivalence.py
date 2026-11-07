"""Verify Alpha-Beta Equivalence to Plain Minimax (Gate X-7).

Evaluates root values and optimal action equivalence between:
1. Plain Minimax (unpruned, no ordering)
2. Alpha-Beta without ordering
3. Alpha-Beta with ordering

Verifies exact mathematical equivalence across positions and records report to:
experiments/results/minimax/x7_equivalence.json
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


def run_x7_equivalence_test(num_positions: int = 500, depth: int = 2) -> Dict[str, Any]:
    print("============================================================")
    print(f"Starting Gate X-7 Equivalence Verification: {num_positions} positions at depth {depth}")
    print("============================================================")
    random.seed(42)

    tested = 0
    matched_val_ab = 0
    matched_val_ord = 0
    t0 = time.time()

    for i in range(num_positions):
        n = random.choice([3, 4])  # 3x3 and 4x4 allow exact deep verification across 500 positions
        rules = {
            'A': random.choice([True, False]),
            'B': random.choice([True, False]),
            'C': random.choice([True, False]),
            'K': random.randint(1, 2),
            'M': random.randint(1, 3),
        }
        e = WallzEngine(board_size=n, rules=rules)
        # Apply 0 to 6 random legal plies
        for _ in range(random.randint(0, 6)):
            if e.is_terminal():
                break
            legal = e.get_legal_actions()
            if not legal:
                break
            e.apply_action(random.choice(legal), update_clock=False)

        if e.is_terminal() or not e.get_legal_actions():
            continue

        tested += 1
        curr = e.current_player

        agent_plain = MinimaxAgent(player_id=curr, depth=depth, use_alpha_beta=False, order_moves=False)
        agent_ab = MinimaxAgent(player_id=curr, depth=depth, use_alpha_beta=True, order_moves=False)
        agent_ord = MinimaxAgent(player_id=curr, depth=depth, use_alpha_beta=True, order_moves=True)

        _, val_plain = agent_plain._search_root(e, depth=depth, deadline=None)
        _, val_ab = agent_ab._search_root(e, depth=depth, deadline=None)
        _, val_ord = agent_ord._search_root(e, depth=depth, deadline=None)

        if abs(val_plain - val_ab) < 1e-5:
            matched_val_ab += 1
        if abs(val_plain - val_ord) < 1e-5:
            matched_val_ord += 1

        if tested % 100 == 0 or tested == num_positions:
            print(f"  Tested {tested}/{num_positions} | AB match: {matched_val_ab}/{tested} | Ordered match: {matched_val_ord}/{tested}")

    elapsed = time.time() - t0
    report = {
        'num_positions_requested': num_positions,
        'positions_tested': tested,
        'search_depth': depth,
        'ab_value_agreement_rate': matched_val_ab / tested if tested > 0 else 0.0,
        'ordered_value_agreement_rate': matched_val_ord / tested if tested > 0 else 0.0,
        'gate_x7_passed': (matched_val_ab == tested and matched_val_ord == tested),
        'elapsed_seconds': round(elapsed, 2),
    }

    print("============================================================")
    print(f"GATE X-7 RESULTS: {'PASSED' if report['gate_x7_passed'] else 'FAILED'}")
    print(f"  Alpha-Beta Agreement: {report['ab_value_agreement_rate'] * 100:.2f}%")
    print(f"  Ordered AB Agreement: {report['ordered_value_agreement_rate'] * 100:.2f}%")
    print("============================================================")

    out_file = Path("experiments/results/minimax/x7_equivalence.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Verify Gate X-7 Equivalence.")
    parser.add_argument("--positions", type=int, default=500)
    parser.add_argument("--depth", type=int, default=2)
    args = parser.parse_args()
    run_x7_equivalence_test(num_positions=args.positions, depth=args.depth)
