"""
Cross-check evaluation/retrograde.py (canonical keys) against agents/exact_solver.py
(raw keys) on every A/B/C cell of one (N, K, M) setting, and spot-check the canonical
key: all raw keys merged into one canonical node must have equal value and rank.

  python scripts/crosscheck_solvers.py -N 3 -K 1 -M 1 --out experiments/results/final_eval/crosscheck_N3_K1_M1.json

Note: ExactSolver is an adapter over the same StateGraph/retrograde code with
canonical=False, so this checks canonical merging, not an independent algorithm. The
independent reference (DFS + value iteration) lives in tests/test_solver_final_rules.py.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.exact_solver import ExactSolver  # noqa: E402
from evaluation.ablation import factorial_rules  # noqa: E402
from evaluation.match_runner import MatchConfig, run_metadata  # noqa: E402
from evaluation.retrograde import solve_config, STATUS  # noqa: E402


def check_cell(n, rules):
    t0 = time.perf_counter()
    can = solve_config(n, rules)
    old = ExactSolver(board_size=n, rules=rules).solve()
    groups, sizes, status_bad, rank_bad = {}, {}, 0, 0
    for k, sol in old.solutions.items():
        u = can.graph.node_of(k)
        sizes[u] = sizes.get(u, 0) + 1
        if u is None or sol['status'] != STATUS[can.value[u]]:
            status_bad += 1
            continue
        if sol['status'] != 'DRAW' and sol['distance'] != can.rank[u]:
            rank_bad += 1
        groups.setdefault(u, set()).add((sol['status'], sol['distance']))
    return {
        'config_label': MatchConfig(board_size=n, rules=rules).label(),
        'both_complete': can.exact and old._graph.complete,
        'raw_states': old.state_count,
        'canonical_states': len(can.graph.keys),
        'canonical_nodes_covered': len(groups),
        'canonical_nodes_with_merged_raw_keys': sum(1 for c in sizes.values() if c > 1),
        'raw_keys_in_merged_nodes': sum(c for c in sizes.values() if c > 1),
        'merged_groups_with_unequal_value_or_rank': sum(1 for v in groups.values() if len(v) > 1),
        'status_mismatches': status_bad,
        'rank_mismatches': rank_bad,
        'root_status': can.status(0),
        'root_rank': can.rank[0],
        'seconds': round(time.perf_counter() - t0, 2),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-N', type=int, default=3)
    ap.add_argument('-K', type=int, default=1)
    ap.add_argument('-M', type=int, default=0)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    meta = run_metadata()
    rows = []
    for rules in factorial_rules(args.K, args.M):
        row = check_cell(args.N, rules)
        rows.append(row)
        print(row, flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as fh:
        json.dump({'source': meta['source'], 'run': meta, 'config': vars(args), 'cells': rows}, fh, indent=2)
    print(f"wrote {args.out}")


if __name__ == '__main__':
    main()
