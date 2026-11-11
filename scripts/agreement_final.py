"""
Agent-vs-exact-solver agreement on the final rules, all agents on the SAME sampled states.

For every A/B/C cell of N = 3, K = 1, M in {0, 1}: solve the cell exactly, sample up to
``--max-states`` non-terminal positions once (seed ``--seed``), and score every agent on
that one list: does it pick a value-optimal action (keeps the exact value) and a
rank-optimal one (fastest win / slowest loss)? A DQN checkpoint is scored only in the cell
it was trained for (its rules must match); elsewhere it is reported as not applicable.
One cell's graph is kept in memory at a time.

  python scripts/agreement_final.py --dqn-dir "D:/SEM V/Wallz-Study-wt/bot2-dqn/checkpoints/final" \
      --out experiments/results/final_eval/agreement_N3_K1.json
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.ablation import agreement, factorial_rules, sample_nonterminal  # noqa: E402
from evaluation.match_runner import AgentUnavailable, MatchConfig, run_metadata  # noqa: E402
from evaluation.retrograde import solve_config  # noqa: E402

BASE_AGENTS = ['greedy', 'minimax:depth=2', 'minimax:depth=2,eval=hop']
DQN_RULES = {'A': True, 'B': True, 'C': True, 'K': 1, 'M': 1}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--Ms', type=int, nargs='+', default=[0, 1])
    ap.add_argument('--max-states', type=int, default=300)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--dqn-dir', default=None, help='holds 3x3_s<seed>/final.pt')
    ap.add_argument('--dqn-seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    ap.add_argument('--manifest-dir', default='checkpoints/final', help='committed manifests to check SHA-256')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    meta = run_metadata()

    dqn = {}
    if args.dqn_dir:
        for s in args.dqn_seeds:
            ck = Path(args.dqn_dir) / f'3x3_s{s}' / 'final.pt'
            with open(Path(args.manifest_dir) / f'3x3_s{s}' / 'manifest.json', encoding='utf-8') as fh:
                expected = json.load(fh)['sha256']
            actual = sha256(ck)
            if actual != expected:
                sys.exit(f"error: seed {s} SHA-256 {actual} != manifest {expected}")
            dqn[f'dqn_s{s}'] = {'spec': f'dqn:path={ck.as_posix()}', 'sha256': actual}

    rows, cells = [], []
    for M in args.Ms:
        for rules in factorial_rules(K=1, M=M):
            label = MatchConfig(board_size=3, rules=rules).label()
            t0 = time.perf_counter()
            sol = solve_config(3, rules)
            nodes = sample_nonterminal(sol, args.max_states, args.seed)
            cells.append({'config_label': label, 'states': len(sol.graph.keys), 'sampled_nodes': len(nodes),
                          'sampled_keys_sha256': hashlib.sha256(
                              repr([sol.graph.keys[u] for u in nodes]).encode()).hexdigest(),
                          'solve_seconds': round(time.perf_counter() - t0, 2)})
            specs = [(a, a) for a in BASE_AGENTS] + [(k, v['spec']) for k, v in dqn.items()]
            for name, spec in specs:
                if spec.startswith('dqn') and any(rules[k] != v for k, v in DQN_RULES.items()):
                    rows.append({'agent': name, 'config_label': label, 'applicable': False,
                                 'reason': 'checkpoint trained on N3_A1B1C1_K1_M1 only'})
                    continue
                t1 = time.perf_counter()
                try:
                    row = agreement(sol, spec, nodes=nodes)
                except AgentUnavailable as exc:
                    rows.append({'agent': name, 'config_label': label, 'applicable': False, 'reason': str(exc)})
                    continue
                row.update({'agent': name, 'spec': spec, 'applicable': True,
                            'seconds': round(time.perf_counter() - t1, 2)})
                rows.append(row)
                a = row['rates']['all']
                print(f"{label} {name}: value-opt {a['value_optimal_rate']:.3f} rank-opt "
                      f"{a['rank_optimal_rate']:.3f} n={a['n']}", flush=True)
            del sol
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as fh:
        json.dump({'source': meta['source'], 'run': meta, 'config': vars(args),
                   'dqn_checkpoints': dqn, 'cells': cells, 'rows': rows}, fh, indent=2)
    print(f"wrote {args.out}")


if __name__ == '__main__':
    main()
