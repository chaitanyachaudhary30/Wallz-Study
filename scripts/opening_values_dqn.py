#!/usr/bin/env python3
"""Game-theoretic value of each evaluation opening, for comparing match outcomes vs exact.

For opening seeds 0..K-1 (opening_plies 2, as in eval_dqn_final.py) and both DQN seats,
look up the exact value of the post-opening position for the DQN side, and count how
many games against the exact solver the DQN won / drew / lost compared with the value.
A DQN that matches the value in every game played the opening optimally.

Example: python scripts/opening_values_dqn.py --eval final_dqn_3x3_s42_eval.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wallz_engine import WallzEngine  # noqa: E402
from evaluation.match_runner import opening_actions  # noqa: E402
from scripts.eval_dqn_final import cached_solve  # noqa: E402
from scripts.final_dqn_lib import RESULTS_DIR, provenance, write_result  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--evals', nargs='+', required=True, help='eval JSON names in final_dqn/')
    ap.add_argument('--seeds', type=int, default=50)
    args = ap.parse_args()
    out = {}
    for name in args.evals:
        d = json.loads((RESULTS_DIR / name).read_text(encoding='utf-8'))
        n, rules = d['board_size'], d['rules']
        sol, label, _ = cached_solve(n, rules)
        per_seat = {}
        for seat in (0, 1):
            vals = Counter()
            for seed in range(args.seeds):
                e = WallzEngine(board_size=n, rules=rules)
                opening_actions(e, 2, seed)
                v = sol.value_for(sol.graph.node_of(e), seat)
                vals[{1: 'win', 0: 'draw', -1: 'loss'}[v]] += 1
            got = d['matches']['exact'][f'dqn_as_p{seat}']['categories']
            got_wdl = {'win': got.get('win', 0), 'draw': sum(v for k, v in got.items() if k.startswith('draw_')),
                       'loss': got.get('loss', 0), 'truncated': got.get('truncated', 0)}
            per_seat[f'dqn_as_p{seat}'] = {'game_values': dict(vals), 'outcomes_vs_exact': got_wdl}
        tot_v = Counter()
        for s in per_seat.values():
            tot_v.update(s['game_values'])
        out[name] = {'label': label, 'game_values_total': dict(tot_v), **per_seat,
                     'value_achieved_in_every_game': all(
                         s['game_values'].get(k, 0) == s['outcomes_vs_exact'][k]
                         for s in per_seat.values() for k in ('win', 'draw', 'loss'))}
        print(name, out[name]['game_values_total'], out[name]['value_achieved_in_every_game'])
    write_result('opening_values_vs_exact.json',
                 {'provenance': provenance({'opening_plies': 2, 'seeds': args.seeds}, seeds=list(range(args.seeds))),
                  'note': 'counts match by category; with deterministic exact play an equal count per category '
                          'means each game reached its game value only if no win offset a loss, which the '
                          'solver makes impossible (DQN cannot beat a lost position against exact).',
                  'results': out})


if __name__ == '__main__':
    main()
