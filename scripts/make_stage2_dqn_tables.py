#!/usr/bin/env python3
"""Markdown tables for docs/reports/stage2_dqn.md from experiments/results/final_dqn/*.json,
plus aggregate_3x3_seeds.json (mean / min / max over the five 3x3 seeds).

Example: python scripts/make_stage2_dqn_tables.py > experiments/results/final_dqn/tables.md
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.final_dqn_lib import RESULTS_DIR, provenance, write_result  # noqa: E402

OPPS = ['random', 'greedy', 'minimax:depth=2', 'exact']
EVALS = [('release 3x3', 'release_dqn_3x3_eval.json')] + \
        [(f'final 3x3 s{s}', f'final_dqn_3x3_s{s}_eval.json') for s in range(42, 47)] + \
        [('release 4x4', 'release_dqn_4x4_eval.json'), ('final 4x4 s42', 'final_dqn_4x4_s42_eval.json')]


def load(name):
    p = RESULTS_DIR / name
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else None


def pct(x):
    return '-' if x is None else f"{100 * x:.1f}"


def truth_table(rows):
    print('| Model | States | Value-opt % | Rank-opt % | Sign agree % | 3-class value % | MAE | Legacy protocol |')
    print('|---|---:|---:|---:|---:|---:|---:|---|')
    for label, d in rows:
        p = d['ground_truth_population']['all']
        lg = d['ground_truth_legacy_protocol']
        if 'overall_sign_agreement' in lg:
            leg = f"sign {pct(lg['overall_sign_agreement'])}% (old {pct(lg['old_value'])}%)"
        else:
            r = lg['result']['rates']['all']
            leg = (f"value-opt {pct(r['value_optimal_rate'])}% / rank-opt {pct(r['rank_optimal_rate'])}% "
                   f"(old {pct(lg['old_value_optimal_rate'])} / {pct(lg['old_rank_optimal_rate'])})")
        print(f"| {label} | {p['n']} | {pct(p['value_optimal_rate'])} | {pct(p['rank_optimal_rate'])} | "
              f"{pct(p.get('sign_agreement'))} | {pct(p['value_class_agreement'])} | "
              f"{p['mae_vs_discounted_value']} | {leg} |")


def match_table(rows):
    print('| Model | Opponent | Games | Win % | Draw % | Trunc % | Loss % | Mean plies | Win % as P0 | Win % as P1 |')
    print('|---|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    for label, d in rows:
        for o in OPPS:
            m = d['matches'].get(o)
            if not m:
                continue
            a = m['overall']
            print(f"| {label} | {o} | {a['matches']} | {pct(a['win_rate'])} | {pct(a['draw_rate'])} | "
                  f"{pct(a['truncation_rate'])} | {pct(a['loss_rate'])} | {a['mean_plies']} | "
                  f"{pct(m['dqn_as_p0']['win_rate'])} | {pct(m['dqn_as_p1']['win_rate'])} |")


def aggregate(seed_rows):
    def stat(xs):
        return {'mean': round(statistics.mean(xs), 4), 'min': round(min(xs), 4),
                'max': round(max(xs), 4), 'values': xs}
    out = {'seeds': [42, 43, 44, 45, 46], 'files': [n for _, n in EVALS[1:6]]}
    pops = [d['ground_truth_population']['all'] for _, d in seed_rows]
    for k in ('value_optimal_rate', 'rank_optimal_rate', 'sign_agreement', 'value_class_agreement',
              'mae_vs_discounted_value'):
        out[k] = stat([p[k] for p in pops])
    out['legacy_sign_agreement'] = stat([d['ground_truth_legacy_protocol']['overall_sign_agreement']
                                         for _, d in seed_rows])
    for o in OPPS:
        for k in ('win_rate', 'draw_rate', 'truncation_rate', 'loss_rate'):
            out[f'{o}_{k}'] = stat([d['matches'][o]['overall'][k] for _, d in seed_rows])
    write_result('aggregate_3x3_seeds.json', {'provenance': provenance({'inputs': out['files']}, seeds=out['seeds']), **out})
    return out


def main():
    rows = [(label, load(n)) for label, n in EVALS if load(n)]
    seed_rows = [(l, d) for l, d in rows if l.startswith('final 3x3')]
    agg = aggregate(seed_rows) if len(seed_rows) == 5 else None
    print('### Exact ground truth (all non-terminal solved states)\n')
    truth_table(rows)
    print('\n### Matches (opening_plies 2, both seatings, runner cap 200)\n')
    match_table(rows)
    if agg:
        print('\n### 3x3, five seeds (aggregate_3x3_seeds.json)\n')
        print('| Metric | Mean % | Min % | Max % |')
        print('|---|---:|---:|---:|')
        for k in ('value_optimal_rate', 'rank_optimal_rate', 'sign_agreement', 'legacy_sign_agreement',
                  'greedy_win_rate', 'minimax:depth=2_win_rate', 'exact_win_rate', 'random_win_rate'):
            s = agg[k]
            print(f"| {k} | {pct(s['mean'])} | {pct(s['min'])} | {pct(s['max'])} |")


if __name__ == '__main__':
    main()
