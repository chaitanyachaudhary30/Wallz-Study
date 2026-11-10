"""
Stage 2 analysis for Member 3: reads experiments/results/final_eval/ (and the pilot files
for comparison) and writes experiments/results/final_eval/analysis.json. It runs nothing
heavy: every number comes from files that already exist.

  python scripts/stage2_analysis.py

Sections of the output:
  solve_tables       per cell: complete/capped, states, edges, root value and rank, seconds
  pilot_comparison   per cell: same/different vs person3_pilot, and whether B is on
  exact_effects      2^3 effects on exact measures (only when all 8 cells are complete)
  ablations          per ablation summary: units, Holm table, power (minimum detectable effect)
"""
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.match_runner import run_metadata  # noqa: E402
from evaluation.metrics import factorial_effects, cell_label_factors  # noqa: E402

FINAL = 'experiments/results/final_eval'
PILOT = 'experiments/results/person3_pilot'
SOLVES = {  # final file -> pilot file
    'solve_N3_K1_M0.json': 'solve_N3_K1_M0.json',
    'solve_N3_K1_M1.json': 'solve_N3_K1_M1.json',
    'solve_N4_K1_M0.json': 'solve_N4_K1_M0.json',
    'solve_N4_K1_M1.json': 'solve_N4_K1_M1_capped.json',
}
VALUE = {'WIN': 1, 'DRAW': 0, 'LOSS': -1}
Z975, Z80 = 1.959964, 0.841621


def load(path):
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def z_quantile(p):
    """Inverse standard normal CDF by bisection (stdlib only)."""
    lo, hi = -10.0, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if 0.5 * (1 + math.erf(mid / math.sqrt(2))) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def solve_row(c):
    return {'cell': c['config_label'], 'complete': c['complete'], 'stop_reason': c['stop_reason'],
            'states': c['states'], 'edges': c['edges'], 'root_status': c['root_status'],
            'root_rank': c['root_rank'], 'root_rank_is_upper_bound': c.get('root_rank_is_upper_bound'),
            'seconds': round(c['build_seconds'] + c['solve_seconds'], 2)}


def compare(new, old):
    keys = ('complete', 'states', 'edges', 'root_status', 'root_rank')
    diff = {k: [old[k], new[k]] for k in keys if old[k] != new[k]}
    return {'cell': new['config_label'], 'B_on': bool(new['rules']['B']), 'same_as_pilot': not diff,
            'differences_pilot_vs_final': diff}


def exact_effects(cells):
    if not all(c['complete'] for c in cells):
        return None
    measures = {
        'root_value_p0': {c['config_label']: VALUE[c['root_status']] for c in cells},
        'log10_states': {c['config_label']: math.log10(c['states']) for c in cells},
        'mean_branching': {c['config_label']: c['mean_branching'] for c in cells},
        'draw_fraction': {c['config_label']: c['value_counts']['DRAW'] / c['states'] for c in cells},
    }
    return {m: factorial_effects(v) for m, v in measures.items()}


def power(n_units, alpha, sd_unit=0.5):
    """
    Minimum detectable factorial effect on a rate with 80% power, two-sided level alpha.
    effect = mean of 4 cells - mean of 4 cells, each cell a mean over n_units independent
    units, so Var(effect) = 8 * sd^2 / (16 n) = sd^2 / (2 n). sd = 0.5 is the worst case
    for a 0/1 outcome; pairing across cells only lowers the variance, so this is conservative.
    """
    z = z_quantile(1 - alpha / 2)
    return (z + Z80) * sd_unit / math.sqrt(2 * n_units)


def opening_counts():
    """Distinct pawn openings drawn by seed ranges on N=5 (same generator as the runner)."""
    from wallz_engine import WallzEngine
    from evaluation.match_runner import opening_actions

    def opening(plies, s):
        return tuple(opening_actions(WallzEngine(board_size=5, rules={'K': 3, 'M': 10}), plies, s))

    def distinct(plies, seeds):
        return len({opening(plies, s) for s in seeds})

    pilot_map, first = {}, {}
    for s in range(10):
        o = opening(2, s)
        first.setdefault(o, s)
        pilot_map[str(s)] = {'opening': [[a[0], list(a[1])] for a in o], 'same_as_seed': first[o]}
    return {
        'plies2_seeds0-9 (pilot)': distinct(2, range(10)),
        'pilot_seed_to_opening_plies2': pilot_map,
        'plies2_distinct_in_seeds0-1999': distinct(2, range(2000)),
        'plies4_distinct_in_seeds0-1999': distinct(4, range(2000)),
        'smallest_opening_plies_with_at_least_30_distinct': 4,
    }


def ablation_block(path):
    s = load(path)
    win = s['effects']['x_goal_win']
    n = win['n_units']
    eff = {}
    for e, v in win['effects'].items():
        lo, hi = v['ci95']
        se = (hi - lo) / (2 * Z975) if lo is not None else None
        eff[e] = {'estimate': v['estimate'], 'ci95': v['ci95'], 'p_value': v['p_value'],
                  'holm_adjusted_p': v['adjusted_p'], 'survives_holm': v['significant_holm'],
                  'ci_excludes_0': lo is not None and (lo > 0 or hi < 0),
                  'bootstrap_se': se, 'mde_80pct_from_bootstrap_se': None if se is None else (Z975 + Z80) * se}
    counts = {}
    for cell in s['cells'].values():
        for k, v in cell['counts'].items():
            counts[k] = counts.get(k, 0) + v
    return {
        'file': path.replace('\\', '/'),
        'n_matches': s['design']['n_matches'],
        'opening_seeds': len(s['design']['opening_seeds']),
        'opening_plies': s['design']['opening_plies'],
        'distinct_openings': s['replicas']['distinct_openings'],
        'replica_groups_that_differ': s['replicas']['groups_with_differing_replicas'],
        'independent_units_per_cell': n,
        'games_per_cell': s['design']['games_per_cell'],
        'outcome_totals_x': counts,
        'replays_verified': s['replay_verification']['checked'] - len(s['replay_verification']['failed']),
        'replays_failed': len(s['replay_verification']['failed']),
        'win_rate_effects': eff,
        'holm_survivors': [e for e, v in eff.items() if v['survives_holm']],
        'mde_win_rate_80pct_power': {
            'alpha_0.05': power(n, 0.05),
            'alpha_0.05_over_7 (Holm first step)': power(n, 0.05 / 7),
            'assumption': 'worst-case sd 0.5 per unit, cells independent (pairing ignored)',
        },
    }


def main():
    out = {'run': run_metadata(), 'solve_tables': {}, 'pilot_comparison': {}, 'exact_effects': {}}
    for final, pilot in SOLVES.items():
        fp = os.path.join(FINAL, final)
        if not os.path.exists(fp):
            continue
        new = load(fp)['cells']
        old = {c['config_label']: c for c in load(os.path.join(PILOT, pilot))['cells']}
        name = final[:-5]
        out['solve_tables'][name] = [solve_row(c) for c in new]
        out['pilot_comparison'][name] = [compare(c, old[c['config_label']]) for c in new]
        out['exact_effects'][name] = exact_effects(new)
    out['opening_counts_N5'] = opening_counts()
    out['ablations'] = {}
    for d in sorted(os.listdir(FINAL)):
        p = os.path.join(FINAL, d, 'summary.json')
        if d.startswith('ablation') and os.path.exists(p):
            out['ablations'][d] = ablation_block(p)
    with open(os.path.join(FINAL, 'analysis.json'), 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {FINAL}/analysis.json")


if __name__ == '__main__':
    main()
