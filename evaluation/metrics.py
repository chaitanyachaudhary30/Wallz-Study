"""
Match metrics, 2^3 factorial effects and paired bootstrap intervals.

Accounting from a focus agent X's perspective (one category per game):
    win / loss         goal reached (rule outcome)
    draw               rule draw (repetition or no legal action)
    truncated          external ply cap; NOT a draw, kept separate
    forfeit_win/loss   timeout, illegal action, no action or agent error by the opponent / by X
Rates are reported over all games of a cell so the denominators always add up.

Factorial effects use the usual +/-1 coding of A, B, C over the 8 cell means:
    effect(S) = mean(y | prod_{f in S} x_f = +1) - mean(y | prod = -1)
so main effects, two-way and three-way interactions are all on the same scale.
Uncertainty comes from a paired bootstrap that resamples opening seeds, keeping each
seed's games in all 8 cells together (the same openings are played in every cell).
"""
import math
import random
from collections import defaultdict

FACTORS = ('A', 'B', 'C')
EFFECTS = ['A', 'B', 'C', 'AB', 'AC', 'BC', 'ABC']
CATEGORIES = ('win', 'loss', 'draw', 'truncated', 'forfeit_win', 'forfeit_loss')


def focus_player(result, spec):
    """Seat (0/1) occupied by agent ``spec`` in this match (P0 if both seats match)."""
    for p in ('0', '1'):
        if result['agents'][p]['spec'] == spec:
            return int(p)
    raise KeyError(f"{spec} did not play in {result['match_id']}")


def outcome_for(result, player):
    o = result['outcome']
    cat = o['category']
    if cat == 'goal':
        return 'win' if o['winner'] == player else 'loss'
    if cat.startswith('draw_'):
        return 'draw'
    if cat == 'truncated':
        return 'truncated'
    return 'forfeit_win' if o['winner'] == player else 'forfeit_loss'


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def game_responses(result, spec):
    """Per-game numeric responses used for cell means and effects."""
    p = focus_player(result, spec)
    cat = outcome_for(result, p)
    legal = [v for v in result['mean_legal_actions'].values() if v is not None]
    return {
        'x_goal_win': 1.0 if cat == 'win' else 0.0,
        'x_goal_loss': 1.0 if cat == 'loss' else 0.0,
        'rule_draw': 1.0 if cat == 'draw' else 0.0,
        'truncated': 1.0 if cat == 'truncated' else 0.0,
        'length_plies': float(result['length_plies']),
        'mean_legal_actions': mean(legal),
        'x_decision_ms': None if result['decision_seconds'][str(p)]['mean'] is None
        else 1000.0 * result['decision_seconds'][str(p)]['mean'],
    }


def summarize_cells(results, spec):
    """Per-config counts, rates with Wilson 95% CIs, lengths and timings for agent ``spec``."""
    cells = defaultdict(list)
    for r in results:
        cells[r['config_label']].append(r)
    out = {}
    for label, rs in sorted(cells.items()):
        counts = {c: 0 for c in CATEGORIES}
        for r in rs:
            counts[outcome_for(r, focus_player(r, spec))] += 1
        n = len(rs)
        resp = [game_responses(r, spec) for r in rs]
        out[label] = {
            'n_games': n,
            'counts': counts,
            'x_goal_win_rate': counts['win'] / n,
            'x_goal_win_rate_ci95': wilson(counts['win'], n),
            'truncation_rate': counts['truncated'] / n,
            'truncation_rate_ci95': wilson(counts['truncated'], n),
            'mean_length_plies': mean([x['length_plies'] for x in resp]),
            'mean_legal_actions': mean([x['mean_legal_actions'] for x in resp]),
            'x_mean_decision_ms': mean([x['x_decision_ms'] for x in resp]),
        }
    return out


def cell_label_factors(label):
    """'N5_A1B0C1_K3_M2' -> {'A': 1, 'B': 0, 'C': 1}"""
    abc = label.split('_')[1]
    return {'A': int(abc[1]), 'B': int(abc[3]), 'C': int(abc[5])}


def factorial_effects(cell_means):
    """
    cell_means: {label: y} for the 8 A/B/C cells (other factors held fixed).
    Returns {effect_name: estimate}. Missing cells or None values give None.
    """
    by_levels = {}
    for label, y in cell_means.items():
        f = cell_label_factors(label)
        by_levels[(f['A'], f['B'], f['C'])] = y
    if len(by_levels) != 8 or any(v is None for v in by_levels.values()):
        return {e: None for e in EFFECTS}
    out = {}
    for e in EFFECTS:
        plus, minus = [], []
        for levels, y in by_levels.items():
            sign = 1
            for f in e:
                sign *= 1 if levels[FACTORS.index(f)] else -1
            (plus if sign > 0 else minus).append(y)
        out[e] = sum(plus) / 4 - sum(minus) / 4
    return out


def unit_table(results, spec, response):
    """{opening_seed: {config_label: mean response over that seed's games}}"""
    acc = defaultdict(lambda: defaultdict(list))
    for r in results:
        y = game_responses(r, spec)[response]
        if y is not None:
            acc[r['config']['opening_seed']][r['config_label']].append(y)
    return {s: {c: sum(v) / len(v) for c, v in cells.items()} for s, cells in acc.items()}


def effects_with_ci(results, spec, response, n_boot=2000, seed=0):
    """Point estimates and paired-bootstrap 95% percentile intervals for all effects."""
    units = unit_table(results, spec, response)
    seeds = sorted(units)
    labels = sorted({c for u in units.values() for c in u})

    def estimate(sample):
        cell = {}
        for c in labels:
            vals = [units[s][c] for s in sample if c in units[s]]
            cell[c] = sum(vals) / len(vals) if vals else None
        return factorial_effects(cell), cell

    point, cell_means = estimate(seeds)
    rng = random.Random(seed)
    boots = {e: [] for e in EFFECTS}
    for _ in range(n_boot if len(seeds) > 1 else 0):
        est, _ = estimate([rng.choice(seeds) for _ in seeds])
        for e in EFFECTS:
            if est[e] is not None:
                boots[e].append(est[e])
    out = {}
    for e in EFFECTS:
        b = sorted(boots[e])
        ci = (b[int(0.025 * len(b))], b[min(len(b) - 1, int(0.975 * len(b)))]) if b else (None, None)
        out[e] = {'estimate': point[e], 'ci95': ci}
    return {'response': response, 'n_opening_seeds': len(seeds), 'cell_means': cell_means,
            'effects': out}


def agreement_rate(records):
    """records: [{'status', 'value_optimal', 'rank_optimal'}] -> rates overall and per status."""
    def rates(rs):
        n = len(rs)
        v = sum(r['value_optimal'] for r in rs)
        k = sum(r['rank_optimal'] for r in rs)
        return {'n': n, 'value_optimal_rate': v / n if n else None,
                'value_optimal_ci95': wilson(v, n), 'rank_optimal_rate': k / n if n else None}
    out = {'all': rates(records)}
    for status in ('WIN', 'LOSS', 'DRAW'):
        out[status] = rates([r for r in records if r['status'] == status])
    return out
