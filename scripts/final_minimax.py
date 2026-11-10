"""Stage 2 (Member 1) minimax experiments on the frozen engine.

Every output goes to experiments/results/final_minimax/ and records the git commit
(and dirty flag), the full config, the seeds, the exact command, Python version and date.
An existing output file is never overwritten.

  python scripts/final_minimax.py vs-greedy -N 5 --seeds 50 --spec minimax:depth=2 --tag N5
  python scripts/final_minimax.py vs-greedy -N 9 --seeds 10 --spec minimax:depth=2,time_limit=1.0 --tag N9
  python scripts/final_minimax.py ordering
  python scripts/final_minimax.py equivalence
  python scripts/final_minimax.py exact-draws
  python scripts/final_minimax.py timing
"""
import argparse
import datetime
import json
import os
import platform
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.minimax import MinimaxAgent  # noqa: E402
from evaluation.match_runner import (MatchConfig, run_batch, load_results, source_commit,  # noqa: E402
                                     verify_replay, opening_actions)
from evaluation.metrics import wilson  # noqa: E402
from wallz_engine import WallzEngine  # noqa: E402

OUT = ROOT / 'experiments' / 'results' / 'final_minimax'
RULES = {'A': True, 'B': True, 'C': True, 'K': 3, 'M': 10}


def provenance(config, seeds):
    return {
        'source': source_commit(),
        'command': 'python ' + ' '.join(sys.argv),
        'python': platform.python_version(),
        'platform': platform.platform(),
        'date': datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
        'config': config,
        'seeds': seeds,
    }


def write_new(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    if path.exists():
        raise SystemExit(f"{path} exists; refusing to overwrite a result")
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(obj, fh, indent=2, default=str)
    print(f"wrote {path}")


def pct(xs, q):
    xs = sorted(xs)
    i = (len(xs) - 1) * q
    lo, hi = int(i), min(int(i) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def spread(xs):
    return {'n': len(xs), 'mean': statistics.mean(xs), 'median': statistics.median(xs),
            'stdev': statistics.stdev(xs) if len(xs) > 1 else 0.0,
            'p10': pct(xs, 0.1), 'p90': pct(xs, 0.9), 'min': min(xs), 'max': max(xs)}


# ---------------------------------------------------------------------------
# Task 3: minimax vs greedy, both seatings
# ---------------------------------------------------------------------------
def summarize_vs(results, spec):
    rows = {'minimax_as_p0': [], 'minimax_as_p1': []}
    for r in results:
        seat = 0 if r['agents']['0']['spec'] == spec else 1
        rows[f'minimax_as_p{seat}'].append((seat, r))
    rows['overall'] = rows['minimax_as_p0'] + rows['minimax_as_p1']
    out = {}
    for name, items in rows.items():
        n = len(items)
        cats = {}
        mm_win = gr_win = forfeits_mm = forfeits_gr = 0
        dec_n = dec_t = nodes_t = depth_t = 0.0
        lengths = []
        for seat, r in items:
            o = r['outcome']
            cats[o['category']] = cats.get(o['category'], 0) + 1
            if o['category'] == 'goal':
                mm_win += o['winner'] == seat
                gr_win += o['winner'] != seat
            elif o['winner'] is not None:
                forfeits_mm += o['winner'] == seat
                forfeits_gr += o['winner'] != seat
            lengths.append(r['length_plies'])
            s = str(seat)
            k = r['decision_seconds'][s]['n']
            dec_n += k
            dec_t += r['decision_seconds'][s]['total']
            if r['mean_nodes_expanded'][s] is not None:
                nodes_t += r['mean_nodes_expanded'][s] * k
            if r['mean_depth_completed'][s] is not None:
                depth_t += r['mean_depth_completed'][s] * k
        draws = {c: v for c, v in cats.items() if c.startswith('draw_')}
        lo, hi = wilson(mm_win, n)
        out[name] = {
            'matches': n,
            'minimax_goal_wins': mm_win,
            'greedy_goal_wins': gr_win,
            'draws_by_type': draws,
            'draws_total': sum(draws.values()),
            'truncated': cats.get('truncated', 0),
            'forfeits_won_by_minimax': forfeits_mm,
            'forfeits_won_by_greedy': forfeits_gr,
            'categories': cats,
            'minimax_win_rate': mm_win / n if n else None,
            'minimax_win_rate_wilson95': [lo, hi],
            'draw_rate': sum(draws.values()) / n if n else None,
            'truncation_rate': cats.get('truncated', 0) / n if n else None,
            'mean_length_plies_incl_opening': statistics.mean(lengths) if lengths else None,
            'minimax_decisions': int(dec_n),
            'minimax_mean_depth_completed': depth_t / dec_n if dec_n else None,
            'minimax_mean_nodes_expanded': nodes_t / dec_n if dec_n else None,
            'minimax_mean_decision_seconds': dec_t / dec_n if dec_n else None,
        }
    return out


def opening_of(n, rules, plies, seed):
    return tuple(opening_actions(WallzEngine(board_size=n, rules=rules), plies, seed))


def distinct_seeds(n, rules, plies, want, start=0, max_scan=10_000):
    """First seed of each distinct opening action sequence, scanning upward from ``start``."""
    seen = {}
    s = start
    while len(seen) < want and s < start + max_scan:
        seen.setdefault(opening_of(n, rules, plies, s), s)
        s += 1
    return sorted(seen.values()), s - start


def cmd_vs_greedy(args):
    rules = dict(RULES, K=args.K, M=args.M)
    scanned = None
    if args.distinct:
        seeds, scanned = distinct_seeds(args.N, rules, args.opening_plies, args.seeds, args.seed_start)
    else:
        seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    openings = {s: [[a[0], list(a[1])] for a in opening_of(args.N, rules, args.opening_plies, s)] for s in seeds}
    n_distinct = len({json.dumps(o) for o in openings.values()})
    configs = []
    for s in seeds:
        for p0, p1 in ((args.spec, 'greedy'), ('greedy', args.spec)):
            configs.append(MatchConfig(board_size=args.N, rules=rules, p0=p0, p1=p1,
                                       max_plies=args.max_plies, opening_plies=args.opening_plies,
                                       opening_seed=s))
    log = OUT / f'vs_greedy_{args.tag}.jsonl'
    t0 = time.time()

    def progress(i, n, res):
        print(f"  {i}/{n} {res['outcome']['category']} ({time.time() - t0:.0f}s)", flush=True)

    run_batch(configs, str(log), keep_replay=True, progress=progress)
    ids = {c.match_id() for c in configs}
    results = [r for r in load_results(str(log)) if r['match_id'] in ids]
    bad = [(r['match_id'], m) for r in results for ok, m in [verify_replay(r)] if not ok]
    config = {'board_size': args.N, 'rules': configs[0].rules, 'minimax_spec': args.spec,
              'opponent': 'greedy', 'max_plies': args.max_plies, 'opening_plies': args.opening_plies,
              'seatings': 'both per opening seed', 'move_time_limit': None,
              'seed_selection': (f'first seed of each distinct opening, scanned {scanned} seeds from {args.seed_start}'
                                 if args.distinct else 'consecutive seeds'),
              'distinct_openings': n_distinct, 'seed_to_opening': openings,
              'deterministic': 'time_limit' not in args.spec,
              'note': ('timed search: depth reached depends on machine load, so results are '
                       'not reproducible move-for-move') if 'time_limit' in args.spec else
                      'pure depth-limited search: deterministic'}
    summary = provenance(config, seeds)
    summary.update({'match_log': str(log.relative_to(ROOT)).replace('\\', '/'),
                    'wall_seconds': round(time.time() - t0, 1),
                    'replay_verification': {'checked': len(results), 'failed': bad},
                    'results': summarize_vs(results, args.spec)})
    write_new(f'vs_greedy_{args.tag}_summary.json', summary)


# ---------------------------------------------------------------------------
# Task 4: move ordering on/off
# ---------------------------------------------------------------------------
def x8_positions(num_positions, board_size):
    """Same generator as scripts/benchmark_ordering_x8.py (random.seed(42), 2-6 random plies)."""
    random.seed(42)
    positions = []
    while len(positions) < num_positions:
        e = WallzEngine(board_size=board_size)
        for _ in range(random.randint(2, 6)):
            if e.is_terminal():
                break
            legal = e.get_legal_actions()
            if not legal:
                break
            e.apply_action(random.choice(legal), update_clock=False)
        if not e.is_terminal() and e.get_legal_actions():
            positions.append(e)
    return positions


def run_agent(pos, **kw):
    a = MinimaxAgent(player_id=pos.current_player, **kw)
    t0 = time.perf_counter()
    action = a.select_action(pos.clone())
    dt = time.perf_counter() - t0
    return {'nodes': a.nodes_expanded, 'cutoffs': a.cutoffs, 'depth': a.last_search_depth,
            'seconds': dt, 'value': a.last_action_value, 'action': list(action) if action else None}


def bootstrap_ratio(unord, ordd, n_boot=2000, seed=0):
    rng = random.Random(seed)
    n = len(unord)
    vals = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        u = sum(unord[i] for i in idx)
        o = sum(ordd[i] for i in idx)
        vals.append(100.0 * (1 - o / u))
    return [pct(vals, 0.025), pct(vals, 0.975)]


def cmd_ordering(args):
    positions = x8_positions(args.positions, args.board_size)
    report = {}
    for depth in args.depths:
        rows = []
        for i, pos in enumerate(positions):
            u = run_agent(pos, depth=depth, order_moves=False)
            o = run_agent(pos, depth=depth, order_moves=True)
            rows.append({'position': i, 'unordered': u, 'ordered': o})
            if (i + 1) % 10 == 0:
                print(f"  depth {depth}: {i + 1}/{len(positions)}", flush=True)
        un = [r['unordered']['nodes'] for r in rows]
        on = [r['ordered']['nodes'] for r in rows]
        per_pos = [100.0 * (1 - o / u) for u, o in zip(un, on) if u > 0]
        report[f'fixed_depth_{depth}'] = {
            'total_nodes_unordered': sum(un), 'total_nodes_ordered': sum(on),
            'aggregate_node_reduction_pct': 100.0 * (1 - sum(on) / sum(un)),
            'aggregate_reduction_bootstrap95_pct': bootstrap_ratio(un, on),
            'per_position_reduction_pct': spread(per_pos),
            'positions_where_ordering_used_more_nodes': sum(1 for u, o in zip(un, on) if o > u),
            'root_values_equal': all(abs(r['unordered']['value'] - r['ordered']['value']) < 1e-12 for r in rows),
            'seconds_unordered': spread([r['unordered']['seconds'] for r in rows]),
            'seconds_ordered': spread([r['ordered']['seconds'] for r in rows]),
            'rows': rows,
        }
    # Same positions under a time budget: how deep does each variant get?
    rows = []
    for i, pos in enumerate(positions[:args.timed_positions]):
        u = run_agent(pos, depth=args.timed_depth_cap, time_limit=args.time_limit, order_moves=False)
        o = run_agent(pos, depth=args.timed_depth_cap, time_limit=args.time_limit, order_moves=True)
        rows.append({'position': i, 'unordered': u, 'ordered': o})
    report[f'timed_{args.time_limit}s'] = {
        'depth_cap': args.timed_depth_cap,
        'depth_unordered': spread([r['unordered']['depth'] for r in rows]),
        'depth_ordered': spread([r['ordered']['depth'] for r in rows]),
        'positions_ordered_deeper': sum(r['ordered']['depth'] > r['unordered']['depth'] for r in rows),
        'positions_unordered_deeper': sum(r['ordered']['depth'] < r['unordered']['depth'] for r in rows),
        'rows': rows,
    }
    config = {'board_size': args.board_size, 'rules': dict(WallzEngine(args.board_size).rules),
              'positions': args.positions, 'position_generator': 'benchmark_ordering_x8 (random.seed(42), 2-6 random plies of any action)',
              'depths': args.depths, 'time_limit': args.time_limit, 'timed_positions': args.timed_positions,
              'use_alpha_beta': True,
              'note': 'ordering on = static order + PV (root, across ID iterations) + history within one call'}
    out = provenance(config, {'position_rng': 42, 'bootstrap_rng': 0})
    out['results'] = report
    write_new('ordering.json', out)


# ---------------------------------------------------------------------------
# Task 2: alpha-beta vs full-width minimax
# ---------------------------------------------------------------------------
def random_positions(n, rules, count, seed, max_plies=10):
    rng = random.Random(seed)
    out = []
    while len(out) < count:
        e = WallzEngine(board_size=n, rules=rules)
        for _ in range(rng.randint(0, max_plies)):
            if e.is_terminal():
                break
            e.apply_action(rng.choice(e.get_legal_actions()), update_clock=False)
        if not e.is_terminal() and len(e.get_legal_actions()) > 1:
            out.append(e)
    return out


def cmd_equivalence(args):
    suites = [(3, {'K': 1, 'M': 1}, 3, 40, 1), (3, {'K': 3, 'M': 2}, 4, 8, 2),
              (4, {'K': 2, 'M': 2}, 2, 40, 3), (4, {'K': 3, 'M': 3}, 3, 10, 4),
              (5, dict(RULES), 2, 20, 5)]
    report = []
    for n, rules, depth, count, seed in suites:
        rules = dict(RULES, **rules)
        diffs, actions_same, nodes = [], 0, {'ab': 0, 'full': 0}
        for pos in random_positions(n, rules, count, seed):
            full = run_agent(pos, depth=depth, use_alpha_beta=False, order_moves=False)
            ab = run_agent(pos, depth=depth, use_alpha_beta=True, order_moves=True)
            ab_u = run_agent(pos, depth=depth, use_alpha_beta=True, order_moves=False)
            diffs.append(max(abs(full['value'] - ab['value']), abs(full['value'] - ab_u['value'])))
            actions_same += full['action'] == ab_u['action']
            nodes['ab'] += ab['nodes']
            nodes['full'] += full['nodes']
        report.append({'board_size': n, 'rules': rules, 'depth': depth, 'positions': count, 'seed': seed,
                       'max_abs_root_value_diff': max(diffs),
                       'positions_with_diff_over_1e-12': sum(d > 1e-12 for d in diffs),
                       'unordered_ab_same_action_as_full_width': actions_same,
                       'total_nodes_full_width': nodes['full'], 'total_nodes_alpha_beta_ordered': nodes['ab']})
        print(f"  N{n} d{depth}: max diff {max(diffs)}", flush=True)
    out = provenance({'suites': 'see results', 'compared': 'full-width (no AB, no ordering) vs AB+ordering and AB unordered'},
                     [s[4] for s in suites])
    out['results'] = report
    write_new('alphabeta_equivalence.json', out)


# ---------------------------------------------------------------------------
# Task 2: draws vs the exact solver on 3x3
# ---------------------------------------------------------------------------
def cmd_exact_draws(args):
    from evaluation.retrograde import solve_config
    from evaluation.state_graph import GraphCaps, NONTERMINAL
    claim = MinimaxAgent.WIN_SCORE - 100
    report = []
    rng = random.Random(0)
    for rules, deep_sample in (({'A': True, 'B': True, 'C': True, 'K': 3, 'M': 0}, 400),
                               ({'A': True, 'B': True, 'C': False, 'K': 1, 'M': 1}, 300),
                               ({'A': False, 'B': True, 'C': True, 'K': 2, 'M': 1}, 300)):
        sol = solve_config(3, rules, GraphCaps(max_states=200_000, max_seconds=300))
        assert sol.exact, sol.graph.stop_reason
        g = sol.graph
        nodes = [u for u in range(len(g.keys)) if g.terminal[u] is NONTERMINAL and g.succ[u]]
        draws = [u for u in nodes if sol.status(u) == 'DRAW']
        rows = {}
        for depth, subset in ((2, nodes), (4, rng.sample(draws, min(deep_sample, len(draws))))):
            c = {'states': 0, 'draw_states': 0, 'draw_claimed_win': 0, 'draw_claimed_loss': 0,
                 'claimed_win': 0, 'claimed_win_wrong': 0, 'claimed_loss': 0, 'claimed_loss_wrong': 0,
                 'examples': []}
            for u in subset:
                e = g.restore(g.keys[u])
                a = MinimaxAgent(player_id=g.to_move[u], depth=depth)
                a.select_action(e)
                if len(e.get_legal_actions()) == 1:
                    # single legal action is returned without search; value is not a claim
                    continue
                v = a.last_action_value
                st = sol.status(u)
                c['states'] += 1
                if st == 'DRAW':
                    c['draw_states'] += 1
                    c['draw_claimed_win'] += v >= claim
                    c['draw_claimed_loss'] += v <= -claim
                    if abs(v) >= claim and len(c['examples']) < 5:
                        c['examples'].append({'node': u, 'value': v})
                if v >= claim:
                    c['claimed_win'] += 1
                    c['claimed_win_wrong'] += st != 'WIN'
                if v <= -claim:
                    c['claimed_loss'] += 1
                    c['claimed_loss_wrong'] += st != 'LOSS'
            rows[f'depth_{depth}'] = c
        report.append({'board_size': 3, 'rules': sol.graph.rules, 'states': len(g.keys),
                       'value_counts': sol.counts(), 'draw_nontrivial_states': len(draws), 'checks': rows})
        print(f"  {rules}: {rows}", flush=True)
    out = provenance({'claim_threshold': claim, 'note': 'depth 2 on every non-terminal state; '
                      'depth 4 on a random sample of DRAW states; positions restored with fresh history'},
                     {'draw_sample_rng': 0})
    out['results'] = report
    write_new('exact_draws_3x3.json', out)


# ---------------------------------------------------------------------------
# Task 2: time management overshoot
# ---------------------------------------------------------------------------
def cmd_timing(args):
    positions = random_positions(args.N, dict(RULES), args.positions, seed=11, max_plies=10)
    report = {}
    for depth_cap in args.depth_caps:
        for limit in args.limits:
            over, depths = [], []
            for pos in positions:
                r = run_agent(pos, depth=depth_cap, time_limit=limit)
                over.append(r['seconds'] - limit)
                depths.append(r['depth'])
            report[f'cap{depth_cap}_limit{limit}'] = {
                'depth_cap': depth_cap, 'time_limit': limit,
                'overshoot_seconds': spread(over),
                'decisions_over_limit': sum(o > 0 for o in over),
                'depth_completed': spread(depths)}
            print(f"  cap {depth_cap} limit {limit}: max overshoot {max(over):.3f}s", flush=True)
    out = provenance({'board_size': args.N, 'rules': RULES, 'positions': args.positions,
                      'position_generator': 'random play 0-10 plies (any action)',
                      'limits': args.limits, 'depth_caps': args.depth_caps,
                      'clock': 'time.perf_counter around select_action (agent uses time.time)'},
                     {'position_rng': 11})
    out['results'] = report
    write_new(f'timing_N{args.N}.json', out)


def cmd_openings(args):
    """Seed -> opening map and distinct counts for the logged vs-greedy runs (from the match logs)."""
    report = {}
    for tag in args.tags:
        rs = load_results(str(OUT / f'vs_greedy_{tag}.jsonl'))
        cfg = rs[0]['config']
        seeds = sorted({r['opening']['seed'] for r in rs})
        seq = {s: next(r['opening']['actions'] for r in rs if r['opening']['seed'] == s) for s in seeds}
        games = {json.dumps([r['agents']['0']['spec'], [st['action'] for st in r['replay']]]) for r in rs}
        report[tag] = {'board_size': cfg['board_size'], 'opening_plies': cfg['opening_plies'],
                       'seeds': seeds, 'distinct_openings': len({json.dumps(v) for v in seq.values()}),
                       'matches': len(rs), 'distinct_games': len(games), 'seed_to_opening': seq}
        print(f"  {tag}: {report[tag]['distinct_openings']} distinct openings, {len(games)} distinct games")
    out = provenance({'tags': args.tags}, None)
    out['results'] = report
    write_new('opening_maps.json', out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('vs-greedy')
    p.add_argument('-N', type=int, default=5)
    p.add_argument('-K', type=int, default=3)
    p.add_argument('-M', type=int, default=10)
    p.add_argument('--spec', default='minimax:depth=2')
    p.add_argument('--seeds', type=int, default=50)
    p.add_argument('--seed-start', type=int, default=0)
    p.add_argument('--opening-plies', type=int, default=2)
    p.add_argument('--max-plies', type=int, default=200)
    p.add_argument('--distinct', action='store_true',
                   help='--seeds counts distinct openings: keep the first seed of each distinct opening')
    p.add_argument('--tag', required=True)
    p.set_defaults(func=cmd_vs_greedy)
    p = sub.add_parser('openings')
    p.add_argument('tags', nargs='+')
    p.set_defaults(func=cmd_openings)
    p = sub.add_parser('ordering')
    p.add_argument('--positions', type=int, default=50)
    p.add_argument('--board-size', type=int, default=5)
    p.add_argument('--depths', type=int, nargs='+', default=[1, 2, 3])
    p.add_argument('--time-limit', type=float, default=0.5)
    p.add_argument('--timed-depth-cap', type=int, default=6)
    p.add_argument('--timed-positions', type=int, default=50)
    p.set_defaults(func=cmd_ordering)
    p = sub.add_parser('equivalence')
    p.set_defaults(func=cmd_equivalence)
    p = sub.add_parser('exact-draws')
    p.set_defaults(func=cmd_exact_draws)
    p = sub.add_parser('timing')
    p.add_argument('-N', type=int, default=9)
    p.add_argument('--positions', type=int, default=15)
    p.add_argument('--limits', type=float, nargs='+', default=[0.1, 0.25, 0.5, 1.0])
    p.add_argument('--depth-caps', type=int, nargs='+', default=[8])
    p.set_defaults(func=cmd_timing)
    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
