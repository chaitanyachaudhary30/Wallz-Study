"""Stage 3 (Member 2, Task 3) Greedy match campaign on the frozen engine.

Greedy plays random:seed=R, minimax:depth=2 and itself (greedy:seed=S vs greedy:seed=S+100),
in both seats, on N3 (K1 M1, compared with the exact solver), N5 (K3 M10, main board) and
N9 (K3 M10, larger-board check). All mechanics on.

Openings: Greedy and minimax without a time limit are deterministic, so a repeated opening
replays the same game. For each board the script uses the SHORTEST opening_plies that gives
the wanted number of distinct openings, scanning seeds upward from 0 and keeping the first
seed of each distinct opening action sequence. The seed -> opening map goes into every
summary. Statistics are counted over distinct games (same Greedy seat + same action list
counts once).

`truncated` (the runner's 200-ply cap) is its own outcome, never a draw.

Every output goes to experiments/results/final_greedy/ and records the git commit, dirty
flag, config, seeds, seed -> opening map, command, Python version and date. An existing
summary is never overwritten.

  python scripts/final_greedy.py run --board N5 --pairing random
  python scripts/final_greedy.py run --board N5 --pairing minimax2
  python scripts/final_greedy.py run --board N5 --pairing self
  python scripts/final_greedy.py tables
"""
import argparse
import datetime
import hashlib
import json
import pickle
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.match_runner import (MatchConfig, load_results, opening_actions,  # noqa: E402
                                     run_batch, source_commit, verify_replay)
from evaluation.metrics import wilson  # noqa: E402
from wallz_engine import WallzEngine  # noqa: E402

OUT = ROOT / 'experiments' / 'results' / 'final_greedy'
ALL_ON = {'A': True, 'B': True, 'C': True}
BOARDS = {
    # want = distinct openings needed
    'N3': {'board_size': 3, 'rules': dict(ALL_ON, K=1, M=1), 'want': 8},
    'N5': {'board_size': 5, 'rules': dict(ALL_ON, K=3, M=10), 'want': 50},
    'N9': {'board_size': 9, 'rules': dict(ALL_ON, K=3, M=10), 'want': 10},
}
RANDOM_SEEDS = [0, 1, 2, 3, 4]
SELF_SEEDS = [0, 1, 2, 3, 4]          # greedy:seed=S vs greedy:seed=S+100
MAX_PLIES = 200
MAX_SCAN = 10_000


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


# ---------------------------------------------------------------------------
# Openings
# ---------------------------------------------------------------------------
def opening_of(n, rules, plies, seed):
    """Opening action tuple, or None when it ends early (game over / no MOVE) during the opening."""
    e = WallzEngine(board_size=n, rules=rules)
    acts = opening_actions(e, plies, seed)
    if len(acts) != plies or e.is_terminal():
        return None
    return tuple(acts)


def distinct_seeds(n, rules, plies, want):
    """First seed of each distinct opening, scanning upward from 0. Returns (seeds, scanned)."""
    seen = {}
    s = 0
    while len(seen) < want and s < MAX_SCAN:
        op = opening_of(n, rules, plies, s)
        if op is not None:
            seen.setdefault(op, s)
        s += 1
    return sorted(seen.values()), s


def count_openings(n, rules, plies):
    """Number of distinct MOVE-only openings of exactly `plies` plies that leave the game running.

    opening_actions picks uniformly from the sorted MOVE list, so every such sequence is
    reachable by some seed; counting them avoids scanning seeds for lengths that cannot work.
    """
    def walk(e, depth):
        if e.is_terminal():
            return 0
        if depth == plies:
            return 1
        total = 0
        for a in sorted(a for a in e.get_legal_actions() if a[0] == 'MOVE'):
            c = e.clone()
            c.apply_action(a, update_clock=False)
            total += walk(c, depth + 1)
        return total
    return walk(WallzEngine(board_size=n, rules=rules), 0)


def choose_openings(n, rules, want):
    """Shortest opening_plies with at least `want` distinct openings, then the seed scan."""
    tried = {}
    for plies in range(0, 12):
        tried[plies] = count_openings(n, rules, plies)
        if tried[plies] >= want:
            seeds, scanned = distinct_seeds(n, rules, plies, want)
            if len(seeds) < want:
                raise SystemExit(f'{plies} plies: only {len(seeds)} distinct openings in {scanned} seeds')
            return plies, seeds, scanned, tried
    raise SystemExit(f'no opening length up to 11 plies gives {want} distinct openings: {tried}')


# ---------------------------------------------------------------------------
# Pairings
# ---------------------------------------------------------------------------
def pairing_specs(pairing):
    """[(focus greedy spec, opponent spec)]."""
    if pairing == 'random':
        return [('greedy', f'random:seed={r}') for r in RANDOM_SEEDS]
    if pairing == 'minimax2':
        return [('greedy', 'minimax:depth=2')]
    if pairing == 'self':
        return [(f'greedy:seed={s}', f'greedy:seed={s + 100}') for s in SELF_SEEDS]
    raise SystemExit(f'unknown pairing {pairing}')


def focus_seat(r, focus_specs):
    a0 = r['agents']['0']['spec']
    return 0 if a0 in focus_specs else 1


def game_key(r, seat):
    return json.dumps([seat, [st['action'] for st in r['replay']]])


def summarize(items):
    """items: [(focus seat, result)] -> outcome table from the focus Greedy's side."""
    n = len(items)
    cats = {}
    win = loss = fwin = floss = 0
    lengths, after_open = [], []
    dec = {'focus': [0, 0.0], 'opponent': [0, 0.0]}
    for seat, r in items:
        o = r['outcome']
        cats[o['category']] = cats.get(o['category'], 0) + 1
        if o['category'] == 'goal':
            win += o['winner'] == seat
            loss += o['winner'] != seat
        elif o['winner'] is not None:          # forfeit
            fwin += o['winner'] == seat
            floss += o['winner'] != seat
        lengths.append(r['length_plies'])
        after_open.append(r['length_plies'] - r['opening']['plies'])
        for who, p in (('focus', seat), ('opponent', 1 - seat)):
            d = r['decision_seconds'][str(p)]
            dec[who][0] += d['n']
            dec[who][1] += d['total']
    draws = {c: v for c, v in sorted(cats.items()) if c.startswith('draw_')}
    nd = sum(draws.values())
    tr = cats.get('truncated', 0)

    def rate(k):
        lo, hi = wilson(k, n) if n else (None, None)
        return {'count': k, 'rate': k / n if n else None, 'wilson95': [lo, hi]}
    return {
        'games': n,
        'greedy_goal_wins': rate(win),
        'greedy_goal_losses': rate(loss),
        'draws_total': rate(nd),
        'draws_by_type': draws,
        'truncated': rate(tr),
        'forfeits_won': fwin,
        'forfeits_lost': floss,
        'categories': cats,
        'mean_length_plies_incl_opening': statistics.mean(lengths) if lengths else None,
        'mean_length_plies_after_opening': statistics.mean(after_open) if after_open else None,
        'greedy_decisions': dec['focus'][0],
        'greedy_mean_decision_seconds': dec['focus'][1] / dec['focus'][0] if dec['focus'][0] else None,
        'opponent_decisions': dec['opponent'][0],
        'opponent_mean_decision_seconds': dec['opponent'][1] / dec['opponent'][0] if dec['opponent'][0] else None,
    }


def engine_tag():
    return hashlib.sha256((ROOT / 'wallz_engine.py').read_bytes()).hexdigest()[:12]


def exact_solution(n, rules):
    """Exact solve of the N3 board (same cache file name as scripts/eval_dqn_final.py)."""
    from evaluation.retrograde import solve_config
    from evaluation.state_graph import GraphCaps
    label = MatchConfig(board_size=n, rules=rules).label()
    path = ROOT / 'experiments' / 'cache' / f'final_sol_{label}_{engine_tag()}.pkl'
    if path.exists():
        with open(path, 'rb') as fh:
            return pickle.load(fh), str(path.relative_to(ROOT)).replace('\\', '/')
    sol = solve_config(n, rules, GraphCaps(max_states=2_000_000, max_edges=40_000_000, max_seconds=1800))
    if not sol.exact:
        raise SystemExit(f'exact solve of {label} incomplete: {sol.graph.stop_reason}')
    return sol, 'solved in this run'


def solver_compare(items, n, rules):
    """Exact value of the post-opening position for Greedy's seat vs the game outcome."""
    sol, src = exact_solution(n, rules)
    table = {}
    for seat, r in items:
        e = WallzEngine(board_size=n, rules=rules)
        for a in r['opening']['actions']:
            e.apply_action((a[0], tuple(a[1])), update_clock=False)
        v = {1: 'win', 0: 'draw', -1: 'loss', None: 'unknown'}[sol.value_for(sol.graph.node_of(e), seat)]
        o = r['outcome']
        if o['category'] == 'goal' or o['winner'] is not None:
            got = 'win' if o['winner'] == seat else 'loss'
        elif o['category'] == 'truncated':
            got = 'truncated'
        else:
            got = 'draw'
        row = table.setdefault(f'value_{v}', {})
        row[got] = row.get(got, 0) + 1
    below = sum(c for v, row in table.items() for g, c in row.items()
                if (v, g) in (('value_win', 'draw'), ('value_win', 'loss'), ('value_win', 'truncated'),
                              ('value_draw', 'loss')))
    return {'solution': src, 'model': 'exact value under optimal play from the post-opening position, '
                                     'Greedy seat perspective (solver: history-free, infinite play = draw)',
            'value_vs_outcome': table, 'games_below_game_value': below}


def cmd_run(args):
    b = BOARDS[args.board]
    n, rules = b['board_size'], b['rules']
    plies, seeds, scanned, tried = choose_openings(n, rules, b['want'])
    openings = {s: [[a[0], list(a[1])] for a in opening_of(n, rules, plies, s)] for s in seeds}
    pairs = pairing_specs(args.pairing)
    configs = []
    for s in seeds:
        for focus, opp in pairs:
            for p0, p1 in ((focus, opp), (opp, focus)):
                configs.append(MatchConfig(board_size=n, rules=rules, p0=p0, p1=p1, max_plies=MAX_PLIES,
                                           opening_plies=plies, opening_seed=s))
    tag = f'{args.pairing}_{args.board}'
    log = OUT / f'{tag}.jsonl'
    print(f'{tag}: opening_plies={plies} ({len(seeds)} distinct openings, scanned {scanned} seeds), '
          f'{len(configs)} matches', flush=True)
    t0 = time.time()

    def progress(i, total, res):
        if i % 10 == 0 or i == total:
            print(f'  {i}/{total} {res["outcome"]["category"]} ({time.time() - t0:.0f}s)', flush=True)

    run_batch(configs, str(log), keep_replay=True, progress=progress)
    ids = {c.match_id() for c in configs}
    results = [r for r in load_results(str(log)) if r['match_id'] in ids]
    bad = [(r['match_id'], m) for r in results for ok, m in [verify_replay(r)] if not ok]
    focus_specs = {f for f, _ in pairs}
    uniq = {}
    for r in results:
        # Self-play: both sides are Greedy, so a game is one game whichever seed sits where;
        # count it once and read it from Player 0's side.
        seat = 0 if args.pairing == 'self' else focus_seat(r, focus_specs)
        uniq.setdefault(game_key(r, seat), (seat, r))
    items = list(uniq.values())
    if args.pairing == 'self':
        res = {'overall': summarize(items), 'note': 'self-play: one row per distinct game, from Player 0'}
    else:
        res = {'overall': summarize(items),
               'greedy_as_p0': summarize([x for x in items if x[0] == 0]),
               'greedy_as_p1': summarize([x for x in items if x[0] == 1])}
    if args.board == 'N3':
        res['vs_exact_solver'] = solver_compare(items, n, rules)
    config = {'board': args.board, 'board_size': n, 'rules': configs[0].rules, 'pairing': args.pairing,
              'focus_vs_opponent_specs': pairs, 'max_plies_runner_cap': MAX_PLIES,
              'truncated': 'runner cap reached; its own outcome, never a draw',
              'opening_plies': plies, 'distinct_openings_wanted': b['want'],
              'distinct_openings_possible_by_plies': tried,
              'seed_selection': f'first seed of each distinct opening action sequence, seeds 0..{scanned - 1} scanned',
              'seed_to_opening': openings, 'seatings': 'both per opening seed and spec pair',
              'move_time_limit': None,
              'counting': ('statistics over distinct games: the same full action list counts once; '
                           'self-play results are from Player 0, other pairings from Greedy, '
                           'with the same action list in different Greedy seats counted once per seat')}
    summary = provenance(config, seeds)
    summary.update({'match_log': str(log.relative_to(ROOT)).replace('\\', '/'),
                    'wall_seconds': round(time.time() - t0, 1),
                    'matches_run': len(results), 'distinct_games': len(items),
                    'replay_verification': {'checked': len(results), 'failed': bad},
                    'results': res})
    write_new(f'{tag}_summary.json', summary)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def pct(x):
    return '-' if x is None else f'{100 * x:.1f}'


def ci(d):
    lo, hi = d['wilson95']
    return f"{pct(d['rate'])} [{pct(lo)}, {pct(hi)}]"


def cmd_tables(args):
    rows = []
    for path in sorted(OUT.glob('*_summary.json')):
        s = json.loads(path.read_text(encoding='utf-8'))
        for part in ('greedy_as_p0', 'greedy_as_p1', 'overall'):
            if part not in s['results']:
                continue
            r = s['results'][part]
            if s['config']['pairing'] == 'self':
                part = 'self-play (W/L = P0)'
            draws = ', '.join(f'{k[5:]} {v}' for k, v in r['draws_by_type'].items()) or '0'
            rows.append(
                f"| {s['config']['board']} | {s['config']['pairing']} | {part.replace('greedy_as_', '')} | "
                f"{r['games']} | {r['greedy_goal_wins']['count']} | {r['greedy_goal_losses']['count']} | "
                f"{draws} | {r['truncated']['count']} | {r['forfeits_won']}/{r['forfeits_lost']} | "
                f"{ci(r['greedy_goal_wins'])} | {ci(r['draws_total'])} | {ci(r['truncated'])} | "
                f"{r['mean_length_plies_incl_opening']:.1f} | "
                f"{1000 * (r['greedy_mean_decision_seconds'] or 0):.1f} | "
                f"{1000 * (r['opponent_mean_decision_seconds'] or 0):.1f} |")
    head = ('| Board | Pairing | Greedy seat | Games | W | L | Draws by type | Trunc | Forfeit W/L | '
            'Win % [95% CI] | Draw % [95% CI] | Trunc % [95% CI] | Mean plies | Greedy ms/move | Opp ms/move |\n'
            '|---|---|---|---:|---:|---:|---|---:|---|---|---|---|---:|---:|---:|')
    text = head + '\n' + '\n'.join(rows) + '\n'
    path = OUT / 'tables.md'
    path.write_text(text, encoding='utf-8')  # derived view of the summaries; regenerated on purpose
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--board', choices=sorted(BOARDS), required=True)
    r.add_argument('--pairing', choices=['random', 'minimax2', 'self'], required=True)
    r.set_defaults(fn=cmd_run)
    t = sub.add_parser('tables')
    t.set_defaults(fn=cmd_tables)
    args = ap.parse_args()
    args.fn(args)


if __name__ == '__main__':
    main()
