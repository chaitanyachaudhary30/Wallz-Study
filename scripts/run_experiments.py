"""
Person 3 evaluation CLI: matches, A/B/C ablations, exact solving, agreement, replays.

Examples (run from the repository root):
  python scripts/run_experiments.py match --p0 minimax:depth=2 --p1 greedy -N 5 --abc 111 --out m.json
  python scripts/run_experiments.py ablation --x minimax:depth=2 --y greedy -N 5 --seeds 10 --out-dir experiments/results/ablation_pilot
  python scripts/run_experiments.py solve -N 3 --abc all -K 1 -M 0 --out experiments/results/solve_N3_M0.json
  python scripts/run_experiments.py agreement -N 3 -K 1 -M 0 --agents greedy minimax:depth=2 --out a.json
  python scripts/run_experiments.py verify-replay experiments/logs/ablation_pilot/matches.jsonl
  python scripts/run_experiments.py export-demo m.json --out demo_replay.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.ablation import ablation_configs, solver_sweep, agreement, default_caps  # noqa: E402
from evaluation.match_runner import (MatchConfig, run_match, run_batch, load_results,  # noqa: E402
                                     verify_replay, source_commit)
from evaluation.metrics import summarize_cells, effects_with_ci  # noqa: E402
from evaluation.replay_export import export_demo_replay  # noqa: E402
from evaluation.retrograde import solve_config  # noqa: E402
from evaluation.state_graph import GraphCaps  # noqa: E402

RESPONSES = ['x_goal_win', 'x_goal_loss', 'truncated', 'length_plies', 'mean_legal_actions', 'x_decision_ms']


def rules_from(abc, K, M):
    a, b, c = (bool(int(ch)) for ch in abc)
    return {'A': a, 'B': b, 'C': c, 'K': K, 'M': M}


def write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(obj, fh, indent=2, default=str)
    print(f"wrote {path}")


def read_results(path):
    if path.endswith('.jsonl'):
        return load_results(path)
    with open(path, encoding='utf-8') as fh:
        obj = json.load(fh)
    return obj if isinstance(obj, list) else [obj]


def cmd_match(args):
    cfg = MatchConfig(board_size=args.N, rules=rules_from(args.abc, args.K, args.M), p0=args.p0,
                      p1=args.p1, max_plies=args.max_plies, opening_plies=args.opening_plies,
                      opening_seed=args.opening_seed, move_time_limit=args.move_time_limit)
    res = run_match(cfg)
    o = res['outcome']
    print(f"{res['match_id']}: {o['category']} winner={o['winner']} plies={res['length_plies']}")
    ok, msg = verify_replay(res)
    print(f"replay check: {'OK' if ok else 'FAILED'} - {msg}")
    if args.out:
        write_json(args.out, res)


def cmd_ablation(args):
    seeds = range(args.seed_start, args.seed_start + args.seeds)
    configs = ablation_configs(args.x, args.y, board_size=args.N, K=args.K, M=args.M,
                               opening_seeds=seeds, opening_plies=args.opening_plies,
                               max_plies=args.max_plies, move_time_limit=args.move_time_limit)
    log_path = os.path.join(args.log_dir, Path(args.out_dir).name, 'matches.jsonl')
    t0 = time.time()

    def progress(i, n, res):
        if i % 10 == 0 or i == n:
            print(f"  {i}/{n} matches ({time.time() - t0:.0f}s)", flush=True)

    run_batch(configs, log_path, keep_replay=True, progress=progress)
    results = [r for r in load_results(log_path)
               if r['match_id'] in {c.match_id() for c in configs}]
    summary = {
        'source': source_commit(),
        'design': {'agent_x': args.x, 'agent_y': args.y, 'board_size': args.N, 'K': args.K,
                   'M': args.M, 'opening_seeds': list(seeds), 'opening_plies': args.opening_plies,
                   'max_plies': args.max_plies, 'move_time_limit': args.move_time_limit,
                   'games_per_cell': 2 * len(seeds), 'n_matches': len(results)},
        'match_log': log_path.replace('\\', '/'),
        'cells': summarize_cells(results, args.x),
        'effects': {r: effects_with_ci(results, args.x, r, n_boot=args.n_boot) for r in RESPONSES},
        'replay_verification': None,
    }
    bad = [(r['match_id'], m) for r in results for ok, m in [verify_replay(r)] if not ok]
    summary['replay_verification'] = {'checked': len(results), 'failed': bad}
    write_json(os.path.join(args.out_dir, 'summary.json'), summary)


def cmd_solve(args):
    caps = GraphCaps(max_states=args.max_states, max_edges=args.max_edges, max_seconds=args.max_seconds)
    if args.abc == 'all':
        def progress(row):
            print(f"  {row['config_label']}: {row['states']} states, complete={row['complete']}, "
                  f"root {row['root_status']} rank {row['root_rank']}, "
                  f"{row['build_seconds'] + row['solve_seconds']:.1f}s", flush=True)
        rows, effects, _ = solver_sweep(args.N, K=args.K, M=args.M, caps=caps, progress=progress)
        out = {'source': source_commit(), 'caps': vars(caps), 'cells': rows, 'factorial_effects': effects}
    else:
        sol = solve_config(args.N, rules_from(args.abc, args.K, args.M), caps)
        out = {'source': source_commit(), 'caps': vars(caps), 'cells': [sol.summary()]}
        s = out['cells'][0]
        print(f"{s['states']} states, complete={s['complete']}, root {s['root_status']} "
              f"rank {s['root_rank']}, optimal {s['root_rank_optimal_actions']}")
    if args.out:
        write_json(args.out, out)


def cmd_agreement(args):
    caps = GraphCaps(max_states=args.max_graph_states, max_seconds=args.max_seconds)
    rows = []
    if args.abc == 'all':
        _, _, solutions = solver_sweep(args.N, K=args.K, M=args.M, caps=caps)
    else:
        cfg = MatchConfig(board_size=args.N, rules=rules_from(args.abc, args.K, args.M))
        solutions = {cfg.label(): solve_config(args.N, cfg.rules, caps)}
    for label, sol in solutions.items():
        if not sol.exact:
            print(f"  {label}: graph incomplete ({sol.graph.stop_reason}); skipped")
            continue
        for spec in args.agents:
            t0 = time.time()
            row = agreement(sol, spec, max_states=args.max_states, seed=args.seed)
            row['seconds'] = round(time.time() - t0, 2)
            rows.append(row)
            a = row['rates']['all']
            print(f"  {label} {spec}: value-optimal {a['value_optimal_rate']:.3f}, "
                  f"rank-optimal {a['rank_optimal_rate']:.3f} (n={a['n']})", flush=True)
    if args.out:
        write_json(args.out, {'source': source_commit(), 'rows': rows})


def cmd_verify(args):
    results = read_results(args.path)
    failures = 0
    for r in results:
        ok, msg = verify_replay(r)
        failures += not ok
        if not ok or args.verbose:
            print(f"{'OK  ' if ok else 'FAIL'} {r['match_id']}: {msg}")
    print(f"{len(results) - failures}/{len(results)} replays verified")
    sys.exit(1 if failures else 0)


def cmd_export(args):
    results = read_results(args.path)
    demo = export_demo_replay(results[args.index])
    print(f"demo_compatible={demo['demo_compatible']} events={len(demo['events'])}")
    for issue in demo['incompatibilities'][:5]:
        print(f"  - {issue}")
    write_json(args.out, demo)


def main():
    # Support --config <path> to load experiment YAML configurations (P7.1)
    if '--config' in sys.argv:
        c_idx = sys.argv.index('--config')
        if c_idx + 1 < len(sys.argv):
            config_path = sys.argv[c_idx + 1]
            sys.argv = sys.argv[:c_idx] + sys.argv[c_idx + 2:]
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                loaded = yaml.safe_load(f) or {}
            if 'command' in loaded:
                cmd = loaded['command']
            elif 'experiment' in loaded and 'command' in loaded['experiment']:
                cmd = loaded['experiment']['command']
            else:
                cmd = 'ablation'
            params = loaded.get('params', {})
            for k, v in loaded.items():
                if k not in ('experiment', 'params', 'command'):
                    params[k] = v

            known_cmds = {'match', 'ablation', 'solve', 'agreement', 'verify-replay', 'export-demo'}
            has_cmd = any(arg in known_cmds for arg in sys.argv[1:])
            if not has_cmd:
                sys.argv.insert(1, cmd)

            # Inject missing config parameters into sys.argv
            for k, v in params.items():
                flag = f"-{k}" if len(k) == 1 else f"--{k.replace('_', '-')}"
                alt_flag = f"--{k}" if len(k) == 1 else f"-{k}"
                if flag not in sys.argv and alt_flag not in sys.argv:
                    if isinstance(v, bool):
                        if v:
                            sys.argv.append(flag)
                    elif isinstance(v, list):
                        sys.argv.append(flag)
                        sys.argv.extend(str(x) for x in v)
                    elif v is not None:
                        sys.argv.extend([flag, str(v)])

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', type=str, help='Path to YAML experiment config file')
    sub = ap.add_subparsers(dest='cmd', required=True)

    def rule_args(p, abc_default='111', K=3, M=10):
        p.add_argument('-N', '--N', type=int, default=5, help='board size')
        p.add_argument('-K', '--K', type=int, default=K, help='block duration in plies')
        p.add_argument('-M', '--M', type=int, default=M, help='wall budget per player')
        p.add_argument('--abc', default=abc_default, help="mechanic toggles A,B,C as bits, e.g. 101")

    p = sub.add_parser('match', help='play and log one match')
    rule_args(p)
    p.add_argument('--p0', default='greedy')
    p.add_argument('--p1', default='greedy')
    p.add_argument('--max-plies', type=int, default=200)
    p.add_argument('--opening-plies', type=int, default=0)
    p.add_argument('--opening-seed', type=int, default=0)
    p.add_argument('--move-time-limit', type=float, default=None)
    p.add_argument('--out')
    p.set_defaults(func=cmd_match)

    p = sub.add_parser('ablation', help='paired 2^3 factorial match ablation')
    rule_args(p)
    p.add_argument('--x', required=True, help='focus agent spec')
    p.add_argument('--y', required=True, help='opponent agent spec')
    p.add_argument('--seeds', type=int, default=100)
    p.add_argument('--seed-start', type=int, default=0)
    p.add_argument('--opening-plies', type=int, default=2)
    p.add_argument('--max-plies', type=int, default=200)
    p.add_argument('--move-time-limit', type=float, default=None)
    p.add_argument('--n-boot', type=int, default=2000)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--log-dir', default='experiments/logs')
    p.set_defaults(func=cmd_ablation)

    p = sub.add_parser('solve', help='exact retrograde solve (one cell or --abc all)')
    rule_args(p, abc_default='all', K=1, M=0)
    p.set_defaults(N=3)
    p.add_argument('--max-states', type=int, default=default_caps().max_states)
    p.add_argument('--max-edges', type=int, default=default_caps().max_edges)
    p.add_argument('--max-seconds', type=float, default=default_caps().max_seconds)
    p.add_argument('--out')
    p.set_defaults(func=cmd_solve)

    p = sub.add_parser('agreement', help='agent optimal-action agreement vs exact solutions')
    rule_args(p, abc_default='all', K=1, M=0)
    p.set_defaults(N=3)
    p.add_argument('--agents', nargs='+', default=['greedy', 'minimax:depth=2'])
    p.add_argument('--max-states', type=int, default=300, help='sampled states per cell')
    p.add_argument('--max-graph-states', type=int, default=300_000)
    p.add_argument('--max-seconds', type=float, default=600.0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out')
    p.set_defaults(func=cmd_agreement)

    p = sub.add_parser('verify-replay', help='re-simulate logged matches')
    p.add_argument('path', help='.json result or .jsonl batch log')
    p.add_argument('--verbose', action='store_true')
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser('export-demo', help='convert a logged match to demo actions')
    p.add_argument('path')
    p.add_argument('--index', type=int, default=0)
    p.add_argument('--out', required=True)
    p.set_defaults(func=cmd_export)

    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
