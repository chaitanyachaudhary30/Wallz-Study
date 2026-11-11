#!/usr/bin/env python3
"""Stage 2 DQN checkpoint evaluation under the frozen engine (Member 4, Tasks 4-5).

Matches: vs random and greedy (opening seeds 0..99), minimax:depth=2 and the exact
solver (seeds 0..49), both seatings, opening_plies 2, runner cap 200 plies. Outcomes
keep ``truncated`` as its own column (never merged into draws).

Ground truth (when the exact solve completes): over EVERY non-terminal solved state,
value-optimal and rank-optimal agreement of the greedy masked argmax, sign agreement of
max legal Q with the solver value on decided states, 3-class value agreement (Q > 0.5
WIN, Q < -0.5 LOSS, else DRAW) and MAE against the discounted optimal value. It also
reruns the earlier protocols unchanged (3x3: scripts/validate_dqn.evaluate_cell_x16 on
the two Gate X-16 cells, 1000 states, seed 42 -> the old 96.47%; 4x4:
evaluation.ablation.agreement, 300 states, seed 42).

Example:
  python scripts/eval_dqn_final.py --checkpoint checkpoints/release/dqn_3x3/model.pt \
      --out release_dqn_3x3_eval.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from agents.dqn.agent import load_checkpoint  # noqa: E402
from agents.dqn.encoding import encode_observation  # noqa: E402
from evaluation.match_runner import MatchConfig, make_agent, run_match  # noqa: E402
from evaluation.metrics import wilson  # noqa: E402
from evaluation.retrograde import solve_config  # noqa: E402
from evaluation.state_graph import GraphCaps, NONTERMINAL  # noqa: E402
from scripts.final_dqn_lib import REPO_ROOT, provenance, write_result  # noqa: E402

GAMMA = 0.99
CACHE = REPO_ROOT / "experiments" / "cache"


def engine_tag() -> str:
    """Short hash of wallz_engine.py so a cached solve is never reused across rule changes."""
    return hashlib.sha256((REPO_ROOT / "wallz_engine.py").read_bytes()).hexdigest()[:12]


def cached_solve(n, rules, max_seconds=1800.0):
    label = MatchConfig(board_size=n, rules=rules).label()
    path = CACHE / f"final_sol_{label}_{engine_tag()}.pkl"
    if path.exists():
        with open(path, "rb") as fh:
            return pickle.load(fh), label, True
    t0 = time.time()
    sol = solve_config(n, rules, GraphCaps(max_states=2_000_000, max_edges=40_000_000,
                                           max_seconds=max_seconds))
    print(f"solved {label}: {len(sol.graph.keys)} states exact={sol.exact} in {time.time() - t0:.0f}s",
          flush=True)
    if sol.exact:
        CACHE.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(sol, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return sol, label, False


def population_truth(sol, agent, batch=4096):
    """Metrics over every non-terminal state of a complete solution."""
    g, codec, model = sol.graph, agent.codec, agent.model
    nodes = [u for u in range(len(g.keys)) if g.terminal[u] is NONTERMINAL]
    by = {s: Counter() for s in ('WIN', 'LOSS', 'DRAW', 'all')}
    abs_err = []
    for start in range(0, len(nodes), batch):
        chunk = nodes[start:start + batch]
        engines = [g.restore(g.keys[u]) for u in chunk]
        obs = np.stack([encode_observation(e, mover=g.to_move[u]) for e, u in zip(engines, chunk)])
        with torch.no_grad():
            q = model(torch.as_tensor(obs, dtype=torch.float32)).numpy()
        for row, e, u in zip(q, engines, chunk):
            mover = g.to_move[u]
            ids = sorted(codec.encode(a, e, mover=mover) for a, _ in g.succ[u])
            best = ids[int(np.argmax(row[ids]))]           # lowest id on ties, as DQNAgent
            act = codec.decode(best, e, mover=mover)
            v = float(row[ids].max())
            status = sol.status(u)
            v_star = sol.discounted_value(u, GAMMA)
            pred = 'WIN' if v > 0.5 else ('LOSS' if v < -0.5 else 'DRAW')
            for key in (status, 'all'):
                c = by[key]
                c['n'] += 1
                c['value_optimal'] += act in sol.value_optimal_actions(u)
                c['rank_optimal'] += act in sol.rank_optimal_actions(u)
                c['class_agree'] += pred == status
                if status in ('WIN', 'LOSS'):
                    c['decided'] += 1
                    c['sign_agree'] += (v > 0) == (v_star > 0)
            abs_err.append(abs(v - v_star))
    out = {}
    for key, c in by.items():
        n = c['n']
        if not n:
            continue
        row = {'n': n,
               'value_optimal_rate': round(c['value_optimal'] / n, 4),
               'value_optimal_ci95': [round(x, 4) for x in wilson(c['value_optimal'], n)],
               'rank_optimal_rate': round(c['rank_optimal'] / n, 4),
               'value_class_agreement': round(c['class_agree'] / n, 4)}
        if c['decided']:
            row['sign_agreement'] = round(c['sign_agree'] / c['decided'], 4)
            row['sign_agreement_ci95'] = [round(x, 4) for x in wilson(c['sign_agree'], c['decided'])]
            row['decided_states'] = c['decided']
        out[key] = row
    out['all']['mae_vs_discounted_value'] = round(float(np.mean(abs_err)), 4)
    out['value_counts'] = sol.counts()
    out['root_status'] = sol.status(0)
    return out


def legacy_truth(n, sol, agent, ckpt):
    """The earlier protocols, unchanged, for a before/after comparison."""
    if n == 3:
        from scripts.validate_dqn import evaluate_cell_x16
        cells = {}
        tot_d = tot_a = 0
        off_rules = {'A': False, 'B': False, 'C': False, 'K': 1, 'M': 0}
        off_sol, off_label, _ = cached_solve(3, off_rules)
        for label, s, k in ((off_label, off_sol, 1000), (MatchConfig(board_size=3, rules=sol.graph.rules).label(), sol, 1000)):
            r = evaluate_cell_x16(s, agent, gamma=GAMMA, max_states=k, seed=42)
            cells[label] = r
            tot_d += r['decided_states']
            tot_a += r['sign_agreement_count']
        return {'protocol': 'scripts/validate_dqn.evaluate_cell_x16, 1000 states, seed 42 (Gate X-16)',
                'overall_sign_agreement': round(tot_a / tot_d, 4), 'cells': cells,
                'old_value_file': 'experiments/results/dqn/gate_x16.json', 'old_value': 0.9647}
    if n == 4:
        from evaluation.ablation import agreement
        r = agreement(sol, f"dqn:path={ckpt}", max_states=300, seed=42)
        return {'protocol': 'evaluation.ablation.agreement, 300 states, seed 42',
                'result': r, 'old_value_file': 'experiments/results/dqn/agreement_N4.json',
                'old_value_optimal_rate': 0.9333, 'old_rank_optimal_rate': 0.7067}
    return None


def play_series(cfg_base, ckpt, opp, seeds, solutions):
    agents_dqn = {p: load_checkpoint(ckpt, player_id=p, expected_board_size=cfg_base['board_size'])
                  for p in (0, 1)}
    cats = {0: Counter(), 1: Counter()}
    plies = {0: [], 1: []}
    for seed in seeds:
        for seat in (0, 1):
            cfg = MatchConfig(p0='dqn' if seat == 0 else opp, p1=opp if seat == 0 else 'dqn',
                              opening_seed=seed, **cfg_base)
            opp_agent, opp_id = make_agent(opp, 1 - seat, cfg, solutions, seed)
            agents = {seat: agents_dqn[seat], 1 - seat: opp_agent}
            ids = {seat: {'name': 'dqn', 'spec': f'dqn:path={ckpt}'}, 1 - seat: opp_id}
            res = run_match(cfg, agents=agents, identities=ids)
            o = res['outcome']
            c = o['category']
            if c == 'goal':
                c = 'win' if o['winner'] == seat else 'loss'
            elif c in ('timeout', 'illegal_action', 'no_action', 'agent_error'):
                c = ('forfeit_by_dqn_' if o['loser'] == seat else 'forfeit_by_opp_') + c
            cats[seat][c] += 1
            plies[seat].append(res['length_plies'])

    def summarize(counter, pl):
        n = sum(counter.values())
        rule_draws = sum(v for k, v in counter.items() if k.startswith('draw_'))
        return {'matches': n, 'categories': dict(counter),
                'win_rate': round(counter['win'] / n, 4),
                'win_rate_ci95': [round(x, 4) for x in wilson(counter['win'], n)],
                'loss_rate': round(counter['loss'] / n, 4),
                'draw_rate': round(rule_draws / n, 4),
                'truncation_rate': round(counter['truncated'] / n, 4),
                'forfeit_by_dqn': sum(v for k, v in counter.items() if k.startswith('forfeit_by_dqn')),
                'mean_plies': round(float(np.mean(pl)), 2)}
    total = cats[0] + cats[1]
    for k in ('truncated', 'win', 'loss'):
        total.setdefault(k, 0)
    return {'opponent': opp, 'opening_seeds': [seeds[0], seeds[-1]],
            'overall': summarize(total, plies[0] + plies[1]),
            'dqn_as_p0': summarize(cats[0], plies[0]),
            'dqn_as_p1': summarize(cats[1], plies[1])}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--out', required=True, help='file name inside experiments/results/final_dqn/')
    ap.add_argument('--random-seeds', type=int, default=100)
    ap.add_argument('--minimax-seeds', type=int, default=50)
    ap.add_argument('--exact-seeds', type=int, default=50)
    ap.add_argument('--skip-matches', action='store_true')
    ap.add_argument('--skip-truth', action='store_true')
    ap.add_argument('--solve-seconds', type=float, default=1800.0)
    ap.add_argument('--distinct-openings', type=int, default=0,
                    help='K > 0: use the distinct-opening rule (K openings) for random, greedy and minimax')
    args = ap.parse_args()
    torch.set_num_threads(2)

    ckpt = Path(args.checkpoint)
    meta = json.loads(((ckpt.parent if ckpt.is_file() else ckpt) / 'meta.json').read_text())
    n, rules = meta['N'], dict(meta['rules'])
    t0 = time.time()
    agent = load_checkpoint(ckpt, player_id=0, expected_board_size=n)
    out = {'checkpoint_meta': meta, 'board_size': n, 'rules': rules}

    sol = None
    if not args.skip_truth or args.exact_seeds:
        sol, label, cached = cached_solve(n, rules, args.solve_seconds)
        out['solve'] = {'label': label, 'exact': sol.exact, 'states': len(sol.graph.keys),
                        'from_cache': cached, 'engine_sha256_12': engine_tag(),
                        'stop_reason': sol.graph.stop_reason}
    if not args.skip_truth and sol is not None and sol.exact:
        out['ground_truth_population'] = population_truth(sol, agent)
        out['ground_truth_legacy_protocol'] = legacy_truth(n, sol, agent, ckpt)
        print('truth done', time.time() - t0, flush=True)

    opening_plies, seed_lists = 2, {}
    if args.distinct_openings:
        # distinct-opening rule: shortest opening_plies with >= K distinct openings,
        # first seed of each, scanned upward from seed 0; same seeds for every opponent
        from evaluation.ablation import first_distinct_openings
        for opening_plies in range(1, 11):
            found = first_distinct_openings(n, opening_plies, args.distinct_openings,
                                            K=rules['K'], M=rules['M'])
            if len(found) >= args.distinct_openings:
                break
        else:
            raise SystemExit(f"fewer than {args.distinct_openings} distinct openings up to 10 plies")
        distinct_seeds = [s for s, _ in found]
        out['openings'] = {'rule': f'shortest opening_plies with >= {args.distinct_openings} distinct '
                                   'openings, first seed of each, scanned from seed 0',
                           'opening_plies': opening_plies, 'seeds': distinct_seeds,
                           'seeds_scanned': distinct_seeds[-1] + 1,
                           'seed_to_opening': {str(s): [[a[0], list(a[1])] for a in acts]
                                               for s, acts in found}}
        seed_lists = {'random': distinct_seeds, 'greedy': distinct_seeds,
                      'minimax:depth=2': distinct_seeds}

    if not args.skip_matches:
        base = {'board_size': n, 'rules': rules, 'max_plies': 200, 'opening_plies': opening_plies}
        plan = [('random', args.random_seeds), ('greedy', args.random_seeds),
                ('minimax:depth=2', args.minimax_seeds)]
        if sol is not None and sol.exact and args.exact_seeds:
            plan.append(('exact', args.exact_seeds))
        solutions = {MatchConfig(board_size=n, rules=rules).label(): sol} if sol is not None else {}
        out['matches'] = {}
        for opp, k in plan:
            out['matches'][opp] = play_series(base, ckpt, opp, seed_lists.get(opp, list(range(k))), solutions)
            print(opp, out['matches'][opp]['overall']['categories'], f"{time.time() - t0:.0f}s", flush=True)

    out['elapsed_seconds'] = round(time.time() - t0, 1)
    cfg = {'gamma': GAMMA, 'max_plies': 200, 'opening_plies': opening_plies,
           'distinct_openings': args.distinct_openings,
           'random_greedy_seeds': args.random_seeds, 'minimax_seeds': args.minimax_seeds,
           'exact_seeds': args.exact_seeds, 'value_class_threshold': 0.5,
           'tie_break': 'lowest action id among max Q (as DQNAgent)'}
    seeds = {'distinct_opening_seeds': out['openings']['seeds']} if args.distinct_openings else \
            {'opening_seeds_random_greedy': [0, args.random_seeds - 1],
             'opening_seeds_minimax_exact': [0, max(args.minimax_seeds, args.exact_seeds) - 1],
             'legacy_sample_seed': 42}
    payload = {'provenance': provenance(cfg, seeds=[seeds], checkpoints=[ckpt]), **out}
    print('wrote', write_result(args.out, payload))


if __name__ == '__main__':
    main()
