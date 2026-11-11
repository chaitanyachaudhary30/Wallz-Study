#!/usr/bin/env python3
"""Recount the final_dqn match results over DISTINCT games only.

DQN, Greedy (no seed), Minimax (depth only) and the exact solver are deterministic, so two
matches with the same opening and the same seats replay the same game. The evaluation files
count every match, so their Wilson intervals treat repeated games as independent and are too
narrow. Random is seeded per opening seed and seat, so its games are mostly distinct.

For each evaluation file this script replays every match exactly as scripts/eval_dqn_final.py
did (same configs, same agents), keys each game by (DQN seat, full action sequence including
the opening), checks that the replayed outcome counts equal the file's, and reports the number
of distinct games and the win/draw/loss/truncation rates and Wilson 95% intervals over the
distinct games, next to the old ones. The merged evaluation files are not changed.

Example: python scripts/distinct_reanalysis_dqn.py
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from agents.dqn.agent import load_checkpoint  # noqa: E402
from evaluation.match_runner import MatchConfig, make_agent, run_match  # noqa: E402
from evaluation.metrics import wilson  # noqa: E402
from scripts.eval_dqn_final import cached_solve  # noqa: E402
from scripts.final_dqn_lib import RESULTS_DIR, provenance, write_result  # noqa: E402

FILES = (['release_dqn_3x3_eval.json'] + [f'final_dqn_3x3_s{s}_eval.json' for s in range(42, 47)]
         + ['release_dqn_4x4_eval.json', 'final_dqn_4x4_s42_eval.json',
            '5x5/final_dqn_5x5_s42_500k_eval.json'])
FORFEITS = ('timeout', 'illegal_action', 'no_action', 'agent_error')


def category(outcome, seat):
    c = outcome['category']
    if c == 'goal':
        return 'win' if outcome['winner'] == seat else 'loss'
    if c in FORFEITS:
        return ('forfeit_by_dqn_' if outcome['loser'] == seat else 'forfeit_by_opp_') + c
    return c


def rates(counter):
    n = sum(counter.values())
    draws = sum(v for k, v in counter.items() if k.startswith('draw_'))
    row = {'n': n}
    for name, k in (('win', counter.get('win', 0)), ('draw', draws),
                    ('truncated', counter.get('truncated', 0)), ('loss', counter.get('loss', 0))):
        row[f'{name}_rate'] = round(k / n, 4) if n else None
        row[f'{name}_ci95'] = [round(x, 4) for x in wilson(k, n)] if n else None
    return row


def seeds_of(d, opp):
    if 'openings' in d:
        return d['openings']['seeds'], d['openings']['opening_plies']
    lo, hi = d['matches'][opp]['opening_seeds']
    return list(range(lo, hi + 1)), d['provenance']['config']['opening_plies']


def reanalyse(name):
    d = json.loads((RESULTS_DIR / name).read_text(encoding='utf-8'))
    ckpt = d['provenance']['checkpoints'][0]['path']
    n, rules = d['board_size'], d['rules']
    dqn = {p: load_checkpoint(ckpt, player_id=p, expected_board_size=n) for p in (0, 1)}
    solutions = {}
    if 'exact' in d['matches']:
        sol, label, _ = cached_solve(n, rules)
        solutions[label] = sol
    out = {}
    for opp, m in d['matches'].items():
        seeds, plies = seeds_of(d, opp)
        all_games, distinct = Counter(), {}
        per_seat_distinct = {0: {}, 1: {}}
        for seed in seeds:
            for seat in (0, 1):
                cfg = MatchConfig(board_size=n, rules=rules, p0='dqn' if seat == 0 else opp,
                                  p1=opp if seat == 0 else 'dqn', max_plies=200,
                                  opening_plies=plies, opening_seed=seed)
                opp_agent, opp_id = make_agent(opp, 1 - seat, cfg, solutions, seed)
                res = run_match(cfg, agents={seat: dqn[seat], 1 - seat: opp_agent},
                                identities={seat: {'name': 'dqn'}, 1 - seat: opp_id})
                cat = category(res['outcome'], seat)
                all_games[cat] += 1
                key = (seat, tuple((s['action'][0], tuple(s['action'][1])) for s in res['replay']))
                distinct.setdefault(key, cat)
                per_seat_distinct[seat].setdefault(key, cat)
        stored = Counter({k: v for k, v in m['overall']['categories'].items() if v})
        reproduced = (Counter({k: v for k, v in all_games.items() if v}) == stored)
        old = m['overall']
        out[opp] = {
            'games': sum(all_games.values()),
            'distinct_games': len(distinct),
            'replay_reproduces_stored_counts': reproduced,
            'old_all_games': {'win_rate': old['win_rate'], 'win_rate_ci95': old['win_rate_ci95'],
                              'draw_rate': old['draw_rate'], 'truncation_rate': old['truncation_rate'],
                              'loss_rate': old['loss_rate'],
                              'status': 'superseded: Wilson interval counts repeated games as independent'},
            'distinct': {'categories': dict(Counter(distinct.values())), **rates(Counter(distinct.values()))},
            'distinct_as_p0': rates(Counter(per_seat_distinct[0].values())),
            'distinct_as_p1': rates(Counter(per_seat_distinct[1].values())),
        }
        print(f"{name} {opp}: {out[opp]['games']} games, {out[opp]['distinct_games']} distinct, "
              f"reproduced={reproduced}", flush=True)
    return {'source_file': f'experiments/results/final_dqn/{name}', 'checkpoint': ckpt,
            'checkpoint_sha256': d['provenance']['checkpoints'][0]['sha256'], 'board_size': n,
            'rules': rules, 'by_opponent': out}


def main():
    torch.set_num_threads(2)
    t0 = time.time()
    results = {name: reanalyse(name) for name in FILES}
    payload = {
        'provenance': provenance({'files': FILES, 'distinct_key': '(dqn seat, full action sequence incl. opening)',
                                  'max_plies': 200},
                                 seeds=None, checkpoints=[r['checkpoint'] for r in results.values()]),
        'note': ('Rates and Wilson 95% intervals over distinct games replace the all-games ones, which '
                 'count each repeat of a deterministic game as new evidence. Each distinct game counts once.'),
        'all_reproduced': all(o['replay_reproduces_stored_counts']
                              for r in results.values() for o in r['by_opponent'].values()),
        'elapsed_seconds': round(time.time() - t0, 1),
        'results': results,
    }
    print('wrote', write_result('distinct_reanalysis.json', payload), 'all_reproduced', payload['all_reproduced'])


if __name__ == '__main__':
    main()
