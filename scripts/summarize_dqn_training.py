#!/usr/bin/env python3
"""Summarise one training run's train_log.jsonl into experiments/results/final_dqn/ (Task 6).

Records: loss curve (mean per-episode loss in bins of --bin env steps), Q-probe curve and
in-training win rates at every eval, episode-length curve, the measured opponent mix
(episodes and env steps per opponent type, before and after the first snapshot), terminal
reasons, and run speed.

Example: python scripts/summarize_dqn_training.py --run checkpoints/final/3x3_s42
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.final_dqn_lib import provenance, write_result  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--run', required=True)
    ap.add_argument('--bin', type=int, default=5000)
    args = ap.parse_args()
    run = Path(args.run)
    rows = [json.loads(line) for line in open(run / 'train_log.jsonl', encoding='utf-8') if line.strip()]
    eps = [r for r in rows if r['type'] == 'episode']
    evals = [r for r in rows if r['type'] == 'eval']
    run_cfg = json.loads((run / 'run_config.json').read_text()) if (run / 'run_config.json').exists() else None
    snap = (run_cfg or {}).get('config', {}).get('snapshot_interval', 20000)

    bins = defaultdict(lambda: {'loss': [], 'len': []})
    mix = {'before_first_snapshot': Counter(), 'after_first_snapshot': Counter()}
    steps_by = {'before_first_snapshot': Counter(), 'after_first_snapshot': Counter()}
    reasons = Counter()
    for r in eps:
        b = bins[(r['env_steps'] - 1) // args.bin]
        if r['avg_loss'] is not None:
            b['loss'].append(r['avg_loss'])
        b['len'].append(r['length'])
        phase = 'before_first_snapshot' if r['env_steps'] - r['length'] < snap else 'after_first_snapshot'
        mix[phase][r['opponent']] += 1
        steps_by[phase][r['opponent']] += r['length']
        reasons[str(r['terminal_reason'])] += 1

    def frac(c):
        n = sum(c.values())
        return {k: round(v / n, 4) for k, v in sorted(c.items())} if n else {}

    curve = [{'env_steps_upto': (k + 1) * args.bin,
              'mean_loss': round(sum(v['loss']) / len(v['loss']), 5) if v['loss'] else None,
              'episodes': len(v['len']),
              'mean_episode_length': round(sum(v['len']) / len(v['len']), 2)}
             for k, v in sorted(bins.items())]
    out = {
        'run_dir': run.as_posix(),
        'run_config': run_cfg,
        'episodes': len(eps),
        'env_steps': eps[-1]['env_steps'] if eps else 0,
        'grad_steps': eps[-1]['grad_steps'] if eps else 0,
        'wall_seconds': evals[-1].get('elapsed_seconds') if evals else None,
        'opponent_mix_episodes': {p: {'counts': dict(c), 'fractions': frac(c)} for p, c in mix.items()},
        'opponent_mix_env_steps': {p: {'counts': dict(c), 'fractions': frac(c)} for p, c in steps_by.items()},
        'terminal_reasons': dict(reasons),
        'loss_curve': curve,
        'eval_curve': [{'env_steps': e['env_steps'], 'grad_steps': e['grad_steps'],
                        'win_vs_greedy': e['eval']['vs_greedy']['win_rate'],
                        'win_vs_random': e['eval']['vs_random']['win_rate'],
                        'q_probe': e.get('q_probe'), 'elapsed_seconds': e.get('elapsed_seconds')}
                       for e in evals],
        'note': ('in-training eval uses opening_plies 0 and merges truncation into draws '
                 '(scripts/train_dqn.evaluate_agent); use the *_eval.json files for results.'),
    }
    name = f"training_curves_{run.name}.json"
    ckpt = run / 'model.pt'
    print('wrote', write_result(name, {'provenance': provenance({'bin': args.bin}, seeds=[(run_cfg or {}).get('config', {}).get('seed')],
                                                                    checkpoints=[ckpt] if ckpt.exists() else []), **out}))


if __name__ == '__main__':
    main()
