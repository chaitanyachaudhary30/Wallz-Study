#!/usr/bin/env python3
"""3x3 multi-seed stability study (P2.3 / X-11) on already trained DQN checkpoints.

For each seed (default 42-46) it loads ``<checkpoint-dir>/3x3_s<seed>/final.pt`` and
1. checks value-sign agreement and MAE against the exact N3 A1B1C1 K1 M1 solve (Gate X-16);
2. plays every distinct ``--opening-plies`` opening (first seed of each, up to
   ``--distinct-openings``) in both seatings against Greedy, and against Random with
   ``--random-seeds`` opponent seeds. DQN and Greedy are deterministic, so a repeated
   opening would only replay a game; no replicas are played.
Then it aggregates over seeds: mean, sample SD and a 95% t-interval per metric, pooled
Wilson intervals for rates, and leave-one-seed-out means.

Outcomes use evaluation/metrics.py categories: win, loss, draw (rule draw), truncated
(the runner's ply cap, never a draw), forfeit_win, forfeit_loss.

By default it never trains (``--no-train``): a missing checkpoint is an error. ``--train``
runs scripts/train_dqn.py for missing seeds, one at a time (DQN training is owned by the
DQN member; use it only when asked).

  python scripts/run_multi_seed_3x3.py --checkpoint-dir checkpoints/final \
      --out experiments/results/final_eval/multi_seed_3x3.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.match_runner import MatchConfig, run_match, run_metadata  # noqa: E402
from evaluation.metrics import CATEGORIES, outcome_for, wilson, sample_sd  # noqa: E402

RULES_3X3 = {'A': True, 'B': True, 'C': True, 'K': 1, 'M': 1}
CHECKPOINT_PATTERN = '3x3_s{seed}/final.pt'
OPPONENTS = ('greedy', 'random')
# Two-sided 95% Student t quantiles t_{0.975, df}, df = 1..30
T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
        2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
        2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]


def t975(df: int) -> float:
    # shortcut: normal quantile beyond df 30; more than 31 seeds is not expected here
    return T975[df - 1] if 1 <= df <= len(T975) else 1.96


# ---------------------------------------------------------------------------
# Pure aggregation (no torch); tested in tests/test_multi_seed_aggregation.py
# ---------------------------------------------------------------------------

def outcome_row(categories: List[str], lengths: List[int]) -> Dict[str, Any]:
    """Per-seed, per-opponent summary from the DQN agent's outcome categories."""
    n = len(categories)
    counts = {c: categories.count(c) for c in CATEGORIES}
    unknown = set(categories) - set(CATEGORIES)
    if unknown:
        raise ValueError(f"unknown outcome categories {sorted(unknown)}")
    row = {'n_games': n, 'counts': counts,
           'mean_length_plies': sum(lengths) / n if n else None}
    for name, cat in (('win_rate', 'win'), ('loss_rate', 'loss'), ('draw_rate', 'draw'),
                      ('truncation_rate', 'truncated')):
        row[name] = counts[cat] / n if n else None
        row[name + '_ci95'] = list(wilson(counts[cat], n))
    return row


def across_seeds(values: List[float]) -> Dict[str, Any]:
    """Mean, sample SD, 95% t-interval and leave-one-out means over per-seed values."""
    xs = [v for v in values if v is not None]
    n = len(xs)
    if n == 0:
        return {'n_seeds': 0, 'mean': None, 'sd': None, 'ci95_t': None, 'leave_one_out_means': []}
    m = sum(xs) / n
    sd = sample_sd(xs)
    ci = None if sd is None else [m - t975(n - 1) * sd / math.sqrt(n), m + t975(n - 1) * sd / math.sqrt(n)]
    loo = [(sum(xs) - x) / (n - 1) for x in xs] if n > 1 else []
    return {'n_seeds': n, 'mean': m, 'sd': sd, 'ci95_t': ci, 'min': min(xs), 'max': max(xs),
            'leave_one_out_means': loo,
            'max_leave_one_out_shift': max(abs(v - m) for v in loo) if loo else None}


def aggregate(seed_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    seed_results: [{'seed', 'sign_agreement', 'mae', 'vs': {opp: outcome_row(...)}}].
    Returns across-seed statistics per metric and pooled Wilson intervals per opponent.
    """
    out = {'seeds': [r['seed'] for r in seed_results],
           'sign_agreement': across_seeds([r.get('sign_agreement') for r in seed_results]),
           'mae': across_seeds([r.get('mae') for r in seed_results]),
           'vs': {}}
    opps = sorted({o for r in seed_results for o in r['vs']})
    for opp in opps:
        rows = [r['vs'][opp] for r in seed_results if opp in r['vs']]
        block = {m: across_seeds([row[m] for row in rows])
                 for m in ('win_rate', 'loss_rate', 'draw_rate', 'truncation_rate', 'mean_length_plies')}
        total = {c: sum(row['counts'][c] for row in rows) for c in CATEGORIES}
        n = sum(row['n_games'] for row in rows)
        block['pooled'] = {'n_games': n, 'counts': total}
        for name, cat in (('win_rate', 'win'), ('draw_rate', 'draw'), ('truncation_rate', 'truncated')):
            block['pooled'][name] = total[cat] / n if n else None
            block['pooled'][name + '_ci95'] = list(wilson(total[cat], n))
        out['vs'][opp] = block
    return out


# ---------------------------------------------------------------------------
# Experiment (needs torch and the checkpoints)
# ---------------------------------------------------------------------------

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def checkpoint_path(checkpoint_dir: str, seed: int) -> Path:
    return Path(checkpoint_dir) / CHECKPOINT_PATTERN.format(seed=seed)


def play_vs(dqn_spec: str, opp: str, opening_seeds: List[int], opening_plies: int, max_plies: int,
            opp_seeds=(None,)) -> Dict[str, Any]:
    """
    Every distinct opening x every opponent seed x both seatings. DQN and Greedy are
    deterministic, so repeating an opening against Greedy replays the same game; only
    distinct (opening, seat, opponent seed) triples are played. ``opp_seeds`` varies a
    random opponent ('random:seed=r'); None keeps the spec as given.
    """
    cats, lengths = [], []
    for oseed in opening_seeds:
        for r in opp_seeds:
            opp_spec = opp if r is None else f"{opp}:seed={r}"
            for dqn_seat in (0, 1):
                p0, p1 = (dqn_spec, opp_spec) if dqn_seat == 0 else (opp_spec, dqn_spec)
                res = run_match(MatchConfig(board_size=3, rules=RULES_3X3, p0=p0, p1=p1,
                                            max_plies=max_plies, opening_plies=opening_plies,
                                            opening_seed=oseed, record_replay=False))
                cats.append(outcome_for(res, dqn_seat))
                lengths.append(res['length_plies'])
    return outcome_row(cats, lengths)


def check_manifests(seeds, checkpoint_dir, manifest_dir) -> Dict[int, str]:
    """SHA-256 of each checkpoint must equal the committed manifest's ``sha256``.
    Returns {seed: sha256}; exits on a missing manifest or a mismatch."""
    shas = {}
    for s in seeds:
        mpath = checkpoint_path(manifest_dir, s).parent / 'manifest.json'
        if not mpath.exists():
            sys.exit(f"error: no manifest {mpath} for seed {s}")
        with open(mpath, encoding='utf-8') as fh:
            expected = json.load(fh)['sha256']
        actual = sha256(checkpoint_path(checkpoint_dir, s))
        if actual != expected:
            sys.exit(f"error: seed {s} checkpoint SHA-256 {actual} != manifest {expected} ({mpath})")
        shas[s] = actual
    return shas


def train_missing(seeds, checkpoint_dir, steps):
    for s in seeds:
        out_dir = checkpoint_path(checkpoint_dir, s).parent
        subprocess.run([sys.executable, 'scripts/train_dqn.py', '--board-size', '3', '--steps', str(steps),
                        '--seed', str(s), '--eval-freq', '10000', '--output-dir', str(out_dir)], check=True)


def run_study(args) -> Dict[str, Any]:
    meta = run_metadata()
    missing = [s for s in args.seeds if not checkpoint_path(args.checkpoint_dir, s).exists()]
    if missing and args.no_train:
        sys.exit(f"error: missing checkpoints for seeds {missing} under {args.checkpoint_dir} "
                 f"(pattern {CHECKPOINT_PATTERN}). This script does not train by default; "
                 f"get the checkpoints from the DQN member, or pass --train.")
    if missing:
        train_missing(missing, args.checkpoint_dir, args.steps)
    manifest_sha = check_manifests(args.seeds, args.checkpoint_dir, args.manifest_dir) if args.manifest_dir else {}

    import torch  # only needed from here on
    from agents.dqn.agent import load_checkpoint
    from evaluation.retrograde import solve_config
    from scripts.validate_dqn import evaluate_cell_x16

    from evaluation.ablation import first_distinct_openings
    from evaluation.match_runner import action_to_json

    picked = first_distinct_openings(3, args.opening_plies, args.distinct_openings,
                                     K=RULES_3X3['K'], M=RULES_3X3['M'])
    opening_seeds = [s for s, _ in picked]
    opening_map = {str(s): [action_to_json(a) for a in acts] for s, acts in picked}
    opp_seeds = {'greedy': (None,), 'random': tuple(range(args.random_seeds))}
    print(f"{len(opening_seeds)} distinct {args.opening_plies}-ply openings: seeds {opening_seeds}", flush=True)

    print("Solving N3 A1B1C1 K1 M1 exactly (about 1 minute, about 1 GB)...", flush=True)
    sol = solve_config(3, RULES_3X3)
    seed_results = []
    for seed in args.seeds:
        ckpt = checkpoint_path(args.checkpoint_dir, seed)
        agent = load_checkpoint(checkpoint_path=str(ckpt), player_id=0, expected_board_size=3,
                                expected_rules=RULES_3X3)
        x16 = evaluate_cell_x16(sol, agent)
        spec = f"dqn:path={ckpt.as_posix()}"
        vs = {opp: play_vs(spec, opp, opening_seeds, args.opening_plies, args.max_plies, opp_seeds[opp])
              for opp in OPPONENTS}
        seed_results.append({'seed': seed, 'checkpoint': ckpt.as_posix(), 'checkpoint_sha256': sha256(ckpt),
                             'sign_agreement': x16['sign_agreement_rate'], 'mae': x16['mae_all'],
                             'manifest_sha256_match': (manifest_sha.get(seed) == sha256(ckpt))
                             if manifest_sha else None,
                             'x16_cell_metrics': x16, 'vs': vs})
        print(f"seed {seed}: sign agreement {x16['sign_agreement_rate']}, "
              f"win vs greedy {vs['greedy']['win_rate']:.3f}, vs random {vs['random']['win_rate']:.3f}", flush=True)
    return {
        'source': meta['source'], 'run': dict(meta, torch=torch.__version__),
        'config': {k: v for k, v in vars(args).items()},
        'rules': RULES_3X3, 'checkpoint_pattern': CHECKPOINT_PATTERN,
        'opening_map': opening_map,
        'games_per_seed': {opp: len(opening_seeds) * len(opp_seeds[opp]) * 2 for opp in OPPONENTS},
        'note': 'max_plies is this study\'s own cap (old default 100, kept); other experiments use 200. '
                'Truncated games are a separate outcome, not draws.',
        'seeds': seed_results, 'aggregate': aggregate(seed_results),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    ap.add_argument('--checkpoint-dir', default='checkpoints/final',
                    help=f'directory holding {CHECKPOINT_PATTERN}')
    ap.add_argument('--manifest-dir', default=None,
                    help='directory holding 3x3_s<seed>/manifest.json; each checkpoint SHA-256 must match')
    ap.add_argument('--no-train', dest='no_train', action='store_true', default=True,
                    help='default: exit with an error if a checkpoint is missing')
    ap.add_argument('--train', dest='no_train', action='store_false',
                    help='train missing seeds with scripts/train_dqn.py (DQN member only)')
    ap.add_argument('--steps', type=int, default=50000, help='training steps, only with --train')
    ap.add_argument('--opening-plies', type=int, default=2,
                    help='random pawn plies before the agents play (2 = longest that cannot reach a goal on 3x3)')
    ap.add_argument('--distinct-openings', type=int, default=40,
                    help='use the first seed of each distinct opening, up to this many')
    ap.add_argument('--random-seeds', type=int, default=5,
                    help='random-opponent seeds per opening and seating')
    ap.add_argument('--max-plies', type=int, default=100,
                    help='ply cap per game; 100 is this study\'s historical default (runner default is 200)')
    ap.add_argument('--out', default='experiments/results/final_eval/multi_seed_3x3.json')
    args = ap.parse_args(argv)
    report = run_study(args)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as fh:
        json.dump(report, fh, indent=2)
    print(f"wrote {args.out}")


if __name__ == '__main__':
    main()
