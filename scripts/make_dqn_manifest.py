#!/usr/bin/env python3
"""Write checkpoints/final/<run>/manifest.json for a finished training run.

The manifest is the committed record of a git-ignored checkpoint: training commit and
dirty flag, full config, seed, exact command, steps, SHA-256 of final.pt, torch and
Python versions, and wall time.

Runs started before train_dqn.py recorded git state in run_config.json need
``--train-commit`` and ``--train-dirty`` (taken from the commit log at the run's start).

Example:
  python scripts/make_dqn_manifest.py --run checkpoints/final/3x3_s42 --train-commit d251df3 --train-dirty false
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.final_dqn_lib import git_state, sha256  # noqa: E402


def same_weights(a: Path, b: Path):
    if not b.is_file():
        return None
    import torch
    sa, sb = torch.load(a, weights_only=True), torch.load(b, weights_only=True)
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--run', required=True)
    ap.add_argument('--train-commit', default=None)
    ap.add_argument('--train-dirty', choices=['true', 'false'], default=None)
    ap.add_argument('--note', default=None)
    ap.add_argument('--kept-resume', action='store_true',
                    help='the run ended with --keep-resume; record resume.pt instead of refusing')
    args = ap.parse_args()
    run = Path(args.run)
    rc = json.loads((run / 'run_config.json').read_text(encoding='utf-8'))
    meta = json.loads((run / 'meta.json').read_text(encoding='utf-8'))
    rows = [json.loads(line) for line in open(run / 'train_log.jsonl', encoding='utf-8') if line.strip()]
    evals = [r for r in rows if r['type'] == 'eval']
    final = run / 'final.pt'
    resume = run / 'resume.pt'
    if not final.is_file() or (resume.exists() and not args.kept_resume):
        raise SystemExit(f"{run}: run not finished (final.pt missing, or resume.pt present "
                         f"without --kept-resume)")
    git = rc.get('git') or {}
    commit = args.train_commit or git.get('commit')
    dirty = {'true': True, 'false': False}.get(args.train_dirty, git.get('dirty'))
    if commit is None or dirty is None:
        raise SystemExit("training commit/dirty unknown: pass --train-commit and --train-dirty")
    import torch
    manifest = {
        'checkpoint': (run / 'final.pt').as_posix(),
        'sha256': sha256(final),
        'size_bytes': final.stat().st_size,
        # train_dqn.py saves the same weights twice; the evaluation JSONs cite model.pt
        'model_pt_sha256': sha256(run / 'model.pt') if (run / 'model.pt').is_file() else None,
        'model_pt_same_weights': same_weights(final, run / 'model.pt'),
        'train_commit': commit,
        'train_dirty': dirty,
        'train_git_source': 'run_config.json' if git else 'commit log at run start (--train-commit)',
        'seed': rc['config'].get('seed'),
        'steps': meta['step'],
        'command': rc['command'],
        'config': rc['config'],
        'started': rc.get('started'),
        'wall_seconds': evals[-1].get('elapsed_seconds') if evals else None,
        'torch': rc.get('torch', torch.__version__),
        'python': rc.get('python', sys.version.split()[0]),
        'meta': meta,
        'manifest_written_at_commit': git_state(),
    }
    if args.kept_resume:
        # full state (networks, optimizer, buffer, pool, RNG) kept for a later --resume
        manifest['resume_state'] = {'path': resume.as_posix(), 'sha256': sha256(resume),
                                    'size_bytes': resume.stat().st_size}
    if args.note:
        manifest['note'] = args.note
    out = run / 'manifest.json'
    out.write_text(json.dumps(manifest, indent=2, default=str) + '\n', encoding='utf-8')
    print(out, manifest['sha256'], manifest['steps'], manifest['wall_seconds'])


if __name__ == '__main__':
    main()
