"""Shared helpers for the Stage 2 DQN validation (experiments/results/final_dqn/).

Provenance for every result JSON, random position generation, and two exact state
transforms built from engine raw keys (role swap and horizontal mirror). The engine is
the only source of legality: positions come from ``get_legal_actions`` + ``apply_action``
and transformed positions are loaded with ``restore_state_key``.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path

from wallz_engine import WallzEngine

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "experiments" / "results" / "final_dqn"

# Rule cells used by the compatibility and encoding audits.
RULE_CELLS = {
    "default_K3_M10": {"A": True, "B": True, "C": True, "K": 3, "M": 10},
    "train3_K1_M1": {"A": True, "B": True, "C": True, "K": 1, "M": 1},
    "train4_K1_M0": {"A": True, "B": True, "C": True, "K": 1, "M": 0},
    "K2_M3": {"A": True, "B": True, "C": True, "K": 2, "M": 3},
    "A0_K1_M2": {"A": False, "B": True, "C": True, "K": 1, "M": 2},
    "B0C0_M2": {"A": True, "B": False, "C": False, "K": 3, "M": 2},
    "all_off_M0": {"A": False, "B": False, "C": False, "K": 1, "M": 0},
}


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_state() -> dict:
    """Commit and dirty flag of tracked files (untracked result files do not count)."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                    cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def provenance(config: dict, seeds, checkpoints=()) -> dict:
    """Provenance block required on every final_dqn result JSON."""
    try:
        import torch
        torch_version = torch.__version__
    except ImportError:
        torch_version = None
    import numpy
    ckpts = []
    for p in checkpoints:
        p = Path(p)
        ckpts.append({"path": p.as_posix(), "sha256": sha256(p) if p.is_file() else None})
    return {
        "git": git_state(),
        "command": "python " + " ".join(sys.argv),
        "config": config,
        "seeds": list(seeds) if seeds is not None else None,
        "checkpoints": ckpts,
        "python": sys.version.split()[0],
        "torch": torch_version,
        "numpy": numpy.__version__,
        "platform": platform.platform(),
        "date": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def write_result(name: str, payload: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / name
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False, default=str)
        fh.write("\n")
    return out


# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------

def engine_accepts(engine: WallzEngine, action) -> bool:
    """True iff the engine's own validator accepts ``action`` (tried on a clone)."""
    trial = engine.clone()
    try:
        trial.apply_action(action, update_clock=False)
    except ValueError:
        return False
    return True


def random_positions(n: int, rules: dict, seed: int, plies: int = 40, biased: bool = True):
    """Yield every non-terminal engine position of one random playout.

    ``biased`` first picks an action type uniformly among the types present, then an
    action of that type, so HOP, REPLACE and BLOCK appear far more often than under
    uniform play.
    """
    rng = random.Random(seed)
    engine = WallzEngine(board_size=n, rules=rules)
    for _ in range(plies):
        if engine.is_terminal():
            return
        legal = engine.get_legal_actions()
        if not legal:
            return
        yield engine
        if biased:
            types = sorted({a[0] for a in legal})
            t = rng.choice(types)
            action = rng.choice([a for a in legal if a[0] == t])
        else:
            action = rng.choice(legal)
        engine.apply_action(action, update_clock=False)


def _flip_cell(cell, n):
    return (n - 1 - cell[0], cell[1])


def _flip_wall(w, n):
    return None if w is None else (n - 2 - w[0], w[1], w[2])


def swap_roles_key(key: tuple, n: int) -> tuple:
    """Raw key of the same position with the players' roles exchanged.

    Rows are flipped (r -> N-1-r, wall rows r -> N-2-r) and every per-player field moves
    to the other seat, so player 1's view of ``key`` is player 0's view of the result.
    """
    (p0, p1, w0, w1, i0, i1, l0, l1, h0, h1, b0, b1, blocks, to_move) = key
    fw = lambda ws: tuple(_flip_wall(w, n) for w in ws)
    nblocks = tuple(sorted((_flip_cell(c, n), t) for c, t in blocks))
    return (_flip_cell(p1, n), _flip_cell(p0, n), fw(w1), fw(w0), i1, i0,
            _flip_wall(l1, n), _flip_wall(l0, n), h1, h0, b1, b0, nblocks, 1 - to_move)


def mirror_key(key: tuple, n: int) -> tuple:
    """Raw key of the horizontally mirrored position (c -> N-1-c, wall c -> N-2-c)."""
    (p0, p1, w0, w1, i0, i1, l0, l1, h0, h1, b0, b1, blocks, to_move) = key
    mc = lambda c: (c[0], n - 1 - c[1])
    mw = lambda w: None if w is None else (w[0], n - 2 - w[1], w[2])
    nblocks = tuple(sorted((mc(c), t) for c, t in blocks))
    return (mc(p0), mc(p1), tuple(mw(w) for w in w0), tuple(mw(w) for w in w1), i0, i1,
            mw(l0), mw(l1), h0, h1, b0, b1, nblocks, to_move)


def restore(n: int, rules: dict, key: tuple) -> WallzEngine:
    return WallzEngine(board_size=n, rules=rules).restore_state_key(key)
