"""Agent discovery, compatibility gating, and factory registry (P08).

Discovers installed agents, inspects checkpoint metadata for exact compatibility
matching, and builds agents dynamically for GameSessions.
"""
from __future__ import annotations

import glob
import json
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from evaluation.match_runner import AgentUnavailable, MatchConfig, make_agent
from src.modified_wallz.config import WallzConfig


# Cached models for DQN to avoid reloading weights repeatedly
_DQN_MODEL_CACHE: Dict[str, Any] = {}
# Cached exact solver solutions keyed by config label
_EXACT_SOLVE_CACHE: Dict[str, Any] = {}

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _release_metas() -> List[Dict[str, Any]]:
    """meta.json of every release checkpoint, sorted by board size."""
    metas = []
    for meta_path in sorted(glob.glob(os.path.join(REPO_ROOT, "checkpoints", "release", "*", "meta.json"))):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                metas.append(json.load(f))
        except Exception:
            continue
    return sorted(metas, key=lambda m: m.get("N", 0))


def presets() -> List[Dict[str, Any]]:
    """Canonical configuration presets for quick selection in the UI.

    DQN presets are derived from the release checkpoints' meta.json so that the
    preset always matches the exact rules the network was trained on.
    """
    out = [
        {
            "id": "default_5x5",
            "label": "Standard 5×5 (Final Rules)",
            "config": {"board_size": 5, "A": True, "B": True, "C": True, "K": 3, "M": WallzConfig().M},
        },
    ]
    for meta in _release_metas():
        n = meta.get("N")
        r = meta.get("rules", {})
        if n is None or not r:
            continue
        out.append({
            "id": f"dqn_{n}x{n}",
            "label": f"DQN {n}×{n} (K={r.get('K')}, M={r.get('M')})",
            "config": {"board_size": n, "A": bool(r.get("A")), "B": bool(r.get("B")),
                       "C": bool(r.get("C")), "K": int(r.get("K")), "M": int(r.get("M"))},
        })
    out.append({
        "id": "classic_9x9",
        "label": "Classic 9×9 (Base Rules)",
        "config": {"board_size": 9, "A": False, "B": False, "C": False, "K": 1, "M": WallzConfig().M},
    })
    return out


# -----------------------------------------------------------------------------
# Exact solver availability
# -----------------------------------------------------------------------------
# Measured (experiments/results/solves, 2026-11-08): every N3/N4 cell with M=0 solves
# exactly in <= 40 s; N3 with M=1 needs minutes and several hundred MB; anything with
# M >= 2 or N >= 5 does not finish (3x3 M=10 exceeds 300k states in 55 s).
EXACT_CAPS = dict(max_states=2_000_000, max_seconds=900.0)
EXACT_CACHE_DIR = os.path.join(REPO_ROOT, "experiments", "cache", "exact")
_EXACT_LOCK = threading.Lock()
_EXACT_LOAD_LOCK = threading.Lock()
# label -> {"status": "solving"|"ready"|"failed", "started", "error", "proc"}
_EXACT_JOBS: Dict[str, Dict[str, Any]] = {}
_SOURCE_FINGERPRINT: Optional[str] = None


def _match_config(board_size: int, rules: Dict[str, Any]) -> MatchConfig:
    return MatchConfig(board_size=board_size, rules=dict(rules))


def exact_supported(board_size: int, rules: Dict[str, Any]) -> Optional[str]:
    """None if an exact solve is feasible for this config, else the reason it is not."""
    if board_size > 4:
        return "state space too large for exact solving (supported: N <= 4 with M <= 1)"
    if rules.get("M", 0) > 1:
        return "state space too large for exact solving with M >= 2 (supported: N <= 4 with M <= 1)"
    return None


def _source_fingerprint() -> str:
    """Hash of the code that defines solutions: a rule change invalidates cached solves."""
    global _SOURCE_FINGERPRINT
    if _SOURCE_FINGERPRINT is None:
        import hashlib
        h = hashlib.sha1()
        for rel in ("wallz_engine.py", "src/modified_wallz/board.py", "src/modified_wallz/pathfinding.py",
                    "src/modified_wallz/config.py", "evaluation/state_graph.py", "evaluation/retrograde.py"):
            with open(os.path.join(REPO_ROOT, rel), "rb") as f:
                h.update(f.read())
        _SOURCE_FINGERPRINT = h.hexdigest()[:12]
    return _SOURCE_FINGERPRINT


def _cache_path(label: str) -> str:
    return os.path.join(EXACT_CACHE_DIR, f"{label}__{_source_fingerprint()}.pkl")


def _solve_to_file(board_size: int, rules: Dict[str, Any], caps: Dict[str, Any], path: str) -> None:
    """Child-process entry point: solve and write the Solution (or a failure note) to disk.

    Runs in its own process so the CPU-bound pure-Python search never holds the web
    server's GIL (in-process it slowed the first API calls from ~0.7 s to ~47 s)."""
    import pickle
    from evaluation.retrograde import solve_config
    from evaluation.state_graph import GraphCaps
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sol = solve_config(board_size, rules, GraphCaps(**caps))
    if not sol.exact:
        with open(path + ".failed", "w", encoding="utf-8") as f:
            f.write(f"solve hit cap '{sol.graph.stop_reason}' after {len(sol.graph.keys):,} states")
        return
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "wb") as f:
        pickle.dump(sol, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)  # atomic: readers never see a half-written file


def _load_solution(label: str) -> Any:
    """Load a solved config from the disk cache into memory (once)."""
    import pickle
    with _EXACT_LOAD_LOCK:
        if label not in _EXACT_SOLVE_CACHE:
            with open(_cache_path(label), "rb") as f:
                _EXACT_SOLVE_CACHE[label] = pickle.load(f)
    return _EXACT_SOLVE_CACHE[label]


def exact_status(board_size: int, rules: Dict[str, Any], start: bool = True) -> Tuple[bool, Optional[str]]:
    """(available, reason). Never blocks: solves run in a child process and are cached on
    disk (experiments/cache/exact), so a solved config is available instantly after a
    restart. Loading into memory happens only when an exact agent is actually built."""
    reason = exact_supported(board_size, rules)
    if reason:
        return False, reason
    label = _match_config(board_size, rules).label()
    path = _cache_path(label)
    if label in _EXACT_SOLVE_CACHE or os.path.exists(path):
        return True, None
    if os.path.exists(path + ".failed"):
        with open(path + ".failed", encoding="utf-8") as f:
            return False, f"exact solve failed: {f.read().strip()}"
    with _EXACT_LOCK:
        job = _EXACT_JOBS.get(label)
        if job is not None and job["status"] == "solving" and not job["proc"].is_alive():
            job["proc"].join()
            if os.path.exists(path):
                job["status"] = "ready"
            else:
                job.update(status="failed", error=f"solver process exited with code {job['proc'].exitcode}")
        if job is None and start:
            running = [k for k, j in _EXACT_JOBS.items() if j["status"] == "solving" and j["proc"].is_alive()]
            if running:
                return False, f"waiting: another exact solve ({running[0]}) is running"
            import multiprocessing as mp
            # spawn, not fork: forking a threaded server process is unsafe and copies its memory
            proc = mp.get_context("spawn").Process(
                target=_solve_to_file, args=(board_size, dict(rules), dict(EXACT_CAPS), path),
                daemon=True, name=f"exact-{label}")
            proc.start()
            job = _EXACT_JOBS[label] = {"status": "solving", "started": time.time(), "error": None, "proc": proc}
    if job is None:
        return False, "exact solution not computed yet"
    if job["status"] == "ready":
        return True, None
    if job["status"] == "solving":
        return False, f"solving in background ({time.time() - job['started']:.0f}s so far) — refresh shortly"
    return False, f"exact solve failed: {job['error']}"


def wait_for_exact(board_size: int, rules: Dict[str, Any], timeout: float = 900.0) -> bool:
    """Block until the exact solution for a config is ready (used by tests and warm-up)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        ok, reason = exact_status(board_size, rules)
        if ok:
            return True
        if reason and reason.startswith(("exact solve failed", "state space")):
            return False
        time.sleep(0.5)
    return False


def warm_exact_presets() -> None:
    """Pre-solve preset configs in a child process (sequentially), so the server stays
    responsive. Already-cached solutions cost nothing."""
    def _run():
        for p in presets():
            c = p["config"]
            rules = {k: c[k] for k in ("A", "B", "C", "K", "M")}
            if exact_supported(c["board_size"], rules) is None:
                wait_for_exact(c["board_size"], rules)
    threading.Thread(target=_run, daemon=True, name="exact-warmup").start()


def list_agents(config: Optional[Dict[str, Any]] = None, start_exact: bool = True) -> List[Dict[str, Any]]:
    """Returns catalog of registered agent specs with compatibility flags for `config`."""
    if config is None:
        config = {"board_size": 5, "A": True, "B": True, "C": True, "K": 3, "M": WallzConfig().M}

    n = config.get("board_size", 5)
    rules = {
        "A": config.get("A", True),
        "B": config.get("B", True),
        "C": config.get("C", True),
        "K": config.get("K", 3),
        "M": config.get("M", WallzConfig().M),
    }

    exact_ok, exact_reason = exact_status(n, rules, start=start_exact)

    agents: List[Dict[str, Any]] = [
        {
            "spec": "human",
            "name": "Human Player",
            "family": "human",
            "label": "Human (Interactive)",
            "available": True,
            "reason": None,
            "explainable": False,
        },
        {
            "spec": "random:seed=0",
            "name": "Random Agent",
            "family": "random",
            "label": "Random (Uniform)",
            "available": True,
            "reason": None,
            "explainable": False,
        },
        {
            "spec": "greedy",
            "name": "Spatial Greedy",
            "family": "greedy",
            "label": "Greedy (Tier-1 Baseline)",
            "available": True,
            "reason": None,
            "explainable": True,
        },
        {
            "spec": "greedy:seed=1",
            "name": "Seeded Greedy",
            "family": "greedy",
            "label": "Greedy (Seeded Tie-Break)",
            "available": True,
            "reason": None,
            "explainable": True,
        },
        {
            "spec": "minimax:depth=1",
            "name": "Minimax (Depth 1)",
            "family": "minimax",
            "label": "Minimax (Depth 1)",
            "available": True,
            "reason": None,
            "explainable": True,
        },
        {
            "spec": "minimax:depth=2",
            "name": "Minimax (Depth 2)",
            "family": "minimax",
            "label": "Minimax (Depth 2)",
            "available": (n < 7),
            "reason": "too slow for interactive play" if n >= 7 else None,
            "explainable": True,
        },
        {
            "spec": "minimax:depth=3,time_limit=2.0",
            "name": "Minimax (Depth 3)",
            "family": "minimax",
            "label": "Minimax (Depth 3, 2s limit)",
            "available": (n < 7),
            "reason": "too slow for interactive play" if n >= 7 else None,
            "explainable": True,
        },
        {
            "spec": "exact",
            "name": "Exact Solver",
            "family": "exact",
            "label": "Exact Retrograde Solver (Oracle)",
            "available": exact_ok,
            "reason": exact_reason,
            "explainable": True,
        },
    ]

    # Discover DQN checkpoints
    # find_spec checks that torch is installed without importing it (import costs ~1.1 s;
    # torch is imported only when a DQN agent is actually built)
    import importlib.util
    torch_available = importlib.util.find_spec("torch") is not None

    checkpoint_dirs = [
        os.path.join(REPO_ROOT, "checkpoints", "release", "*"),
        os.path.join(REPO_ROOT, "checkpoints", "dqn_*"),
    ]

    def _step_of(path: str) -> int:
        m = re.search(r"step_(\d+)\.pt$", path)
        return int(m.group(1)) if m else -1

    for cdir_pattern in checkpoint_dirs:
        for cdir in sorted(glob.glob(cdir_pattern)):
            meta_path = os.path.join(cdir, "meta.json")
            if not os.path.exists(meta_path):
                continue
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                continue

            final_pt = None
            for cand in ("model.pt", "final.pt"):
                if os.path.exists(os.path.join(cdir, cand)):
                    final_pt = os.path.join(cdir, cand)
                    break
            step_pts = sorted(glob.glob(os.path.join(cdir, "step_*.pt")), key=_step_of)
            if final_pt is None and step_pts:
                final_pt = step_pts.pop()          # latest step stands in for the final model
            if final_pt is None:
                continue
            is_release = os.path.basename(os.path.dirname(cdir)) == "release"
            # Final model first, then intermediate snapshots (checkpoint time-lapse)
            # (skip the snapshot whose step equals the final model's step: same weights)
            entries = [(final_pt, None)] + ([] if is_release else
                                            [(p, _step_of(p)) for p in step_pts if _step_of(p) != meta.get("step")])

            meta_n = meta.get("N")
            meta_rules = meta.get("rules", {})
            compat = (meta_n == n and meta_rules == rules)
            reason = None
            if not torch_available:
                compat = False
                reason = "PyTorch is not installed in the environment"
            elif not compat:
                rule_str = (f"N{meta_n} A{int(meta_rules.get('A', True))}B{int(meta_rules.get('B', True))}"
                            f"C{int(meta_rules.get('C', True))} K{meta_rules.get('K')} M{meta_rules.get('M')}")
                reason = f"trained for {rule_str}"

            dir_name = os.path.basename(cdir)
            for pt, step in entries:
                rel_pt = os.path.relpath(pt, REPO_ROOT)
                label = f"DQN ({dir_name})" if step is None else f"DQN ({dir_name} @ step {step:,})"
                agents.append({
                    "spec": f"dqn:path={rel_pt}",
                    "name": label,
                    "family": "dqn",
                    "label": label,
                    "available": compat,
                    "reason": reason,
                    "explainable": True,
                    "checkpoint_step": step if step is not None else meta.get("step"),
                    "is_snapshot": step is not None,
                })

    return agents


def build_agent(spec: str, player: int, config: Dict[str, Any]) -> Any:
    """Builds an agent instance according to spec for the specified player and configuration."""
    if spec == "human":
        return None

    board_size = config.get("board_size", 5)
    rules = {
        "A": config.get("A", True),
        "B": config.get("B", True),
        "C": config.get("C", True),
        "K": config.get("K", 3),
        "M": config.get("M", WallzConfig().M),
    }
    match_cfg = MatchConfig(board_size=board_size, rules=rules)

    if spec.split(":", 1)[0] == "exact":
        # Never solve inside a request: only use a solution the background job finished.
        ok, reason = exact_status(board_size, rules)
        if not ok:
            raise AgentUnavailable(f"Exact solver unavailable: {reason}")
        _load_solution(match_cfg.label())

    agent, _ = make_agent(
        spec=spec,
        player_id=player,
        config=match_cfg,
        solutions=_EXACT_SOLVE_CACHE,
    )
    return agent
