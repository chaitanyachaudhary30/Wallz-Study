"""Agent discovery, compatibility gating, and factory registry (P08).

Discovers installed agents, inspects checkpoint metadata for exact compatibility
matching, and builds agents dynamically for GameSessions.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from evaluation.match_runner import MatchConfig, make_agent
from src.modified_wallz.config import WallzConfig


# Cached models for DQN to avoid reloading weights repeatedly
_DQN_MODEL_CACHE: Dict[str, Any] = {}
# Cached exact solver solutions keyed by config label
_EXACT_SOLVE_CACHE: Dict[str, Any] = {}


def presets() -> List[Dict[str, Any]]:
    """Returns canonical configuration presets for quick selection in the UI."""
    return [
        {
            "id": "default_5x5",
            "label": "Standard 5×5 (Final Rules)",
            "config": {"board_size": 5, "A": True, "B": True, "C": True, "K": 3, "M": WallzConfig().M},
        },
        {
            "id": "dqn_3x3",
            "label": "DQN 3×3 (K=1, M=1)",
            "config": {"board_size": 3, "A": True, "B": True, "C": True, "K": 1, "M": 1},
        },
        {
            "id": "dqn_4x4",
            "label": "DQN 4×4 (K=1, M=1)",
            "config": {"board_size": 4, "A": True, "B": True, "C": True, "K": 1, "M": 1},
        },
        {
            "id": "classic_9x9",
            "label": "Classic 9×9 (Base Rules)",
            "config": {"board_size": 9, "A": False, "B": False, "C": False, "K": 1, "M": WallzConfig().M},
        },
    ]


def list_agents(config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
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
            "available": (n <= 4),
            "reason": "only available for N <= 4 with exact retrograde solve" if n > 4 else None,
            "explainable": True,
        },
    ]

    # Discover DQN checkpoints
    torch_available = True
    try:
        import torch  # noqa: F401
    except ImportError:
        torch_available = False

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    checkpoint_dirs = [
        os.path.join(repo_root, "checkpoints", "release", "*"),
        os.path.join(repo_root, "checkpoints", "dqn_*"),
    ]

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

            target_pt = os.path.join(cdir, "model.pt")
            if not os.path.exists(target_pt):
                target_pt = os.path.join(cdir, "final.pt")
            if not os.path.exists(target_pt):
                pt_files = sorted(glob.glob(os.path.join(cdir, "*.pt")))
                if pt_files:
                    target_pt = pt_files[-1]
                else:
                    continue

            rel_pt = os.path.relpath(target_pt, repo_root)
            dir_name = os.path.basename(cdir)
            dqn_spec = f"dqn:path={rel_pt}"

            meta_n = meta.get("N")
            meta_rules = meta.get("rules", {})
            compat = (meta_n == n and meta_rules == rules)

            reason = None
            if not torch_available:
                compat = False
                reason = "PyTorch is not installed in the environment"
            elif not compat:
                rule_str = f"N{meta_n} A{int(meta_rules.get('A', True))}B{int(meta_rules.get('B', True))}C{int(meta_rules.get('C', True))} K{meta_rules.get('K', 1)} M{meta_rules.get('M', 1)}"
                reason = f"trained for {rule_str}"

            agents.append({
                "spec": dqn_spec,
                "name": f"DQN ({dir_name})",
                "family": "dqn",
                "label": f"DQN ({dir_name})",
                "available": compat,
                "reason": reason,
                "explainable": True,
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

    agent, _ = make_agent(
        spec=spec,
        player_id=player,
        config=match_cfg,
        solutions=_EXACT_SOLVE_CACHE,
    )
    return agent
