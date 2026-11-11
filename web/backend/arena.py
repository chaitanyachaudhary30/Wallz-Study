"""AI vs AI Arena batch tournament runner (P10 / Contract C-API).

Runs synchronous agent matches with color-alternation and balanced opening seeds,
computes Wilson 95% confidence intervals, and stores replay traces in an LRU cache.
"""
from __future__ import annotations

import collections
import statistics
import uuid
from typing import Any, Dict, List, Optional, Tuple

from evaluation.match_runner import MatchConfig, run_match
from evaluation.metrics import wilson
from src.modified_wallz.config import WallzConfig


class ArenaStore:
    """In-memory LRU store for Arena batch tournament results."""

    def __init__(self, capacity: int = 16) -> None:
        self.capacity = capacity
        self.batches: collections.OrderedDict[str, Dict[str, Any]] = collections.OrderedDict()

    def put(self, batch_id: str, data: Dict[str, Any]) -> None:
        if batch_id in self.batches:
            self.batches.move_to_end(batch_id)
        self.batches[batch_id] = data
        if len(self.batches) > self.capacity:
            self.batches.popitem(last=False)

    def get(self, batch_id: str) -> Dict[str, Any]:
        if batch_id not in self.batches:
            raise KeyError(f"Arena batch '{batch_id}' not found")
        self.batches.move_to_end(batch_id)
        return self.batches[batch_id]


ARENA_STORE = ArenaStore(capacity=16)


def run_arena_batch(
    config_dict: Dict[str, Any],
    p0_spec: str,
    p1_spec: str,
    n: int = 10,
    seed: int = 42,
) -> Dict[str, Any]:
    """Runs a batch of AI vs AI matches with alternating colors and opening seeds."""
    if not (1 <= n <= 50):
        raise ValueError(f"Batch size n must be between 1 and 50, got {n}")

    board_size = int(config_dict.get("board_size", 5))
    # Hard cap check
    if board_size >= 9:
        is_deep_minimax_0 = "minimax" in p0_spec and ("depth=2" in p0_spec or "depth=3" in p0_spec)
        is_deep_minimax_1 = "minimax" in p1_spec and ("depth=2" in p1_spec or "depth=3" in p1_spec)
        if (is_deep_minimax_0 or is_deep_minimax_1) and n > 10:
            raise ValueError("Batch size capped at 10 for N=9 minimax depth >= 2 to maintain responsive execution")

    rules = {
        "A": bool(config_dict.get("A", True)),
        "B": bool(config_dict.get("B", True)),
        "C": bool(config_dict.get("C", True)),
        "K": int(config_dict.get("K", 3)),
        "M": int(config_dict.get("M", WallzConfig().M)),
    }
    match_cfg = MatchConfig(board_size=board_size, rules=rules, max_plies=200)

    records: List[Dict[str, Any]] = []
    categories: Dict[str, int] = collections.defaultdict(int)

    # truncated (runner ply cap) is its own outcome, never a draw
    spec_stats = {
        p0_spec: {"wins": 0, "draws": 0, "losses": 0, "truncated": 0},
        p1_spec: {"wins": 0, "draws": 0, "losses": 0, "truncated": 0},
    }

    for i in range(n):
        opening_seed = seed + (i // 2)
        if i % 2 == 0:
            m_p0, m_p1 = p0_spec, p1_spec
            p0_seat = 0
            p1_seat = 1
        else:
            m_p0, m_p1 = p1_spec, p0_spec
            p0_seat = 1
            p1_seat = 0

        match_cfg = MatchConfig(
            board_size=board_size,
            rules=rules,
            p0=m_p0,
            p1=m_p1,
            max_plies=200,
            opening_plies=2,
            opening_seed=opening_seed,
            record_replay=True,
        )
        res = run_match(match_cfg)
        records.append(res)

        # run_match reports the result under "outcome" (it has no top-level winner/reason)
        outcome = res["outcome"]
        winner = outcome["winner"]
        categories[outcome["category"]] += 1

        if outcome["category"] == "truncated":
            spec_stats[p0_spec]["truncated"] += 1
            spec_stats[p1_spec]["truncated"] += 1
        elif winner is None:
            spec_stats[p0_spec]["draws"] += 1
            spec_stats[p1_spec]["draws"] += 1
        elif winner == p0_seat:
            spec_stats[p0_spec]["wins"] += 1
            spec_stats[p1_spec]["losses"] += 1
        else:
            spec_stats[p1_spec]["wins"] += 1
            spec_stats[p0_spec]["losses"] += 1

    lengths = [r["length_plies"] for r in records]
    mean_length = round(statistics.mean(lengths), 2) if lengths else 0.0

    by_spec = {}
    for sp, st in spec_stats.items():
        w = st["wins"]
        d = st["draws"]
        l = st["losses"]
        wr = round(w / n, 4) if n > 0 else 0.0
        ci_lo, ci_hi = wilson(w, n)
        by_spec[sp] = {
            "wins": w,
            "draws": d,
            "losses": l,
            "truncated": st["truncated"],
            "win_rate": wr,
            "ci": [round(ci_lo, 4), round(ci_hi, 4)],
        }

    batch_id = uuid.uuid4().hex[:8]
    summary = {
        "batch_id": batch_id,
        "games": n,
        "config": {"board_size": board_size, **rules},
        "by_spec": by_spec,
        "mean_length": mean_length,
        "categories": dict(categories),
        "replays": [r.get("match_id", str(idx)) for idx, r in enumerate(records)],
    }

    ARENA_STORE.put(batch_id, {
        "summary": summary,
        "records": records,
    })

    return summary
