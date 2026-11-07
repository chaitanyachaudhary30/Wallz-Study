"""Replay discovery, verification, and frame generation (P10 / Contract C-API).

Scans experiments/results/ for JSONL/JSON matches containing replay traces,
verifies determinism through step-by-step engine re-execution, and outputs
Contract C-STATE frames for interactive scrubbing in the frontend.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from evaluation.match_runner import MatchConfig
from evaluation.state_graph import state_id
from wallz_engine import WallzConfig, WallzEngine
from web.backend.serialize import parse_action, state_json


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "experiments" / "results"

# Cache for discovered replay index records keyed by file mtime
_REPLAY_INDEX_CACHE: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}


def _is_safe_results_path(rel_path: str) -> Path:
    """Validates that rel_path is strictly within experiments/results/ without directory traversal."""
    resolved = (RESULTS_DIR / rel_path).resolve()
    try:
        resolved.relative_to(RESULTS_DIR.resolve())
    except ValueError:
        raise ValueError(f"Path traversal detected: {rel_path} is outside {RESULTS_DIR}")
    return resolved


def _summary(rec: Dict[str, Any], rel_path: str, idx: int) -> Dict[str, Any]:
    """List-row summary of a run_match result (schema: evaluation/match_runner.run_match)."""
    outcome = rec.get("outcome") or {}
    cfg = rec.get("config") if isinstance(rec.get("config"), dict) else {}
    label = rec.get("config_label")
    if not label and cfg:
        label = MatchConfig(board_size=cfg.get("board_size", 5), rules=cfg.get("rules", {})).label()
    return {
        "file": rel_path,
        "index": idx,
        "match_id": rec.get("match_id", f"{rel_path}:{idx}"),
        "config_label": label or "unknown",
        "agents": rec.get("agents", {}),
        "outcome": {
            "category": outcome.get("category"),
            "winner": outcome.get("winner"),
            "reason": outcome.get("detail") or outcome.get("category"),
        },
        "length": rec.get("length_plies", len(rec.get("replay", []))),
    }


def list_replays(offset: int = 0, limit: int = 50, query: Optional[str] = None) -> Dict[str, Any]:
    """Discovers all replay records in experiments/results/ with pagination and filtering."""
    if not RESULTS_DIR.exists():
        return {"total": 0, "items": []}

    all_items: List[Dict[str, Any]] = []

    # Find all .jsonl and .json files under experiments/results/
    for root, _, files in os.walk(RESULTS_DIR):
        for f in sorted(files):
            if not (f.endswith(".jsonl") or f.endswith(".json")):
                continue
            full_path = Path(root) / f
            rel_path = str(full_path.relative_to(RESULTS_DIR))

            # Skip summary/aggregate files that don't hold replays
            if f in ("summary.json", "gate_x16.json", "ladder_x12_N3.json", "ladder_x12_N4.json", "benchmark_N3.json"):
                continue

            mtime = full_path.stat().st_mtime
            cached_mtime, cached_records = _REPLAY_INDEX_CACHE.get(rel_path, (-1.0, []))

            if cached_mtime == mtime:
                file_items = cached_records
            else:
                file_items = []
                try:
                    if f.endswith(".jsonl"):
                        with open(full_path, "r", encoding="utf-8") as fp:
                            for idx, line in enumerate(fp):
                                line = line.strip()
                                if not line:
                                    continue
                                try:
                                    rec = json.loads(line)
                                except Exception:
                                    continue
                                if "replay" in rec and isinstance(rec["replay"], list):
                                    file_items.append(_summary(rec, rel_path, idx))
                    elif f.endswith(".json"):
                        with open(full_path, "r", encoding="utf-8") as fp:
                            try:
                                data = json.load(fp)
                            except Exception:
                                data = None
                        if isinstance(data, dict) and "replay" in data:
                            file_items.append(_summary(data, rel_path, 0))
                except Exception:
                    file_items = []

                _REPLAY_INDEX_CACHE[rel_path] = (mtime, file_items)

            all_items.extend(file_items)

    if query:
        q_lower = query.lower()
        all_items = [
            it for it in all_items
            if q_lower in it["file"].lower() or q_lower in str(it.get("match_id", "")).lower()
        ]

    total = len(all_items)
    paginated = all_items[offset : offset + limit]
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "items": paginated,
    }


def verify_and_frame_replay(record: Dict[str, Any]) -> Dict[str, Any]:
    """Re-plays every logged action through a fresh WallzEngine and builds C-STATE frames.

    Verification mirrors evaluation.match_runner.verify_replay: each action must be legal,
    each logged post-action ``state_id`` must match, and the final state id and rule
    outcome must agree with the log. The first failure stops the replay at that ply.
    """
    def fail(message: str, ply: int, frames: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {"result": record,
                "verified": {"ok": False, "message": message, "failed_ply": ply},
                "frames": frames}

    replay_actions = record.get("replay", [])
    if not isinstance(replay_actions, list):
        return fail("Record has no replay list", 0, [])

    raw_cfg = record.get("config", {})
    if not isinstance(raw_cfg, dict):
        return fail("Record has no config object", 0, [])
    try:
        board_size = int(raw_cfg.get("board_size", raw_cfg.get("N", 5)))
        rules_dict = raw_cfg.get("rules", raw_cfg)
        config_obj = WallzConfig(
            board_size=board_size,
            max_plies=raw_cfg.get("max_plies") or 200,
            A=bool(rules_dict.get("A", True)),
            B=bool(rules_dict.get("B", True)),
            C=bool(rules_dict.get("C", True)),
            K=int(rules_dict.get("K", 3)),
            M=int(rules_dict.get("M", WallzConfig().M)),
        )
    except (TypeError, ValueError) as err:
        return fail(f"Invalid config in record: {err}", 0, [])
    engine = WallzEngine(config=config_obj)

    frames: List[Dict[str, Any]] = [state_json(engine)]
    if "initial_state_id" in record and state_id(engine._get_state_hashable_key()) != record["initial_state_id"]:
        return fail("Initial state differs from the logged initial_state_id", 0, frames)

    for idx, act in enumerate(replay_actions):
        ply = idx + 1
        try:
            act_to_apply = act["action"] if isinstance(act, dict) and "action" in act else act
            parsed = parse_action(act_to_apply, engine)
            engine.apply_action(parsed, update_clock=False)
        except Exception as err:  # ValueError (illegal) or malformed record
            return fail(f"Verification failed at ply {ply}: {err}", ply, frames)
        frames.append(state_json(engine))
        if isinstance(act, dict) and "state_id" in act and \
                state_id(engine._get_state_hashable_key()) != act["state_id"]:
            return fail(f"Verification failed at ply {ply}: state differs from the logged state_id", ply, frames)

    n = len(replay_actions)
    if "final_state_id" in record and state_id(engine._get_state_hashable_key()) != record["final_state_id"]:
        return fail("Final state differs from the logged final_state_id", n, frames)

    outcome = record.get("outcome") or {}
    category = outcome.get("category")
    if category == "goal" and engine.winner != outcome.get("winner"):
        return fail(f"Winner mismatch: log says {outcome.get('winner')}, engine produced {engine.winner}", n, frames)
    if isinstance(category, str) and category.startswith("draw_") and not engine.is_draw:
        return fail(f"Logged rule draw ({category}) not reproduced by the engine", n, frames)
    rec_winner = record.get("winner")  # legacy flat schema
    if rec_winner is not None and engine.winner != rec_winner:
        return fail(f"Winner mismatch: log says {rec_winner}, engine produced {engine.winner}", n, frames)

    return {
        "result": record,
        "verified": {"ok": True, "message": f"Verified by engine: {n} plies, outcome {category or 'unrecorded'}"},
        "frames": frames,
    }


def get_replay_item(rel_path: str, index: int = 0) -> Dict[str, Any]:
    """Retrieves a specific replay from disk and returns verified frames."""
    safe_path = _is_safe_results_path(rel_path)
    if not safe_path.exists():
        raise FileNotFoundError(f"Replay file '{rel_path}' not found")

    if safe_path.suffix == ".jsonl":
        with open(safe_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i == index:
                    record = json.loads(line)
                    return verify_and_frame_replay(record)
        raise IndexError(f"Index {index} out of range in {rel_path}")
    elif safe_path.suffix == ".json":
        with open(safe_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            record = data[index]
        else:
            record = data
        return verify_and_frame_replay(record)
    else:
        raise ValueError(f"Unsupported replay format: {safe_path.suffix}")
