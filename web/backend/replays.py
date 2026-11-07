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
                                    file_items.append({
                                        "file": rel_path,
                                        "index": idx,
                                        "match_id": rec.get("match_id", f"{rel_path}:{idx}"),
                                        "config_label": rec.get("board", rec.get("config", {}).get("label", "unknown")),
                                        "agents": rec.get("agents", {}),
                                        "outcome": {
                                            "winner": rec.get("winner"),
                                            "reason": rec.get("terminal_reason"),
                                        },
                                        "length": rec.get("length", len(rec["replay"])),
                                    })
                    elif f.endswith(".json"):
                        with open(full_path, "r", encoding="utf-8") as fp:
                            try:
                                data = json.load(fp)
                            except Exception:
                                data = None
                        if isinstance(data, dict) and "replay" in data:
                            file_items.append({
                                "file": rel_path,
                                "index": 0,
                                "match_id": data.get("match_id", rel_path),
                                "config_label": data.get("board", data.get("config", {}).get("label", "unknown")),
                                "agents": data.get("agents", {}),
                                "outcome": {
                                    "winner": data.get("winner"),
                                    "reason": data.get("terminal_reason"),
                                },
                                "length": data.get("length", len(data["replay"])),
                            })
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
    """Step-by-step replays actions through canonical WallzEngine, generating C-STATE frames."""
    replay_actions = record.get("replay", [])

    # Extract configuration
    raw_cfg = record.get("config", {})
    if isinstance(raw_cfg, str):
        board_size = 5
        rules = {"A": True, "B": True, "C": True, "K": 3, "M": WallzConfig().M}
    else:
        board_size = int(raw_cfg.get("board_size", raw_cfg.get("N", 5)))
        rules_dict = raw_cfg.get("rules", raw_cfg)
        rules = {
            "A": bool(rules_dict.get("A", True)),
            "B": bool(rules_dict.get("B", True)),
            "C": bool(rules_dict.get("C", True)),
            "K": int(rules_dict.get("K", 3)),
            "M": int(rules_dict.get("M", WallzConfig().M)),
        }

    config_obj = WallzConfig(
        board_size=board_size,
        max_plies=200,
        A=rules["A"],
        B=rules["B"],
        C=rules["C"],
        K=rules["K"],
        M=rules["M"],
    )
    engine = WallzEngine(config=config_obj)

    frames: List[Dict[str, Any]] = [state_json(engine)]

    for idx, act in enumerate(replay_actions):
        try:
            act_to_apply = act["action"] if isinstance(act, dict) and "action" in act else act
            parsed = parse_action(act_to_apply, engine)
            engine.apply_action(parsed, update_clock=False)
            frames.append(state_json(engine))
        except ValueError as err:
            return {
                "result": record,
                "verified": {
                    "ok": False,
                    "message": f"Verification failed at ply {idx + 1}: {err}",
                    "failed_ply": idx + 1,
                },
                "frames": frames,
            }

    # Verify terminal status match if recorded
    rec_winner = record.get("winner")
    if rec_winner is not None and engine.winner != rec_winner:
        return {
            "result": record,
            "verified": {
                "ok": False,
                "message": f"Terminal winner mismatch: expected {rec_winner}, engine produced {engine.winner}",
                "failed_ply": len(replay_actions),
            },
            "frames": frames,
        }

    return {
        "result": record,
        "verified": {
            "ok": True,
            "message": "Replay verified successfully",
        },
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
