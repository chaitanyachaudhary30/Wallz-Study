"""Training metrics, checkpoints discovery, and summary aggregator (P10 / Contract C-API).

Parses training logs, extracts evaluation metrics and opponent distributions,
downsamples curves to <=500 points for chart rendering, and aggregates experiment summaries.
"""
from __future__ import annotations

import dataclasses
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
RESULTS_DIR = REPO_ROOT / "experiments" / "results"


def _is_safe_checkpoint_path(run_id: str) -> Path:
    """Validates that run_id stays strictly within checkpoints/."""
    target = (CHECKPOINTS_DIR / run_id).resolve()
    try:
        target.relative_to(CHECKPOINTS_DIR.resolve())
    except ValueError:
        raise ValueError(f"Path traversal detected: {run_id} is outside {CHECKPOINTS_DIR}")
    return target


def list_training_runs() -> List[Dict[str, Any]]:
    """Discovers all DQN training runs with metadata and checkpoint artifacts."""
    if not CHECKPOINTS_DIR.exists():
        return []

    runs = []
    for root, _, files in os.walk(CHECKPOINTS_DIR):
        if "meta.json" in files:
            dir_path = Path(root)
            rel_path = str(dir_path.relative_to(CHECKPOINTS_DIR))
            try:
                with open(dir_path / "meta.json", "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                meta = {}

            ckpts = []
            for f in sorted(files):
                if f.endswith(".pt"):
                    step_match = re.search(r"step_(\d+)", f)
                    step_val = int(step_match.group(1)) if step_match else ("final" if "final" in f else "model")
                    ckpts.append({"file": f, "step": step_val})

            runs.append({
                "id": rel_path,
                "path": rel_path,
                "meta": meta,
                "has_log": (dir_path / "train_log.jsonl").exists(),
                "checkpoints": ckpts,
                "release": "release" in rel_path,
            })

    # Sort releases first, then by id
    runs.sort(key=lambda r: (not r["release"], r["id"]))
    return runs


def parse_training_log(log_file: Path, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Parses a train_log.jsonl file into curves, evaluations, and opponent counts."""
    if meta is None:
        meta = {}

    if not log_file.exists():
        return {
            "meta": meta,
            "curves": {"env_steps": [], "win_rate_rolling100": [], "episode_length_rolling100": [], "epsilon": [], "avg_loss": []},
            "evals": [],
            "opponents": {"current": 0, "snapshot": 0, "greedy": 0},
            "warnings": ["train_log.jsonl not found"],
        }

    raw_steps = []
    raw_wr = []
    raw_len = []
    raw_eps = []
    raw_loss = []
    evals = []
    opponents = {"current": 0, "snapshot": 0, "greedy": 0}
    warning_count = 0

    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                warning_count += 1
                continue

            # Record training step curve points (only from episode records)
            if rec.get("type") in ("episode", None) and ("step" in rec or "env_steps" in rec):
                step = rec.get("env_steps", rec.get("step"))
                wr = rec.get("win_rate_rolling100", rec.get("win_rate"))
                ep_len = rec.get("episode_length_rolling100", rec.get("ep_length", rec.get("length")))
                eps = rec.get("epsilon")
                loss = rec.get("avg_loss", rec.get("loss"))

                if step is not None:
                    raw_steps.append(step)
                    raw_wr.append(round(float(wr), 4) if wr is not None else None)
                    raw_len.append(round(float(ep_len), 2) if ep_len is not None else None)
                    raw_eps.append(round(float(eps), 4) if eps is not None else None)
                    raw_loss.append(round(float(loss), 5) if loss is not None else None)

            # Record periodic evaluations
            if "eval_step" in rec or "vs_greedy" in rec:
                evals.append({
                    "env_steps": rec.get("eval_step", rec.get("env_steps", rec.get("step"))),
                    "vs_greedy": rec.get("vs_greedy"),
                    "vs_random": rec.get("vs_random"),
                })

            # Record opponent types
            opp = rec.get("opponent", rec.get("opponent_type"))
            if opp in opponents:
                opponents[opp] += 1

    # Downsample curves to <= 500 points preserving the first and last
    n_points = len(raw_steps)
    if n_points > 500:
        stride = max(1, n_points // 500)
        indices = list(range(0, n_points, stride))
        if indices[-1] != n_points - 1:
            indices.append(n_points - 1)

        curves = {
            "env_steps": [raw_steps[i] for i in indices],
            "win_rate_rolling100": [raw_wr[i] for i in indices],
            "episode_length_rolling100": [raw_len[i] for i in indices],
            "epsilon": [raw_eps[i] for i in indices],
            "avg_loss": [raw_loss[i] for i in indices],
        }
    else:
        curves = {
            "env_steps": raw_steps,
            "win_rate_rolling100": raw_wr,
            "episode_length_rolling100": raw_len,
            "epsilon": raw_eps,
            "avg_loss": raw_loss,
        }

    warnings = []
    if warning_count > 0:
        warnings.append(f"{warning_count} malformed log entries skipped")

    return {
        "meta": meta,
        "curves": curves,
        "evals": evals,
        "opponents": opponents,
        "warnings": warnings,
    }


def get_training_run(run_id: str) -> Dict[str, Any]:
    """Loads training curves and metrics from train_log.jsonl with downsampling <=500 points."""
    target_dir = _is_safe_checkpoint_path(run_id)
    if not target_dir.exists():
        raise FileNotFoundError(f"Training run directory '{run_id}' not found")

    meta_file = target_dir / "meta.json"
    meta = {}
    if meta_file.exists():
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            pass

    log_file = target_dir / "train_log.jsonl"
    return parse_training_log(log_file, meta=meta)


def get_training_summary() -> Dict[str, Any]:
    """Aggregates experiment results across ladder, gate, benchmark, solves, and greedy campaign."""
    target_files = {
        "ladder_n3": RESULTS_DIR / "dqn" / "ladder_x12_N3.json",
        "ladder_n4": RESULTS_DIR / "dqn" / "ladder_x12_N4.json",
        "gate": RESULTS_DIR / "dqn" / "gate_x16.json",
        "benchmark_n3": RESULTS_DIR / "dqn" / "benchmark_N3.json",
        "minimax_reduction": RESULTS_DIR / "minimax" / "x8_node_reduction.json",
        "solve_n3_k1_m0": RESULTS_DIR / "solves" / "solve_N3_K1_M0.json",
        "greedy_campaign": RESULTS_DIR / "greedy" / "summary.json",
    }

    summary: Dict[str, Any] = {}
    warnings: List[str] = []

    for key, path in target_files.items():
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    summary[key] = json.load(f)
            except Exception as err:
                summary[key] = None
                warnings.append(f"Failed to parse {path.name}: {err}")
        else:
            summary[key] = None
            warnings.append(f"Result file {path.name} not found")

    return {
        "summary": summary,
        "warnings": warnings,
    }


def get_training_config() -> Optional[Dict[str, Any]]:
    """Returns canonical default configuration from DQN trainer without hardcoded frontend numbers."""
    try:
        from agents.dqn.trainer import DQNConfig
        cfg = DQNConfig()
        return dataclasses.asdict(cfg)
    except Exception as err:
        return {"error": f"Failed to load DQNConfig: {err}"}
