"""Live mini-training manager and SSE stream generator (P15).
"""
from __future__ import annotations

import atexit
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Generator, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LIVE_CACHE_DIR = PROJECT_ROOT / "experiments" / "cache" / "live"

CURRENT_JOB: Dict[str, Any] = {
    "proc": None,
    "run_id": None,
    "dir": None,
    "steps": 0,
    "seed": 42,
    "start_time": 0.0,
}


def cleanup_live_job():
    """Ensure child process is killed on server exit."""
    proc: Optional[subprocess.Popen] = CURRENT_JOB.get("proc")
    if proc is not None and proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


atexit.register(cleanup_live_job)


def get_live_status() -> Dict[str, Any]:
    """Return status of active or most recent live training job."""
    proc: Optional[subprocess.Popen] = CURRENT_JOB.get("proc")
    run_id = CURRENT_JOB.get("run_id")
    job_dir = CURRENT_JOB.get("dir")

    active = (proc is not None and proc.poll() is None)
    exit_code = proc.poll() if proc is not None else None

    checkpoints = []
    if job_dir is not None and Path(job_dir).exists():
        for pt in sorted(Path(job_dir).glob("*.pt")):
            checkpoints.append(str(pt.relative_to(PROJECT_ROOT)))

    return {
        "active": active,
        "run_id": run_id,
        "exit_code": exit_code,
        "steps": CURRENT_JOB.get("steps", 0),
        "seed": CURRENT_JOB.get("seed", 42),
        "elapsed_seconds": round(time.time() - CURRENT_JOB.get("start_time", time.time()), 1) if active else None,
        "checkpoints": checkpoints,
    }


def start_live_training(steps: int = 2000, seed: int = 42) -> Dict[str, Any]:
    """Launch mini-training in a managed subprocess."""
    proc: Optional[subprocess.Popen] = CURRENT_JOB.get("proc")
    if proc is not None and proc.poll() is None:
        raise ValueError("A live training run is already in progress.")

    steps = max(100, min(int(steps), 5000))
    seed = int(seed)
    run_id = f"live_{int(time.time())}"
    job_dir = LIVE_CACHE_DIR / run_id
    job_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "train_dqn.py"),
        "--board-size", "3",
        "--steps", str(steps),
        "--seed", str(seed),
        "--eval-freq", "500",
        "--output-dir", str(job_dir),
    ]

    new_proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    CURRENT_JOB["proc"] = new_proc
    CURRENT_JOB["run_id"] = run_id
    CURRENT_JOB["dir"] = str(job_dir)
    CURRENT_JOB["steps"] = steps
    CURRENT_JOB["seed"] = seed
    CURRENT_JOB["start_time"] = time.time()

    return {
        "run_id": run_id,
        "steps": steps,
        "seed": seed,
        "output_dir": str(job_dir.relative_to(PROJECT_ROOT)),
    }


def stop_live_training() -> Dict[str, Any]:
    """Terminate the currently active live training job if running."""
    proc: Optional[subprocess.Popen] = CURRENT_JOB.get("proc")
    if proc is None or proc.poll() is not None:
        return {"stopped": False, "message": "No active job running."}

    proc.terminate()
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        proc.kill()

    return {"stopped": True, "run_id": CURRENT_JOB.get("run_id")}


def stream_live_events(log_file_override: Optional[Path] = None) -> Generator[str, None, None]:
    """Server-Sent Events generator yielding parsed train_log records."""
    if log_file_override is not None:
        log_path = log_file_override
    else:
        job_dir = CURRENT_JOB.get("dir")
        if not job_dir:
            yield f"data: {json.dumps({'type': 'error', 'message': 'No live run configured.'})}\n\n"
            return
        log_path = Path(job_dir) / "train_log.jsonl"

    # Wait up to 5 seconds for log file creation
    waited = 0.0
    while not log_path.exists() and waited < 5.0:
        time.sleep(0.2)
        waited += 0.2

    if not log_path.exists():
        yield f"data: {json.dumps({'type': 'error', 'message': 'Log file not created.'})}\n\n"
        return

    with open(log_path, "r", encoding="utf-8") as f:
        while True:
            line = f.readline()
            if line:
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    record = json.loads(line_str)
                    yield f"data: {json.dumps({'type': 'log', 'record': record})}\n\n"
                except json.JSONDecodeError:
                    pass
            else:
                proc = CURRENT_JOB.get("proc")
                if log_file_override is not None or (proc is not None and proc.poll() is not None):
                    yield f"data: {json.dumps({'type': 'done', 'exit_code': proc.poll() if proc else 0})}\n\n"
                    break
                time.sleep(0.3)
