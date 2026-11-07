"""Tests for Arena, Replays, Scenarios, and Training data endpoints (P10 / Contract C-API).
"""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from web.backend.training_data import parse_training_log


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def test_arena_batch_sync_runner(client):
    """POST /api/arena/batch runs n=6 games with balanced colors and valid CIs."""
    res = client.post("/api/arena/batch", json={
        "config": {"board_size": 3, "K": 1, "M": 1},
        "p0": "greedy",
        "p1": "random:seed=12",
        "n": 6,
        "seed": 42,
    })
    assert res.status_code == 200
    data = res.get_json()

    assert data["games"] == 6
    assert "by_spec" in data
    assert "greedy" in data["by_spec"]
    assert "random:seed=12" in data["by_spec"]

    g_stats = data["by_spec"]["greedy"]
    assert g_stats["wins"] + g_stats["draws"] + g_stats["losses"] == 6
    assert len(g_stats["ci"]) == 2
    assert 0.0 <= g_stats["ci"][0] <= g_stats["ci"][1] <= 1.0

    assert len(data["replays"]) == 6
    bid = data["batch_id"]

    # Retrieve batch summary
    res_b = client.get(f"/api/arena/batch/{bid}")
    assert res_b.status_code == 200
    assert res_b.get_json()["batch_id"] == bid

    # Retrieve specific game frame
    res_g = client.get(f"/api/arena/batch/{bid}/0")
    assert res_g.status_code == 200
    g_data = res_g.get_json()
    assert g_data["verified"]["ok"] is True
    assert len(g_data["frames"]) > 1


def test_arena_caps_enforced(client):
    """POST /api/arena/batch rejects oversize requests and deep minimax on N=9."""
    # N=9, depth 2 with n=12 -> rejected
    res = client.post("/api/arena/batch", json={
        "config": {"board_size": 9},
        "p0": "minimax:depth=2",
        "p1": "greedy",
        "n": 12,
    })
    assert res.status_code == 400
    assert "capped at 10" in res.get_json()["error"]

    # n > 50 -> rejected
    res2 = client.post("/api/arena/batch", json={
        "config": {"board_size": 3},
        "p0": "greedy",
        "p1": "random",
        "n": 55,
    })
    assert res2.status_code == 400


def test_replays_listing_and_verification(client):
    """GET /api/replays discovers saved greedy campaign results and verifies cleanly."""
    res = client.get("/api/replays?limit=10")
    assert res.status_code == 200
    data = res.get_json()
    assert data["total"] > 0
    assert len(data["items"]) > 0

    first_item = data["items"][0]
    file_path = first_item["file"]
    index = first_item["index"]

    # Retrieve item
    item_res = client.get(f"/api/replays/item?file={file_path}&index={index}")
    assert item_res.status_code == 200
    item_data = item_res.get_json()
    assert item_data["verified"]["ok"] is True
    assert len(item_data["frames"]) == first_item["length"] + 1


def test_replays_path_traversal_rejected(client):
    """GET /api/replays/item rejects ../ path traversal attempts with 400."""
    res = client.get("/api/replays/item?file=../../etc/passwd&index=0")
    assert res.status_code == 400
    assert "Path traversal" in res.get_json()["error"]


def test_replays_upload_detects_tampering(client):
    """POST /api/replays/upload on replay_tampered.json detects illegality at exact ply."""
    tampered_file = FIXTURES_DIR / "replay_tampered.json"
    with open(tampered_file, "r", encoding="utf-8") as f:
        tampered_data = json.load(f)

    res = client.post("/api/replays/upload", json=tampered_data)
    assert res.status_code == 200
    res_data = res.get_json()

    assert res_data["verified"]["ok"] is False
    assert res_data["verified"]["failed_ply"] == 2
    assert "Verification failed at ply 2" in res_data["verified"]["message"]


def test_scenarios_listing_and_frames(client):
    """GET /api/scenarios lists all 9 scenarios with 100% expectation pass rates."""
    res = client.get("/api/scenarios")
    assert res.status_code == 200
    scenarios = res.get_json()
    assert len(scenarios) == 9

    names = {s["name"] for s in scenarios}
    assert "move_basics" in names
    assert "hop_over_wall" in names
    assert "block_expires_after_K" in names

    # Run one scenario through API and verify checks
    res_scn = client.get("/api/scenarios/hop_over_wall")
    assert res_scn.status_code == 200
    data = res_scn.get_json()
    assert len(data["frames"]) > 1
    for frame in data["frames"]:
        for chk in frame["checks"]:
            assert chk["ok"] is True, f"Check failed: {chk}"


def test_game_created_from_scenario(client):
    """POST /api/games/from-scenario/<name> creates playable session positioned after script."""
    res = client.post("/api/games/from-scenario/move_basics", json={
        "p0": "human",
        "p1": "human",
    })
    assert res.status_code == 201
    session = res.get_json()
    assert session["id"] is not None
    assert len(session["history"]) > 0

    gid = session["id"]
    # Check that game can be played further
    flat_legal = []
    for act_type, vals in session["state"]["legal"].items():
        for v in vals:
            flat_legal.append([act_type, v])
    assert len(flat_legal) > 0

    act_res = client.post(f"/api/games/{gid}/action", json={"action": flat_legal[0]})
    assert act_res.status_code == 200


def test_training_log_parsing_and_resilience():
    """parse_training_log skips malformed lines, tracks warnings, and builds curves."""
    log_path = FIXTURES_DIR / "train_log_small.jsonl"
    data = parse_training_log(log_path, meta={"N": 3})

    assert len(data["warnings"]) == 1
    assert "1 malformed log entries skipped" in data["warnings"][0]
    assert len(data["curves"]["env_steps"]) == 3
    assert data["curves"]["env_steps"] == [10, 25, 40]
    assert data["opponents"]["current"] == 1
    assert data["opponents"]["greedy"] == 1
    assert data["opponents"]["snapshot"] == 1
    assert len(data["evals"]) == 1
    assert data["evals"][0]["vs_greedy"] == 0.8


def test_training_endpoints(client):
    """GET /api/training/runs and /api/training/config return valid structured metadata."""
    res_runs = client.get("/api/training/runs")
    assert res_runs.status_code == 200
    runs = res_runs.get_json()
    assert len(runs) > 0

    run_ids = [r["id"] for r in runs]
    assert any("dqn_3x3" in rid for rid in run_ids)

    # Config endpoint
    res_cfg = client.get("/api/training/config")
    assert res_cfg.status_code == 200
    cfg = res_cfg.get_json()
    assert cfg["gamma"] == 0.99
    assert cfg["batch_size"] == 128

    # Summary endpoint
    res_sum = client.get("/api/training/summary")
    assert res_sum.status_code == 200
    sum_data = res_sum.get_json()
    assert "summary" in sum_data


def test_rules_doc_endpoint(client):
    """GET /api/rules returns the raw markdown text of docs/game_rules.md."""
    res = client.get("/api/rules")
    assert res.status_code == 200
    data = res.get_json()
    assert "rules" in data
    assert "# Modified Wallz" in data["rules"]
