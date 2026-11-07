"""Tests for live mini-training endpoints and SSE stream (P15).
"""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from web.backend.live_training import (
    CURRENT_JOB,
    get_live_status,
    start_live_training,
    stop_live_training,
    stream_live_events,
)


def test_live_training_status_initially_inactive():
    status = get_live_status()
    assert isinstance(status, dict)
    assert "active" in status
    assert "checkpoints" in status


def test_live_training_start_and_conflict(client):
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None  # Process is running

    with patch("subprocess.Popen", return_value=mock_proc):
        # First start should succeed (202)
        res = client.post("/api/training/live", json={"steps": 500, "seed": 42})
        assert res.status_code == 202
        data = res.get_json()
        assert "run_id" in data
        assert data["steps"] == 500

        # Second start while running should fail with 409 conflict
        res2 = client.post("/api/training/live", json={"steps": 500, "seed": 42})
        assert res2.status_code == 409
        assert "already in progress" in res2.get_json()["error"]

        # Status endpoint reflects active job
        status_res = client.get("/api/training/live/status")
        assert status_res.status_code == 200
        assert status_res.get_json()["active"] is True

        # Stop endpoint stops process
        stop_res = client.post("/api/training/live/stop")
        assert stop_res.status_code == 200
        assert stop_res.get_json()["stopped"] is True
        mock_proc.terminate.assert_called_once()

    # Reset CURRENT_JOB
    CURRENT_JOB["proc"] = None
    CURRENT_JOB["run_id"] = None


def test_live_training_stream_with_fixture():
    fixture_log = Path(__file__).parent / "fixtures" / "train_log_small.jsonl"
    events = list(stream_live_events(log_file_override=fixture_log))

    assert len(events) >= 2
    # Check that events have SSE format
    for evt in events:
        assert evt.startswith("data: ")
        assert evt.endswith("\n\n")

    # First event should be a log record
    first_data = json.loads(events[0][6:].strip())
    assert first_data["type"] == "log"
    assert "record" in first_data

    # Last event should be done
    last_data = json.loads(events[-1][6:].strip())
    assert last_data["type"] == "done"
