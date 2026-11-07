"""Data API routes for Arena, Replays, Scenarios, and Training data (P10 / Contract C-API).
"""
from __future__ import annotations

from typing import Any, Dict
from flask import Blueprint, jsonify, request

from web.backend.arena import ARENA_STORE, run_arena_batch
from web.backend.replays import get_replay_item, list_replays, verify_and_frame_replay
from web.backend.scenarios import create_game_from_scenario, get_scenario, list_scenarios
from web.backend.training_data import (
    get_training_config,
    get_training_run,
    get_training_summary,
    list_training_runs,
)


data_bp = Blueprint("data", __name__)


# -----------------------------------------------------------------------------
# Arena Endpoints
# -----------------------------------------------------------------------------

@data_bp.route("/api/arena/batch", methods=["POST"])
def arena_batch():
    data = request.get_json(force=True, silent=True) or {}
    config = data.get("config", {})
    p0 = data.get("p0", "greedy")
    p1 = data.get("p1", "random")
    n = int(data.get("n", 10))
    seed = int(data.get("seed", 42))

    try:
        summary = run_arena_batch(config, p0, p1, n=n, seed=seed)
        return jsonify(summary)
    except ValueError as err:
        return jsonify({"error": str(err)}), 400
    except Exception as err:
        return jsonify({"error": f"Arena execution failed: {err}"}), 500


@data_bp.route("/api/arena/batch/<bid>", methods=["GET"])
def get_arena_batch_summary(bid: str):
    try:
        data = ARENA_STORE.get(bid)
        return jsonify(data["summary"])
    except KeyError:
        return jsonify({"error": f"Arena batch '{bid}' not found"}), 404


@data_bp.route("/api/arena/batch/<bid>/<int:idx>", methods=["GET"])
def get_arena_batch_game(bid: str, idx: int):
    try:
        data = ARENA_STORE.get(bid)
        records = data["records"]
        if not (0 <= idx < len(records)):
            return jsonify({"error": f"Game index {idx} out of range (0..{len(records)-1})"}), 404
        return jsonify(verify_and_frame_replay(records[idx]))
    except KeyError:
        return jsonify({"error": f"Arena batch '{bid}' not found"}), 404


# -----------------------------------------------------------------------------
# Replay Endpoints
# -----------------------------------------------------------------------------

@data_bp.route("/api/replays", methods=["GET"])
def get_replays():
    offset = int(request.args.get("offset", 0))
    limit = int(request.args.get("limit", 50))
    query = request.args.get("q")
    return jsonify(list_replays(offset=offset, limit=limit, query=query))


@data_bp.route("/api/replays/item", methods=["GET"])
def get_replay():
    file_path = request.args.get("file")
    if not file_path:
        return jsonify({"error": "Missing 'file' query parameter"}), 400
    index = int(request.args.get("index", 0))

    try:
        data = get_replay_item(file_path, index=index)
        return jsonify(data)
    except ValueError as err:
        return jsonify({"error": str(err)}), 400
    except FileNotFoundError as err:
        return jsonify({"error": str(err)}), 404
    except IndexError as err:
        return jsonify({"error": str(err)}), 404
    except Exception as err:
        return jsonify({"error": f"Failed to retrieve replay: {err}"}), 500


@data_bp.route("/api/replays/upload", methods=["POST"])
def upload_replay():
    data = request.get_json(force=True, silent=True) or {}
    if "replay" not in data:
        return jsonify({"error": "Request body must include 'replay' array"}), 400
    try:
        res = verify_and_frame_replay(data)
        return jsonify(res)
    except Exception as err:
        return jsonify({"error": f"Replay verification failed: {err}"}), 500


# -----------------------------------------------------------------------------
# Scenario Endpoints
# -----------------------------------------------------------------------------

@data_bp.route("/api/scenarios", methods=["GET"])
def get_scenarios():
    return jsonify(list_scenarios())


@data_bp.route("/api/scenarios/<name>", methods=["GET"])
def get_scenario_by_name(name: str):
    try:
        return jsonify(get_scenario(name))
    except KeyError as err:
        return jsonify({"error": str(err)}), 404
    except Exception as err:
        return jsonify({"error": f"Failed to run scenario '{name}': {err}"}), 500


@data_bp.route("/api/games/from-scenario/<name>", methods=["POST"])
def create_game_from_scenario_route(name: str):
    data = request.get_json(force=True, silent=True) or {}
    p0 = data.get("p0", "human")
    p1 = data.get("p1", "human")
    try:
        session_json = create_game_from_scenario(name, p0=p0, p1=p1)
        return jsonify(session_json), 201
    except KeyError as err:
        return jsonify({"error": str(err)}), 404
    except Exception as err:
        return jsonify({"error": f"Failed to create game from scenario '{name}': {err}"}), 500


# -----------------------------------------------------------------------------
# Training Data Endpoints
# -----------------------------------------------------------------------------

@data_bp.route("/api/training/runs", methods=["GET"])
def get_training_runs():
    return jsonify(list_training_runs())


@data_bp.route("/api/training/runs/<path:run_id>", methods=["GET"])
def get_training_run_data(run_id: str):
    try:
        return jsonify(get_training_run(run_id))
    except ValueError as err:
        return jsonify({"error": str(err)}), 400
    except FileNotFoundError as err:
        return jsonify({"error": str(err)}), 404
    except Exception as err:
        return jsonify({"error": f"Failed to load training run '{run_id}': {err}"}), 500


@data_bp.route("/api/training/summary", methods=["GET"])
def get_training_summary_route():
    return jsonify(get_training_summary())


@data_bp.route("/api/training/config", methods=["GET"])
def get_training_config_route():
    cfg = get_training_config()
    return jsonify(cfg)
