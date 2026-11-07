"""Game management REST API routes (P08 / Contract C-API).

Provides endpoints for configuration defaults, agent discovery, game creation,
player actions, AI stepping, atomic undo, and session deletion.
"""
from __future__ import annotations

import time
from typing import Any, Dict
from flask import Blueprint, jsonify, request

from web.backend.registry import build_agent, list_agents, presets
from web.backend.serialize import action_to_json, parse_action, state_json
from web.backend.sessions import SessionStore


games_bp = Blueprint("games", __name__)
SESSION_STORE = SessionStore(capacity=64)


@games_bp.route("/api/config/defaults", methods=["GET"])
def get_config_defaults():
    defaults = {
        "board_size": 5,
        "A": True,
        "B": True,
        "C": True,
        "K": 3,
        "M": 10,
        "max_plies": 200,
    }
    limits = {
        "N": [3, 9],
        "K": [1, 9],
        "M": [0, 20],
    }
    return jsonify({
        "defaults": defaults,
        "limits": limits,
        "presets": presets(),
    })


@games_bp.route("/api/agents", methods=["GET"])
def get_agents():
    def _parse_bool(val: Any, default: bool) -> bool:
        if val is None:
            return default
        if isinstance(val, bool):
            return val
        return str(val).lower() in ("true", "1", "yes")

    try:
        n = int(request.args.get("N", request.args.get("board_size", 5)))
        k = int(request.args.get("K", 3))
        m = int(request.args.get("M", 10))
    except (ValueError, TypeError) as err:
        return jsonify({"error": f"Invalid integer configuration: {err}"}), 400

    cfg = {
        "board_size": n,
        "A": _parse_bool(request.args.get("A"), True),
        "B": _parse_bool(request.args.get("B"), True),
        "C": _parse_bool(request.args.get("C"), True),
        "K": k,
        "M": m,
    }
    return jsonify(list_agents(cfg))


@games_bp.route("/api/games", methods=["POST"])
def create_game():
    data = request.get_json(force=True, silent=True) or {}
    cfg = data.get("config", {})
    board_size = cfg.get("board_size", cfg.get("N", 5))
    if not (3 <= board_size <= 9):
        return jsonify({"error": f"board_size must be between 3 and 9, got {board_size}"}), 400

    game_config = {
        "board_size": board_size,
        "A": bool(cfg.get("A", True)),
        "B": bool(cfg.get("B", True)),
        "C": bool(cfg.get("C", True)),
        "K": int(cfg.get("K", 3)),
        "M": int(cfg.get("M", 10)),
        "max_plies": int(cfg.get("max_plies", 200)),
    }

    p0_spec = data.get("p0", "human")
    p1_spec = data.get("p1", "greedy")

    # Check agent compatibility
    avail_agents = {a["spec"]: a for a in list_agents(game_config)}
    for spec in (p0_spec, p1_spec):
        if spec != "human" and spec in avail_agents and not avail_agents[spec]["available"]:
            reason = avail_agents[spec]["reason"] or "Agent not available for this configuration"
            return jsonify({"error": reason}), 400

    try:
        agent_p0 = build_agent(p0_spec, 0, game_config)
        agent_p1 = build_agent(p1_spec, 1, game_config)
    except Exception as err:
        return jsonify({"error": f"Failed to build agents: {err}"}), 400

    session = SESSION_STORE.create(
        config=game_config,
        p0_spec=p0_spec,
        p1_spec=p1_spec,
        agents={0: agent_p0, 1: agent_p1},
    )
    return jsonify(session.to_json()), 201


@games_bp.route("/api/games/<gid>", methods=["GET"])
def get_game(gid: str):
    try:
        session = SESSION_STORE.get(gid)
    except KeyError:
        return jsonify({"error": f"Game '{gid}' not found"}), 404
    return jsonify(session.to_json())


@games_bp.route("/api/games/<gid>/action", methods=["POST"])
def post_action(gid: str):
    try:
        session = SESSION_STORE.get(gid)
    except KeyError:
        return jsonify({"error": f"Game '{gid}' not found"}), 404

    if session.engine.is_terminal():
        return jsonify({"error": "Game is already over"}), 409

    curr_p = session.engine.current_player
    if session.specs.get(curr_p) != "human":
        return jsonify({"error": f"Cannot submit manual action: player {curr_p} is an AI agent ({session.specs.get(curr_p)})"}), 409

    data = request.get_json(force=True, silent=True) or {}
    if "action" not in data:
        return jsonify({"error": "Missing 'action' in request body"}), 400

    try:
        action = parse_action(data["action"], session.engine)
    except ValueError as err:
        return jsonify({"error": str(err), "state": state_json(session.engine)}), 400

    try:
        session.engine.apply_action(action, update_clock=False)
    except ValueError as err:
        return jsonify({"error": str(err), "state": state_json(session.engine)}), 400

    session.history.append({
        "ply": session.engine.turn_count,
        "player": curr_p,
        "action": action_to_json(action),
        "source": "human",
        "explanation": None,
        "decision_seconds": 0.0,
    })

    return jsonify(session.to_json())


@games_bp.route("/api/games/<gid>/step", methods=["POST"])
def step_game(gid: str):
    try:
        session = SESSION_STORE.get(gid)
    except KeyError:
        return jsonify({"error": f"Game '{gid}' not found"}), 404

    if session.engine.is_terminal():
        return jsonify({"error": "Game is already over"}), 409

    curr_p = session.engine.current_player
    spec = session.specs.get(curr_p)
    if spec == "human":
        return jsonify({"error": f"Cannot step: player {curr_p} is a human player"}), 409

    agent = session.agents.get(curr_p)
    if agent is None:
        agent = build_agent(spec, curr_p, session.config)
        session.agents[curr_p] = agent

    t0 = time.perf_counter()
    try:
        action = agent.select_action(session.engine)
    except Exception as err:
        return jsonify({"error": f"Agent error during select_action: {err}"}), 500
    duration = time.perf_counter() - t0

    if action is None:
        return jsonify({"error": "Agent returned no action"}), 500

    session.engine.apply_action(action, update_clock=False)
    session.history.append({
        "ply": session.engine.turn_count,
        "player": curr_p,
        "action": action_to_json(action),
        "source": spec,
        "explanation": None,
        "decision_seconds": round(duration, 5),
    })

    return jsonify({
        "action": action_to_json(action),
        "explanation": None,
        "session": session.to_json(),
    })


@games_bp.route("/api/games/<gid>/undo", methods=["POST"])
def undo_game(gid: str):
    try:
        session = SESSION_STORE.get(gid)
    except KeyError:
        return jsonify({"error": f"Game '{gid}' not found"}), 404

    if not session.history:
        return jsonify({"error": "No actions to undo"}), 409

    data = request.get_json(force=True, silent=True) or {}
    n = int(data.get("n", 1))
    to_human = bool(data.get("to_human", False))

    session.undo(n=n, to_human=to_human)
    return jsonify(session.to_json())


@games_bp.route("/api/games/<gid>", methods=["DELETE"])
def delete_game(gid: str):
    try:
        SESSION_STORE.delete(gid)
    except KeyError:
        return jsonify({"error": f"Game '{gid}' not found"}), 404
    return "", 204
