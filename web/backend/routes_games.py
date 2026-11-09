"""Game management REST API routes (P08 / Contract C-API).

Provides endpoints for configuration defaults, agent discovery, game creation,
player actions, AI stepping, atomic undo, and session deletion.
"""
from __future__ import annotations

import functools
import time
from typing import Any, Dict
from flask import Blueprint, jsonify, request

from evaluation.match_runner import parse_agent_spec
from web.backend.explain import explain_and_choose
from web.backend.registry import build_agent, list_agents, presets
from web.backend.serialize import action_to_json, parse_action, state_json
from web.backend.sessions import SESSION_STORE
from src.modified_wallz.config import WallzConfig


games_bp = Blueprint("games", __name__)


def _session_locked(handler):
    """Run a per-game handler while holding that game's lock (see GameSession.lock)."""
    @functools.wraps(handler)
    def wrapper(gid: str, *args, **kwargs):
        try:
            session = SESSION_STORE.get(gid)
        except KeyError:
            return jsonify({"error": f"Game '{gid}' not found"}), 404
        with session.lock:
            return handler(gid, *args, **kwargs)
    return wrapper


@games_bp.route("/api/config/defaults", methods=["GET"])
def get_config_defaults():
    defaults = {
        "board_size": 5,
        "A": True,
        "B": True,
        "C": True,
        "K": 3,
        "M": WallzConfig().M,
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
        m = int(request.args.get("M", WallzConfig().M))
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


def _as_bool(val: Any, default: bool) -> bool:
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    text = str(val).strip().lower()
    if text in ("true", "1", "yes", "on"):
        return True
    if text in ("false", "0", "no", "off", ""):
        return False
    raise ValueError(f"expected a boolean, got {val!r}")


def _as_int(val: Any, name: str, default: int) -> int:
    if val is None:
        return default
    if isinstance(val, bool):
        raise ValueError(f"{name} must be an integer, got {val!r}")
    try:
        f = float(val)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer, got {val!r}")
    if not f.is_integer():
        raise ValueError(f"{name} must be an integer, got {val!r}")
    return int(f)


def parse_game_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a client config. Raises ValueError with a user-facing message.

    Range checks for K, M and max_plies come from WallzConfig itself (single source).
    """
    if not isinstance(cfg, dict):
        raise ValueError("config must be an object")
    board_size = _as_int(cfg.get("board_size", cfg.get("N")), "board_size", 5)
    if not (3 <= board_size <= 9):
        raise ValueError(f"board_size must be between 3 and 9, got {board_size}")
    game_config = {
        "board_size": board_size,
        "A": _as_bool(cfg.get("A"), True),
        "B": _as_bool(cfg.get("B"), True),
        "C": _as_bool(cfg.get("C"), True),
        "K": _as_int(cfg.get("K"), "K", 3),
        "M": _as_int(cfg.get("M"), "M", WallzConfig().M),
        "max_plies": _as_int(cfg.get("max_plies"), "max_plies", 200),
    }
    WallzConfig(**game_config)  # raises ValueError for K < 1, M < 0, max_plies < 1
    return game_config


@games_bp.route("/api/games", methods=["POST"])
def create_game():
    data = request.get_json(force=True, silent=True) or {}
    try:
        game_config = parse_game_config(data.get("config", {}))
    except ValueError as err:
        return jsonify({"error": f"Invalid configuration: {err}"}), 400

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
@_session_locked
def get_game(gid: str):
    try:
        session = SESSION_STORE.get(gid)
    except KeyError:
        return jsonify({"error": f"Game '{gid}' not found"}), 404
    return jsonify(session.to_json())


@games_bp.route("/api/games/<gid>/action", methods=["POST"])
@_session_locked
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
@_session_locked
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

    data = request.get_json(force=True, silent=True) or {}
    explain = bool(data.get("explain", True))

    agent = session.agents.get(curr_p)
    if agent is None:
        try:
            agent = build_agent(spec, curr_p, session.config)
        except Exception as err:
            return jsonify({"error": f"Failed to build agent '{spec}': {err}"}), 400
        session.agents[curr_p] = agent

    family, _ = parse_agent_spec(spec)
    t0 = time.perf_counter()
    try:
        if explain:
            action, explanation = explain_and_choose(agent, family, session.engine, spec=spec)
        else:
            action = agent.select_action(session.engine)
            explanation = None
    except Exception as err:
        return jsonify({"error": f"Agent error during decision: {err}"}), 500
    duration = time.perf_counter() - t0

    if action is None:
        return jsonify({"error": "Agent returned no action"}), 500

    try:
        session.engine.apply_action(action, update_clock=False)
    except ValueError as err:
        return jsonify({"error": f"Agent '{spec}' chose an illegal action {action}: {err}"}), 500
    session.history.append({
        "ply": session.engine.turn_count,
        "player": curr_p,
        "action": action_to_json(action),
        "source": spec,
        "explanation": explanation,
        "decision_seconds": round(duration, 5),
    })

    return jsonify({
        "action": action_to_json(action),
        "explanation": explanation,
        "session": session.to_json(),
    })


@games_bp.route("/api/games/<gid>/analyze", methods=["POST"])
@_session_locked
def analyze_game(gid: str):
    try:
        session = SESSION_STORE.get(gid)
    except KeyError:
        return jsonify({"error": f"Game '{gid}' not found"}), 404

    if session.engine.is_terminal():
        return jsonify({"error": "Cannot analyze a terminal position"}), 409

    data = request.get_json(force=True, silent=True) or {}
    agent_spec = data.get("agent", "greedy")

    curr_p = session.engine.current_player
    family, _ = parse_agent_spec(agent_spec)
    try:
        temp_agent = build_agent(agent_spec, curr_p, session.config)
        _, explanation = explain_and_choose(temp_agent, family, session.engine.clone(), spec=agent_spec)
    except Exception as err:
        return jsonify({"error": f"Analysis failed: {err}"}), 500

    return jsonify(explanation)


@games_bp.route("/api/games/<gid>/undo", methods=["POST"])
@_session_locked
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
