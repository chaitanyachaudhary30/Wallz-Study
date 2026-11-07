"""Rules scenario runners and interactive loader (P10 / Contract C-API).

Exposes the 9 canonical rule demonstration scenarios, generates step-by-step
verification frames with Contract C-STATE snapshots, and enables interactive
continuation through the GameSession engine.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from tests.helpers import key_from_json
from tests.scenario_runner import check_expectation, engine_for, load_scenarios
from web.backend.serialize import action_to_json, parse_action, state_json
from web.backend.sessions import GameSession, SESSION_STORE


def list_scenarios() -> List[Dict[str, Any]]:
    """Returns directory of all 9 canonical rule demonstration scenarios."""
    scenarios = load_scenarios()
    catalog = []
    for s in scenarios:
        catalog.append({
            "name": s["name"],
            "title": s.get("title", s["name"].replace("_", " ").title()),
            "description": s.get("description", ""),
            "config": s.get("config", {}),
        })
    return catalog


def get_scenario(name: str) -> Dict[str, Any]:
    """Runs a scenario step-by-step and returns C-STATE frames with verification checks."""
    scenarios = load_scenarios()
    target = next((s for s in scenarios if s["name"] == name), None)
    if target is None:
        raise KeyError(f"Scenario '{name}' not found")

    engine = engine_for(target)
    script = target.get("script", [])
    expectations = target.get("expect", [])
    final_exp = target.get("final")

    # Group expectations by after_ply
    ply_expectations: Dict[int, List[Dict[str, Any]]] = {}
    for exp in expectations:
        p = exp.get("after_ply", 0)
        ply_expectations.setdefault(p, []).append(exp)

    frames = []

    # Frame 0: Initial state before any script plies
    init_checks = []
    for exp in ply_expectations.get(0, []):
        init_checks.extend(check_expectation(engine, exp))
    frames.append({
        "ply": 0,
        "action": None,
        "state": state_json(engine),
        "checks": init_checks,
    })

    # Execute plies in sequence
    for idx, raw_act in enumerate(script, start=1):
        act = parse_action(raw_act, engine)
        engine.apply_action(act, update_clock=False)

        checks = []
        for exp in ply_expectations.get(idx, []):
            checks.extend(check_expectation(engine, exp))

        if idx == len(script) and final_exp:
            checks.extend(check_expectation(engine, final_exp))

        frames.append({
            "ply": idx,
            "action": action_to_json(act),
            "state": state_json(engine),
            "checks": checks,
        })

    return {
        "scenario": target,
        "frames": frames,
    }


def create_game_from_scenario(name: str, p0: str = "human", p1: str = "human") -> Dict[str, Any]:
    """Positions a new playable GameSession immediately after the scenario script finishes."""
    scenarios = load_scenarios()
    target = next((s for s in scenarios if s["name"] == name), None)
    if target is None:
        raise KeyError(f"Scenario '{name}' not found")

    engine = engine_for(target)
    script = target.get("script", [])

    history = []
    for idx, raw_act in enumerate(script, start=1):
        mover = engine.current_player
        act = parse_action(raw_act, engine)
        engine.apply_action(act, update_clock=False)
        history.append({
            "ply": idx,
            "player": mover,
            "action": action_to_json(act),
            "source": "scenario",
            "explanation": None,
            "decision_seconds": 0.0,
        })

    cfg = dict(target.get("config", {}))
    session = SESSION_STORE.create(
        config=cfg,
        p0_spec=p0,
        p1_spec=p1,
        initial_engine=engine,
    )
    session.history.extend(history)
    return session.to_json()
