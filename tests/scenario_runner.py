"""Scenario Runner (P05 / Contract C-SCN).

Loads and executes scenario definitions against WallzEngine. Shared between
test assertions in `tests/test_scenarios.py` and backend API endpoints in
`web/backend/scenarios.py`.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Any, Dict, List, Optional

from src.modified_wallz.config import WallzConfig
from tests.helpers import key_from_json
from wallz_engine import WallzEngine


def load_scenarios(scenarios_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """Loads all JSON scenario files from `scenarios_dir` sorted by name."""
    if scenarios_dir is None:
        scenarios_dir = os.path.join(os.path.dirname(__file__), "scenarios")

    pattern = os.path.join(scenarios_dir, "*.json")
    files = sorted(glob.glob(pattern))
    scenarios = []
    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:
            scenarios.append(json.load(f))
    return scenarios


def engine_for(scenario: Dict[str, Any]) -> WallzEngine:
    """Instantiates a WallzEngine configured and initialized according to scenario spec."""
    cfg_data = dict(scenario.get("config", {}))
    board_size = cfg_data.pop("board_size", 5)
    max_plies = cfg_data.pop("max_plies", None)

    config = WallzConfig(
        board_size=board_size,
        max_plies=max_plies,
        A=cfg_data.get("A", True),
        B=cfg_data.get("B", True),
        C=cfg_data.get("C", True),
        K=cfg_data.get("K", 3),
        M=cfg_data.get("M", 10),
    )

    start = scenario.get("start", "initial")
    if isinstance(start, dict) and "raw_key" in start:
        engine = WallzEngine(config=config)
        raw_key = key_from_json(start["raw_key"])
        return engine.restore_state_key(raw_key)

    return WallzEngine(config=config)


def check_expectation(engine: WallzEngine, exp: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Evaluates an expectation dict against the given engine state, returning check results."""
    checks = []

    if "blocked" in exp:
        actual_blocked = {f"{r},{c}": t for (r, c), t in engine.blocked_cells.items()}
        ok = actual_blocked == exp["blocked"]
        checks.append({
            "key": "blocked",
            "ok": ok,
            "msg": f"Expected blocked {exp['blocked']}, got {actual_blocked}",
        })

    if "inventory" in exp:
        actual_inv = [engine.wall_inventory[0], engine.wall_inventory[1]]
        ok = actual_inv == exp["inventory"]
        checks.append({
            "key": "inventory",
            "ok": ok,
            "msg": f"Expected inventory {exp['inventory']}, got {actual_inv}",
        })

    if "hop" in exp:
        actual_hop = [engine.hop_available[0], engine.hop_available[1]]
        ok = actual_hop == exp["hop"]
        checks.append({
            "key": "hop",
            "ok": ok,
            "msg": f"Expected hop {exp['hop']}, got {actual_hop}",
        })

    if "block" in exp:
        actual_block = [engine.block_available[0], engine.block_available[1]]
        ok = actual_block == exp["block"]
        checks.append({
            "key": "block",
            "ok": ok,
            "msg": f"Expected block {exp['block']}, got {actual_block}",
        })

    if "latest" in exp:
        actual_latest = [
            list(engine.latest_wall[0]) if engine.latest_wall[0] else None,
            list(engine.latest_wall[1]) if engine.latest_wall[1] else None,
        ]
        ok = actual_latest == exp["latest"]
        checks.append({
            "key": "latest",
            "ok": ok,
            "msg": f"Expected latest {exp['latest']}, got {actual_latest}",
        })

    if "to_move" in exp:
        ok = engine.current_player == exp["to_move"]
        checks.append({
            "key": "to_move",
            "ok": ok,
            "msg": f"Expected current_player {exp['to_move']}, got {engine.current_player}",
        })

    if "legal_includes" in exp:
        legal_actions = engine.get_legal_actions()
        for act in exp["legal_includes"]:
            norm_act = engine._normalize_action(act)
            ok = norm_act in legal_actions
            checks.append({
                "key": f"legal_includes_{act[0]}",
                "ok": ok,
                "msg": f"Expected {norm_act} to be in legal actions",
            })

    if "legal_excludes" in exp:
        legal_actions = engine.get_legal_actions()
        for act in exp["legal_excludes"]:
            norm_act = engine._normalize_action(act)
            ok = norm_act not in legal_actions
            checks.append({
                "key": f"legal_excludes_{act[0]}",
                "ok": ok,
                "msg": f"Expected {norm_act} NOT to be in legal actions",
            })

    if "illegal" in exp:
        act = exp["illegal"]
        error_contains = exp.get("error_contains", "")
        clone = engine.clone()
        ok = False
        msg = f"Expected action {act} to raise ValueError"
        try:
            clone.apply_action(act, update_clock=False)
            msg += ", but it succeeded"
        except ValueError as err:
            if error_contains and error_contains not in str(err):
                msg = f"Expected error to contain {error_contains!r}, got {err!r}"
            else:
                ok = True
                msg = "Raised ValueError as expected"
        except Exception as err:
            msg = f"Expected ValueError, got {type(err).__name__}: {err}"

        checks.append({
            "key": "illegal",
            "ok": ok,
            "msg": msg,
        })

    if "terminal_reason" in exp:
        ok = engine.terminal_reason == exp["terminal_reason"]
        checks.append({
            "key": "terminal_reason",
            "ok": ok,
            "msg": f"Expected terminal_reason {exp['terminal_reason']}, got {engine.terminal_reason}",
        })

    if "winner" in exp:
        ok = engine.winner == exp["winner"]
        checks.append({
            "key": "winner",
            "ok": ok,
            "msg": f"Expected winner {exp['winner']}, got {engine.winner}",
        })

    return checks


def run_scenario(scenario: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Executes a scenario script step by step, evaluating all inline and final expectations."""
    engine = engine_for(scenario)
    script = scenario.get("script", [])
    expectations = scenario.get("expect", [])
    final_exp = scenario.get("final")

    results = []

    # Map after_ply -> list of expectations
    ply_expectations: Dict[int, List[Dict[str, Any]]] = {}
    for exp in expectations:
        p = exp.get("after_ply", 0)
        ply_expectations.setdefault(p, []).append(exp)

    # Initial expectations before script execution
    init_checks = []
    for exp in ply_expectations.get(0, []):
        init_checks.extend(check_expectation(engine, exp))
    results.append({
        "ply": 0,
        "action": None,
        "state_snapshot": engine.get_state(),
        "checks": init_checks,
    })

    # Execute script
    for idx, action in enumerate(script, start=1):
        engine.apply_action(action, update_clock=False)
        checks = []
        for exp in ply_expectations.get(idx, []):
            checks.extend(check_expectation(engine, exp))
        results.append({
            "ply": idx,
            "action": action,
            "state_snapshot": engine.get_state(),
            "checks": checks,
        })

    # Final expectations
    if final_exp:
        final_checks = check_expectation(engine, final_exp)
        results[-1]["checks"].extend(final_checks)

    return results
