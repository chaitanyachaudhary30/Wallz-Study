"""Test suite executing all shared rule scenarios (P05 / Contract C-SCN).

Ensures that every scenario JSON file in `tests/scenarios/` executes cleanly
through `scenario_runner` and satisfies all inline and terminal expectations.
"""
from __future__ import annotations

import pytest

from tests.scenario_runner import load_scenarios, run_scenario


SCENARIOS = load_scenarios()


def test_all_scenarios_loaded():
    """Confirms all 9 expected scenario fixtures are loaded."""
    names = [s["name"] for s in SCENARIOS]
    expected_names = [
        "block_expires_after_K",
        "fivefold_repetition_draw",
        "goal_win",
        "hop_over_wall",
        "move_basics",
        "no_legal_move_draw",
        "path_preservation_rejects_wall",
        "replace_latest_only",
        "wall_place_and_budget",
    ]
    for exp in expected_names:
        assert exp in names, f"Scenario {exp!r} was not loaded"


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s["name"])
def test_scenario_execution(scenario):
    """Executes a scenario and verifies every check succeeded."""
    results = run_scenario(scenario)

    failed_checks = []
    for step in results:
        for check in step["checks"]:
            if not check["ok"]:
                failed_checks.append(
                    f"Ply {step['ply']} key '{check['key']}': {check['msg']}"
                )

    assert not failed_checks, f"Scenario '{scenario['name']}' had check failures:\n" + "\n".join(failed_checks)
