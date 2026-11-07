"""Tests for action normalization in WallzEngine (P01 / Contract C-ACT).

Verifies that apply_action accepts JSON-style list representations as equivalent
to canonical tuples and rejects malformed inputs with ValueError prior to state
mutation, leaving state completely unchanged.
"""
from __future__ import annotations

import pytest
from tests.helpers import make_engine, snapshot
from wallz_engine import WallzEngine


def test_list_actions_equivalent_to_tuples():
    """Applying ['MOVE', [1, 2]] gives the same state as ('MOVE', (1, 2))."""
    e1 = make_engine(n=5)
    e2 = e1.clone()

    e1.apply_action(["MOVE", [1, 2]], update_clock=False)
    e2.apply_action(("MOVE", (1, 2)), update_clock=False)

    s1 = snapshot(e1)
    s2 = snapshot(e2)
    s1.pop("last_turn_timestamp", None)
    s2.pop("last_turn_timestamp", None)
    assert s1 == s2


@pytest.mark.parametrize(
    "malformed_action",
    [
        None,
        "MOVE",
        ("MOVE",),
        ("MOVE", (1,)),
        ("MOVE", (1.0, 2)),
        ("MOVE", (True, 1)),
        ("FLY", (1, 1)),
        ("WALL", (0, 0, "X")),
        ("WALL", (0, 0)),
        ("WALL", (0.0, 0, "H")),
        ("WALL", (0, True, "H")),
        ("BLOCK", (1,)),
        ("BLOCK", (1.5, 2)),
        ("REPLACE", (0, 0)),
        ("REPLACE", (0, 0, "Z")),
        ["MOVE"],
        ["MOVE", [1]],
    ],
)
def test_malformed_actions_raise_value_error(malformed_action):
    """Malformed actions must raise ValueError and leave snapshot unmutated."""
    engine = make_engine(n=5)
    s_before = snapshot(engine)

    with pytest.raises(ValueError):
        engine.apply_action(malformed_action, update_clock=False)

    s_after = snapshot(engine)
    assert s_before == s_after


def test_unknown_type_message_preserved():
    """Engine must raise ValueError containing 'Unknown action type' for unknown types."""
    engine = make_engine(n=5)
    with pytest.raises(ValueError, match=r"Unknown action type"):
        engine.apply_action(("FLY", (1, 1)), update_clock=False)

    with pytest.raises(ValueError, match=r"Unknown action type"):
        engine.apply_action(["TELEPORT", [1, 1]], update_clock=False)


def test_replace_pair_normalised():
    """REPLACE action passed as nested lists/tuples must be accepted and normalized."""
    engine = make_engine(n=5, A=True)
    # Player 0 places a wall at (0, 0, 'H')
    engine.apply_action(("WALL", (0, 0, "H")), update_clock=False)
    # Player 1 moves
    engine.apply_action(("MOVE", (3, 2)), update_clock=False)

    # Player 0 repositions the wall using nested lists
    # latest wall is (0, 0, 'H'); new position is (0, 1, 'H')
    replace_action_list = ["REPLACE", [[0, 0, "H"], [0, 1, "H"]]]
    engine.apply_action(replace_action_list, update_clock=False)

    assert engine.latest_wall[0] == (0, 1, "H")
    assert (0, 1, "H") in engine.walls[0]
    assert (0, 0, "H") not in engine.walls[0]


def test_replace_single_normalised():
    """REPLACE action passed as single triple list is accepted."""
    engine = make_engine(n=5, A=True)
    engine.apply_action(("WALL", (0, 0, "H")), update_clock=False)
    engine.apply_action(("MOVE", (3, 2)), update_clock=False)

    replace_single_list = ["REPLACE", [0, 1, "H"]]
    engine.apply_action(replace_single_list, update_clock=False)

    assert engine.latest_wall[0] == (0, 1, "H")
    assert (0, 1, "H") in engine.walls[0]
