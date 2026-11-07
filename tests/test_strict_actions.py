"""Strict action validation, atomicity, clock isolation, and terminal freeze tests (P02 / F1).

Verifies that:
1. Illegal actions never mutate any state (including clock time) - atomic rollback/rejection.
2. An illegal action after flag fall raises ValueError rather than awarding a timeout win.
3. A legal action after flag fall awards a timeout win and halts the game.
4. Terminal games reject all subsequent actions and freeze legal actions to empty.
5. Error messages remain consistent and stable.
"""
from __future__ import annotations

import random
import time
import unittest
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.modified_wallz.config import WallzConfig
from tests.helpers import illegal_action_pool, make_engine, random_reachable, snapshot
from wallz_engine import WallzEngine


@settings(max_examples=150, deadline=None)
@given(
    n=st.sampled_from([3, 4, 5]),
    a=st.booleans(),
    b=st.booleans(),
    c=st.booleans(),
    k=st.sampled_from([1, 2, 3]),
    m=st.sampled_from([0, 1, 3]),
    plies=st.integers(min_value=0, max_value=30),
    seed=st.integers(min_value=0, max_value=100000),
)
def test_illegal_actions_atomic_property(n, a, b, c, k, m, plies, seed):
    """Applying any illegal action must raise ValueError and leave snapshot completely unchanged."""
    rules = {"A": a, "B": b, "C": c, "K": k, "M": m}
    engine = random_reachable(n=n, rules=rules, plies=plies, seed=seed)
    if engine.is_terminal():
        return

    rng = random.Random(seed)
    pool = illegal_action_pool(engine, rng, k=10)
    for act in pool:
        s_before = snapshot(engine)
        with pytest.raises(ValueError):
            engine.apply_action(act, update_clock=False)
        s_after = snapshot(engine)
        assert s_before == s_after


def test_illegal_action_does_not_touch_clock():
    """An illegal action with update_clock=True must not modify time_remaining or last_turn_timestamp."""
    engine = make_engine(n=5)
    engine.last_turn_timestamp = time.time() - 5.0
    t_rem_before = dict(engine.time_remaining)
    last_ts_before = engine.last_turn_timestamp

    with pytest.raises(ValueError):
        # Move out of bounds / illegal
        engine.apply_action(("MOVE", (99, 99)), update_clock=True)

    assert engine.time_remaining == t_rem_before
    assert engine.last_turn_timestamp == last_ts_before


def test_illegal_action_after_flag_fall_raises():
    """If flag has fallen (time expired), an illegal action must raise ValueError and NOT award timeout win."""
    engine = make_engine(n=5)
    engine.time_remaining[0] = 0.001
    engine.last_turn_timestamp = time.time() - 1.0

    with pytest.raises(ValueError):
        engine.apply_action(("MOVE", (99, 99)), update_clock=True)

    assert engine.winner is None
    assert engine.terminal_reason is None


def test_legal_action_after_flag_fall_is_timeout():
    """If flag has fallen, a legal action triggers timeout win for opponent and freezes game."""
    engine = make_engine(n=5)
    engine.time_remaining[0] = 0.001
    engine.last_turn_timestamp = time.time() - 1.0

    # Legal move for player 0 from (0, 2) to (1, 2)
    state, winner, is_draw = engine.apply_action(("MOVE", (1, 2)), update_clock=True)

    assert winner == 1
    assert engine.winner == 1
    assert engine.terminal_reason == "timeout"
    assert engine.turn_count == 0  # Turn did not complete

    # Next action must be rejected because game is terminal
    with pytest.raises(ValueError, match=r"Game is already over"):
        engine.apply_action(("MOVE", (3, 2)), update_clock=False)


@settings(max_examples=100, deadline=None)
@given(
    n=st.sampled_from([3, 4, 5]),
    a=st.booleans(),
    b=st.booleans(),
    c=st.booleans(),
    k=st.sampled_from([1, 2, 3]),
    m=st.sampled_from([0, 1, 3]),
    plies=st.integers(min_value=0, max_value=20),
    seed=st.integers(min_value=0, max_value=100000),
)
def test_legal_actions_never_raise(n, a, b, c, k, m, plies, seed):
    """Every action returned by get_legal_actions must apply cleanly on a clone."""
    rules = {"A": a, "B": b, "C": c, "K": k, "M": m}
    engine = random_reachable(n=n, rules=rules, plies=plies, seed=seed)
    if engine.is_terminal():
        return

    actions = engine.get_legal_actions()
    for act in actions:
        clone = engine.clone()
        clone.apply_action(act, update_clock=False)


def _build_goal_terminal_engine() -> WallzEngine:
    """Build N=3 engine where player 0 has reached goal row 2."""
    e = make_engine(n=3, B=False, C=False, M=0)
    e.apply_action(("MOVE", (1, 1)), update_clock=False)  # P0
    e.apply_action(("MOVE", (2, 0)), update_clock=False)  # P1
    e.apply_action(("MOVE", (2, 1)), update_clock=False)  # P0 wins (goal row 2)
    assert e.is_terminal() and e.terminal_reason == "goal"
    return e


def _build_repetition_terminal_engine() -> WallzEngine:
    """Build N=5 engine reaching 5-fold repetition."""
    e = make_engine(n=5, A=False, B=False, C=False, M=0)
    for _ in range(30):
        if e.is_terminal():
            break
        p = e.current_player
        r = e.pawns[p][0]
        c = e.pawns[p][1]
        target_c = 1 if c == 2 else 2
        e.apply_action(("MOVE", (r, target_c)), update_clock=False)
    assert e.is_terminal() and e.terminal_reason == "repetition"
    return e


def _build_truncated_terminal_engine() -> WallzEngine:
    """Build engine truncated at max_plies=4."""
    e = WallzEngine(config=WallzConfig(board_size=5, max_plies=4))
    e.apply_action(("MOVE", (0, 1)), update_clock=False)
    e.apply_action(("MOVE", (4, 1)), update_clock=False)
    e.apply_action(("MOVE", (0, 2)), update_clock=False)
    e.apply_action(("MOVE", (4, 2)), update_clock=False)
    assert e.is_terminal() and e.terminal_reason == "truncated"
    return e


@pytest.mark.parametrize(
    "engine_builder",
    [
        _build_goal_terminal_engine,
        _build_repetition_terminal_engine,
        _build_truncated_terminal_engine,
        pytest.param(
            lambda: None,
            marks=pytest.mark.skip(reason="fixture in P04"),
            id="no_legal_actions",
        ),
    ],
)
def test_actions_rejected_after_terminal(engine_builder):
    """Any terminal game state must reject all actions and report no legal actions."""
    engine = engine_builder()
    assert engine.is_terminal()
    assert engine.get_legal_actions() == []
    assert not engine.has_legal_action()

    s_before = snapshot(engine)
    with pytest.raises(ValueError, match=r"Game is already over"):
        engine.apply_action(("MOVE", (0, 0)), update_clock=False)
    s_after = snapshot(engine)
    assert s_before == s_after


def test_error_messages_stable():
    """Engine must raise specific, stable error messages for canonical illegal actions."""
    engine = make_engine(n=5, A=False, C=True)
    # Illegal move
    with pytest.raises(ValueError, match=r"^Illegal move to"):
        engine.apply_action(("MOVE", (0, 0)), update_clock=False)

    # Wall placement with inventory 0
    engine.wall_inventory[0] = 0
    with pytest.raises(ValueError, match=r"no walls remaining"):
        engine.apply_action(("WALL", (0, 0, "H")), update_clock=False)

    # REPLACE with Rule A disabled
    with pytest.raises(ValueError, match=r"Rule A"):
        engine.apply_action(("REPLACE", (0, 0, "H")), update_clock=False)

    # BLOCK reuse when block already consumed
    engine.block_available[0] = False
    with pytest.raises(ValueError, match=r"no cell block available"):
        engine.apply_action(("BLOCK", (2, 2)), update_clock=False)
