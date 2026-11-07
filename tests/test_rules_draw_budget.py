"""Test suite for No-Legal-Move Draw (Task 1C) and Wall-Budget mechanics (Task 1D / P04).

Verifies:
1. Engine, state restore, runner, and agents agree on no-legal-move draw semantics.
2. Terminal states strictly adhere to consistency invariants across hundreds of reachable states.
3. Wall inventory behaves correctly across M in {0, 1, 2, 3, 10, default}, stops at zero, allows REPLACE when M=1 and inventory=0.
4. Validation rejects negative M in both WallzConfig and WallzEngine.
5. Replays in full matches under Greedy and Random strictly observe wall budget M.
6. Static audit verifies no hardcoded wall inventory assumptions exist outside documented defaults.
"""
from __future__ import annotations

import json
import os
import re
import pytest

from evaluation.match_runner import MatchConfig, RandomAgent, run_match
from evaluation.metrics import outcome_for
from greedy_agent import GreedyAgent
from src.modified_wallz.config import WallzConfig
from tests.helpers import key_from_json, make_engine, pass_move, random_reachable
from wallz_engine import WallzEngine


# ============================================================================
# No-Legal-Move Draw Tests (Task 1C)
# ============================================================================

def _load_no_legal_fixture():
    path = os.path.join(os.path.dirname(__file__), "fixtures", "no_legal_draw.json")
    with open(path) as f:
        return json.load(f)


def test_apply_action_produces_no_legal_draw():
    """Applying the fixture action in parent state leaves opponent with no legal moves, triggering draw."""
    data = _load_no_legal_fixture()
    parent_key = key_from_json(data["parent_key"])
    engine = WallzEngine(board_size=data["board_size"], rules=data["rules"]).restore_state_key(parent_key)

    action = (data["action"][0], tuple(data["action"][1]))
    state, winner, is_draw = engine.apply_action(action, update_clock=False)

    assert is_draw is True
    assert winner is None
    assert engine.winner is None
    assert engine.terminal_reason == "no_legal_actions"
    assert engine.is_terminal() is True
    assert engine.get_legal_actions() == []
    assert engine.has_legal_action() is False


def test_restore_detects_no_legal_draw():
    """Restoring directly from a child key with no legal actions sets draw and terminal flags."""
    data = _load_no_legal_fixture()
    child_key = key_from_json(data["child_key"])
    engine = WallzEngine(board_size=data["board_size"], rules=data["rules"]).restore_state_key(child_key)

    assert engine.is_draw is True
    assert engine.winner is None
    assert engine.terminal_reason == "no_legal_actions"
    assert engine.is_terminal() is True
    assert engine.get_legal_actions() == []
    assert engine.has_legal_action() is False


def test_winner_and_terminal_agree():
    """Across 300 reachable states, terminality, winner, and legal action invariants strictly agree."""
    for seed in range(300):
        n = 3 if seed % 2 == 0 else 4
        m = 0 if (seed // 2) % 2 == 0 else 1
        rules = {"A": True, "B": True, "C": True, "K": 2, "M": m}
        e = random_reachable(n, rules, plies=(seed % 20), seed=seed)

        # Invariant 1: is_terminal() == (winner is not None or is_draw)
        assert e.is_terminal() == (e.winner is not None or e.is_draw)

        # Invariant 2: winner is not None => not is_draw
        if e.winner is not None:
            assert not e.is_draw

        # Invariant 3: not is_terminal() => has_legal_action() and len(get_legal_actions()) > 0
        if not e.is_terminal():
            assert e.has_legal_action()
            assert len(e.get_legal_actions()) > 0

        # Invariant 4: has_legal_action() == bool(get_legal_actions())
        assert e.has_legal_action() == bool(e.get_legal_actions())


def test_runner_reports_no_legal_draw():
    """Match runner correctly formats outcome for no-legal draws, and a full game reproduces it."""
    data = _load_no_legal_fixture()
    parent_key = key_from_json(data["parent_key"])
    engine = WallzEngine(board_size=data["board_size"], rules=data["rules"]).restore_state_key(parent_key)
    action = (data["action"][0], tuple(data["action"][1]))
    engine.apply_action(action, update_clock=False)

    # 1. Exact expression in match runner: f"draw_{engine.terminal_reason}"
    category = f"draw_{engine.terminal_reason}"
    assert category == "draw_no_legal_actions"

    # 2. outcome_for treats draw_no_legal_actions as a draw for both players
    fake_result = {"outcome": {"category": category, "winner": None, "loser": None}}
    assert outcome_for(fake_result, 0) == "draw"
    assert outcome_for(fake_result, 1) == "draw"

    # 3. Full match replay from initial position under recorded seed
    if data.get("full_game"):
        fg = data["full_game"]
        cfg = MatchConfig(
            board_size=data["board_size"],
            rules=data["rules"],
            p0=fg["p0"],
            p1=fg["p1"],
            opening_seed=fg["seed"],
        )
        res = run_match(cfg)
        assert res["outcome"]["category"] == "draw_no_legal_actions"


def test_agents_return_none_without_legal_actions():
    """Both GreedyAgent and RandomAgent return None when called on a state with no legal actions."""
    data = _load_no_legal_fixture()
    child_key = key_from_json(data["child_key"])
    engine = WallzEngine(board_size=data["board_size"], rules=data["rules"]).restore_state_key(child_key)
    p = engine.current_player

    assert engine.get_legal_actions(p) == []

    ra = RandomAgent(p)
    assert ra.select_action(engine) is None

    ga = GreedyAgent(p)
    assert ga.select_action(engine) is None


# ============================================================================
# Wall Budget Tests (Task 1D / rules['M'])
# ============================================================================

@pytest.mark.parametrize("m_val", [0, 1, 3, 10, "default"])
def test_initial_inventory(m_val):
    """Initial wall inventory must equal M (defaulting to 10 when unspecified)."""
    if m_val == "default":
        engine = WallzEngine(board_size=5)
        expected_m = 10
    else:
        engine = make_engine(n=5, M=m_val)
        expected_m = m_val

    assert engine.wall_inventory == {0: expected_m, 1: expected_m}
    assert engine.max_walls == expected_m


def test_wall_decrements_and_stops_at_zero():
    """Wall placements decrement inventory until 0, after which WALL actions are illegal and raise."""
    engine = make_engine(n=5, M=2)
    assert engine.wall_inventory[0] == 2

    # Placement 1
    engine.apply_action(("WALL", (0, 0, "H")), update_clock=False)
    assert engine.wall_inventory[0] == 1
    engine.apply_action(pass_move(engine), update_clock=False)  # P1

    # Placement 2
    engine.apply_action(("WALL", (2, 2, "H")), update_clock=False)
    assert engine.wall_inventory[0] == 0
    engine.apply_action(pass_move(engine), update_clock=False)  # P1

    # No further WALL actions in legal actions
    p0_walls = [a for a in engine.get_legal_actions(0) if a[0] == "WALL"]
    assert p0_walls == []

    # Attempting WALL placement at inventory 0 raises ValueError
    with pytest.raises(ValueError, match=r"no walls remaining"):
        engine.apply_action(("WALL", (1, 1, "V")), update_clock=False)


def test_m_zero_has_no_wall_or_replace():
    """When M=0, neither player may place or replace walls even with Rule A enabled."""
    engine = make_engine(n=5, A=True, M=0)
    for p in (0, 1):
        assert engine.wall_inventory[p] == 0
        actions = engine.get_legal_actions(p)
        assert not any(a[0] in ("WALL", "REPLACE") for a in actions)


def test_replace_after_budget_spent():
    """When M=1 and Rule A is enabled, spending the single wall permits REPLACE while keeping inventory 0."""
    engine = make_engine(n=5, A=True, M=1)
    engine.apply_action(("WALL", (0, 0, "H")), update_clock=False)
    assert engine.wall_inventory[0] == 0

    engine.apply_action(pass_move(engine), update_clock=False)  # P1

    # REPLACE is present in legal actions
    replace_actions = [a for a in engine.get_legal_actions(0) if a[0] == "REPLACE"]
    assert len(replace_actions) > 0

    engine.apply_action(("REPLACE", (0, 1, "H")), update_clock=False)
    assert engine.wall_inventory[0] == 0
    assert engine.latest_wall[0] == (0, 1, "H")


def test_config_rejects_negative_m():
    """WallzConfig and WallzEngine reject negative M with ValueError."""
    with pytest.raises(ValueError, match=r"inventory M must be >= 0"):
        WallzConfig(M=-1)

    with pytest.raises(ValueError):
        WallzEngine(rules={"M": -1})


def test_m_respected_by_greedy_and_random():
    """Full matches of Greedy vs Random across M in {0, 2} never exceed the wall budget."""
    for m in (0, 2):
        for seed in range(20):
            cfg = MatchConfig(
                board_size=5,
                rules={"M": m},
                p0="greedy",
                p1="random",
                opening_seed=seed,
                max_plies=40,
            )
            res = run_match(cfg)
            wall_counts = {0: 0, 1: 0}
            for ply in res["replay"]:
                if ply["action"][0] == "WALL":
                    wall_counts[ply["player"]] += 1
            assert wall_counts[0] <= m
            assert wall_counts[1] <= m


def test_no_hardcoded_wall_budget():
    """Audit codebase to ensure no component hardcodes wall inventory to 10 outside documented defaults."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    files_to_check = [
        os.path.join(repo_root, "agents", "greedy.py"),
        os.path.join(repo_root, "evaluation", "match_runner.py"),
        os.path.join(repo_root, "evaluation", "features.py"),
    ]
    web_dir = os.path.join(repo_root, "web")
    if os.path.exists(web_dir):
        for root, _, files in os.walk(web_dir):
            for f in files:
                if f.endswith(".py"):
                    files_to_check.append(os.path.join(root, f))

    pattern = re.compile(
        r"(?:wall|inventory|'M'|\"M\").{0,40}\b10\b|\b10\b.{0,40}(?:wall|inventory|'M'|\"M\")",
        re.IGNORECASE,
    )

    hits = []
    for path in files_to_check:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line_no, line in enumerate(f, 1):
                if pattern.search(line):
                    hits.append(f"{path}:{line_no}: {line.strip()}")

    assert hits == [], f"Unexpected hardcoded wall budget references found: {hits}"
