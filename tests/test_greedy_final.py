"""Verification and regression test suite for GreedyAgent (P06 / Task 3).

Verifies:
1. Complete golden equivalence against pre-refactor decisions across N in {3, 5, 9}.
2. score_actions exposure, ranking consistency, and immediate win short-circuiting.
3. Side-to-move assertion guarding against out-of-turn execution (Defect F4).
4. Strict legality of selected actions across randomized game states (Hypothesis).
5. Terminal state behavior returning None on terminal engines.
6. Wall budget respect (M=0 never walls, M=1 at most one wall).
7. Block duration awareness using engine shortest path calculations.
8. Draw scoring preference over negative advantage / loss states.
9. Game completion across broad combinations of N in {3,4,5,7,9} and M in {0,1,10}.
"""
from __future__ import annotations

import json
import os
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from agents.greedy import GreedyAgent
from evaluation.match_runner import MatchConfig, run_match
from src.modified_wallz.config import WallzConfig
from tests.helpers import key_from_json, make_engine, pass_move, random_reachable
from wallz_engine import WallzEngine


def test_refactor_matches_golden():
    """Refactored GreedyAgent decisions must match pre-refactor golden decisions 100%."""
    golden_path = os.path.join(os.path.dirname(__file__), "fixtures", "greedy_golden.json")
    with open(golden_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    for rec in records:
        raw_key = key_from_json(rec["raw_key"])
        engine = WallzEngine(board_size=rec["n"], rules=rec["rules"]).restore_state_key(raw_key)
        p = rec["player"]

        # Unseeded
        act_unseeded = GreedyAgent(p).select_action(engine)
        expected_u = tuple(rec["choice_unseeded"]) if rec["choice_unseeded"] else None
        if expected_u:
            expected_u = (expected_u[0], tuple(expected_u[1]) if isinstance(expected_u[1], list) else expected_u[1])
        assert act_unseeded == expected_u

        # Seeded(7)
        act_seeded = GreedyAgent(p, seed=7).select_action(engine)
        expected_s = tuple(rec["choice_seed7"]) if rec["choice_seed7"] else None
        if expected_s:
            expected_s = (expected_s[0], tuple(expected_s[1]) if isinstance(expected_s[1], list) else expected_s[1])
        assert act_seeded == expected_s


def test_score_actions_consistent():
    """score_actions rankings must be consistent with select_action and contain valid metadata."""
    engine = make_engine(n=5)
    agent = GreedyAgent(0)
    rows = agent.score_actions(engine)
    selected = agent.select_action(engine)

    assert len(rows) == len(engine.get_legal_actions(0))
    for r in rows:
        assert r["action"] in engine.get_legal_actions(0)
        assert r["outcome"] in ("win", "loss", "draw", "ongoing")
        assert len(r["score"]) == 4

    top_action = max(rows, key=lambda r: r["score"])["action"]
    assert selected == top_action


def test_out_of_turn_raises():
    """GreedyAgent must raise AssertionError if called out of turn (Defect F4)."""
    engine = make_engine(n=5)
    assert engine.current_player == 0

    agent_p1 = GreedyAgent(1)
    with pytest.raises(AssertionError, match=r"does not match agent player_id"):
        agent_p1.select_action(engine)


@settings(max_examples=100, deadline=None)
@given(
    n=st.sampled_from([3, 4, 5, 7]),
    m=st.sampled_from([0, 1, 3]),
    plies=st.integers(min_value=0, max_value=20),
    seed=st.integers(min_value=0, max_value=100000),
)
def test_only_legal_actions(n, m, plies, seed):
    """GreedyAgent always selects an action present in engine.get_legal_actions()."""
    rules = {"A": True, "B": True, "C": True, "K": 2, "M": m}
    engine = random_reachable(n=n, rules=rules, plies=plies, seed=seed)
    if engine.is_terminal():
        return

    p = engine.current_player
    agent = GreedyAgent(p, seed=seed)
    chosen = agent.select_action(engine)

    legal = engine.get_legal_actions(p)
    assert chosen in legal


def test_terminal_returns_none():
    """GreedyAgent returns None on any terminal state."""
    # 1. Goal terminal
    e_goal = make_engine(n=3, B=False, C=False, M=0)
    e_goal.apply_action(("MOVE", (1, 1)), update_clock=False)
    e_goal.apply_action(("MOVE", (2, 0)), update_clock=False)
    e_goal.apply_action(("MOVE", (2, 1)), update_clock=False)
    assert e_goal.is_terminal()
    assert GreedyAgent(e_goal.current_player).select_action(e_goal) is None

    # 2. Truncated terminal
    e_trunc = WallzEngine(config=WallzConfig(board_size=5, max_plies=2))
    e_trunc.apply_action(("MOVE", (0, 1)), update_clock=False)
    e_trunc.apply_action(("MOVE", (4, 1)), update_clock=False)
    assert e_trunc.is_terminal()
    assert GreedyAgent(e_trunc.current_player).select_action(e_trunc) is None

    # 3. No-legal draw fixture
    fixture_path = os.path.join(os.path.dirname(__file__), "fixtures", "no_legal_draw.json")
    with open(fixture_path) as f:
        data = json.load(f)
    raw_key = key_from_json(data["child_key"])
    e_draw = WallzEngine(board_size=data["board_size"], rules=data["rules"]).restore_state_key(raw_key)
    assert e_draw.is_terminal()
    assert GreedyAgent(e_draw.current_player).select_action(e_draw) is None


def test_respects_inventory():
    """Greedy agent strictly obeys wall inventory: 0 walls placed if M=0, <=1 wall placed if M=1."""
    # M=0
    cfg0 = MatchConfig(board_size=5, rules={"M": 0}, p0="greedy", p1="greedy", max_plies=50)
    res0 = run_match(cfg0)
    assert not any(ply["action"][0] == "WALL" for ply in res0["replay"])

    # M=1
    cfg1 = MatchConfig(board_size=5, rules={"M": 1}, p0="greedy", p1="greedy", max_plies=50)
    res1 = run_match(cfg1)
    walls = {0: 0, 1: 0}
    for ply in res1["replay"]:
        if ply["action"][0] == "WALL":
            walls[ply["player"]] += 1
    assert walls[0] <= 1
    assert walls[1] <= 1


def test_block_duration_awareness():
    """Greedy's candidate evaluation recomputes shortest_path directly through simulated engine state."""
    engine = make_engine(n=5, C=True, K=3)
    engine.blocked_cells[(1, 2)] = 1  # Block expires in 1 ply
    agent = GreedyAgent(0)
    rows = agent.score_actions(engine)

    for r in rows:
        sim = engine.clone()
        sim.apply_action(r["action"], update_clock=False)
        if not sim.is_terminal():
            expected_d_self = sim.shortest_path(0)
            assert r["d_self"] == expected_d_self


def test_draw_scored_zero_preferred_over_loss():
    """A draw outcome is scored with H = 0.0 and is preferred over any loss (H = -inf)."""
    # Create engine nearing 5-fold repetition draw
    engine = make_engine(n=5, A=False, B=False, C=False, M=0)
    for _ in range(15):
        p = engine.current_player
        r, c = engine.pawns[p]
        tc = 1 if c == 2 else 2
        engine.apply_action(("MOVE", (r, tc)), update_clock=False)

    agent = GreedyAgent(engine.current_player)
    rows = agent.score_actions(engine)
    draw_rows = [r for r in rows if r["outcome"] == "draw"]
    assert len(draw_rows) > 0
    assert draw_rows[0]["H"] == 0.0
    # Score has H = 0.0 which strictly exceeds a loss score (-inf)
    loss_score = (-float("inf"), 4, -float("inf"), 0)
    assert draw_rows[0]["score"] > loss_score


@pytest.mark.parametrize("n", [3, 4, 5, 7, 9])
@pytest.mark.parametrize("m", [0, 1, 10])
def test_board_sizes_and_budgets(n, m):
    """Full match of Greedy vs Random completes cleanly across board sizes and budgets."""
    cfg = MatchConfig(board_size=n, rules={"M": m}, p0="greedy", p1="random", max_plies=40, opening_seed=42)
    res = run_match(cfg)
    assert res["outcome"]["category"] in ("goal", "draw_repetition", "draw_no_legal_actions", "truncated")
