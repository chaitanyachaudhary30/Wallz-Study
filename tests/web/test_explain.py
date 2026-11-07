"""Tests for agent decision explanations (P09 / Contract C-EXPL).

Verifies model explanation accuracy and consistency across Random, Greedy,
Minimax, DQN, and Exact agents, as well as the /api/games/<id>/analyze endpoint.
"""
from __future__ import annotations

import os
import random
import time
import pytest

from agents.greedy import GreedyAgent
from agents.minimax import MinimaxAgent
from tests.helpers import make_engine, random_reachable
from web.backend.explain import explain_and_choose
from web.backend.serialize import action_to_json


def test_greedy_explanation_matches_choice():
    """Greedy explanation chosen action strictly matches select_action across states."""
    rng = random.Random(42)
    for i in range(30):
        n = 3 if i < 15 else 5
        rules = {"K": 1, "M": 1} if n == 3 else {"K": 3, "M": 10}
        plies = rng.randint(0, 10)
        engine = random_reachable(n, rules, plies=plies, seed=100 + i)
        if engine.is_terminal():
            continue

        mover = engine.current_player
        agent = GreedyAgent(mover, seed=7)
        chosen, exp = explain_and_choose(agent, "greedy", engine)

        # Clone and verify select_action gives identical result
        agent_fresh = GreedyAgent(mover, seed=7)
        expected_chosen = agent_fresh.select_action(engine)
        assert chosen == expected_chosen
        assert exp["chosen"] == (action_to_json(chosen) if chosen else None)
        assert exp["player"] == mover
        assert exp["details"]["objective"] == "H = d_opp − d_self"
        assert len(exp["details"]["rows"]) <= 25

        # Overlay covers all legal actions
        legal = engine.get_legal_actions(mover)
        assert len(exp["overlay"]) == len(legal)
        overlay_actions = [item["action"] for item in exp["overlay"]]
        for a in legal:
            assert action_to_json(a) in overlay_actions


def test_minimax_explanation_chosen_is_real_decision():
    """Minimax explanation chosen action equals fresh select_action and is maximal at root."""
    for n in (3, 5):
        rules = {"K": 1, "M": 1} if n == 3 else {"K": 3, "M": 10}
        engine = make_engine(n=n, **rules)
        mover = engine.current_player

        agent = MinimaxAgent(player_id=mover, depth=2)
        chosen, exp = explain_and_choose(agent, "minimax", engine)

        agent_fresh = MinimaxAgent(player_id=mover, depth=2)
        expected_chosen = agent_fresh.select_action(engine)

        assert chosen == expected_chosen
        assert exp["details"]["nodes_expanded"] > 0
        assert exp["details"]["depth_completed"] >= 1
        assert "features_chosen" in exp["details"]
        assert len(exp["details"]["features_chosen"]) == 6

        # The chosen action's analysed root value must be maximal among analyzed actions
        chosen_json = action_to_json(chosen)
        root_entries = [r for r in exp["details"]["root"] if r["kind"] == "exact"]
        assert len(root_entries) > 0
        max_val = max(r["value"] for r in root_entries)
        chosen_entry = next(r for r in root_entries if r["action"] == chosen_json)
        assert chosen_entry["value"] == pytest.approx(max_val, abs=1e-3)


def test_dqn_explanation():
    """DQN explanation chosen action equals DQNAgent.select_action and overlays all Q-values."""
    try:
        import torch
        from agents.dqn.agent import load_checkpoint
    except ImportError:
        pytest.skip("Torch not installed")

    ckpt_dir = "checkpoints/release/dqn_3x3"
    if not os.path.exists(os.path.join(ckpt_dir, "meta.json")):
        pytest.skip("Release DQN checkpoint not found")

    engine = make_engine(n=3, K=1, M=1, A=True, B=True, C=True)
    mover = engine.current_player
    agent = load_checkpoint(ckpt_dir, player_id=mover)

    chosen, exp = explain_and_choose(agent, "dqn", engine)

    agent_fresh = load_checkpoint(ckpt_dir, player_id=mover)
    expected_chosen = agent_fresh.select_action(engine)
    assert chosen == expected_chosen

    legal = engine.get_legal_actions(mover)
    assert len(exp["overlay"]) == len(legal)
    assert exp["details"]["q_max"] == pytest.approx(max(item["value"] for item in exp["overlay"]), abs=1e-4)
    assert len(exp["details"]["top5"]) <= 5
    assert exp["details"]["n_legal"] == len(legal)


def test_exact_explanation():
    """Exact explanation accurately displays WIN/LOSS/DRAW ranks and values."""
    from evaluation.match_runner import make_agent, MatchConfig
    from evaluation.state_graph import GraphCaps

    cfg = MatchConfig(board_size=3, rules={"A": False, "B": False, "C": False, "K": 1, "M": 1})
    solutions = {}
    agent, _ = make_agent("exact", 0, cfg, solutions=solutions)
    solution = solutions[cfg.label()]

    rng = random.Random(99)
    tested = 0
    for i in range(50):
        engine = random_reachable(3, cfg.rules, plies=rng.randint(0, 4), seed=200 + i)
        if engine.is_terminal():
            continue
        node = solution.graph.node_of(engine)
        if node is None:
            continue

        mover = engine.current_player
        mover_agent, _ = make_agent("exact", mover, cfg, solutions=solutions)
        chosen, exp = explain_and_choose(mover_agent, "exact", engine)

        assert exp["details"]["status"] in ("WIN", "LOSS", "DRAW")
        assert exp["details"]["states_in_graph"] == len(solution.graph.keys)
        assert chosen in solution.rank_optimal_actions(node)

        # Validate that WIN actions match value-optimal actions
        if exp["details"]["status"] == "WIN":
            opt_actions = [action_to_json(a) for a in solution.value_optimal_actions(node)]
            win_overlay = [item["action"] for item in exp["overlay"] if item["label"].startswith("WIN")]
            assert len(win_overlay) == len(opt_actions)

        tested += 1
        if tested >= 20:
            break

    assert tested >= 10


def test_analyze_does_not_change_game(client):
    """POST /api/games/<id>/analyze provides insights without altering game state."""
    res = client.post("/api/games", json={
        "config": {"board_size": 3, "K": 1, "M": 1},
        "p0": "human",
        "p1": "greedy",
    })
    assert res.status_code == 201
    game = res.get_json()
    gid = game["id"]

    # Human moves
    res_move = client.post(f"/api/games/{gid}/action", json={"action": ["MOVE", [1, 1]]})
    assert res_move.status_code == 200
    state_before = res_move.get_json()["state"]

    # Analyze for greedy
    res_an1 = client.post(f"/api/games/{gid}/analyze", json={"agent": "greedy"})
    assert res_an1.status_code == 200
    exp1 = res_an1.get_json()
    assert exp1["agent"] == "Greedy"
    assert "overlay" in exp1
    assert "details" in exp1

    # Analyze for minimax
    res_an2 = client.post(f"/api/games/{gid}/analyze", json={"agent": "minimax:depth=1"})
    assert res_an2.status_code == 200
    exp2 = res_an2.get_json()
    assert exp2["agent"] == "Minimax"

    # Confirm session state remains completely unchanged
    res_get = client.get(f"/api/games/{gid}")
    assert res_get.status_code == 200
    state_after = res_get.get_json()["state"]
    assert state_before == state_after


def test_step_returns_explanation_schema(client):
    """POST /api/games/<id>/step returns full Contract C-EXPL explanation schema."""
    res = client.post("/api/games", json={
        "config": {"board_size": 3, "K": 1, "M": 1},
        "p0": "greedy",
        "p1": "human",
    })
    gid = res.get_json()["id"]

    step_res = client.post(f"/api/games/{gid}/step", json={"explain": True})
    assert step_res.status_code == 200
    data = step_res.get_json()

    exp = data["explanation"]
    assert exp is not None
    required_keys = {"agent", "spec", "player", "chosen", "decision_seconds", "summary", "details", "overlay"}
    assert required_keys.issubset(exp.keys())
    assert isinstance(exp["overlay"], list)
    assert len(exp["overlay"]) > 0


@pytest.mark.slow
def test_minimax_explain_budget():
    """Minimax explanation on N=9 depth 2 runs within interactive budget (< 15s)."""
    engine = make_engine(n=9, K=3, M=10)
    agent = MinimaxAgent(player_id=0, depth=2)
    t0 = time.time()
    chosen, exp = explain_and_choose(agent, "minimax", engine)
    elapsed = time.time() - t0
    assert elapsed < 15.0, f"Minimax N=9 depth 2 explanation took {elapsed:.2f}s (budget: 15s)"
    assert chosen is not None
    assert len(exp["overlay"]) == len(engine.get_legal_actions())
