"""Comprehensive test suite for the Web Backend Game API (P08 / Contract C-API & C-STATE).
"""
from __future__ import annotations

import json
import pytest

from wallz_engine import WallzEngine


def test_defaults_and_presets(client):
    """GET /api/config/defaults returns valid default schema, limits, and presets."""
    res = client.get("/api/config/defaults")
    assert res.status_code == 200
    data = res.get_json()

    assert "defaults" in data
    assert data["defaults"]["board_size"] == 5
    assert data["defaults"]["K"] == 3
    assert data["defaults"]["M"] == 10

    assert "limits" in data
    assert data["limits"]["N"] == [3, 9]

    assert "presets" in data
    assert len(data["presets"]) >= 4
    preset_ids = [p["id"] for p in data["presets"]]
    assert "default_5x5" in preset_ids
    assert "dqn_3x3" in preset_ids


def test_agents_list_gates_dqn(client):
    """GET /api/agents correctly evaluates agent compatibility and gating reasons."""
    # N=3, K=1, M=1 -> release dqn_3x3 should be available
    res3 = client.get("/api/agents?N=3&K=1&M=1&A=true&B=true&C=true")
    assert res3.status_code == 200
    agents3 = res3.get_json()
    specs3 = {a["spec"]: a for a in agents3}

    assert "human" in specs3 and specs3["human"]["available"] is True
    assert "greedy" in specs3 and specs3["greedy"]["available"] is True
    # Exact solving never blocks a request: it is either cached or solving in the background
    assert "exact" in specs3
    if not specs3["exact"]["available"]:
        assert "solving" in specs3["exact"]["reason"] or "waiting" in specs3["exact"]["reason"]

    # Find release DQN 3x3
    dqn_3x3_release = [a for a in agents3 if "dqn_3x3" in a["spec"] and "release" in a["spec"]]
    if dqn_3x3_release:
        assert dqn_3x3_release[0]["available"] is True

    # N=3, K=3, M=10 -> release dqn_3x3 must be unavailable with explanation reason
    res3_incompat = client.get("/api/agents?N=3&K=3&M=10&A=true&B=true&C=true")
    assert res3_incompat.status_code == 200
    agents3_inc = res3_incompat.get_json()
    for a in agents3_inc:
        if "dqn_3x3" in a["spec"] and "release" in a["spec"]:
            assert a["available"] is False
            assert "trained for" in (a["reason"] or "")

    # N=9 -> exact solver and depth-2/3 minimax must be unavailable for latency
    res9 = client.get("/api/agents?N=9&K=3&M=10")
    assert res9.status_code == 200
    specs9 = {a["spec"]: a for a in res9.get_json()}
    assert specs9["exact"]["available"] is False
    assert specs9["minimax:depth=2"]["available"] is False


def test_create_human_vs_greedy_and_play_to_end(client):
    """Create human vs greedy game on N=3 and play to terminal state via action and step."""
    res = client.post("/api/games", json={
        "config": {"board_size": 3, "K": 1, "M": 1, "A": True, "B": True, "C": True},
        "p0": "human",
        "p1": "greedy",
    })
    assert res.status_code == 201
    game = res.get_json()
    gid = game["id"]

    for _ in range(50):
        state = game["state"]
        if state["terminal"]["over"]:
            break

        to_move = state["to_move"]
        if to_move == 0:
            # Human turn: choose first available legal action
            # Flatten legal actions
            flat_legal = []
            for act_type, val_list in state["legal"].items():
                for val in val_list:
                    flat_legal.append([act_type, val])
            assert len(flat_legal) > 0

            act_res = client.post(f"/api/games/{gid}/action", json={"action": flat_legal[0]})
            assert act_res.status_code == 200
            game = act_res.get_json()
        else:
            # AI turn: call step
            step_res = client.post(f"/api/games/{gid}/step", json={})
            assert step_res.status_code == 200
            step_data = step_res.get_json()
            game = step_data["session"]

    assert game["state"]["terminal"]["over"] is True
    assert game["state"]["terminal"]["reason"] in ("goal", "repetition", "no_legal_actions", "truncated")


def test_illegal_action_400_state_unchanged(client):
    """Submitting an illegal wall when inventory is 0 returns 400 and leaves state unchanged."""
    res = client.post("/api/games", json={
        "config": {"board_size": 5, "M": 0},
        "p0": "human",
        "p1": "greedy",
    })
    gid = res.get_json()["id"]
    state_before = client.get(f"/api/games/{gid}").get_json()["state"]

    act_res = client.post(f"/api/games/{gid}/action", json={"action": ["WALL", [0, 0, "H"]]})
    assert act_res.status_code == 400
    err_data = act_res.get_json()
    assert "no walls remaining" in err_data["error"]

    state_after = client.get(f"/api/games/{gid}").get_json()["state"]
    assert state_before == state_after


def test_malformed_action_400(client):
    """Malformed action payload returns 400 Bad Request."""
    res = client.post("/api/games", json={"config": {"board_size": 5}, "p0": "human", "p1": "greedy"})
    gid = res.get_json()["id"]

    act_res = client.post(f"/api/games/{gid}/action", json={"action": "INVALID_MOVE"})
    assert act_res.status_code == 400


def test_step_on_human_turn_409(client):
    """Calling /step when it is a human's turn returns 409 Conflict."""
    res = client.post("/api/games", json={"config": {"board_size": 5}, "p0": "human", "p1": "greedy"})
    gid = res.get_json()["id"]

    step_res = client.post(f"/api/games/{gid}/step", json={})
    assert step_res.status_code == 409
    assert "human" in step_res.get_json()["error"]


def test_action_on_ai_turn_409(client):
    """Calling /action when it is an AI's turn returns 409 Conflict."""
    res = client.post("/api/games", json={"config": {"board_size": 5}, "p0": "greedy", "p1": "human"})
    gid = res.get_json()["id"]

    act_res = client.post(f"/api/games/{gid}/action", json={"action": ["MOVE", [1, 2]]})
    assert act_res.status_code == 409
    assert "AI agent" in act_res.get_json()["error"]


def test_actions_after_terminal_409(client):
    """Submitting actions after game over returns 409 Conflict."""
    res = client.post("/api/games", json={"config": {"board_size": 3, "M": 0}, "p0": "human", "p1": "human"})
    gid = res.get_json()["id"]

    # P0 wins on N=3 in 3 plies
    client.post(f"/api/games/{gid}/action", json={"action": ["MOVE", [1, 1]]})
    client.post(f"/api/games/{gid}/action", json={"action": ["MOVE", [2, 0]]})
    win_res = client.post(f"/api/games/{gid}/action", json={"action": ["MOVE", [2, 1]]})
    assert win_res.get_json()["state"]["terminal"]["over"] is True

    # Action after terminal
    after_res = client.post(f"/api/games/{gid}/action", json={"action": ["MOVE", [0, 0]]})
    assert after_res.status_code == 409


def test_undo_equals_replay(client):
    """Undoing 2 plies restores state exactly matching replay from start."""
    res = client.post("/api/games", json={"config": {"board_size": 5, "M": 0}, "p0": "human", "p1": "human"})
    gid = res.get_json()["id"]

    # Play 6 moves back and forth
    moves = [
        ["MOVE", [0, 1]],
        ["MOVE", [4, 1]],
        ["MOVE", [0, 2]],
        ["MOVE", [4, 2]],
        ["MOVE", [0, 1]],
        ["MOVE", [4, 1]],
    ]
    for m in moves:
        client.post(f"/api/games/{gid}/action", json={"action": m})

    # Undo 2 plies
    undo_res = client.post(f"/api/games/{gid}/undo", json={"n": 2})
    assert undo_res.status_code == 200
    state_undone = undo_res.get_json()["state"]

    # Create fresh session and apply first 4 moves
    fresh_res = client.post("/api/games", json={"config": {"board_size": 5, "M": 0}, "p0": "human", "p1": "human"})
    fresh_gid = fresh_res.get_json()["id"]
    for m in moves[:4]:
        client.post(f"/api/games/{fresh_gid}/action", json={"action": m})
    state_fresh = client.get(f"/api/games/{fresh_gid}").get_json()["state"]

    assert state_undone == state_fresh


def test_state_json_contract(client):
    """Validates exact field types and contracts of C-STATE."""
    res = client.post("/api/games", json={"config": {"board_size": 5, "M": 2}, "p0": "human", "p1": "human"})
    gid = res.get_json()["id"]
    # P0 places wall at (0, 0, 'H')
    client.post(f"/api/games/{gid}/action", json={"action": ["WALL", [0, 0, "H"]]})
    state = client.get(f"/api/games/{gid}").get_json()["state"]

    assert state["n"] == 5
    assert len(state["pawns"]) == 2
    assert len(state["goals"]) == 2
    assert len(state["walls"]) == 1
    assert state["walls"][0]["latest"] is True
    assert state["walls"][0]["owner"] == 0
    assert state["inventory"] == [1, 2]
    assert state["legal_count"] > 0
    assert len(state["paths"][0]) > 0
    assert len(state["paths"][1]) > 0


def test_shortest_path_cells():
    """Engine shortest_path_cells returns valid adjacent cell sequence to goal row."""
    e = WallzEngine(board_size=5)
    path = e.shortest_path_cells(0)

    assert len(path) == 5
    assert path[0] == e.pawns[0]
    assert path[-1][0] == e.goals[0]

    # Verify consecutive adjacency
    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i + 1]
        assert abs(r1 - r2) + abs(c1 - c2) == 1
        assert not e.board.is_edge_blocked((r1, c1), (r2, c2))

    assert len(path) - 1 == int(e.shortest_path(0))


def test_exact_solver_gating_never_blocks(client):
    """Exact solver: unsupported configs are refused with a reason; supported ones solve in
    the background and become available once cached (regression: 4x4 M=10 used to hang)."""
    import time
    from web.backend import registry

    t0 = time.time()
    big = {a["spec"]: a for a in client.get("/api/agents?N=4&K=3&M=10").get_json()}
    assert big["exact"]["available"] is False
    assert "too large" in big["exact"]["reason"]
    res = client.post("/api/games", json={"config": {"board_size": 4, "K": 3, "M": 10},
                                          "p0": "human", "p1": "exact"})
    assert res.status_code == 400
    assert time.time() - t0 < 5

    rules = {"A": True, "B": True, "C": True, "K": 1, "M": 0}
    assert registry.wait_for_exact(3, rules, timeout=120)
    small = {a["spec"]: a for a in client.get("/api/agents?N=3&K=1&M=0").get_json()}
    assert small["exact"]["available"] is True
    res = client.post("/api/games", json={"config": {"board_size": 3, "K": 1, "M": 0},
                                          "p0": "exact", "p1": "greedy"})
    assert res.status_code == 201


def test_presets_match_release_checkpoints(client):
    """Every DQN preset must make its release checkpoint available (regression: the 4x4
    preset said M=1 while the checkpoint was trained with M=0)."""
    presets = client.get("/api/config/defaults").get_json()["presets"]
    dqn_presets = [p for p in presets if p["id"].startswith("dqn_")]
    assert dqn_presets
    for p in dqn_presets:
        c = p["config"]
        q = f"N={c['board_size']}&A={c['A']}&B={c['B']}&C={c['C']}&K={c['K']}&M={c['M']}"
        agents = client.get(f"/api/agents?{q}").get_json()
        release = [a for a in agents if a["family"] == "dqn" and "release" in a["spec"]
                   and f"dqn_{c['board_size']}x{c['board_size']}" in a["spec"]]
        assert release and release[0]["available"], p


def test_invalid_configs_return_400(client):
    """Malformed configs are rejected with 400 and a message, never a 500."""
    bad = [{"board_size": "x"}, {"board_size": None, "K": 0}, {"board_size": 5, "max_plies": 0},
           {"board_size": 5, "M": -1}, {"board_size": 12}, {"board_size": 5, "A": "maybe"}]
    for cfg in bad:
        res = client.post("/api/games", json={"config": cfg, "p0": "human", "p1": "greedy"})
        assert res.status_code == 400, cfg
        assert "error" in res.get_json()
    res = client.post("/api/games", json={"config": {"board_size": "5", "A": "false"},
                                          "p0": "human", "p1": "greedy"})
    assert res.status_code == 201
    assert res.get_json()["config"]["A"] is False


def test_undo_in_scenario_game_keeps_scenario_start(client):
    """Undo in a game started from a raw-key scenario rebuilds from that start position
    (regression: it rebuilt from the default initial position)."""
    res = client.post("/api/games/from-scenario/no_legal_move_draw", json={"p0": "human", "p1": "human"})
    assert res.status_code == 201
    s = res.get_json()
    before = s["state"]
    assert s["history"]
    res = client.post(f"/api/games/{s['id']}/undo", json={"n": 1})
    assert res.status_code == 200
    after = res.get_json()["state"]
    assert after["terminal"]["over"] is False
    assert after["turn"] == before["turn"] - 1 or after["turn"] == 0
    # Replaying the undone action from the restored position reproduces the scenario end state
    redo = client.post(f"/api/games/{s['id']}/action", json={"action": s["history"][-1]["action"]})
    assert redo.status_code == 200
    assert redo.get_json()["state"]["pawns"] == before["pawns"]
    assert redo.get_json()["state"]["terminal"] == before["terminal"]


def test_concurrent_steps_apply_exactly_one_ai_move(client):
    """Concurrent /step calls on one game: one AI move, the rest 409, never a 500
    (regression: unsynchronised requests raced on the shared engine)."""
    import threading
    from web.backend.app import create_app

    app = create_app()
    for spec in ("greedy", "minimax:depth=2", "random:seed=0"):
        c0 = app.test_client()
        gid = c0.post("/api/games", json={"config": {"board_size": 5}, "p0": spec, "p1": "human"}).get_json()["id"]
        codes = []

        def hit():
            codes.append(app.test_client().post(f"/api/games/{gid}/step", json={"explain": True}).status_code)

        threads = [threading.Thread(target=hit) for _ in range(4)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        g = c0.get(f"/api/games/{gid}").get_json()
        assert sorted(codes) == [200, 409, 409, 409], (spec, codes)
        assert [h["player"] for h in g["history"]] == [0]
        assert g["state"]["to_move"] == 1


def test_requests_stay_fast_while_exact_solve_runs(client):
    """Exact solves run in a child process: API calls made while one runs must stay fast
    (regression: an in-process solver thread held the GIL and the first /api/agents call
    at startup took ~47 s)."""
    import time
    from web.backend import registry

    registry.exact_status(4, {"A": True, "B": True, "C": True, "K": 2, "M": 0})  # starts a solve if not cached
    worst = 0.0
    for _ in range(5):
        t0 = time.perf_counter()
        assert client.get("/api/agents?N=5&K=3&M=10").status_code == 200
        res = client.post("/api/games", json={"config": {"board_size": 5}, "p0": "human", "p1": "greedy"})
        assert res.status_code == 201
        worst = max(worst, time.perf_counter() - t0)
        time.sleep(0.2)
    assert worst < 1.0, worst


def test_exact_cache_key_tracks_rule_source():
    """Cached solutions are keyed by a hash of engine/solver source, so a rule change
    (like the 2026-11-08 hop change) can never serve a stale solution."""
    from web.backend import registry
    path = registry._cache_path("N3_A1B1C1_K1_M0")
    assert registry._source_fingerprint() in path
    assert path.startswith(registry.EXACT_CACHE_DIR)
