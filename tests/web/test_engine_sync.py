"""The web API reports exactly what the engine says (Stage 3, Member 2 Task 4).

Each test drives a human-vs-human game through the Flask API with a scripted position and,
after every ply, compares the served state with a fresh WallzEngine that replays the same
actions: legal actions, inventory, walls (owner + latest), hop/block flags, block timers,
side to move and the terminal reason. Covers MOVE, WALL + inventory, REPLACE of the latest
wall only, the one-time HOP across an opponent's wall, BLOCK lasting exactly K plies, path
preservation, and every terminal reason (goal, no_legal_actions, repetition, truncated).
"""
from __future__ import annotations

from src.modified_wallz.config import WallzConfig
from tests.scenario_runner import engine_for, load_scenarios
from wallz_engine import WallzEngine

OFF = {"A": False, "B": False, "C": False}


def flat(legal):
    return {(t, tuple(v)) for t, vals in legal.items() for v in vals}


def assert_matches_engine(state, e):
    legal = e.get_legal_actions()
    assert flat(state["legal"]) == {(t, tuple(v)) for t, v in legal}
    assert state["legal_count"] == len(legal)
    assert state["pawns"] == [list(e.pawns[0]), list(e.pawns[1])]
    assert state["inventory"] == [e.wall_inventory[0], e.wall_inventory[1]]
    assert state["hop"] == [e.hop_available[0], e.hop_available[1]]
    assert state["block"] == [e.block_available[0], e.block_available[1]]
    assert {(b["r"], b["c"]): b["remaining"] for b in state["blocked"]} == dict(e.blocked_cells)
    walls = {(w["r"], w["c"], w["o"], w["owner"], w["latest"]) for w in state["walls"]}
    assert walls == {(*w, p, w == e.latest_wall[p]) for p in (0, 1) for w in e.walls[p]}
    assert state["to_move"] == e.current_player and state["turn"] == e.turn_count
    assert state["terminal"] == {"over": e.is_terminal(), "winner": e.winner, "reason": e.terminal_reason}


class Game:
    """One API game plus a shadow engine built from the same config."""

    def __init__(self, client, n, **cfg):
        cfg = {"board_size": n, **cfg}
        res = client.post("/api/games", json={"config": cfg, "p0": "human", "p1": "human"})
        assert res.status_code == 201, res.get_json()
        self.client, self.gid, self.state = client, res.get_json()["id"], res.get_json()["state"]
        rules = {k: cfg.get(k, d) for k, d in (("A", True), ("B", True), ("C", True), ("K", 3), ("M", 10))}
        self.engine = WallzEngine(config=WallzConfig(board_size=n, max_plies=cfg.get("max_plies", 200), **rules))
        assert_matches_engine(self.state, self.engine)

    def play(self, action):
        res = self.client.post(f"/api/games/{self.gid}/action", json={"action": action})
        assert res.status_code == 200, res.get_json()
        self.state = res.get_json()["state"]
        self.engine.apply_action(self.engine._normalize_action(action), update_clock=False)
        assert_matches_engine(self.state, self.engine)

    def reject(self, action, text, code=400):
        before = self.client.get(f"/api/games/{self.gid}").get_json()["state"]
        res = self.client.post(f"/api/games/{self.gid}/action", json={"action": action})
        assert res.status_code == code
        assert text in res.get_json()["error"]
        assert self.client.get(f"/api/games/{self.gid}").get_json()["state"] == before


def test_move_wall_and_inventory(client):
    g = Game(client, 5, M=2, K=3)
    assert g.state["inventory"] == [2, 2]
    assert sorted(g.state["legal"]["MOVE"]) == [[0, 1], [0, 3], [1, 2]]
    g.play(["WALL", [0, 0, "H"]])
    g.play(["MOVE", [3, 2]])
    g.play(["WALL", [2, 2, "V"]])
    g.play(["MOVE", [3, 1]])
    assert g.state["inventory"] == [0, 2] and g.state["legal"]["WALL"] == []
    assert g.state["legal"]["REPLACE"]          # zero budget still allows REPLACE
    g.reject(["WALL", [1, 0, "H"]], "no walls remaining")


def test_replace_moves_only_the_latest_wall(client):
    g = Game(client, 5, M=5, A=True, B=False, C=False)
    for a in (["WALL", [0, 0, "H"]], ["MOVE", [3, 2]], ["WALL", [1, 1, "H"]], ["MOVE", [3, 1]]):
        g.play(a)
    g.reject(["REPLACE", [[0, 0, "H"], [0, 2, "H"]]], "only latest wall")
    g.play(["REPLACE", [1, 2, "H"]])
    assert g.state["inventory"] == [3, 5]
    assert [(w["r"], w["c"]) for w in g.state["walls"] if w["latest"]] == [(1, 2)]


def test_hop_needs_an_opponent_wall_and_is_one_time(client):
    g = Game(client, 5, M=2, A=False, B=True, C=False)
    assert g.state["legal"]["HOP"] == []
    g.play(["MOVE", [1, 2]])
    g.play(["WALL", [1, 2, "H"]])                # P1's wall in front of P0
    assert g.state["legal"]["HOP"] == [[2, 2]] and [2, 2] not in g.state["legal"]["MOVE"]
    g.play(["HOP", [2, 2]])
    assert g.state["hop"] == [False, True]
    g.play(["MOVE", [4, 1]])
    g.reject(["HOP", [3, 2]], "Illegal hop")


def test_block_lasts_exactly_k_plies(client):
    for k in (1, 2, 3):
        g = Game(client, 5, M=0, K=k, A=False, B=False, C=True)
        g.play(["BLOCK", [3, 2]])
        moves = [["MOVE", [4, 1]], ["MOVE", [0, 1]], ["MOVE", [4, 2]], ["MOVE", [0, 2]]]
        for ply in range(1, k + 2):              # state after ply `ply` of the game
            blocked = {(b["r"], b["c"]): b["remaining"] for b in g.state["blocked"]}
            assert blocked == ({(3, 2): k - ply + 1} if ply <= k else {})
            if g.state["to_move"] == 1 and g.state["pawns"][1] == [4, 2]:
                assert ([3, 2] in g.state["legal"]["MOVE"]) == (ply > k)
            if ply <= k:
                g.play(moves[ply - 1])


def test_path_preservation_rejects_a_sealing_wall(client):
    g = Game(client, 3, M=5, **OFF)
    g.play(["WALL", [0, 0, "H"]])
    g.play(["MOVE", [2, 0]])
    assert [0, 1, "H"] not in g.state["legal"]["WALL"]
    g.reject(["WALL", [0, 1, "H"]], "Illegal wall placement")


def test_goal_win_then_no_more_actions(client):
    g = Game(client, 3, M=0, **OFF)
    for a in (["MOVE", [1, 1]], ["MOVE", [2, 0]], ["MOVE", [2, 1]]):
        g.play(a)
    assert g.state["terminal"] == {"over": True, "winner": 0, "reason": "goal"}
    g.reject(["MOVE", [1, 0]], "already over", code=409)


def test_repetition_draw(client):
    g = Game(client, 5, M=0, **OFF)
    for _ in range(4):
        for a in (["MOVE", [0, 1]], ["MOVE", [4, 1]], ["MOVE", [0, 2]], ["MOVE", [4, 2]]):
            g.play(a)
    assert g.state["terminal"] == {"over": True, "winner": None, "reason": "repetition"}


def test_ply_cap_is_reported_as_truncated(client):
    g = Game(client, 5, M=0, max_plies=2, **OFF)
    g.play(["MOVE", [0, 1]])
    g.play(["MOVE", [4, 1]])
    # The demo passes its cap to the engine; the reason is 'truncated', never 'repetition' etc.
    assert g.state["terminal"] == {"over": True, "winner": None, "reason": "truncated"}


def test_no_legal_action_draw_from_scenario(client):
    scn = next(s for s in load_scenarios() if s["name"] == "no_legal_move_draw")
    res = client.post("/api/games/from-scenario/no_legal_move_draw", json={"p0": "human", "p1": "human"})
    assert res.status_code == 201
    e = engine_for(scn)
    for a in scn["script"]:
        e.apply_action(e._normalize_action(a), update_clock=False)
    state = res.get_json()["state"]
    assert_matches_engine(state, e)
    assert state["terminal"] == {"over": True, "winner": None, "reason": "no_legal_actions"}
    assert state["legal_count"] == 0


def test_scenario_checks_carry_the_message_the_page_shows(client):
    frames = client.get("/api/scenarios/hop_over_wall").get_json()["frames"]
    checks = [c for f in frames for c in f["checks"]]
    assert checks and all(c["ok"] and c["message"] for c in checks)


def test_arena_counts_come_from_the_match_outcome():
    from web.backend.arena import ARENA_STORE, run_arena_batch
    s = run_arena_batch({"board_size": 3, "K": 1, "M": 1}, "greedy", "random:seed=12", n=6, seed=42)
    records = ARENA_STORE.get(s["batch_id"])["records"]
    cats = {}
    for r in records:
        cats[r["outcome"]["category"]] = cats.get(r["outcome"]["category"], 0) + 1
    assert s["categories"] == cats
    g = s["by_spec"]["greedy"]
    assert g["wins"] + g["draws"] + g["losses"] + g["truncated"] == 6
    seat = [0 if r["agents"]["0"]["spec"] == "greedy" else 1 for r in records]
    assert g["wins"] == sum(r["outcome"]["winner"] == p for r, p in zip(records, seat))
    assert g["truncated"] == cats.get("truncated", 0)
    assert s["mean_length"] == round(sum(r["length_plies"] for r in records) / 6, 2)
