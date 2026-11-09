"""Comprehensive test suite for Modification C (Blocks), Modification B (Hop), and Modification A (REPLACE) rules (P03 / F3).

Verifies:
1. Exact K-ply block duration timelines and independent multi-block timers.
2. Rule R7: expiring cell block counts in path preservation until the ply after it expires.
3. Block placement constraints, rejections, and position key uniqueness.
4. Modification B (Hop) rules: crossing an opponent's wall into the adjacent cell, ownership, landing
   restrictions, consumption and goal wins.
5. Path preservation during Hop (sealed pocket).
6. Modification A (REPLACE) rules: equivalence of single triple and pair formats, latest-wall restriction, inventory invariance, order preservation.
"""
from __future__ import annotations

import json
import pytest

from tests.helpers import key_from_json, make_engine, pass_move, snapshot
from wallz_engine import WallzEngine


# ============================================================================
# Blocks (Modification C) Tests
# ============================================================================

@pytest.mark.parametrize("K", [1, 2, 3, 5])
def test_block_timeline_exact_K(K: int):
    """A block with duration K must remain active for exactly K plies and expire on ply K+1."""
    engine = make_engine(n=5, A=False, B=False, C=True, M=0, K=K)
    # Ply 1: P0 plays ('BLOCK', (3, 2)) adjacent to P1 at (4, 2)
    engine.apply_action(("BLOCK", (3, 2)), update_clock=False)

    # For j = 1..K plies after placement:
    for j in range(1, K + 1):
        # Before ply 1 + j:
        assert (3, 2) in engine.blocked_cells
        assert engine.blocked_cells[(3, 2)] == K - j + 1
        # P1 cannot move to (3, 2) while it is blocked
        clone = engine.clone()
        clone.pawns[1] = (4, 2)
        assert clone._is_valid_move(1, (3, 2)) is False
        if engine.current_player == 1 and engine.pawns[1] == (4, 2):
            assert ("MOVE", (3, 2)) not in engine.get_legal_actions(1)

        # Apply a pass move to advance the timeline
        pm = pass_move(engine)
        engine.apply_action(pm, update_clock=False)

    # After ply 1 + K plies have completed, the block must be expired
    assert (3, 2) not in engine.blocked_cells
    clone = engine.clone()
    clone.pawns[1] = (4, 2)
    assert clone._is_valid_move(1, (3, 2)) is True


def test_block_immediately_after_placement():
    """Placing a block consumes block_available, sets timer to K, and forbids second block."""
    engine = make_engine(n=5, C=True, K=3)
    engine.apply_action(("BLOCK", (2, 2)), update_clock=False)

    assert engine.block_available[0] is False
    assert engine.blocked_cells[(2, 2)] == 3

    # P1 moves
    engine.apply_action(("MOVE", (3, 2)), update_clock=False)

    # Next turn for P0: no BLOCK in legal actions
    p0_blocks = [a for a in engine.get_legal_actions(0) if a[0] == "BLOCK"]
    assert p0_blocks == []

    # Second block attempt must raise ValueError
    s_before = snapshot(engine)
    with pytest.raises(ValueError, match=r"no cell block available"):
        engine.apply_action(("BLOCK", (1, 1)), update_clock=False)
    assert snapshot(engine) == s_before


def test_consecutive_blocks_independent_timers():
    """Independent blocks placed on consecutive plies track distinct timers and expire independently."""
    engine = make_engine(n=5, C=True, K=3)
    c0 = (1, 2)
    c1 = (3, 2)

    # Ply 1: P0 blocks c0
    engine.apply_action(("BLOCK", c0), update_clock=False)
    assert engine.blocked_cells[c0] == 3

    # Ply 2: P1 blocks c1
    engine.apply_action(("BLOCK", c1), update_clock=False)
    assert engine.blocked_cells[c0] == 2
    assert engine.blocked_cells[c1] == 3

    # Ply 3: P0 pass move
    engine.apply_action(pass_move(engine), update_clock=False)
    assert engine.blocked_cells[c0] == 1
    assert engine.blocked_cells[c1] == 2

    # Ply 4: P1 pass move -> c0 expires
    engine.apply_action(pass_move(engine), update_clock=False)
    assert c0 not in engine.blocked_cells
    assert engine.blocked_cells[c1] == 1

    # Ply 5: P0 pass move -> c1 expires
    engine.apply_action(pass_move(engine), update_clock=False)
    assert c1 not in engine.blocked_cells


def test_block_expiring_this_ply_counts_in_path_checks():
    """Rule R7: A block expiring on the current ply still blocks paths during action validation."""
    with open("tests/fixtures/r7_block_expiring.json") as f:
        data = json.load(f)

    raw_key = key_from_json(data["raw_key"])
    engine = WallzEngine(board_size=data["board_size"], rules=data["rules"]).restore_state_key(raw_key)
    player = data["player"]
    wall = tuple(data["wall"])
    exp_c = tuple(data["expiring_cell"])

    # Cell is blocked with timer 1
    assert exp_c in engine.blocked_cells
    assert engine.blocked_cells[exp_c] == 1

    # Wall placement is illegal because the expiring block blocks the required open path
    assert ("WALL", wall) not in engine.get_legal_actions(player)
    assert engine._is_valid_wall_placement(player, *wall) is False
    with pytest.raises(ValueError):
        engine.apply_action(("WALL", wall), update_clock=False)

    # Mover plays a legal non-wall action
    non_wall = [a for a in engine.get_legal_actions(player) if a[0] != "WALL"]
    engine.apply_action(non_wall[0], update_clock=False)

    # Now the block has expired
    assert exp_c not in engine.blocked_cells
    # And the wall is now valid
    assert engine._is_valid_wall_placement(player, *wall) is True


def test_block_rejections():
    """Engine must reject blocks on pawns, existing blocks, out-of-bounds, disabled C, and path severance."""
    engine = make_engine(n=3, C=True, K=3)

    # 1. Blocking pawn cells
    s_before = snapshot(engine)
    with pytest.raises(ValueError):
        engine.apply_action(("BLOCK", engine.pawns[0]), update_clock=False)
    with pytest.raises(ValueError):
        engine.apply_action(("BLOCK", engine.pawns[1]), update_clock=False)
    assert snapshot(engine) == s_before

    # 2. Blocking an already-blocked cell
    engine.apply_action(("BLOCK", (1, 0)), update_clock=False)
    engine.apply_action(pass_move(engine), update_clock=False)  # P1 turn
    s_before = snapshot(engine)
    with pytest.raises(ValueError):
        engine.apply_action(("BLOCK", (1, 0)), update_clock=False)
    assert snapshot(engine) == s_before

    # 3. Blocking out-of-bounds
    with pytest.raises(ValueError):
        engine.apply_action(("BLOCK", (-1, 0)), update_clock=False)
    with pytest.raises(ValueError):
        engine.apply_action(("BLOCK", (3, 3)), update_clock=False)

    # 4. Rule C disabled
    engine_no_c = make_engine(n=3, C=False)
    assert not any(a[0] == "BLOCK" for a in engine_no_c.get_legal_actions(0))
    with pytest.raises(ValueError):
        engine_no_c.apply_action(("BLOCK", (1, 1)), update_clock=False)

    # 5. Blocking the only remaining path
    e_path = make_engine(n=3, C=True, K=3)
    e_path.blocked_cells[(1, 0)] = 5
    e_path.blocked_cells[(1, 2)] = 5
    # (1, 1) is the only open passage between rows 0 and 2
    assert not e_path._is_valid_cell_block(1, 1)
    with pytest.raises(ValueError):
        e_path.apply_action(("BLOCK", (1, 1)), update_clock=False)


def test_blocks_in_position_key():
    """States differing only in block timer values must produce distinct position_keys."""
    e1 = make_engine(n=5, C=True)
    e1.blocked_cells[(2, 2)] = 1
    e2 = make_engine(n=5, C=True)
    e2.blocked_cells[(2, 2)] = 2

    assert e1.position_key() != e2.position_key()


# ============================================================================
# Hop (Modification B) Tests
# ============================================================================

def _opponent_wall_engine(n=5, mover=0, mover_cell=(1, 2), wall=(1, 2, "H"), owner=None, **rules):
    """Engine where `wall` (default: owned by the mover's opponent) sits next to the mover."""
    e = make_engine(n=n, B=True, **rules)
    e.pawns[mover] = mover_cell
    e.pawns[1 - mover] = (n - 1, 0) if mover == 0 else (0, 0)
    owner = (1 - mover) if owner is None else owner
    e.walls[owner] = [wall]
    e.latest_wall[owner] = wall
    e.current_player = mover
    return e


def test_hop_across_opponent_wall_legal():
    """HOP steps into the adjacent cell across a wall placed by the opponent (proposal LegalHop)."""
    engine = _opponent_wall_engine()          # H(1,2) blocks (1,2)-(2,2); owned by P1
    assert ("MOVE", (2, 2)) not in engine.get_legal_actions(0)
    assert ("HOP", (2, 2)) in engine.get_legal_actions(0)
    engine.apply_action(("HOP", (2, 2)), update_clock=False)
    assert engine.pawns[0] == (2, 2)
    assert engine.hop_available[0] is False


def test_hop_across_own_wall_illegal():
    """A player cannot hop over their own wall."""
    engine = _opponent_wall_engine(owner=0)
    assert ("HOP", (2, 2)) not in engine.get_legal_actions(0)
    s_before = snapshot(engine)
    with pytest.raises(ValueError, match=r"Illegal hop"):
        engine.apply_action(("HOP", (2, 2)), update_clock=False)
    assert snapshot(engine) == s_before


def test_hop_needs_a_wall():
    """Without an opponent wall there is nothing to hop: no HOP actions, and the former
    2-cell sprint is rejected (regression: AI moved two cells on its first turn)."""
    engine = make_engine(n=5, B=True)
    assert not any(a[0] == "HOP" for a in engine.get_legal_actions(0))
    for target in [(1, 2), (2, 2), (0, 4)]:
        with pytest.raises(ValueError):
            engine.apply_action(("HOP", target), update_clock=False)


def test_hop_does_not_jump_pawns_or_blocks():
    """An adjacent pawn or blocked cell is not a wall: no hop over them; landing on the
    opponent or a blocked cell across an opponent wall is illegal too."""
    e1 = make_engine(n=5, B=True)
    e1.pawns[0], e1.pawns[1] = (1, 2), (2, 2)
    assert not e1._is_valid_hop(0, (2, 2)) and not e1._is_valid_hop(0, (3, 2))

    e2 = _opponent_wall_engine(C=True)
    e2.blocked_cells[(2, 2)] = 2
    assert not e2._is_valid_hop(0, (2, 2))

    e3 = _opponent_wall_engine()
    e3.pawns[1] = (2, 2)
    assert not e3._is_valid_hop(0, (2, 2))


def test_hop_geometry_restrictions():
    """Only orthogonally adjacent cells: no 2-cell, diagonal or off-board hops."""
    engine = _opponent_wall_engine()
    for target in [(3, 2), (2, 3), (2, 1), (-1, 2), (1, 2)]:
        assert not engine._is_valid_hop(0, target), target


def test_hop_consumed():
    """Using a HOP consumes hop_available and rejects subsequent HOP attempts."""
    engine = _opponent_wall_engine()
    engine.apply_action(("HOP", (2, 2)), update_clock=False)
    assert engine.hop_available[0] is False
    engine.walls[1].append((2, 2, "H"))   # another opponent wall right below P0
    engine.apply_action(pass_move(engine), update_clock=False)  # P1 turn
    assert not any(a[0] == "HOP" for a in engine.get_legal_actions(0))
    with pytest.raises(ValueError):
        engine.apply_action(("HOP", (3, 2)), update_clock=False)


def test_hop_to_goal_wins():
    """A HOP onto the goal row wins immediately."""
    engine = _opponent_wall_engine(n=3, mover_cell=(1, 1), wall=(1, 0, "H"), C=False, M=0)
    assert ("HOP", (2, 1)) in engine.get_legal_actions(0)
    engine.apply_action(("HOP", (2, 1)), update_clock=False)
    assert engine.is_terminal()
    assert engine.winner == 0
    assert engine.terminal_reason == "goal"


def test_hop_into_sealed_pocket_rejected():
    """A hop across an opponent wall into a pocket with no route to goal is illegal.
    Pocket {(0,0),(1,0)}: V(0,0) seals its right side, H(1,0) its bottom."""
    engine = make_engine(n=5, B=True, C=False)
    engine.pawns[0], engine.pawns[1] = (0, 1), (4, 2)
    engine.walls[1] = [(0, 0, "V"), (1, 0, "H")]
    engine.latest_wall[1] = (1, 0, "H")
    assert engine.is_path_open(0) and engine.is_path_open(1)
    assert engine.wall_owner_on_edge((0, 1), (0, 0)) == 1
    assert ("HOP", (0, 0)) not in engine.get_legal_actions(0)
    s_before = snapshot(engine)
    with pytest.raises(ValueError, match=r"Illegal hop"):
        engine.apply_action(("HOP", (0, 0)), update_clock=False)
    assert snapshot(engine) == s_before


def test_hop_ownership_in_position_key():
    """While a hop is unused, wall ownership changes future legality and must stay in the
    canonical key; once no hop is left it may be merged (smaller solver graphs)."""
    a = _opponent_wall_engine()
    b = _opponent_wall_engine(owner=0)
    a.latest_wall = {0: None, 1: None}; b.latest_wall = {0: None, 1: None}
    assert a.position_key() != b.position_key()
    for e in (a, b):
        e.hop_available = {0: False, 1: False}
    assert a.position_key() == b.position_key()
    # restore_state_key round-trips the ownership-tagged canonical key
    c = _opponent_wall_engine()
    c.latest_wall = {0: None, 1: None}
    r = make_engine(n=5, B=True).restore_state_key(c.position_key())
    assert r.walls[1] == [(1, 2, "H")] and r.walls[0] == []
    assert ("HOP", (2, 2)) in r.get_legal_actions(0)


def test_hop_disabled():
    """When Rule B is disabled, no HOP actions are available or legal."""
    engine = make_engine(n=5, B=False)
    assert not any(a[0] == "HOP" for a in engine.get_legal_actions(0))
    with pytest.raises(ValueError):
        engine.apply_action(("HOP", (2, 2)), update_clock=False)


# ============================================================================
# REPLACE forms (Modification A) Tests
# ============================================================================

def test_replace_single_and_pair_forms_equivalent():
    """Both ('REPLACE', w2) and ('REPLACE', (w1, w2)) produce identical game states."""
    e1 = make_engine(n=5, A=True)
    e1.apply_action(("WALL", (0, 0, "H")), update_clock=False)
    e1.apply_action(("MOVE", (3, 2)), update_clock=False)
    e2 = e1.clone()

    e1.apply_action(("REPLACE", (0, 1, "H")), update_clock=False)
    e2.apply_action(("REPLACE", ((0, 0, "H"), (0, 1, "H"))), update_clock=False)

    s1 = snapshot(e1)
    s2 = snapshot(e2)
    s1.pop("last_turn_timestamp", None)
    s2.pop("last_turn_timestamp", None)
    assert s1 == s2


def test_replace_older_wall_rejected():
    """Attempting to replace an older wall instead of the latest wall must raise ValueError."""
    engine = make_engine(n=5, A=True)
    w1 = (0, 0, "H")
    w2 = (1, 1, "H")
    engine.apply_action(("WALL", w1), update_clock=False)
    engine.apply_action(("MOVE", (3, 2)), update_clock=False)
    engine.apply_action(("WALL", w2), update_clock=False)
    engine.apply_action(("MOVE", (3, 1)), update_clock=False)

    # Only latest wall w2 may be repositioned
    with pytest.raises(ValueError, match=r"only latest wall"):
        engine.apply_action(("REPLACE", (w1, (0, 2, "H"))), update_clock=False)

    # Replacing latest wall onto itself is illegal
    with pytest.raises(ValueError):
        engine.apply_action(("REPLACE", w2), update_clock=False)


def test_replace_keeps_inventory_and_order():
    """Repositioning a wall does not alter wall inventory and preserves wall list ordering."""
    engine = make_engine(n=5, A=True, M=5)
    engine.apply_action(("WALL", (0, 0, "H")), update_clock=False)
    engine.apply_action(("MOVE", (3, 2)), update_clock=False)
    engine.apply_action(("WALL", (2, 2, "H")), update_clock=False)
    engine.apply_action(("MOVE", (3, 1)), update_clock=False)

    inv_before = engine.wall_inventory[0]
    engine.apply_action(("REPLACE", (2, 0, "H")), update_clock=False)

    assert engine.wall_inventory[0] == inv_before
    assert engine.walls[0] == [(0, 0, "H"), (2, 0, "H")]
    assert engine.latest_wall[0] == (2, 0, "H")


def test_replace_when_inventory_zero():
    """REPLACE remains legal when wall inventory is zero, provided a wall has been placed."""
    engine = make_engine(n=5, A=True, M=1)
    engine.apply_action(("WALL", (0, 0, "H")), update_clock=False)
    assert engine.wall_inventory[0] == 0

    engine.apply_action(("MOVE", (3, 2)), update_clock=False)
    replace_actions = [a for a in engine.get_legal_actions(0) if a[0] == "REPLACE"]
    assert len(replace_actions) > 0

    engine.apply_action(("REPLACE", (0, 1, "H")), update_clock=False)
    assert engine.wall_inventory[0] == 0
    assert engine.latest_wall[0] == (0, 1, "H")


def test_replace_disabled():
    """When Rule A is disabled, REPLACE is neither generated nor accepted."""
    engine = make_engine(n=5, A=False)
    assert not any(a[0] == "REPLACE" for a in engine.get_legal_actions(0))
    with pytest.raises(ValueError, match=r"Rule A"):
        engine.apply_action(("REPLACE", (0, 0, "H")), update_clock=False)
