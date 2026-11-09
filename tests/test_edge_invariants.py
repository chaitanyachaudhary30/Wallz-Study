"""Randomised invariant sweep over full games (regression net for rule edge cases).

Checks after every ply: pawn displacement per action (MOVE/HOP exactly 1 cell, hop only
across an opponent's wall and at most once), turn alternation, inventory bookkeeping,
wall/board consistency, block timers, path preservation, terminal consistency; on every
third ply also filtered-vs-brute-force legality, raw/canonical key round trips, clone
isolation, atomic rejection of illegal actions and ActionCodec round trips.
"""
import collections
import random
import sys

import pytest

from wallz_engine import WallzEngine
from src.modified_wallz.board import WallzBoard
from src.modified_wallz.actions import ActionCodec
from agents.greedy import GreedyAgent
from agents.minimax import MinimaxAgent
from evaluation.match_runner import RandomAgent
from evaluation.state_graph import canonical_key
from tests.helpers import snapshot, illegal_action_pool
from tests.test_filter_equivalence import brute_force_legal_actions

V = collections.Counter(); EX = {}; STATS = collections.Counter()
def bad(kind, ctx):
    V[kind] += 1
    EX.setdefault(kind, ctx)

def check_state(e, ctx, rng, deep):
    n = e.board_size; M = e.rules['M']; K = e.rules['K']
    for p in (0, 1):
        if len(e.walls[p]) != M - e.wall_inventory[p]: bad('walls != M - inventory', ctx)
        if e.wall_inventory[p] < 0: bad('negative inventory', ctx)
        if e.latest_wall[p] is not None and e.latest_wall[p] not in e.walls[p]: bad('latest wall not owned', ctx)
        if e.walls[p] and e.rules['A'] and e.latest_wall[p] is None: bad('walls but no latest', ctx)
        if not e.is_path_open(p): bad('path preservation broken', ctx)
    allw = e.walls[0] + e.walls[1]
    if len(set(allw)) != len(allw): bad('duplicate wall', ctx)
    b = WallzBoard(n)
    for w in allw:
        if not b.can_place_wall(*w): bad('overlapping/crossing walls', ctx); break
        b.add_wall(*w)
    if set(e.board.walls) != set(allw): bad('board out of sync with wall lists', ctx)
    for cell, t in e.blocked_cells.items():
        if not (1 <= t <= K): bad('block timer out of range', ctx)
        if cell in (e.pawns[0], e.pawns[1]): bad('pawn on blocked cell', ctx)
    if e.pawns[0] == e.pawns[1]: bad('pawns share a cell', ctx)
    w = [p for p in (0, 1) if e.pawns[p][0] == e.goals[p]]
    if w and e.winner not in w: bad('pawn on goal but no winner', ctx)
    if e.winner is not None and e.is_draw: bad('winner and draw', ctx)
    legal = e.get_legal_actions()
    if e.is_terminal():
        if legal: bad('legal actions in terminal state', ctx)
        return legal
    if not legal: bad('non-terminal without legal actions', ctx)
    if e.has_legal_action() != bool(legal): bad('has_legal_action disagrees', ctx)
    if len(set(legal)) != len(legal): bad('duplicate legal actions', ctx)
    STATS['states'] += 1
    STATS['hop_legal_states'] += any(a[0]=='HOP' for a in legal)
    if deep:
        STATS['deep'] += 1
        STATS['tagged_keys'] += any(isinstance(x[0], tuple) for x in e.position_key()[2])
        if set(legal) != set(brute_force_legal_actions(e)): bad('filtered legal != brute force', ctx)
        # canonical key round trip
        pk = e.position_key()
        if pk != canonical_key(e._get_state_hashable_key(), e.rules): bad('position_key != canonical_key(raw)', ctx)
        r1 = WallzEngine(board_size=n, rules=e.rules).restore_state_key(e._get_state_hashable_key())
        r2 = WallzEngine(board_size=n, rules=e.rules).restore_state_key(pk)
        if set(r1.get_legal_actions()) != set(legal): bad('raw-key restore changes legal set', ctx)
        if set(r2.get_legal_actions()) != set(legal): bad('canonical restore changes legal set', ctx)
        if r2.position_key() != pk: bad('canonical restore changes key', ctx)
        # clone independence
        c = e.clone(); before = snapshot(e)
        c.apply_action(rng.choice(legal), update_clock=False)
        if snapshot(e) != before: bad('clone mutation leaks into original', ctx)
        # illegal actions atomic
        for a in illegal_action_pool(e, rng, 6):
            s0 = snapshot(e)
            try:
                e.apply_action(a, update_clock=False); bad('illegal action accepted', ctx + (a,))
            except ValueError:
                pass
            except Exception as ex:
                bad(f'illegal action raised {type(ex).__name__}', ctx + (a,))
            if snapshot(e) != s0: bad('illegal action mutated state', ctx + (a,))
        # codec: every legal action round trips; mask size == legal count, both movers' view
        codec = ActionCodec(n); mover = e.current_player
        ids = set()
        for a in legal:
            i = codec.encode(a, e, mover)
            if codec.decode(i, e, mover) != a: bad('codec round trip fails', ctx + (a,))
            ids.add(i)
        mask = codec.legal_mask(e, mover)
        if sum(mask) != len(legal) or {i for i, m in enumerate(mask) if m} != ids: bad('codec mask != legal set', ctx)
    return legal

def play(seed, deep):
    rng = random.Random(seed)
    n = rng.choice([3, 3, 4, 4, 5, 5, 7, 9])
    rules = dict(A=rng.random() < .6, B=rng.random() < .7, C=rng.random() < .6, K=rng.randint(1, 4), M=rng.choice([0, 1, 2, 3, 5, 10]))
    def mk(p):
        kind = rng.choice(['random', 'random', 'greedy', 'minimax1'] + (['minimax2'] if n <= 5 else []))
        return kind, {'random': lambda: RandomAgent(p, seed), 'greedy': lambda: GreedyAgent(p, seed=rng.randint(0, 9)),
                      'minimax1': lambda: MinimaxAgent(player_id=p, depth=1),
                      'minimax2': lambda: MinimaxAgent(player_id=p, depth=2)}[kind]()
    kinds, agents = zip(*(mk(0), mk(1)))
    e = WallzEngine(board_size=n, rules=rules)
    hop_used = {0: 0, 1: 0}
    for ply in range(220):
        ctx = (seed, n, rules, kinds, ply)
        legal = check_state(e, ctx, rng, deep and (ply % 3 == 0))
        if e.is_terminal():
            break
        p = e.current_player; before = dict(e.pawns); turn = e.turn_count; inv = dict(e.wall_inventory)
        owner_before = None
        try:
            a = agents[p].select_action(e)
        except Exception as ex:
            bad(f'agent {kinds[p]} raised {type(ex).__name__}', ctx); return
        if a is None: bad(f'agent {kinds[p]} returned None with legal moves', ctx); return
        if a not in legal: bad(f'agent {kinds[p]} chose illegal action', ctx + (a,)); return
        if a[0] == 'HOP':
            owner_before = e.wall_owner_on_edge(e.pawns[p], a[1])
        e.apply_action(a, update_clock=False)
        d = abs(e.pawns[p][0] - before[p][0]) + abs(e.pawns[p][1] - before[p][1])
        if e.pawns[1 - p] != before[1 - p]: bad('opponent pawn moved', ctx + (a,))
        if a[0] in ('MOVE', 'HOP') and d != 1: bad(f'{a[0]} distance {d}', ctx + (a,))
        if a[0] not in ('MOVE', 'HOP') and d: bad('non-pawn action moved pawn', ctx + (a,))
        STATS['ply_'+a[0]] += 1
        if a[0] == 'HOP':
            hop_used[p] += 1
            if hop_used[p] > 1: bad('second hop', ctx)
            if owner_before != 1 - p: bad('hop not across opponent wall', ctx + (a, owner_before))
        if a[0] == 'WALL' and e.wall_inventory[p] != inv[p] - 1: bad('WALL inventory not decremented', ctx)
        if a[0] == 'REPLACE' and e.wall_inventory[p] != inv[p]: bad('REPLACE changed inventory', ctx)
        if e.turn_count != turn + 1: bad('turn counter not +1', ctx)
        if not e.is_terminal() and e.current_player == p: bad('same player to move twice', ctx)



@pytest.mark.parametrize("block", range(4))
def test_random_games_respect_all_invariants(block):
    V.clear(); EX.clear()
    for seed in range(block * 40, block * 40 + 40):
        play(seed, deep=True)
    assert not V, {k: (V[k], EX[k]) for k in V}
