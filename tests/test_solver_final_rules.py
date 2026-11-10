"""
Exact solver checked against the frozen Stage 1 rules (Member 3, Task 1).

Covers terminal wins and losses, draws (no legal action, cycles), block expiry,
wall inventory, latest-wall REPLACE, HOP (opponent's wall only, once), that the
graph is built only through the engine, an independent Bellman reference, the
legacy ``agents/exact_solver.py`` adapter, and canonical-key merging.

Hand-built positions are 3x3. Value and rank are for the side to move; rank is plies
to the goal under optimal play. Run: python -m unittest tests.test_solver_final_rules
"""
import ast
import json
import os
import unittest
from unittest import mock

from wallz_engine import WallzEngine
from evaluation.retrograde import solve, solve_config, WIN, LOSS, DRAW
from evaluation.state_graph import (StateGraph, NONTERMINAL, WIN_P0, WIN_P1,
                                    DRAW_NO_LEGAL_ACTIONS, canonical_key)
from tests.helpers import key_from_json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def key(p0, p1, w0=(), w1=(), i0=0, i1=0, l0=None, l1=None, h0=False, h1=False,
        b0=False, b1=False, blocks=(), to_move=0):
    """Raw engine state key (same layout as WallzEngine._get_state_hashable_key)."""
    return (p0, p1, tuple(w0), tuple(w1), i0, i1, l0, l1, h0, h1, b0, b1, tuple(blocks), to_move)


def rules(**kw):
    return dict({'A': False, 'B': False, 'C': False, 'K': 1, 'M': 0}, **kw)


def root(rules_, k):
    s = solve_config(3, rules_, root_key=k)
    return s, s.value[0], s.rank[0]


def reference(board_size, rules_, root_key=None):
    """
    Independent check that shares no code with StateGraph/retrograde: DFS over raw keys
    straight from the engine, then finite-horizon value iteration
    V_h(s) = max_a -V_{h-1}(s') to a fixed point. Returns {raw key: (value, rank)}.
    """
    def fresh(k):
        return WallzEngine(board_size=board_size, rules=rules_).restore_state_key(k)

    if root_key is None:
        root_key = WallzEngine(board_size=board_size, rules=rules_)._get_state_hashable_key()
    term, kids, stack, seen = {}, {}, [root_key], {root_key}
    while stack:
        k = stack.pop()
        e = fresh(k)
        kids[k] = []
        if e.winner is not None:
            term[k] = 1 if e.winner == k[-1] else -1
            continue
        if e.is_draw or not e.get_legal_actions():
            term[k] = 0
            continue
        for a in e.get_legal_actions():
            c = e.clone()
            c.apply_action(a, update_clock=False)
            ck = c._get_state_hashable_key()
            kids[k].append(ck)
            if ck not in seen:
                seen.add(ck)
                stack.append(ck)
    V = {k: term.get(k, 0) for k in kids}
    rank = {k: 0 for k, v in V.items() if v != 0}
    h = 0
    while True:
        h += 1
        new = {k: term[k] if k in term else max(-V[c] for c in kids[k]) for k in kids}
        for k, v in new.items():
            if v != 0 and k not in rank:
                rank[k] = h
        if new == V:
            return {k: (V[k], rank.get(k)) for k in V}
        V = new


def assert_game_equations(tc, sol):
    """Every solved node satisfies the negamax equations, including exact ranks."""
    g = sol.graph
    for u in range(len(g.keys)):
        v, r = sol.value[u], sol.rank[u]
        if g.terminal[u] in (WIN_P0, WIN_P1):
            winner = 0 if g.terminal[u] == WIN_P0 else 1
            tc.assertEqual((v, r), (WIN if winner == g.to_move[u] else LOSS, 0))
            continue
        if g.terminal[u] == DRAW_NO_LEGAL_ACTIONS:
            tc.assertEqual(v, DRAW)
            continue
        child = {c for _, c in g.succ[u]}
        cv = [sol.value[c] for c in child]
        if LOSS in cv:
            tc.assertEqual(v, WIN, u)
            tc.assertEqual(r, 1 + min(sol.rank[c] for c in child if sol.value[c] == LOSS), u)
        elif all(x == WIN for x in cv):
            tc.assertEqual(v, LOSS, u)
            tc.assertEqual(r, 1 + max(sol.rank[c] for c in child), u)
        else:
            tc.assertEqual(v, DRAW, u)
            tc.assertIsNone(r, u)


class TestTerminalAndDraws(unittest.TestCase):
    def test_terminal_win_state(self):
        # P0 stands on its goal row; P1 is to move
        s, v, r = root(rules(), key((2, 1), (1, 0), to_move=1))
        self.assertEqual(s.graph.terminal[0], WIN_P0)
        self.assertEqual((v, r), (LOSS, 0))
        self.assertEqual(s.value_for(0, 0), 1)
        self.assertEqual(s.action_values(0), [])

    def test_terminal_loss_state(self):
        # P1 stands on its goal row 0; P0 is to move and has lost
        s, v, r = root(rules(), key((1, 1), (0, 0), to_move=0))
        self.assertEqual(s.graph.terminal[0], WIN_P1)
        self.assertEqual((v, r), (LOSS, 0))
        self.assertEqual(s.value_for(0, 1), 1)

    def test_win_in_one_and_forced_loss_in_two(self):
        s, v, r = root(rules(), key((1, 1), (2, 0), to_move=0))
        self.assertEqual((v, r), (WIN, 1))
        self.assertEqual(s.rank_optimal_actions(0), [('MOVE', (2, 1))])
        # P1 to move, two plies from goal, cannot reach (2, 0)'s column in time
        s, v, r = root(rules(), key((1, 0), (2, 2), to_move=1))
        self.assertEqual((v, r), (LOSS, 2))
        self.assertTrue(all(cv == LOSS and cr == 1 for _, cv, cr in s.action_values(0)))

    def test_no_legal_action_draw(self):
        with open(os.path.join(ROOT, 'tests', 'fixtures', 'no_legal_draw.json'), encoding='utf-8') as fh:
            fx = json.load(fh)
        k = key_from_json(fx['child_key'])
        e = WallzEngine(board_size=fx['board_size'], rules=fx['rules']).restore_state_key(k)
        self.assertEqual(e.get_legal_actions(), [])
        self.assertTrue(e.is_path_open(0) and e.is_path_open(1))
        s, v, r = root(fx['rules'], k)
        self.assertEqual(s.graph.terminal[0], DRAW_NO_LEGAL_ACTIONS)
        self.assertEqual((v, s.draw_type(0)), (DRAW, 'no_legal_actions'))
        # one ply earlier the BLOCK that causes it is a draw-making move for P0
        parent = key_from_json(fx['parent_key'])
        sp = solve_config(3, fx['rules'], root_key=parent)
        child = sp.graph.node_of(k)
        self.assertEqual(sp.value[child], DRAW)

    def test_cyclic_draws(self):
        # Pawn interference: P1 can stand on P0's only goal-entry cell forever
        s, v, r = root(rules(), key((1, 1), (2, 0), to_move=1))
        self.assertEqual((v, r, s.draw_type(0)), (DRAW, None, 'cycle'))
        # Full game N3 A1B0C0 K1 M1 is a cycle draw from the start
        s = solve_config(3, rules(A=True, M=1))
        self.assertEqual((s.status(0), s.draw_type(0)), ('DRAW', 'cycle'))
        assert_game_equations(self, s)
        # Value-optimal play from the root never ends and revisits a position
        u, seen = 0, set()
        while u not in seen:
            seen.add(u)
            self.assertEqual(s.graph.terminal[u], NONTERMINAL)
            self.assertEqual(s.value[u], DRAW)
            a = s.value_optimal_actions(u)[0]
            u = dict(s.graph.succ[u])[a]
        self.assertIn(u, seen)


class TestMechanics(unittest.TestCase):
    def test_block_active_on_last_ply_then_expires(self):
        k2 = rules(C=True, K=2)
        # P1 to move, its block on P0's entry cell (2, 0) with 1 ply left: it expires at
        # the end of P1's ply, P0 walks in -> P1 loses in 2 (same as no block at all)
        self.assertEqual(root(k2, key((1, 0), (2, 2), blocks=[((2, 0), 1)], to_move=1))[1:], (LOSS, 2))
        self.assertEqual(root(k2, key((1, 0), (2, 2), to_move=1))[1:], (LOSS, 2))
        # With 2 plies left the cell is still shut on P0's ply -> P1 wins in 3
        self.assertEqual(root(k2, key((1, 0), (2, 2), blocks=[((2, 0), 2)], to_move=1))[1:], (WIN, 3))
        # A timer-1 block still stops the mover on its last ply
        k1 = rules(C=True, K=1)
        self.assertEqual(root(k1, key((1, 0), (1, 2), to_move=0))[1:], (WIN, 1))
        self.assertEqual(root(k1, key((1, 0), (1, 2), blocks=[((2, 0), 1)], to_move=0))[1:], (LOSS, 2))

    def test_block_timers_on_every_edge(self):
        r = rules(C=True, K=2)
        g = StateGraph(3, r).build()
        seen_block = False
        for u, edges in enumerate(g.succ):
            parent = dict(g.keys[u][12])
            for a, c in edges:
                expect = {cell: t - 1 for cell, t in parent.items() if t > 1}
                if a[0] == 'BLOCK':
                    seen_block = True
                    expect[a[1]] = r['K']
                    self.assertTrue(g.keys[u][10 + g.to_move[u]])
                    self.assertFalse(g.keys[c][10 + g.to_move[u]])
                self.assertEqual(dict(g.keys[c][12]), expect, (u, a))
        self.assertTrue(seen_block)

    def test_wall_inventory(self):
        r = rules(M=1)
        # Same position, P1 with 0 or 1 wall left
        self.assertEqual(root(r, key((1, 0), (2, 2), to_move=1))[1:], (LOSS, 2))
        s, v, rk = root(r, key((1, 0), (2, 2), i1=1, to_move=1))
        self.assertEqual((v, rk), (WIN, 5))
        self.assertEqual(s.rank_optimal_actions(0), [('WALL', (1, 0, 'H'))])
        # Over a whole game: WALL only with stock, costs exactly 1, nothing else costs
        g = StateGraph(3, rules(M=2)).build()
        walls = 0
        for u, edges in enumerate(g.succ):
            p = g.to_move[u]
            for a, c in edges:
                before, after = g.keys[u][4 + p], g.keys[c][4 + p]
                if a[0] == 'WALL':
                    walls += 1
                    self.assertGreater(before, 0)
                    self.assertEqual(after, before - 1)
                else:
                    self.assertEqual(after, before)
                self.assertEqual(g.keys[c][5 - p], g.keys[u][5 - p])
        self.assertGreater(walls, 0)

    def test_latest_wall_replace(self):
        w = (0, 1, 'V')  # P1's latest wall, useless where it stands; P1 has no stock
        pos = key((1, 0), (2, 2), w1=[w], l1=w, to_move=1)
        self.assertEqual(root(rules(A=False, M=1), pos)[1:], (LOSS, 2))
        s, v, rk = root(rules(A=True, M=1), pos)
        self.assertEqual((v, rk), (WIN, 5))
        self.assertEqual(s.rank_optimal_actions(0), [('REPLACE', (1, 0, 'H'))])
        # P0 has an older wall (1,1,V) and a latest wall (0,0,H): only the latest moves
        older, latest = (1, 1, 'V'), (0, 0, 'H')
        r = rules(A=True, M=2)
        g = StateGraph(3, r, root_key=key((0, 1), (2, 2), w0=[older, latest], l0=latest),
                       canonical=False).build()
        self.assertTrue(g.complete)
        n = 0
        for u, edges in enumerate(g.succ):
            p = g.to_move[u]
            for a, c in edges:
                if a[0] != 'REPLACE':
                    continue
                n += 1
                pk, ck = g.keys[u], g.keys[c]
                self.assertIsNotNone(pk[6 + p])
                self.assertEqual(ck[6 + p], a[1])                      # new latest wall
                self.assertEqual(ck[4 + p], pk[4 + p])                 # no stock used
                self.assertEqual(set(pk[2 + p]) - {pk[6 + p]}, set(ck[2 + p]) - {ck[6 + p]})
                self.assertIn(older, ck[2])                            # older wall never moves
        self.assertGreater(n, 0)

    def test_hop_needs_unused_hop_and_opponent_wall(self):
        hw = (1, 0, 'H')  # blocks (1,0)-(2,0) and (1,1)-(2,1)
        r = rules(B=True, M=1)
        s, v, rk = root(r, key((1, 1), (2, 0), w1=[hw], l1=hw, h0=True, to_move=0))
        self.assertEqual((v, rk), (WIN, 1))
        self.assertEqual(s.rank_optimal_actions(0), [('HOP', (2, 1))])
        # Own wall, or hop already used: must walk round, win in 3
        for pos in (key((1, 1), (2, 0), w0=[hw], l0=hw, h0=True, to_move=0),
                    key((1, 1), (2, 0), w1=[hw], l1=hw, h0=False, to_move=0)):
            s, v, rk = root(r, pos)
            self.assertEqual((v, rk), (WIN, 3))
            self.assertNotIn('HOP', {a[0] for e in s.graph.succ if e for a, _ in e})
        # Rule B off: no HOP even across the opponent's wall
        s, _, _ = root(rules(M=1), key((1, 1), (2, 0), w1=[hw], l1=hw, to_move=0))
        self.assertNotIn('HOP', {a[0] for e in s.graph.succ if e for a, _ in e})

    def test_hop_edges_cross_an_opponent_wall_once(self):
        # Walls already placed, REPLACE on: ownership must survive REPLACE
        r = rules(A=True, B=True, M=1)
        start = key((0, 1), (2, 1), w0=[(0, 0, 'H')], w1=[(1, 1, 'H')], l0=(0, 0, 'H'),
                    l1=(1, 1, 'H'), h0=True, h1=True)
        g = StateGraph(3, r, root_key=start).build()
        self.assertTrue(g.complete)
        hops = 0
        for u, edges in enumerate(g.succ):
            p = g.to_move[u]
            for a, c in edges:
                if a[0] != 'HOP':
                    continue
                hops += 1
                e = g.restore(g.keys[u])
                self.assertTrue(e.hop_available[p])
                self.assertEqual(e.wall_owner_on_edge(e.pawns[p], a[1]), 1 - p)
                self.assertFalse(g.keys[c][8 + p])
                self.assertEqual(g.keys[c][p], a[1])
        self.assertGreater(hops, 0)

    def test_no_walls_means_no_hop(self):
        # M = 0: nothing to hop over, so B changes nothing (built-in check for Task 2)
        for c in (False, True):
            off = solve_config(3, rules(C=c, M=0))
            on = solve_config(3, rules(B=True, C=c, M=0))
            self.assertEqual(len(on.graph.keys), len(off.graph.keys))
            self.assertEqual((on.status(0), on.rank[0]), (off.status(0), off.rank[0]))

    def test_owner_kept_in_key_only_while_a_hop_remains(self):
        hw = (1, 0, 'H')
        r = rules(B=True, M=1)
        mine = key((1, 1), (2, 0), w0=[hw], l0=hw, h0=True, to_move=0)
        theirs = key((1, 1), (2, 0), w1=[hw], l1=hw, h0=True, to_move=0)
        self.assertNotEqual(canonical_key(mine, r), canonical_key(theirs, r))
        mine_used = key((1, 1), (2, 0), w0=[hw], l0=hw, to_move=0)
        theirs_used = key((1, 1), (2, 0), w1=[hw], l1=hw, to_move=0)
        self.assertEqual(canonical_key(mine_used, r), canonical_key(theirs_used, r))
        a, b = root(r, mine_used), root(r, theirs_used)
        self.assertEqual(a[1:], b[1:])


class TestEngineIsTheOnlyRuleSource(unittest.TestCase):
    CASES = (
        (rules(B=True, M=2), key((0, 1), (2, 1), i0=1, i1=1, h0=True, h1=True)),        # MOVE WALL HOP
        (rules(A=True, M=2), key((0, 1), (2, 2), w0=[(1, 1, 'V'), (0, 0, 'H')],
                                 l0=(0, 0, 'H'))),                                        # REPLACE
        (rules(C=True, K=2), None),                                                       # BLOCK
    )

    def test_every_edge_is_an_engine_transition(self):
        kinds = set()
        real_apply = WallzEngine.apply_action
        for r, start in self.CASES:
            with mock.patch.object(WallzEngine, 'apply_action', autospec=True,
                                   side_effect=real_apply) as spy:
                g = StateGraph(3, r, root_key=start).build()
            self.assertTrue(g.complete)
            self.assertEqual(spy.call_count, g.edge_count)        # one engine call per edge
            for u, edges in enumerate(g.succ):
                e = g.restore(g.keys[u])
                if g.terminal[u] is not NONTERMINAL:
                    self.assertEqual(edges, [])
                    self.assertTrue(e.winner is not None or not e.get_legal_actions())
                    continue
                self.assertEqual([a for a, _ in edges], e.get_legal_actions())
                for a, c in edges:
                    kinds.add(a[0])
                    x = WallzEngine(board_size=3, rules=r).restore_state_key(g.keys[u])
                    x.apply_action(a, update_clock=False)
                    self.assertEqual(x.position_key(), canonical_key(g.keys[c], g.rules))
                    winner = None if g.terminal[c] not in (WIN_P0, WIN_P1) else (
                        0 if g.terminal[c] == WIN_P0 else 1)
                    self.assertEqual(x.winner, winner)
        self.assertEqual(kinds, {'MOVE', 'WALL', 'REPLACE', 'HOP', 'BLOCK'})

    def test_solver_code_has_no_rule_logic(self):
        """state_graph reads no engine internals (it only restores, lists, applies and
        reads terminal flags); retrograde does not import the engine at all."""
        banned = {'pawns', 'walls', 'blocked_cells', 'wall_inventory', 'latest_wall',
                  'hop_available', 'block_available', 'board', 'goals', '_is_valid_move',
                  '_is_valid_hop', 'is_path_open', 'shortest_path', '_mutate'}
        for name in ('state_graph.py', 'retrograde.py'):
            with open(os.path.join(ROOT, 'evaluation', name), encoding='utf-8') as fh:
                tree = ast.parse(fh.read())
            attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
            self.assertFalse(attrs & banned, (name, attrs & banned))
            if name == 'retrograde.py':
                mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
                self.assertNotIn('wallz_engine', mods)


class TestIndependentChecks(unittest.TestCase):
    def test_matches_bellman_reference(self):
        for r, start in (
            (rules(B=True, M=1), None),                                        # hop-relevant game
            (rules(C=True, K=2), None),                                        # 2-ply blocks
            (rules(A=True, M=2), key((0, 1), (2, 2), w0=[(1, 1, 'V'), (0, 0, 'H')],
                                     l0=(0, 0, 'H'))),                         # REPLACE
        ):
            s = solve_config(3, r, root_key=start)
            self.assertTrue(s.exact)
            assert_game_equations(self, s)
            ref = reference(3, r, start)
            nodes = set()
            for k, (rv, rr) in ref.items():
                u = s.graph.node_of(k)
                self.assertIsNotNone(u, (r, k))
                nodes.add(u)
                self.assertEqual(s.value[u], rv, (r, k))
                if rv != 0:
                    self.assertEqual(s.rank[u], rr, (r, k))
            self.assertEqual(nodes, set(range(len(s.graph.keys))))

    def test_legacy_exact_solver_agrees(self):
        from agents.exact_solver import ExactSolver
        to_status = {WIN: 'WIN', LOSS: 'LOSS', DRAW: 'DRAW'}
        for r in (rules(M=1), rules(B=True, M=1), rules(A=True, M=1), rules(C=True, K=1, M=0)):
            mine = solve_config(3, r)
            old = ExactSolver(board_size=3, rules=r).solve()
            self.assertEqual(old.state_count, len(old._graph.keys))
            for k, sol in old.solutions.items():
                u = mine.graph.node_of(k)
                self.assertEqual(sol['status'], to_status[mine.value[u]], (r, k))
                if sol['status'] != 'DRAW':
                    self.assertEqual(sol['distance'], mine.rank[u], (r, k))
            self.assertEqual({mine.graph.node_of(k) for k in old.solutions},
                             set(range(len(mine.graph.keys))))

    def test_canonical_merge_keeps_value_and_rank(self):
        merged_somewhere = False
        for r in (rules(M=1), rules(B=True, M=1), rules(A=True, B=True, M=1)):
            raw = solve(StateGraph(3, r, canonical=False).build())
            can = solve_config(3, r)
            groups = {}
            for u, k in enumerate(raw.graph.keys):
                c = can.graph.node_of(k)
                self.assertEqual((raw.value[u], raw.rank[u]), (can.value[c], can.rank[c]), (r, k))
                groups.setdefault(c, set()).add((raw.value[u], raw.rank[u]))
            self.assertTrue(all(len(v) == 1 for v in groups.values()))
            self.assertEqual(len(groups), len(can.graph.keys))
            merged_somewhere |= len(raw.graph.keys) > len(can.graph.keys)
        self.assertTrue(merged_somewhere)


if __name__ == '__main__':
    unittest.main()
