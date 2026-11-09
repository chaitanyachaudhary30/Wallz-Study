"""Tests for evaluation.state_graph and evaluation.retrograde."""
import unittest
from collections import deque

from wallz_engine import WallzEngine
from evaluation.retrograde import solve, solve_config, WIN, LOSS, DRAW
from evaluation.state_graph import (StateGraph, GraphCaps, NONTERMINAL, WIN_P0, WIN_P1,
                                    DRAW_NO_LEGAL_ACTIONS)


class FakeGraph:
    """Hand-built abstract graph with the attributes solve() reads."""

    def __init__(self, to_move, terminal, succ, complete=True):
        self.keys = list(range(len(to_move)))
        self.to_move = to_move
        self.terminal = terminal
        self.succ = succ
        self.complete = complete
        self.stop_reason = None if complete else 'max_states'


def reference(board_size, rules):
    """
    Independent check: DFS enumeration straight from the engine and finite-horizon
    Bellman iteration V_h(s) = max_a -V_{h-1}(s') until a fixed point. Returns
    {key: (value, rank)} with rank = first h at which the value becomes non-zero.
    """
    base = WallzEngine(board_size=board_size, rules=rules)
    root = base._get_state_hashable_key()
    term, kids, stack = {}, {}, [root]
    seen = {root}
    while stack:
        k = stack.pop()
        e = WallzEngine(board_size=board_size, rules=rules).restore_state_key(k)
        if e.winner is not None:
            term[k] = 1 if e.winner == k[-1] else -1
        elif e.is_draw:
            term[k] = 0
        if k in term:
            kids[k] = []
            continue
        kids[k] = []
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


class TestSyntheticGraphs(unittest.TestCase):
    def test_terminal_perspective(self):
        # node 0: P1 to move but P0 already won -> LOSS for side to move
        # node 1: P0 to move and P0 has won (constructed) -> WIN for side to move
        g = FakeGraph([1, 0], [WIN_P0, WIN_P0], [[], []])
        s = solve(g)
        self.assertEqual(s.value, [LOSS, WIN])
        self.assertEqual(s.value_for(0, 0), 1)
        self.assertEqual(s.value_for(0, 1), -1)

    def test_cycle_is_draw(self):
        # 0 (P0) <-> 1 (P1); 0 can also walk into 2 (P1 wins), 1 into 3 (P0 wins)
        g = FakeGraph([0, 1, 1, 0], [NONTERMINAL, NONTERMINAL, WIN_P1, WIN_P0],
                      [[('a', 1), ('lose', 2)], [('b', 0), ('lose', 3)], [], []])
        s = solve(g)
        self.assertEqual(s.value[:2], [DRAW, DRAW])
        self.assertIsNone(s.rank[0])
        self.assertEqual(s.value_optimal_actions(0), ['a'])
        self.assertEqual(s.rank_optimal_actions(1), ['b'])
        self.assertEqual(s.horizon_value(0, 1000), DRAW)

    def test_duplicate_edges_counted_once(self):
        # 0 (P0): two actions into 1 and one into 2; both children are P1 wins -> LOSS
        g = FakeGraph([0, 1, 1], [NONTERMINAL, WIN_P1, WIN_P1],
                      [[('x', 1), ('y', 1), ('z', 2)], [], []])
        s = solve(g)
        self.assertEqual(s.value[0], LOSS)
        self.assertEqual(s.rank[0], 1)
        # Same duplicates but the third action reaches a draw -> not a LOSS
        g2 = FakeGraph([0, 1, 1], [NONTERMINAL, WIN_P1, DRAW_NO_LEGAL_ACTIONS],
                       [[('x', 1), ('y', 1), ('z', 2)], [], []])
        s2 = solve(g2)
        self.assertEqual(s2.value[0], DRAW)
        self.assertEqual(s2.value_optimal_actions(0), ['z'])

    def test_tied_and_rank_optimal_actions(self):
        # 0 (P0) -> 1, 2: immediate P0 wins; -> 3: P1 to move, forced loss in 2 more plies
        g = FakeGraph([0, 1, 1, 1, 0, 1],
                      [NONTERMINAL, WIN_P0, WIN_P0, NONTERMINAL, NONTERMINAL, WIN_P0],
                      [[('w1', 1), ('w2', 2), ('slow', 3)], [], [], [('m', 4)], [('w', 5)], []])
        s = solve(g)
        self.assertEqual((s.value[0], s.rank[0]), (WIN, 1))
        self.assertEqual((s.value[3], s.rank[3]), (LOSS, 2))
        self.assertEqual(sorted(s.value_optimal_actions(0)), ['slow', 'w1', 'w2'])
        self.assertEqual(sorted(s.rank_optimal_actions(0)), ['w1', 'w2'])
        self.assertEqual(s.horizon_value(3, 1), DRAW)
        self.assertEqual(s.horizon_value(3, 2), LOSS)
        self.assertAlmostEqual(s.discounted_value(3, 0.9), -0.9)

    def test_loss_rank_is_max_delay(self):
        # 0 (P0): 'fast' -> 1 (P1 wins next ply), 'slow' -> 2 (P1 to move, wins in 3)
        g = FakeGraph([0, 1, 1, 0, 1, 0],
                      [NONTERMINAL, NONTERMINAL, NONTERMINAL, NONTERMINAL, NONTERMINAL, WIN_P1],
                      [[('fast', 1), ('slow', 2)], [('win', 5)], [('p', 3)], [('q', 4)],
                       [('win', 5)], []])
        s = solve(g)
        self.assertEqual(s.value[0], LOSS)
        self.assertEqual(s.rank[1], 1)
        self.assertEqual(s.rank[2], 3)
        self.assertEqual(s.rank[0], 4)
        self.assertEqual(s.rank_optimal_actions(0), ['slow'])

    def test_incomplete_graph_marks_unknown(self):
        # 0 (P0) -> 1 unexpanded, -> 2 P0 already won: WIN is still provable
        # 3 (P0) -> 1 only: unknown, must not become DRAW
        g = FakeGraph([0, 1, 1, 0], [NONTERMINAL, NONTERMINAL, WIN_P0, NONTERMINAL],
                      [[('u', 1), ('w', 2)], None, [], [('u', 1)]], complete=False)
        s = solve(g)
        self.assertFalse(s.exact)
        self.assertEqual(s.value[0], WIN)
        self.assertIsNone(s.value[1])
        self.assertIsNone(s.value[3])
        self.assertEqual(s.status(3), 'UNKNOWN')
        self.assertIsNone(s.value_optimal_actions(0))  # sibling value unknown


class TestEngineGraphs(unittest.TestCase):
    RACE = {'A': False, 'B': False, 'C': False, 'M': 0}

    def test_pure_race_worked_example(self):
        s = solve_config(3, self.RACE)
        g = s.graph
        self.assertTrue(s.exact)
        self.assertEqual(len(g.keys), 45)
        self.assertEqual((s.status(0), s.rank[0]), ('WIN', 3))
        self.assertEqual(s.rank_optimal_actions(0), [('MOVE', (1, 1))])
        self.assertEqual(s.value_optimal_actions(0), [('MOVE', (1, 1))])

    def test_graph_consistency(self):
        rules = {'A': True, 'B': True, 'C': True, 'K': 1, 'M': 0}
        g = StateGraph(3, rules).build()
        self.assertTrue(g.complete)
        self.assertEqual(sum(len(e) for e in g.succ), g.edge_count)
        indeg = [0] * len(g.keys)
        for u, edges in enumerate(g.succ):
            if g.terminal[u] is not NONTERMINAL:
                self.assertEqual(edges, [])
                continue
            e = g.restore(g.keys[u])
            self.assertEqual([a for a, _ in edges], e.get_legal_actions())
            for _, c in edges:
                indeg[c] += 1
                self.assertNotEqual(g.to_move[c], g.to_move[u])
        self.assertTrue(all(d > 0 for d in indeg[1:]))  # everything reachable from root
        self.assertEqual(len(set(g.keys)), len(g.keys))

    def test_matches_independent_reference(self):
        for rules in (self.RACE,
                      {'A': True, 'B': True, 'C': True, 'K': 1, 'M': 0},
                      {'A': False, 'B': True, 'C': False, 'M': 0},
                      {'A': True, 'B': False, 'C': False, 'M': 1}):
            s = solve_config(3, rules)
            ref = reference(3, rules)
            merged = {}
            for k, (rv, rr) in ref.items():
                u = s.graph.node_of(k)
                self.assertIsNotNone(u, (rules, k))
                self.assertEqual(s.value[u], rv, (rules, k))
                if rv != 0:
                    self.assertEqual(s.rank[u], rr, (rules, k))
                # every raw key merged into one canonical node has the same exact value
                self.assertEqual(merged.setdefault(u, (rv, rr)), (rv, rr), (rules, k))
            self.assertEqual(len(merged), len(s.graph.keys))

    def test_matches_existing_exact_solver(self):
        from agents.exact_solver import ExactSolver
        to_status = {WIN: 'WIN', LOSS: 'LOSS', DRAW: 'DRAW'}
        for rules, m in ((self.RACE, 0), ({'A': True, 'B': True, 'C': True, 'K': 2, 'M': 0}, 0)):
            mine = solve_config(3, rules)
            old = ExactSolver(board_size=3, rules=rules, max_walls=m).solve()
            for k, sol in old.solutions.items():
                self.assertEqual(sol['status'], to_status[mine.value[mine.graph.node_of(k)]], k)
            self.assertEqual({mine.graph.node_of(k) for k in old.solutions},
                             set(range(len(mine.graph.keys))))

    def test_discounted_and_horizon_values_match_iteration(self):
        rules = {'A': False, 'B': True, 'C': True, 'K': 1, 'M': 0}
        s = solve_config(3, rules)
        g = s.graph
        gamma = 0.9
        # Discounted negamax value iteration: reward +1 to the mover on a winning move
        V = [0.0] * len(g.keys)
        for _ in range(200):
            new = []
            for u, edges in enumerate(g.succ):
                if g.terminal[u] is not NONTERMINAL:
                    new.append(0.0)
                    continue
                best = -2.0
                for _, c in edges:
                    r = 1.0 if g.terminal[c] in (WIN_P0, WIN_P1) else 0.0
                    best = max(best, r - gamma * V[c])
                new.append(best)
            V = new
        for u in range(len(g.keys)):
            self.assertAlmostEqual(V[u], s.discounted_value(u, gamma), places=9)
        # Finite horizon: draw declared after h plies
        def goal_value(u):
            if g.terminal[u] not in (WIN_P0, WIN_P1):
                return 0
            return 1 if (g.terminal[u] == WIN_P0) == (g.to_move[u] == 0) else -1

        for h in (0, 1, 2, 5):
            W = [goal_value(u) for u in range(len(g.keys))]
            for _ in range(h):
                W = [W[u] if g.terminal[u] is not NONTERMINAL else max(-W[c] for _, c in g.succ[u])
                     for u in range(len(g.keys))]
            for u in range(len(g.keys)):
                self.assertEqual(W[u], s.horizon_value(u, h), (h, u))

    def test_cap_gives_unknown_not_draw(self):
        rules = {'A': True, 'B': True, 'C': True, 'K': 1, 'M': 0}
        full = solve_config(3, rules)
        part = solve_config(3, rules, GraphCaps(max_states=200))
        self.assertFalse(part.exact)
        self.assertEqual(part.graph.stop_reason, 'max_states')
        self.assertGreater(part.counts()['UNKNOWN'], 0)
        for u, k in enumerate(part.graph.keys):
            v = part.value[u]
            if v is None:
                continue
            fu = full.graph.node_of(k)
            self.assertEqual(v, full.value[fu])   # every proven value is correct
            if v == DRAW:
                self.assertEqual(part.graph.terminal[u], DRAW_NO_LEGAL_ACTIONS)

    def test_capped_rank_queries_never_wrong(self):
        rules = {'A': True, 'B': True, 'C': True, 'K': 1, 'M': 0}
        full = solve_config(3, rules)
        total = len(full.graph.keys)
        for cap in sorted({total * f // 10 for f in (2, 4, 6)}):
            part = solve_config(3, rules, GraphCaps(max_states=cap))
            self.assertFalse(part.exact)
            for u, k in enumerate(part.graph.keys):
                fu = full.graph.node_of(k)
                if part.graph.succ[u]:
                    self.assertIsNone(part.rank_optimal_actions(u))
                for h in range(8):
                    hv = part.horizon_value(u, h)
                    self.assertIn(hv, (None, full.horizon_value(fu, h)), (cap, u, h))
                dv = part.discounted_value(u, 0.9)
                if dv is not None:
                    self.assertAlmostEqual(dv, full.discounted_value(fu, 0.9))
            summary = part.summary()
            self.assertIsNone(summary['root_rank_optimal_actions'])

    def test_canonical_positions_merge_equivalent_keys(self):
        rules = {'A': False, 'B': False, 'C': False, 'M': 1}
        raw_graph = StateGraph(3, rules, canonical=False).build()
        canon = solve_config(3, rules)
        self.assertEqual(len(raw_graph.keys), 1953)
        self.assertEqual(len(canon.graph.keys), 1341)
        raw_sol = solve(raw_graph)
        for u, k in enumerate(raw_graph.keys):
            self.assertEqual(raw_sol.value[u], canon.value[canon.graph.node_of(k)])
            self.assertEqual(raw_sol.rank[u], canon.rank[canon.graph.node_of(k)])

    def test_no_legal_action_terminals_reachable(self):
        g = StateGraph(3, {'A': False, 'B': False, 'C': False, 'M': 2}).build()
        n = g.terminal.count(DRAW_NO_LEGAL_ACTIONS)
        self.assertGreater(n, 0)
        for u, t in enumerate(g.terminal):
            if t == DRAW_NO_LEGAL_ACTIONS:
                self.assertEqual(g.restore(g.keys[u]).get_legal_actions(), [])

    def test_root_from_midgame_key(self):
        e = WallzEngine(board_size=3, rules=self.RACE)
        e.apply_action(('MOVE', (0, 0)), update_clock=False)
        s = solve_config(3, self.RACE, root_key=e._get_state_hashable_key())
        full = solve_config(3, self.RACE)
        self.assertEqual(s.value[0], full.value[full.graph.node_of(s.graph.root_key)])

    def test_search_order_independence(self):
        # BFS graph and DFS reference agree on the reachable set (checked above); also
        # building twice yields identical node order and edges.
        a = StateGraph(3, self.RACE).build()
        b = StateGraph(3, self.RACE).build()
        self.assertEqual(a.keys, b.keys)
        self.assertEqual(a.succ, b.succ)


if __name__ == '__main__':
    unittest.main()
