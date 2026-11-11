"""Tests for the match runner, outcome accounting, metrics and demo export."""
import json
import os
import tempfile
import time
import unittest
from collections import deque

from wallz_engine import WallzEngine
from evaluation.ablation import factorial_rules, ablation_configs
from evaluation.match_runner import (MatchConfig, run_match, verify_replay, run_batch, load_results,
                                     make_agent, opening_actions, AgentUnavailable, action_from_json)
from evaluation.metrics import (factorial_effects, outcome_for, summarize_cells, effects_with_ci,
                                wilson)
from evaluation.replay_export import export_demo_replay, to_demo_event
from evaluation.state_graph import StateGraph, DRAW_NO_LEGAL_ACTIONS


class Scripted:
    """Plays a fixed action list; then behaves as configured."""

    def __init__(self, actions, then=None):
        self.actions = deque(actions)
        self.then = then

    def select_action(self, engine):
        if self.actions:
            return self.actions.popleft()
        if self.then == 'raise':
            raise RuntimeError("boom")
        if self.then == 'sleep':
            time.sleep(0.05)
        if self.then == 'none':
            return None
        if self.then == 'illegal':
            return ('MOVE', (0, 0))
        return engine.get_legal_actions()[0]


def ids(spec):
    return {'spec': spec, 'name': spec, 'params': {}}


def play(cfg, a0, a1):
    return run_match(cfg, agents={0: a0, 1: a1}, identities={0: ids('x'), 1: ids('y')})


class TestOutcomes(unittest.TestCase):
    def test_goal(self):
        cfg = MatchConfig(board_size=3, rules={'A': False, 'B': False, 'C': False, 'M': 0})
        r = play(cfg, Scripted([('MOVE', (1, 1)), ('MOVE', (2, 1))]), Scripted([('MOVE', (2, 0))]))
        self.assertEqual(r['outcome']['category'], 'goal')
        self.assertEqual(r['outcome']['winner'], 0)
        self.assertEqual(r['length_plies'], 3)
        self.assertEqual(outcome_for(r, 0), 'win')
        self.assertEqual(outcome_for(r, 1), 'loss')

    def test_repetition_draw(self):
        cfg = MatchConfig(board_size=5, rules={'B': False, 'C': False, 'M': 0})
        cycle0 = [('MOVE', (0, 1)), ('MOVE', (0, 2))] * 4
        cycle1 = [('MOVE', (4, 1)), ('MOVE', (4, 2))] * 4
        r = play(cfg, Scripted(cycle0), Scripted(cycle1))
        self.assertEqual(r['outcome']['category'], 'draw_repetition')
        self.assertEqual(outcome_for(r, 0), 'draw')

    def test_no_legal_actions_draw(self):
        rules = {'A': False, 'B': False, 'C': False, 'M': 2}
        g = StateGraph(3, rules).build()
        target = g.terminal.index(DRAW_NO_LEGAL_ACTIONS)
        parent = {0: None}
        q = deque([0])
        while target not in parent:
            u = q.popleft()
            for a, c in g.succ[u]:
                if c not in parent:
                    parent[c] = (u, a)
                    q.append(c)
        path, v = [], target
        while parent[v]:
            v, a = parent[v]
            path.append(a)
        path.reverse()
        r = play(MatchConfig(board_size=3, rules=rules), Scripted(path[0::2]), Scripted(path[1::2]))
        self.assertEqual(r['outcome']['category'], 'draw_no_legal_actions')
        self.assertTrue(verify_replay(r)[0])

    def test_truncation_is_not_a_draw(self):
        cfg = MatchConfig(board_size=5, max_plies=3)
        r = play(cfg, Scripted([]), Scripted([]))
        self.assertEqual(r['outcome']['category'], 'truncated')
        self.assertIsNone(r['outcome']['winner'])
        self.assertEqual(outcome_for(r, 0), 'truncated')
        self.assertEqual(r['length_plies'], 3)

    def test_failures_are_forfeits(self):
        for then, cat in (('raise', 'agent_error'), ('none', 'no_action'), ('illegal', 'illegal_action')):
            r = play(MatchConfig(board_size=5), Scripted([], then), Scripted([]))
            self.assertEqual(r['outcome']['category'], cat)
            self.assertEqual(r['outcome']['loser'], 0)
            self.assertEqual(outcome_for(r, 1), 'forfeit_win')
            self.assertEqual(outcome_for(r, 0), 'forfeit_loss')

    def test_timeout(self):
        cfg = MatchConfig(board_size=5, move_time_limit=0.01)
        r = play(cfg, Scripted([]), Scripted([], 'sleep'))
        self.assertEqual(r['outcome']['category'], 'timeout')
        self.assertEqual(r['outcome']['loser'], 1)

    def test_dqn_is_reported_missing(self):
        with self.assertRaises(AgentUnavailable):
            make_agent('dqn', 0, MatchConfig())


class TestReproducibility(unittest.TestCase):
    def test_deterministic_and_serialisable(self):
        cfg = MatchConfig(board_size=5, p0='minimax:depth=1', p1='greedy', opening_plies=2, opening_seed=4)
        a, b = run_match(cfg), run_match(cfg)
        json.dumps(a)
        strip = lambda r: [(s['action'], s.get('state_id')) for s in r['replay']]
        self.assertEqual(strip(a), strip(b))
        self.assertEqual(a['outcome'], b['outcome'])
        self.assertTrue(verify_replay(a)[0])

    def test_replay_tampering_detected(self):
        r = run_match(MatchConfig(board_size=5, p0='greedy', p1='greedy', max_plies=12))
        bad = json.loads(json.dumps(r))
        bad['replay'][3]['state_id'] = '0' * 16
        self.assertFalse(verify_replay(bad)[0])
        bad = json.loads(json.dumps(r))
        bad['replay'][0]['action'] = ['MOVE', [3, 3]]
        self.assertFalse(verify_replay(bad)[0])

    def test_openings_identical_across_mechanics(self):
        seqs = set()
        for rules in factorial_rules(K=3, M=10):
            e = WallzEngine(board_size=5, rules=rules)
            seqs.add(tuple(opening_actions(e, 4, seed=11)))
        self.assertEqual(len(seqs), 1)

    def test_first_distinct_openings(self):
        from evaluation.ablation import first_distinct_openings
        picked = first_distinct_openings(5, 2, 50, max_seed=200)
        self.assertEqual(len(picked), 9)                       # 3 x 3 pawn moves exist at 2 plies
        self.assertEqual(len({a for _, a in picked}), 9)
        seeds = [s for s, _ in picked]
        self.assertEqual(seeds, sorted(seeds))
        self.assertEqual(seeds[0], 0)
        for s, acts in picked:                                # each is the FIRST seed of its opening
            for earlier in range(s):
                e = WallzEngine(board_size=5, rules={'K': 3, 'M': 10})
                self.assertNotEqual(tuple(opening_actions(e, 2, earlier)), acts)
        self.assertEqual(len(first_distinct_openings(5, 4, 3)), 3)

    def test_batch_resume(self):
        cfgs = ablation_configs('greedy', 'random', board_size=3, K=1, M=0, opening_seeds=range(1),
                                opening_plies=0, max_plies=30)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'm.jsonl')
            first = run_batch(cfgs, path, keep_replay=True)
            again = run_batch(cfgs, path, keep_replay=True)
            self.assertEqual(len(first), 16)
            self.assertEqual(again, [])
            self.assertEqual(len(load_results(path)), 16)


class TestMetrics(unittest.TestCase):
    def test_factorial_effects_recover_contrasts(self):
        cells = {}
        for rules in factorial_rules():
            a, b, c = (1 if rules[f] else -1 for f in 'ABC')
            label = MatchConfig(board_size=5, rules=rules).label()
            cells[label] = 10 + 1.0 * a + 0.25 * a * b + 0.5 * a * b * c
        eff = factorial_effects(cells)
        expected = {'A': 2.0, 'B': 0.0, 'C': 0.0, 'AB': 0.5, 'AC': 0.0, 'BC': 0.0, 'ABC': 1.0}
        for k, v in expected.items():
            self.assertAlmostEqual(eff[k], v)

    def test_summary_denominators_and_bootstrap(self):
        cfgs = ablation_configs('greedy', 'random:seed=1', board_size=3, K=1, M=0,
                                opening_seeds=range(3), opening_plies=0, max_plies=40)
        results = [run_match(c) for c in cfgs]
        cells = summarize_cells(results, 'greedy')
        self.assertEqual(len(cells), 8)
        for cell in cells.values():
            self.assertEqual(sum(cell['counts'].values()), cell['n_games'])
            self.assertEqual(cell['n_games'], 6)
        eff = effects_with_ci(results, 'greedy', 'x_goal_win', n_boot=50)
        self.assertEqual(eff['n_opening_seeds'], 3)
        lo, hi = eff['effects']['A']['ci95']
        self.assertLessEqual(lo, hi)

    def test_wilson(self):
        lo, hi = wilson(5, 10)
        self.assertLess(lo, 0.5)
        self.assertGreater(hi, 0.5)
        self.assertEqual(wilson(0, 0), (None, None))


def demo_edge_blocked(x1, y1, x2, y2, walls):
    """Python transcription of isEdgeBlocked() in the demo's js/pathfinder.js."""
    for gx, gy, o in walls:
        if o == 'H' and ((y1 == gy and y2 == gy + 1) or (y1 == gy + 1 and y2 == gy)) and x1 in (gx, gx + 1):
            return True
        if o == 'V' and ((x1 == gx and x2 == gx + 1) or (x1 == gx + 1 and x2 == gx)) and y1 in (gy, gy + 1):
            return True
    return False


class TestDemoExport(unittest.TestCase):
    def test_wall_geometry_mapping(self):
        n = 9
        e = WallzEngine(board_size=n)
        for wr in range(n - 1):
            for wc in range(n - 1):
                for o in 'HV':
                    e.walls[0] = [(wr, wc, o)]
                    ev = to_demo_event(('WALL', (wr, wc, o)), 0, n)
                    demo_wall = [(ev['gx'], ev['gy'], ev['orientation'])]
                    for r in range(n):
                        for c in range(n):
                            for nr, nc in ((r + 1, c), (r, c + 1)):
                                if nr < n and nc < n:
                                    self.assertEqual(
                                        e._is_wall_blocking((r, c), (nr, nc)),
                                        demo_edge_blocked(c, n - 1 - r, nc, n - 1 - nr, demo_wall))

    def test_start_cells_and_compatibility_flags(self):
        ev = to_demo_event(('MOVE', (0, 4)), 0, 9)
        self.assertEqual((ev['player'], ev['x'], ev['y']), (1, 4, 8))  # demo P1 start
        ev = to_demo_event(('MOVE', (8, 4)), 1, 9)
        self.assertEqual((ev['player'], ev['x'], ev['y']), (2, 4, 0))  # demo P2 start
        r = run_match(MatchConfig(board_size=5, p0='greedy', p1='greedy'))
        out = export_demo_replay(r)
        self.assertFalse(out['demo_compatible'])
        self.assertEqual(len(out['events']), len(r['replay']))
        for step, ev in zip(r['replay'], out['events']):
            self.assertEqual(ev['player'], step['player'] + 1)
            self.assertEqual(action_from_json(step['action'])[0] == 'BLOCK', ev['type'] == 'BLOCK')


if __name__ == '__main__':
    unittest.main()
