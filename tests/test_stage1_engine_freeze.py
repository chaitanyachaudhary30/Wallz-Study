"""Stage 1 engine freeze: one unittest file covering the frozen rule set (rules A-E)
plus the Task 2 regression list and the Task 3 Greedy compatibility checks.

Plain unittest on purpose: the older rule files (test_rules_blocks_hop.py,
test_rules_draw_budget.py, test_strict_actions.py, test_action_normalization.py) need
pytest/hypothesis, which are not installed on every team machine.
Run: python -m unittest tests.test_stage1_engine_freeze
"""
import unittest

from agents.greedy import GreedyAgent
from src.modified_wallz.config import WallzConfig
from tests.helpers import snapshot
from wallz_engine import WallzEngine


def make(n=5, **rules):
    base = {'A': True, 'B': True, 'C': True, 'K': 3, 'M': 10}
    base.update(rules)
    return WallzEngine(board_size=n, rules=base)


def play(engine, *actions):
    for a in actions:
        engine.apply_action(a, update_clock=False)
    return engine


def side_step(engine):
    """Apply a sideways MOVE (same row) so plies pass without progress."""
    r, _ = engine.pawns[engine.current_player]
    for a in engine.get_legal_actions():
        if a[0] == 'MOVE' and a[1][0] == r:
            engine.apply_action(a, update_clock=False)
            return a
    raise AssertionError("no sideways move available")


def set_position(engine, pawns, walls0=(), walls1=(), blocks=None, to_move=0):
    """Direct state set-up for hand-built positions (tests only)."""
    engine.pawns = {0: pawns[0], 1: pawns[1]}
    engine.walls = {0: list(walls0), 1: list(walls1)}
    engine.latest_wall = {0: None, 1: None}
    engine.blocked_cells = dict(blocks or {})
    engine.current_player = to_move
    engine._sync_board()
    return engine


# ---------------------------------------------------------------------------
# Rule A: block duration is exactly K plies
# ---------------------------------------------------------------------------
class TestRuleA_BlockDuration(unittest.TestCase):
    def test_state_right_after_placement(self):
        e = make(K=3)
        play(e, ('BLOCK', (2, 4)))
        self.assertEqual(e.blocked_cells, {(2, 4): 3})
        self.assertFalse(e.block_available[0])
        self.assertTrue(e.block_available[1])

    def test_active_exactly_k_plies_for_several_k(self):
        for k in (1, 2, 3, 4):
            with self.subTest(K=k):
                e = make(K=k)
                play(e, ('BLOCK', (2, 4)))                  # ply t
                for j in range(1, k + 1):                   # plies t+1 .. t+K
                    self.assertIn((2, 4), e.blocked_cells, f"blocked at start of ply t+{j}")
                    self.assertEqual(e.blocked_cells[(2, 4)], k - j + 1)
                    side_step(e)
                self.assertNotIn((2, 4), e.blocked_cells, "expired after ply t+K")

    def test_move_into_block_rejected_on_kth_ply_allowed_after(self):
        e = make(K=3)
        play(e, ('BLOCK', (3, 1)),                     # t: P0
             ('MOVE', (4, 1)),                         # t+1: P1, now under the block
             ('MOVE', (0, 1)))                         # t+2: P0
        snap = snapshot(e)                             # t+3: P1, still blocked
        self.assertNotIn(('MOVE', (3, 1)), e.get_legal_actions())
        with self.assertRaises(ValueError):
            e.apply_action(('MOVE', (3, 1)), update_clock=False)
        self.assertEqual(snapshot(e), snap)
        play(e, ('WALL', (0, 3, 'V')))                 # t+3 played: block expires
        self.assertNotIn((3, 1), e.blocked_cells)
        play(e, ('MOVE', (0, 2)))                      # t+4: P0
        self.assertIn(('MOVE', (3, 1)), e.get_legal_actions())   # t+5: P1 may enter

    def test_consecutive_blocks_by_both_players(self):
        e = make(K=3)
        play(e, ('BLOCK', (2, 0)))                      # t: P0
        play(e, ('BLOCK', (2, 4)))                      # t+1: P1, both active
        self.assertEqual(e.blocked_cells, {(2, 0): 2, (2, 4): 3})
        with self.assertRaisesRegex(ValueError, 'no cell block available'):
            e.apply_action(('BLOCK', (1, 0)), update_clock=False)   # P0 already used it
        side_step(e)                                    # t+2
        side_step(e)                                    # t+3: P0's block expires
        self.assertEqual(e.blocked_cells, {(2, 4): 1})
        side_step(e)                                    # t+4: P1's block expires
        self.assertEqual(e.blocked_cells, {})

    def test_cannot_block_blocked_or_pawn_cell(self):
        e = make()
        play(e, ('BLOCK', (2, 0)))
        for cell in ((2, 0), e.pawns[0], e.pawns[1]):
            with self.subTest(cell=cell):
                with self.assertRaises(ValueError):
                    e.apply_action(('BLOCK', cell), update_clock=False)

    def test_expiring_block_still_counts_in_path_checks(self):
        """Decision 9 (docs/decisions.md): a block with timer 1 still blocks path checks
        on the ply it expires, so a wall that is only legal once it expires is rejected."""
        e = make(n=3, K=3)
        set_position(e, {0: (0, 1), 1: (2, 1)}, blocks={(1, 0): 1, (1, 2): 1})
        self.assertFalse(e._is_valid_wall_placement(0, 0, 0, 'H'))
        self.assertNotIn(('WALL', (0, 0, 'H')), e.get_legal_actions())
        e.blocked_cells = {}
        self.assertTrue(e._is_valid_wall_placement(0, 0, 0, 'H'))


# ---------------------------------------------------------------------------
# Rule B: strict action validation
# ---------------------------------------------------------------------------
class TestRuleB_StrictValidation(unittest.TestCase):
    def assertRejectedUnchanged(self, e, action, update_clock=False):
        snap = snapshot(e)
        with self.assertRaises(ValueError):
            e.apply_action(action, update_clock=update_clock)
        self.assertEqual(snapshot(e), snap)

    def test_illegal_actions_rejected_without_state_change(self):
        e = make()
        play(e, ('WALL', (0, 1, 'H')))       # P0
        play(e, ('MOVE', (3, 2)))           # P1
        illegal = [
            ('MOVE', (1, 2)),               # across own wall
            ('MOVE', (-1, 2)),              # off board
            ('MOVE', (2, 2)),               # not adjacent
            ('HOP', (1, 2)),                # own wall: no hop
            ('WALL', (0, 1, 'V')),          # same centre (cross)
            ('WALL', (0, 2, 'H')),          # overlaps
            ('WALL', (9, 9, 'H')),          # off board
            ('REPLACE', ((0, 3, 'H'), (2, 2, 'H'))),   # old wall is not the latest
            ('REPLACE', (0, 1, 'H')),       # same place
            ('BLOCK', (0, 2)),              # own pawn cell
            ('JUMP', (1, 2)),               # unknown type
            ('MOVE', (True, 2)),            # bool coordinate
            ('WALL', (1, 1, 'X')),          # bad orientation
            'MOVE',                         # not a pair
            ('MOVE',),                      # wrong length
        ]
        for a in illegal:
            with self.subTest(action=a):
                self.assertRejectedUnchanged(e, a)

    def test_invalid_action_does_not_touch_clock(self):
        e = make()
        self.assertRejectedUnchanged(e, ('MOVE', (3, 3)), update_clock=True)

    def test_replace_rejected_when_rule_off_or_no_wall_yet(self):
        e = make()
        self.assertRejectedUnchanged(e, ('REPLACE', (2, 2, 'H')))   # no wall yet
        e = make(A=False)
        play(e, ('WALL', (2, 0, 'H')), ('MOVE', (3, 2)))
        play(e, ('MOVE', (1, 2)), ('MOVE', (3, 1)))
        self.assertRejectedUnchanged(e, ('REPLACE', (2, 2, 'H')))

    def test_actions_after_termination_rejected(self):
        e = make(n=3)
        play(e, ('MOVE', (1, 1)), ('MOVE', (2, 0)), ('MOVE', (2, 1)))
        self.assertEqual(e.winner, 0)
        self.assertRejectedUnchanged(e, ('MOVE', (1, 0)))


# ---------------------------------------------------------------------------
# Rule C: no-legal-move position is a draw; terminal detection agrees
# ---------------------------------------------------------------------------
class TestRuleC_NoLegalDraw(unittest.TestCase):
    def _trapped(self):
        e = make(n=3, A=False, B=False, C=False, M=0)
        # P0 in the corner, wall on its right; P1 then steps into the only free neighbour.
        set_position(e, {0: (0, 0), 1: (2, 0)}, walls0=[(0, 0, 'V')], to_move=1)
        play(e, ('MOVE', (1, 0)))
        return e

    def test_draw_flags(self):
        e = self._trapped()
        self.assertTrue(e.is_draw)
        self.assertIsNone(e.winner)
        self.assertEqual(e.terminal_reason, 'no_legal_actions')
        self.assertTrue(e.is_terminal())
        self.assertFalse(e.has_legal_action())
        self.assertEqual(e.get_legal_actions(0), [])
        self.assertEqual(e.get_legal_actions(1), [])
        self.assertTrue(e.is_path_open(0))   # path exists; only the pawn is in the way
        with self.assertRaises(ValueError):
            e.apply_action(('MOVE', (0, 1)), update_clock=False)

    def test_greedy_returns_none_on_draw(self):
        e = self._trapped()
        self.assertIsNone(GreedyAgent(0).select_action(e))

    def test_goal_win_flags_agree(self):
        e = make(n=3)
        play(e, ('MOVE', (1, 1)), ('MOVE', (2, 0)), ('MOVE', (2, 1)))
        self.assertEqual((e.winner, e.is_draw, e.terminal_reason), (0, False, 'goal'))
        self.assertTrue(e.is_terminal())
        self.assertFalse(e.has_legal_action())
        self.assertIsNone(GreedyAgent(1).select_action(e))


# ---------------------------------------------------------------------------
# Rule D: configurable wall budget rules['M']
# ---------------------------------------------------------------------------
class TestRuleD_WallBudget(unittest.TestCase):
    def test_default_is_10_for_both(self):
        e = WallzEngine(board_size=5)
        self.assertEqual(e.wall_inventory, {0: 10, 1: 10})
        self.assertEqual(WallzConfig().M, 10)

    def test_configured_budget(self):
        for m in (0, 1, 3, 12):
            with self.subTest(M=m):
                e = make(M=m)
                self.assertEqual(e.wall_inventory, {0: m, 1: m})
                self.assertEqual(e.max_walls, m)
                walls = [a for a in e.get_legal_actions() if a[0] == 'WALL']
                self.assertEqual(bool(walls), m > 0)

    def test_negative_budget_rejected(self):
        with self.assertRaises(ValueError):
            make(M=-1)

    def test_budget_exhausted_but_replace_allowed(self):
        e = make(M=1)
        play(e, ('WALL', (2, 0, 'H')), ('MOVE', (3, 2)))
        self.assertEqual(e.wall_inventory[0], 0)
        legal = e.get_legal_actions()
        self.assertFalse([a for a in legal if a[0] == 'WALL'])
        self.assertTrue([a for a in legal if a[0] == 'REPLACE'])
        with self.assertRaisesRegex(ValueError, 'no walls remaining'):
            e.apply_action(('WALL', (2, 3, 'H')), update_clock=False)
        play(e, ('REPLACE', (2, 3, 'H')))
        self.assertEqual(e.wall_inventory[0], 0)

    def test_greedy_never_walls_with_zero_budget(self):
        e = make(M=0)
        for _ in range(8):
            if e.is_terminal():
                break
            a = GreedyAgent(e.current_player).select_action(e)
            self.assertNotEqual(a[0], 'WALL')
            e.apply_action(a, update_clock=False)


# ---------------------------------------------------------------------------
# Rule E: HOP = one step across an OPPONENT's wall, once per game
# ---------------------------------------------------------------------------
class TestRuleE_Hop(unittest.TestCase):
    def _opponent_wall_in_front(self):
        """P0 at (1,2); P1's wall (moved there by REPLACE) blocks (1,2)-(2,2)."""
        e = make()
        play(e, ('MOVE', (1, 2)), ('WALL', (3, 0, 'H')),
             ('WALL', (3, 3, 'H')), ('REPLACE', (1, 2, 'H')))
        return e

    def test_hop_over_opponent_wall_and_replace_keeps_owner(self):
        e = self._opponent_wall_in_front()
        self.assertEqual(e.wall_owner_on_edge((1, 2), (2, 2)), 1)
        self.assertNotIn(('MOVE', (2, 2)), e.get_legal_actions())
        self.assertIn(('HOP', (2, 2)), e.get_legal_actions())
        play(e, ('HOP', (2, 2)))
        self.assertEqual(e.pawns[0], (2, 2))
        self.assertFalse(e.hop_available[0])
        self.assertTrue(e.hop_available[1])

    def test_hop_is_one_time(self):
        e = self._opponent_wall_in_front()
        play(e, ('HOP', (2, 2)), ('MOVE', (4, 1)))
        # P1's wall is now behind P0; crossing it back would need a second hop.
        self.assertNotIn(('HOP', (1, 2)), e.get_legal_actions())
        with self.assertRaises(ValueError):
            e.apply_action(('HOP', (1, 2)), update_clock=False)

    def test_no_hop_over_own_wall(self):
        e = make()
        play(e, ('WALL', (0, 1, 'H')), ('MOVE', (3, 2)))
        self.assertEqual(e.wall_owner_on_edge((0, 2), (1, 2)), 0)
        self.assertFalse([a for a in e.get_legal_actions() if a[0] == 'HOP'])
        with self.assertRaises(ValueError):
            e.apply_action(('HOP', (1, 2)), update_clock=False)

    def test_no_hop_over_open_edge_or_two_cells(self):
        e = make()
        for target in ((1, 2), (2, 2)):
            with self.subTest(target=target):
                with self.assertRaises(ValueError):
                    e.apply_action(('HOP', target), update_clock=False)

    def test_no_hop_onto_opponent_pawn_or_blocked_cell(self):
        e = make()
        set_position(e, {0: (1, 2), 1: (2, 2)}, walls1=[(1, 2, 'H')])
        self.assertFalse(e._is_valid_hop(0, (2, 2)))
        set_position(e, {0: (1, 2), 1: (4, 2)}, walls1=[(1, 2, 'H')], blocks={(2, 2): 2})
        self.assertFalse(e._is_valid_hop(0, (2, 2)))
        e.blocked_cells = {}
        self.assertTrue(e._is_valid_hop(0, (2, 2)))

    def test_no_hop_into_dead_end(self):
        """Path preservation: landing cell (2,2) is walled in on all four sides."""
        e = make()
        pocket = [(2, 1, 'H'), (1, 1, 'V'), (2, 2, 'V')]
        set_position(e, {0: (1, 2), 1: (4, 2)}, walls0=pocket, walls1=[(1, 2, 'H')])
        self.assertTrue(e.is_path_open(0) and e.is_path_open(1))
        self.assertNotIn(('HOP', (2, 2)), e.get_legal_actions())
        snap = snapshot(e)
        with self.assertRaises(ValueError):
            e.apply_action(('HOP', (2, 2)), update_clock=False)
        self.assertEqual(snapshot(e), snap)
        set_position(e, {0: (1, 2), 1: (4, 2)}, walls0=pocket[1:], walls1=[(1, 2, 'H')])
        self.assertIn(('HOP', (2, 2)), e.get_legal_actions())

    def test_no_hop_when_rule_off(self):
        e = make(B=False)
        set_position(e, {0: (1, 2), 1: (4, 2)}, walls1=[(1, 2, 'H')])
        self.assertFalse([a for a in e.get_legal_actions() if a[0] == 'HOP'])

    def test_position_key_keeps_owner_only_while_a_hop_remains(self):
        a, b = make(), make()
        set_position(a, {0: (0, 2), 1: (4, 2)}, walls0=[(2, 0, 'H')])
        set_position(b, {0: (0, 2), 1: (4, 2)}, walls1=[(2, 0, 'H')])
        self.assertNotEqual(a.position_key(), b.position_key())
        for e in (a, b):
            e.hop_available = {0: False, 1: False}
        self.assertEqual(a.position_key(), b.position_key())


# ---------------------------------------------------------------------------
# Task 2 regression list (items not already covered above)
# ---------------------------------------------------------------------------
class TestRegression(unittest.TestCase):
    def test_normal_pawn_moves(self):
        e = make()
        moves = sorted(a[1] for a in e.get_legal_actions() if a[0] == 'MOVE')
        self.assertEqual(moves, [(0, 1), (0, 3), (1, 2)])   # edge row: no move to row -1

    def test_wall_placement_updates_state(self):
        e = make()
        play(e, ('WALL', (2, 0, 'H')))
        self.assertEqual(e.walls[0], [(2, 0, 'H')])
        self.assertEqual(e.latest_wall[0], (2, 0, 'H'))
        self.assertEqual(e.wall_inventory[0], 9)
        self.assertTrue(e.board.is_edge_blocked((2, 0), (3, 0)))

    def test_wall_reposition_latest_only(self):
        e = make()
        play(e, ('WALL', (2, 0, 'H')), ('MOVE', (3, 2)),
             ('WALL', (2, 3, 'H')), ('MOVE', (4, 2)))
        with self.assertRaisesRegex(ValueError, 'older wall'):
            e.apply_action(('REPLACE', ((2, 0, 'H'), (1, 0, 'H'))), update_clock=False)
        play(e, ('REPLACE', ((2, 3, 'H'), (1, 3, 'H'))))
        self.assertEqual(e.walls[0], [(2, 0, 'H'), (1, 3, 'H')])
        self.assertEqual(e.latest_wall[0], (1, 3, 'H'))
        self.assertEqual(e.wall_inventory[0], 8)
        self.assertFalse(e.board.is_edge_blocked((2, 3), (3, 3)))
        self.assertTrue(e.board.is_edge_blocked((1, 3), (2, 3)))

    def test_wall_and_block_path_preservation(self):
        e = make()
        play(e, ('WALL', (0, 1, 'H')), ('MOVE', (3, 2)),
             ('WALL', (0, 3, 'H')), ('MOVE', (4, 2)))
        # P0's only exit is now (0,0)->(1,0)
        legal = e.get_legal_actions()
        self.assertNotIn(('WALL', (0, 0, 'V')), legal)
        self.assertNotIn(('BLOCK', (1, 0)), legal)
        self.assertIn(('BLOCK', (2, 0)), legal)
        for a in (('WALL', (0, 0, 'V')), ('BLOCK', (1, 0))):
            with self.assertRaises(ValueError):
                e.apply_action(a, update_clock=False)

    def test_fivefold_repetition_draw(self):
        e = make()
        cycle = [('MOVE', (1, 2)), ('MOVE', (3, 2)), ('MOVE', (0, 2)), ('MOVE', (4, 2))]
        for _ in range(3):
            play(e, *cycle)
        play(e, *cycle[:3])
        self.assertFalse(e.is_terminal())           # start position seen 4 times
        play(e, cycle[3])
        self.assertEqual((e.is_draw, e.terminal_reason), (True, 'repetition'))

    def test_repetition_limit_configurable(self):
        e = make(repetition_limit=2)
        play(e, ('MOVE', (1, 2)), ('MOVE', (3, 2)), ('MOVE', (0, 2)), ('MOVE', (4, 2)))
        self.assertEqual(e.terminal_reason, 'repetition')

    def test_max_plies_truncation(self):
        e = make(max_plies=3)
        play(e, ('MOVE', (1, 2)), ('MOVE', (3, 2)))
        self.assertFalse(e.is_terminal())
        play(e, ('MOVE', (0, 2)))
        self.assertEqual((e.is_truncated, e.is_draw, e.terminal_reason), (True, True, 'truncated'))
        self.assertTrue(WallzEngine(board_size=5).max_plies is None)   # off by default


# ---------------------------------------------------------------------------
# Task 3 compatibility: Greedy against the frozen engine
# ---------------------------------------------------------------------------
class TestGreedyCompatibility(unittest.TestCase):
    CONFIGS = [
        (3, {'K': 1, 'M': 0}),
        (3, {'K': 3, 'M': 2}),
        (5, {'K': 2, 'M': 3}),
        (5, {'K': 3, 'M': 10}),
        (5, {'A': False, 'B': False, 'C': False, 'M': 4}),
        (7, {'K': 4, 'M': 6}),
    ]

    def test_greedy_plays_only_legal_actions(self):
        for n, rules in self.CONFIGS:
            with self.subTest(n=n, rules=rules):
                e = make(n=n, **rules)
                agents = {0: GreedyAgent(0), 1: GreedyAgent(1, seed=1)}
                for _ in range(12 if n < 7 else 6):
                    legal = e.get_legal_actions()
                    self.assertEqual(bool(legal), e.has_legal_action())
                    a = agents[e.current_player].select_action(e)
                    if e.is_terminal():
                        self.assertIsNone(a)
                        break
                    self.assertIn(a, legal)
                    mover = e.current_player
                    before = e.wall_inventory[mover]
                    e.apply_action(a, update_clock=False)
                    self.assertGreaterEqual(e.wall_inventory[mover], 0)
                    self.assertEqual(e.wall_inventory[mover], before - (a[0] == 'WALL'))
                    for cell, t in e.blocked_cells.items():
                        self.assertTrue(1 <= t <= e.rules['K'])
                        self.assertNotIn(cell, (e.pawns[0], e.pawns[1]))

    def test_greedy_scores_truncation_draw_as_zero(self):
        e = make(max_plies=1)
        rows = GreedyAgent(0).score_actions(e)
        self.assertTrue(rows)
        self.assertTrue(all(r['outcome'] == 'draw' and r['H'] == 0.0 for r in rows))


if __name__ == '__main__':
    unittest.main()
