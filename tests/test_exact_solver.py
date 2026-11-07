import unittest
from wallz_engine import WallzEngine
from agents.exact_solver import ExactSolver, ExactSolverAgent
from agents.minimax import MinimaxAgent


class TestExactSolver(unittest.TestCase):
    """
    Validation Test Suite for Tier-3 Exact Retrograde Solver.
    Covers:
      1. Terminal state detection and labeling (d=0)
      2. Player-to-move perspective and zero-sum symmetry
      3. Hand-constructed miniature state exact action selection
      4. State-key completeness (Markov state representation)
      5. State immutability (no side-effects on caller engine)
      6. Cyclic game graph draw resolution (backward induction on cycles)
      7. Exhaustive reachability and degree consistency
      8. Ground-truth benchmark alignment with Minimax (terminal depth)
      9. Out-of-domain state handling and graceful fallback
    """

    def setUp(self):
        # 3x3 pure pawn race (fast to solve: 45 states, ~0.002s)
        self.solver_race = ExactSolver(board_size=3, rules={'A': False, 'B': False, 'C': False}, max_walls=0)
        self.solver_race.solve()

        # 3x3 pawn race with hop (186 states, ~0.02s)
        self.solver_hop = ExactSolver(board_size=3, rules={'A': False, 'B': True, 'C': False}, max_walls=0)
        self.solver_hop.solve()

    def test_01_terminal_wins_and_losses(self):
        """1. Terminal states must have distance 0 and exact +1.0 / -1.0 values."""
        # Create an already terminal state where Player 0 reached row 2
        engine = WallzEngine(board_size=3, rules={'A': False, 'B': False, 'C': False})
        engine.pawns[0] = (2, 1)  # Player 0 at goal row
        engine.pawns[1] = (2, 0)
        engine.winner = 0
        engine.current_player = 0

        solver = ExactSolver(board_size=3)
        solver.solve(initial_engine=engine)

        status, dist = solver.get_state_outcome(engine)
        self.assertEqual(status, 'WIN')
        self.assertEqual(dist, 0)
        self.assertEqual(solver.get_value(engine, player=0), 1.0)
        self.assertEqual(solver.get_value(engine, player=1), -1.0)

        # From Player 1 perspective (active player when Player 0 has won)
        engine.current_player = 1
        solver_p1 = ExactSolver(board_size=3)
        solver_p1.solve(initial_engine=engine)
        status_p1, dist_p1 = solver_p1.get_state_outcome(engine)
        self.assertEqual(status_p1, 'LOSS')
        self.assertEqual(dist_p1, 0)
        self.assertEqual(solver_p1.get_value(engine, player=1), -1.0)
        self.assertEqual(solver_p1.get_value(engine, player=0), 1.0)

    def test_02_player_perspective_invariance(self):
        """2. For every solved state, V(s, 0) == -V(s, 1) (zero-sum) and matches active player perspective."""
        for key, sol in self.solver_race.solutions.items():
            val_p0 = self.solver_race.get_value(key, player=0)
            val_p1 = self.solver_race.get_value(key, player=1)
            self.assertEqual(val_p0, -val_p1, f"Zero-sum violated at state {key}: {val_p0} vs {val_p1}")

            active_p = sol['active_player']
            val_active = self.solver_race.get_value(key, player=active_p)
            self.assertEqual(val_active, sol['value'])

    def test_03_hand_constructed_optimal_actions(self):
        """3. Solver selects the exact 1-ply winning action in a hand-crafted miniature state."""
        # Player 0 is at (1, 1), Player 1 is at (1, 0).
        # Player 0 needs to reach row 2. Moving to (2, 1) wins immediately in 1 ply.
        engine = WallzEngine(board_size=3, rules={'A': False, 'B': False, 'C': False})
        engine.pawns[0] = (1, 1)
        engine.pawns[1] = (1, 0)
        engine.current_player = 0
        engine.wall_inventory = {0: 0, 1: 0}

        solver = ExactSolver(board_size=3, rules={'A': False, 'B': False, 'C': False}, max_walls=0)
        solver.solve(initial_engine=engine)

        opt_action = solver.get_optimal_action(engine)
        status, dist = solver.get_state_outcome(engine)

        self.assertEqual(status, 'WIN')
        self.assertEqual(dist, 1)
        self.assertEqual(opt_action, ('MOVE', (2, 1)))

    def test_04_state_key_completeness(self):
        """4. Every dimension of the Markov state must produce a distinct hashable key."""
        base_eng = WallzEngine(board_size=3, rules={'A': True, 'B': True, 'C': True, 'K': 3})
        base_key = base_eng._get_state_hashable_key()

        # a. Current player
        eng_p = base_eng.clone()
        eng_p.current_player = 1
        self.assertNotEqual(base_key, eng_p._get_state_hashable_key())

        # b. Pawn positions
        eng_pos = base_eng.clone()
        eng_pos.pawns[0] = (0, 0)
        self.assertNotEqual(base_key, eng_pos._get_state_hashable_key())

        # c. Wall placements
        eng_w = base_eng.clone()
        eng_w.walls[0].append((0, 0, 'H'))
        self.assertNotEqual(base_key, eng_w._get_state_hashable_key())

        # d. Wall inventory
        eng_inv = base_eng.clone()
        eng_inv.wall_inventory[0] = 5
        self.assertNotEqual(base_key, eng_inv._get_state_hashable_key())

        # e. Latest wall
        eng_lw = base_eng.clone()
        eng_lw.latest_wall[0] = (0, 0, 'H')
        self.assertNotEqual(base_key, eng_lw._get_state_hashable_key())

        # f. Hop availability
        eng_hop = base_eng.clone()
        eng_hop.hop_available[0] = False
        self.assertNotEqual(base_key, eng_hop._get_state_hashable_key())

        # g. Blocked cells
        eng_blk = base_eng.clone()
        eng_blk.blocked_cells[(1, 1)] = 2
        self.assertNotEqual(base_key, eng_blk._get_state_hashable_key())

    def test_05_state_immutability(self):
        """5. Solving and querying exact solver must not mutate caller's engine state."""
        engine = WallzEngine(board_size=3, rules={'A': False, 'B': False, 'C': False})
        p0_before = engine.pawns[0]
        p1_before = engine.pawns[1]
        curr_before = engine.current_player
        inv_before = dict(engine.wall_inventory)
        hop_before = dict(engine.hop_available)

        solver = ExactSolver(board_size=3, rules={'A': False, 'B': False, 'C': False}, max_walls=0)
        solver.solve(initial_engine=engine)
        solver.get_optimal_action(engine)
        solver.get_value(engine)

        self.assertEqual(engine.pawns[0], p0_before)
        self.assertEqual(engine.pawns[1], p1_before)
        self.assertEqual(engine.current_player, curr_before)
        self.assertEqual(engine.wall_inventory, inv_before)
        self.assertEqual(engine.hop_available, hop_before)

    def test_06_draw_and_cycle_handling(self):
        """6. Cyclic states without forced winning paths must resolve to DRAW (dist=inf, val=0)."""
        # In 3x3 pawn race, if Player 0 is at (0, 0) and Player 1 is at (2, 1),
        # both players can cycle indefinitely. That branch must resolve to DRAW.
        engine = WallzEngine(board_size=3, rules={'A': False, 'B': False, 'C': False})
        engine.pawns[0] = (0, 0)
        engine.pawns[1] = (2, 1)
        engine.current_player = 0
        engine.wall_inventory = {0: 0, 1: 0}

        solver = ExactSolver(board_size=3, rules={'A': False, 'B': False, 'C': False}, max_walls=0)
        solver.solve(initial_engine=engine)

        status, dist = solver.get_state_outcome(engine)
        val = solver.get_value(engine)
        self.assertEqual(status, 'DRAW')
        self.assertEqual(dist, float('inf'))
        self.assertEqual(val, 0.0)

    def test_07_game_theoretic_bellman_invariants(self):
        """7. Solver must satisfy strict retrograde Bellman equilibrium invariants across all states."""
        solver = self.solver_race
        dummy_engine = WallzEngine(board_size=3, rules={'A': False, 'B': False, 'C': False})

        for key, sol in solver.solutions.items():
            status = sol['status']
            dist = sol['distance']
            eng = dummy_engine.restore_state_key(key)
            legal_actions = eng.get_legal_actions()

            if status == 'WIN' and dist > 0:
                # Must have at least one action leading to opponent LOSS with distance d - 1
                has_optimal_successor = False
                for a in legal_actions:
                    child = eng.clone()
                    child.apply_action(a, update_clock=False)
                    child_key = child._get_state_hashable_key()
                    succ_status, succ_dist = solver.get_state_outcome(child_key)
                    if succ_status == 'LOSS' and succ_dist == dist - 1:
                        has_optimal_successor = True
                        break
                self.assertTrue(has_optimal_successor, f"WIN state {key} lacks optimal predecessor with d-1")

            elif status == 'LOSS' and dist > 0:
                # In a forced loss, all legal actions must lead to opponent WIN
                for a in legal_actions:
                    child = eng.clone()
                    child.apply_action(a, update_clock=False)
                    child_key = child._get_state_hashable_key()
                    succ_status, succ_dist = solver.get_state_outcome(child_key)
                    self.assertEqual(succ_status, 'WIN', f"LOSS state {key} has non-WIN successor under {a}")

            elif status == 'DRAW':
                # In a draw state, no action can lead to an opponent LOSS, and at least one preserves DRAW
                succ_statuses = []
                for a in legal_actions:
                    child = eng.clone()
                    child.apply_action(a, update_clock=False)
                    child_key = child._get_state_hashable_key()
                    succ_st, _ = solver.get_state_outcome(child_key)
                    succ_statuses.append(succ_st)
                self.assertNotIn('LOSS', succ_statuses, f"DRAW state {key} allows winning move (opp loss)")
                if legal_actions:
                    self.assertIn('DRAW', succ_statuses, f"DRAW state {key} has no draw-preserving move")

    def test_08_agreement_with_minimax_at_terminal_depth(self):
        """8. Minimax with sufficient depth to reach terminal states must agree with Exact Solver."""
        # On 3x3 pure pawn race, the initial state has a forced win for Player 0 in 3 plies.
        # Minimax with depth >= 4 explores to terminal leaves and evaluates terminal +1000.
        engine = WallzEngine(board_size=3, rules={'A': False, 'B': False, 'C': False})
        engine.wall_inventory = {0: 0, 1: 0}

        exact_action = self.solver_race.get_optimal_action(engine)
        self.assertEqual(exact_action, ('MOVE', (1, 1)))

        minimax = MinimaxAgent(player_id=0, depth=4, time_limit=2.0)
        minimax_action = minimax.select_action(engine)

        self.assertEqual(minimax_action, exact_action,
                         f"Minimax action {minimax_action} should match exact solver {exact_action}")

    def test_09_out_of_domain_and_agent_fallback(self):
        """9. Querying states outside the solved domain must cleanly return None or fallback."""
        unsolved_engine = WallzEngine(board_size=9)  # 9x9 board is outside 3x3 solver domain

        self.assertFalse(self.solver_race.is_solved(unsolved_engine))
        self.assertIsNone(self.solver_race.get_value(unsolved_engine))
        self.assertIsNone(self.solver_race.get_optimal_action(unsolved_engine))
        status, dist = self.solver_race.get_state_outcome(unsolved_engine)
        self.assertIsNone(status)
        self.assertIsNone(dist)

        # ExactSolverAgent fallback
        agent = ExactSolverAgent(player_id=0, solver=self.solver_race)
        fallback_action = agent.select_action(unsolved_engine)
        legal_actions = unsolved_engine.get_legal_actions(0)
        self.assertIn(fallback_action, legal_actions)


if __name__ == "__main__":
    unittest.main()
