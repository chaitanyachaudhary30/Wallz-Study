"""Gate X-1: Hypothesis RuleBasedStateMachine for Engine Invariants and Legality Fuzzing (P9.1)."""
import random
import unittest
from hypothesis import strategies as st, settings, Phase
from hypothesis.stateful import RuleBasedStateMachine, rule, initialize, invariant

from wallz_engine import WallzEngine
from src.modified_wallz.config import WallzConfig


class WallzStateMachine(RuleBasedStateMachine):
    """
    Hypothesis stateful testing machine verifying game invariants and legality checks
    under arbitrary rule configurations:
    - N in {3, 4, 5, 9}
    - K in {1, 2, 3, 5}
    - M in {0, 1, 10}
    - Rule toggles: A, B, C in {True, False}
    """

    @initialize(
        n=st.sampled_from([3, 4, 5, 9]),
        k=st.sampled_from([1, 2, 3, 5]),
        m=st.sampled_from([0, 1, 10]),
        a=st.booleans(),
        b=st.booleans(),
        c=st.booleans(),
    )
    def initialize_game(self, n: int, k: int, m: int, a: bool, b: bool, c: bool):
        self.board_size = n
        self.k = k
        self.m = m
        self.rules = {'A': a, 'B': b, 'C': c, 'K': k, 'M': m}
        self.engine = WallzEngine(board_size=n, rules=self.rules)
        # Track initial ability flags to verify monotonicity (flags never turn back on)
        self.ever_lost_hop = {0: False, 1: False}
        self.ever_lost_block = {0: False, 1: False}

    @rule()
    def step_random_legal_action(self):
        """Execute a random legal action if the engine is non-terminal."""
        if self.engine.is_terminal():
            return

        actions = self.engine.get_legal_actions()
        if not actions:
            return

        # Pick a random legal action
        action = random.choice(actions)

        # Update tracking before action
        for p in (0, 1):
            if not self.engine.hop_available[p]:
                self.ever_lost_hop[p] = True
            if not self.engine.block_available[p]:
                self.ever_lost_block[p] = True

        self.engine.apply_action(action)

    @rule()
    def fuzz_illegal_action(self):
        """
        Attempt an illegal action; assert that it raises ValueError
        and does not mutate engine state.
        """
        if self.engine.is_terminal():
            return

        legal = set(self.engine.get_legal_actions())
        n = self.board_size
        p = self.engine.current_player

        # Generate candidates that might be illegal
        candidates = [
            ('MOVE', (random.randint(-2, n + 2), random.randint(-2, n + 2))),
            ('HOP', (random.randint(-2, n + 2), random.randint(-2, n + 2))),
            ('WALL', (random.randint(-2, n + 2), random.randint(-2, n + 2), random.choice(['H', 'V']))),
            ('REPLACE', (random.randint(-2, n + 2), random.randint(-2, n + 2), random.choice(['H', 'V']))),
            ('BLOCK', (random.randint(-2, n + 2), random.randint(-2, n + 2))),
            ('INVALID_TYPE', (0, 0)),
        ]
        # Filter for candidates not in legal
        illegal_candidates = [c for c in candidates if c not in legal]
        if not illegal_candidates:
            return

        action = random.choice(illegal_candidates)
        saved_key = self.engine.position_key()

        try:
            self.engine.apply_action(action)
            raise AssertionError(f"Illegal action {action} was accepted!")
        except ValueError:
            # Expected; verify engine state unchanged
            assert self.engine.position_key() == saved_key, "Illegal action mutated engine state!"

    @invariant()
    def verify_engine_invariants(self):
        """Assert all 8 Gate X-1 invariants after every step."""
        eng = self.engine
        n = self.board_size

        # Invariant 1: Pawns are on different cells
        assert eng.pawns[0] != eng.pawns[1], f"Pawns occupy same cell: {eng.pawns[0]}"

        # Invariant 2: PathExists holds for both players (unless game already won)
        if not eng.is_terminal():
            assert eng.shortest_path(0) < float('inf'), "Path to goal blocked for Player 0"
            assert eng.shortest_path(1) < float('inf'), "Path to goal blocked for Player 1"

        # Invariant 3: No overlapping or crossing walls
        all_walls = eng.walls[0] + eng.walls[1]
        assert len(all_walls) == len(set(all_walls)), "Duplicate walls in player inventories"
        assert len(eng.board.walls) == len(all_walls), "Board wall count mismatch"

        # Invariant 4: 0 <= m_i <= M
        assert 0 <= eng.wall_inventory[0] <= self.m, f"Wall inventory P0 out of range: {eng.wall_inventory[0]}"
        assert 0 <= eng.wall_inventory[1] <= self.m, f"Wall inventory P1 out of range: {eng.wall_inventory[1]}"

        # Invariant 5: Ability flags never turn back on
        for p in (0, 1):
            if self.ever_lost_hop[p]:
                assert not eng.hop_available[p], f"Hop flag re-enabled for player {p}"
            if self.ever_lost_block[p]:
                assert not eng.block_available[p], f"Block flag re-enabled for player {p}"

        # Invariant 6: Every block timer is in 1..K
        for cell, timer in eng.blocked_cells.items():
            assert 1 <= timer <= self.k, f"Block timer {timer} at {cell} out of range 1..{self.k}"

        # Invariant 7: terminal_reason is set if and only if is_terminal()
        assert eng.is_terminal() == (eng.terminal_reason is not None), (
            f"Terminal mismatch: is_terminal={eng.is_terminal()}, reason={eng.terminal_reason}"
        )

        # Invariant 8: restore_state_key(key) reproduces the exact same legal actions
        key = eng.position_key()
        restored = WallzEngine(board_size=n, rules=self.rules)
        restored.restore_state_key(key)
        assert set(restored.get_legal_actions()) == set(eng.get_legal_actions()), (
            f"Restored actions mismatch at key {key}"
        )


TestWallzInvariantsStateful = WallzStateMachine.TestCase
TestWallzInvariantsStateful.settings = settings(
    max_examples=100,
    stateful_step_count=25,
    deadline=None,
    phases=[Phase.generate, Phase.target],
)


if __name__ == '__main__':
    unittest.main()
