"""
Mutation check for tests/test_solver_final_rules.py: patch two engine rules in memory
(never on disk), run the tests, and list which tests fail. A useful test file must fail
under both mutants.  Run: python scripts/mutation_check_solver_tests.py
"""
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wallz_engine import WallzEngine  # noqa: E402

orig_canon = WallzEngine.canonicalize_key
orig_hop = WallzEngine._is_valid_hop


def canon_no_owner(raw_key, rules):
    """Mutant 1: forget wall owners in the canonical key even while a hop remains."""
    r = dict(rules)
    r['B'] = False
    return orig_canon(raw_key, r)


def hop_any_wall(self, player, cell):
    """Mutant 2: allow a hop across ANY wall (own walls too)."""
    r, c = self.pawns[player]
    if not self.rules['B'] or not self.hop_available[player]:
        return False
    if abs(cell[0] - r) + abs(cell[1] - c) != 1:
        return False
    return (self.wall_owner_on_edge(self.pawns[player], cell) is not None
            and cell != self.pawns[1 - player] and cell not in self.blocked_cells)


MUTANTS = {'canon_no_owner': ('canonicalize_key', staticmethod(canon_no_owner), staticmethod(orig_canon)),
           'hop_any_wall': ('_is_valid_hop', hop_any_wall, orig_hop)}

if __name__ == '__main__':
    for name, (attr, mutant, original) in MUTANTS.items():
        setattr(WallzEngine, attr, mutant)
        try:
            suite = unittest.defaultTestLoader.loadTestsFromName('tests.test_solver_final_rules')
            with open(os.devnull, 'w') as sink:
                res = unittest.TextTestRunner(stream=sink, verbosity=0).run(suite)
        finally:
            setattr(WallzEngine, attr, original)
        failed = sorted({t.id().split('.')[-1] for t, _ in res.failures + res.errors})
        print(f"{name}: {len(failed)} of {res.testsRun} tests fail: {failed}")
