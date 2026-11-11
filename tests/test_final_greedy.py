"""Checks for scripts/final_greedy.py: distinct-opening choice and outcome counting."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.final_greedy import BOARDS, choose_openings, count_openings, opening_of, summarize  # noqa: E402


def test_shortest_length_and_first_seed_of_each_opening():
    b = BOARDS['N3']
    n, rules = b['board_size'], b['rules']
    assert count_openings(n, rules, 1) == 3
    plies, seeds, scanned, tried = choose_openings(n, rules, 8)
    assert plies == 2 and tried == {0: 1, 1: 3, 2: 8}
    ops = [opening_of(n, rules, plies, s) for s in seeds]
    assert len(set(ops)) == len(seeds) == 8
    for s, op in zip(seeds, ops):
        assert all(opening_of(n, rules, plies, t) != op for t in range(s))  # s is the first seed


def _result(category, winner, length=10):
    dec = {'n': 2, 'total': 0.2, 'mean': 0.1, 'max': 0.1}
    return {'outcome': {'category': category, 'winner': winner}, 'length_plies': length,
            'opening': {'plies': 2}, 'decision_seconds': {'0': dec, '1': dec}}


def test_truncated_is_not_a_draw_and_forfeits_are_separate():
    items = [(0, _result('goal', 0)), (0, _result('goal', 1)), (1, _result('draw_repetition', None)),
             (1, _result('truncated', None)), (0, _result('illegal_action', 0))]
    s = summarize(items)
    assert s['games'] == 5
    assert s['greedy_goal_wins']['count'] == 1 and s['greedy_goal_losses']['count'] == 1
    assert s['draws_total']['count'] == 1 and s['draws_by_type'] == {'draw_repetition': 1}
    assert s['truncated']['count'] == 1
    assert (s['forfeits_won'], s['forfeits_lost']) == (1, 0)
    assert s['mean_length_plies_after_opening'] == 8
