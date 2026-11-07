"""Test suite for Greedy match campaign runner and statistics (P07).

Verifies:
1. build_configs builds proper configs, pairings, and colours in quick and full modes.
2. summarise correctly calculates wins, losses, draws, win rates, Wilson intervals, and plies.
3. make_greedy_table formats Markdown and LaTeX tables with all required columns.
"""
from __future__ import annotations

from scripts.run_greedy_matches import build_configs, make_greedy_table, summarise


def test_build_configs_quick():
    """Quick mode generates expected board configs, colours, and opening counts."""
    batches = build_configs(quick=True, seeds=[42])
    # 3 pairings x 2 boards (N3_solver, N5_final) = 6 batches
    assert len(batches) == 6

    for (pairing, board), cfgs in batches.items():
        assert board in ("N3_solver", "N5_final")
        assert pairing in ("greedy_vs_random", "greedy_vs_minimax2", "greedy_vs_greedy_self")
        # 1 seed x 2 openings x 2 colours = 4 configs per batch
        assert len(cfgs) == 4

        p0_is_greedy = [c.p0.startswith("greedy") for c in cfgs]
        p1_is_greedy = [c.p1.startswith("greedy") for c in cfgs]
        # Colours alternate
        assert sum(p0_is_greedy) == 4 or sum(p0_is_greedy) == 2
        assert sum(p1_is_greedy) == 4 or sum(p1_is_greedy) == 2


def test_summarise_synthetic():
    """summarise produces accurate win/draw/loss rates, Wilson CIs, and metrics."""
    synthetic_results = [
        # Win for P0
        {
            "config": {"p0": "greedy", "p1": "random", "opening_seed": 4200},
            "outcome": {"category": "goal", "winner": 0, "loser": 1},
            "total_plies": 10,
            "metrics": {"move_times": {"0": [0.001, 0.002]}},
        },
        # Loss for P0
        {
            "config": {"p0": "greedy", "p1": "random", "opening_seed": 4201},
            "outcome": {"category": "goal", "winner": 1, "loser": 0},
            "total_plies": 20,
            "metrics": {"move_times": {"0": [0.001]}},
        },
        # Draw for P1 (greedy is P1)
        {
            "config": {"p0": "random", "p1": "greedy", "opening_seed": 4202},
            "outcome": {"category": "draw_repetition", "winner": None, "loser": None, "detail": "repetition"},
            "total_plies": 30,
            "metrics": {"move_times": {"1": [0.002]}},
        },
        # Win for P1 (greedy is P1)
        {
            "config": {"p0": "random", "p1": "greedy", "opening_seed": 4203},
            "outcome": {"category": "goal", "winner": 1, "loser": 0},
            "total_plies": 40,
            "metrics": {"move_times": {"1": [0.003]}},
        },
    ]

    summary = summarise(synthetic_results, "greedy_vs_random", "N5_final")
    assert summary["games"] == 4
    assert summary["wins"] == 2
    assert summary["losses"] == 1
    assert summary["draws"] == 1
    assert summary["win_rate"] == 0.5
    assert summary["draw_rate"] == 0.25
    assert summary["mean_plies"] == 25.0
    # Wilson interval: 0 < ci_low < 0.5 < ci_high < 1
    assert 0.0 < summary["win_rate_ci"][0] < 0.5 < summary["win_rate_ci"][1] < 1.0


def test_make_greedy_table_formats():
    """make_greedy_table renders valid Markdown and LaTeX strings containing all required columns."""
    sample_summary = {
        "campaign_results": [
            {
                "pairing": "greedy_vs_random",
                "board": "N5_final",
                "games": 100,
                "wins": 95,
                "draws": 2,
                "losses": 3,
                "win_rate": 0.95,
                "win_rate_ci": [0.88, 0.98],
                "draw_rate": 0.02,
                "draw_rate_ci": [0.005, 0.07],
                "mean_plies": 18.5,
                "sd_plies": 4.2,
                "mean_greedy_decision_sec": 0.0021,
            }
        ]
    }

    md, tex = make_greedy_table(sample_summary)
    assert "| Pairing | Board | Games |" in md
    assert "Greedy Vs Random" in md
    assert "95.0%" in md
    assert r"\begin{table}" in tex
    assert r"\caption{Greedy Baseline Match Campaign Performance Across Configs}" in tex
    assert r"\end{table}" in tex
