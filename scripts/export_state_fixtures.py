"""Exports canonical GameState and Explanation JSON fixtures for frontend tests (P11/P12).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from wallz_engine import WallzConfig, WallzEngine
from web.backend.serialize import state_json
from web.backend.explain import explain_and_choose
from agents.greedy import GreedyAgent
from agents.minimax import MinimaxAgent


REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "web" / "frontend" / "src" / "__tests__" / "fixtures"


def export_fixtures(include_explanations: bool = True):
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Initial 5x5 state
    engine5 = WallzEngine(config=WallzConfig(board_size=5))
    with open(FIXTURES_DIR / "state_initial_5x5.json", "w", encoding="utf-8") as f:
        json.dump(state_json(engine5), f, indent=2)

    # 2. State with walls, hop, block, and active timers
    engine_mid = WallzEngine(config=WallzConfig(board_size=5, A=True, B=True, C=True, K=3, M=2))
    engine_mid.apply_action(("WALL", (1, 1, "H")), update_clock=False)
    engine_mid.apply_action(("BLOCK", (2, 2)), update_clock=False)
    engine_mid.apply_action(("WALL", (3, 2, "V")), update_clock=False)
    with open(FIXTURES_DIR / "state_midgame.json", "w", encoding="utf-8") as f:
        json.dump(state_json(engine_mid), f, indent=2)

    # 3. Terminal state
    engine_term = WallzEngine(config=WallzConfig(board_size=3, K=1, M=1))
    engine_term.apply_action(("MOVE", (1, 1)), update_clock=False)
    engine_term.apply_action(("MOVE", (2, 0)), update_clock=False)
    engine_term.apply_action(("MOVE", (2, 1)), update_clock=False) # P0 reaches goal row 2
    with open(FIXTURES_DIR / "state_terminal.json", "w", encoding="utf-8") as f:
        json.dump(state_json(engine_term), f, indent=2)

    if include_explanations:
        # Explanations on initial 5x5
        g_agent = GreedyAgent(0)
        _, g_exp = explain_and_choose(g_agent, "greedy", engine5)
        with open(FIXTURES_DIR / "explanation_greedy.json", "w", encoding="utf-8") as f:
            json.dump(g_exp, f, indent=2)

        m_agent = MinimaxAgent(0, depth=1)
        _, m_exp = explain_and_choose(m_agent, "minimax", engine5)
        with open(FIXTURES_DIR / "explanation_minimax.json", "w", encoding="utf-8") as f:
            json.dump(m_exp, f, indent=2)

    print(f"Exported fixtures to {FIXTURES_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--explanations", action="store_true", default=True)
    args = parser.parse_args()
    export_fixtures(include_explanations=args.explanations)
