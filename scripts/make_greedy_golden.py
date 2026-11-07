#!/usr/bin/env python3
"""Generates golden decision fixture for GreedyAgent before refactoring (P06).

Ensures that refactoring GreedyAgent into `score_actions` and adding side-to-move assertions
leaves agent decisions 100% bit-for-bit identical across unseeded and seeded modes.
"""
from __future__ import annotations

import json
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.greedy import GreedyAgent
from tests.helpers import key_to_json, random_reachable
from wallz_engine import WallzEngine


def make_greedy_golden(out_path: str = "tests/fixtures/greedy_golden.json"):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    records = []

    configs = [
        {"K": 3, "M": 10},  # default rules
        {"K": 1, "M": 1},
    ]

    for n in (3, 5, 9):
        max_seeds = 100 if n == 9 else 200
        for rules_part in configs:
            full_rules = {"A": True, "B": True, "C": True, **rules_part}
            for seed in range(max_seeds):
                rng = random.Random(seed)
                plies = rng.randint(0, 20)
                engine = random_reachable(n=n, rules=full_rules, plies=plies, seed=seed)
                if engine.is_terminal():
                    continue

                p = engine.current_player
                # Unseeded decision
                agent_unseeded = GreedyAgent(p)
                choice_unseeded = agent_unseeded.select_action(engine)

                # Seeded(7) decision
                agent_seeded = GreedyAgent(p, seed=7)
                choice_seeded = agent_seeded.select_action(engine)

                raw_key = key_to_json(engine._get_state_hashable_key())

                records.append({
                    "n": n,
                    "rules": full_rules,
                    "raw_key": raw_key,
                    "player": p,
                    "choice_unseeded": [choice_unseeded[0], list(choice_unseeded[1]) if isinstance(choice_unseeded[1], (tuple, list)) else choice_unseeded[1]] if choice_unseeded else None,
                    "choice_seed7": [choice_seeded[0], list(choice_seeded[1]) if isinstance(choice_seeded[1], (tuple, list)) else choice_seeded[1]] if choice_seeded else None,
                })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, separators=(",", ":"))

    size_kb = os.path.getsize(out_path) / 1024
    print(f"Generated {len(records)} golden records in {out_path} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    make_greedy_golden()
