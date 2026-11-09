#!/usr/bin/env python3
"""Rule fixture generator for WallzEngine (P03 / P04).

Generates deterministic, frozen state fixtures for edge cases:
- r7: Block expiring in 1 ply that currently prevents a wall placement.
- hop-path: HOP target geometrically valid but illegal due to goal path severance.
- no-legal: Terminal state reached via an action leaving opponent with no legal moves (P04).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.helpers import key_to_json, random_reachable
from wallz_engine import WallzEngine


def generate_r7(seed_start: int, max_tries: int) -> Dict[str, Any]:
    """Search for a state where an expiring cell block (timer 1) prevents a wall placement."""
    for i in range(max_tries):
        current_seed = seed_start + i
        for n in (3, 4, 5):
            for k in (1, 2, 3):
                plies = (current_seed % 25) + 1
                rules = {"A": True, "B": True, "C": True, "K": k, "M": 10}
                eng = random_reachable(n=n, rules=rules, plies=plies, seed=current_seed)
                if eng.is_terminal():
                    continue

                expiring = [cell for cell, t in eng.blocked_cells.items() if t == 1]
                if not expiring:
                    continue

                p = eng.current_player
                for r in range(n - 1):
                    for c in range(n - 1):
                        for o in ("H", "V"):
                            w = (r, c, o)
                            if not eng.board.can_place_wall(*w):
                                continue
                            if not eng._is_valid_wall_placement(p, *w):
                                for exp_c in expiring:
                                    clone = eng.clone()
                                    del clone.blocked_cells[exp_c]
                                    if clone._is_valid_wall_placement(p, *w):
                                        return {
                                            "board_size": n,
                                            "rules": eng.rules,
                                            "raw_key": key_to_json(eng._get_state_hashable_key()),
                                            "player": p,
                                            "wall": [r, c, o],
                                            "expiring_cell": list(exp_c),
                                            "seed": current_seed,
                                        }

    print(f"Error: r7 search exhausted after {max_tries} tries starting from seed {seed_start}", file=sys.stderr)
    sys.exit(2)


def generate_hop_path(seed_start: int, max_tries: int) -> Dict[str, Any]:
    """Search for a state where a hop target satisfies geometry but violates path-to-goal preservation."""
    for i in range(max_tries):
        current_seed = seed_start + i
        for n in (3, 4, 5):
            plies = (current_seed % 25) + 1
            rules = {"A": True, "B": True, "C": True, "K": 3, "M": 10}
            eng = random_reachable(n=n, rules=rules, plies=plies, seed=current_seed)
            if eng.is_terminal():
                continue

            p = eng.current_player
            if not eng.hop_available[p]:
                continue

            r, c = eng.pawns[p]
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < n and 0 <= nc < n):
                    continue
                if (nr, nc) in eng.blocked_cells or (nr, nc) == eng.pawns[1 - p]:
                    continue
                # Hop geometry: the edge must be blocked by an opponent's wall
                if eng.wall_owner_on_edge((r, c), (nr, nc)) != 1 - p:
                    continue

                # Geometrically valid hop candidate. Now check legality:
                if not eng._is_valid_hop(p, (nr, nc)):
                    return {
                        "board_size": n,
                        "rules": eng.rules,
                        "raw_key": key_to_json(eng._get_state_hashable_key()),
                        "player": p,
                        "hop": [nr, nc],
                        "seed": current_seed,
                    }

    print(f"Error: hop-path search exhausted after {max_tries} tries starting from seed {seed_start}", file=sys.stderr)
    sys.exit(2)


def generate_no_legal(max_seconds: int = 120, search_from_initial: bool = True) -> Dict[str, Any]:
    """Search StateGraph for first edge leading to DRAW_NO_LEGAL_ACTIONS and full game from initial."""
    from evaluation.state_graph import StateGraph, GraphCaps, DRAW_NO_LEGAL_ACTIONS
    from evaluation.match_runner import run_match, MatchConfig

    configs = [
        (3, {"A": False, "B": False, "C": True, "K": 2, "M": 0}),
        (3, {"A": True, "B": True, "C": True, "K": 2, "M": 0}),
        (3, {"A": False, "B": False, "C": True, "K": 3, "M": 0}),
        (3, {"A": True, "B": True, "C": True, "K": 1, "M": 1}),
    ]

    for n, rules in configs:
        sg = StateGraph(n, rules, canonical=False)
        sg.build(GraphCaps(max_seconds=max_seconds))
        if sg.terminal.count(DRAW_NO_LEGAL_ACTIONS) > 0:
            for parent_idx, edges in enumerate(sg.succ):
                for act, child_idx in edges:
                    if sg.terminal[child_idx] == DRAW_NO_LEGAL_ACTIONS:
                        parent_key = sg.keys[parent_idx]
                        child_key = sg.keys[child_idx]

                        full_game_info = None
                        if search_from_initial:
                            for s in range(10000):
                                cfg = MatchConfig(board_size=n, rules=rules, p0="random", p1="random", opening_seed=s)
                                res = run_match(cfg)
                                if res["outcome"]["category"] == "draw_no_legal_actions":
                                    full_game_info = {"p0": "random", "p1": "random", "seed": s}
                                    break

                        act_val = list(act[1]) if isinstance(act[1], (tuple, list)) else act[1]
                        return {
                            "board_size": n,
                            "rules": rules,
                            "parent_key": key_to_json(parent_key),
                            "action": [act[0], act_val],
                            "child_key": key_to_json(child_key),
                            "seed": None,
                            "source": "StateGraph first-hit",
                            "full_game": full_game_info,
                        }

    print("Error: no-legal search exhausted all configs without finding a DRAW_NO_LEGAL_ACTIONS state", file=sys.stderr)
    sys.exit(2)


def main():
    parser = argparse.ArgumentParser(description="Generate frozen rule test fixtures for Modified Wallz.")
    parser.add_argument("subcommand", choices=["r7", "hop-path", "no-legal"], help="Fixture type to generate")
    parser.add_argument("--seed", type=int, default=0, help="Initial random seed (default 0)")
    parser.add_argument("--tries", type=int, default=20000, help="Maximum search tries (default 20000)")
    parser.add_argument("--out", type=str, default=None, help="Output JSON path")

    args = parser.parse_args()

    default_paths = {
        "r7": "tests/fixtures/r7_block_expiring.json",
        "hop-path": "tests/fixtures/hop_path_violation.json",
        "no-legal": "tests/fixtures/no_legal_draw.json",
    }

    out_path = args.out or default_paths[args.subcommand]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    if args.subcommand == "r7":
        fixture_data = generate_r7(args.seed, args.tries)
    elif args.subcommand == "hop-path":
        fixture_data = generate_hop_path(args.seed, args.tries)
    elif args.subcommand == "no-legal":
        fixture_data = generate_no_legal()
    else:
        raise ValueError(f"Unknown subcommand {args.subcommand}")

    with open(out_path, "w") as f:
        json.dump(fixture_data, f, indent=2)

    print(f"Wrote fixture for '{args.subcommand}' to {out_path} (seed {fixture_data['seed']})")


if __name__ == "__main__":
    main()
