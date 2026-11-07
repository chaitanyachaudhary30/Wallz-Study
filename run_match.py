"""Thin CLI and programmatic wrapper over evaluation.match_runner (P7.3)."""
import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluation.match_runner import MatchConfig, run_match as execute_match, verify_replay


def run_match(
    p0_type: str = 'minimax',
    p1_type: str = 'greedy',
    board_size: int = 9,
    depth: int = 2,
    time_limit: Optional[float] = 1.0,
    max_steps: int = 200,
    rules: Optional[Dict[str, Any]] = None,
    opening_seed: int = 0,
    opening_plies: int = 0,
    verbose: bool = True,
) -> Tuple[Optional[int], int, Dict[Any, Any]]:
    """
    Programmatic entry point for running a single match.
    Wraps evaluation.match_runner.run_match.
    Returns (winner, total_plies, metrics).
    """
    if rules is None:
        rules = {'A': True, 'B': True, 'C': True, 'K': 3, 'M': 10}

    # Format agent specs
    def format_spec(agent_type: str) -> str:
        agent_type = agent_type.strip()
        if agent_type.startswith(('minimax', 'greedy', 'random', 'dqn', 'exact')):
            if ':' in agent_type:
                return agent_type
            if agent_type == 'minimax':
                parts = [f"depth={depth}"]
                if time_limit is not None:
                    parts.append(f"time_limit={time_limit}")
                return f"minimax:{','.join(parts)}"
            return agent_type
        raise ValueError(f"Unknown agent type: {agent_type}")

    spec0 = format_spec(p0_type)
    spec1 = format_spec(p1_type)

    cfg = MatchConfig(
        board_size=board_size,
        rules=rules,
        p0=spec0,
        p1=spec1,
        max_plies=max_steps,
        opening_plies=opening_plies,
        opening_seed=opening_seed,
        move_time_limit=time_limit,
    )

    if verbose:
        print(f"--- Starting Match: {spec0} (Player 0) vs {spec1} (Player 1) ---")
        print(f"Configuration: Board {board_size}x{board_size} | Rules: {rules} | Max Plies: {max_steps}\n")

    t0 = time.time()
    res = execute_match(cfg)
    total_match_time = time.time() - t0

    if verbose:
        for step in res['replay']:
            ply = step['ply']
            player = step['player']
            action = step['action']
            act_type = action[0] if action else "NONE"
            act_coords = str(action[1]) if action else "None"
            print(f"Turn {ply:03d} | Player {player} executed {act_type:7s} -> {act_coords:14s}")

        outcome = res['outcome']
        print("\n" + "=" * 50)
        print("MATCH SUMMARY & PERFORMANCE METRICS")
        print("=" * 50)
        if outcome['category'] == 'goal':
            winner = outcome['winner']
            print(f"Outcome         : WINNER -> Player {winner} ({spec0 if winner == 0 else spec1})")
        elif outcome['category'].startswith('draw_'):
            print(f"Outcome         : DRAW -> {outcome['detail']}")
        elif outcome['category'] == 'truncated':
            print(f"Outcome         : TRUNCATED -> {outcome['detail']}")
        else:
            print(f"Outcome         : FORFEIT -> Winner Player {outcome['winner']} ({outcome['category']})")

        print(f"Total Plies     : {res['length_plies']}")
        print(f"Match Duration  : {total_match_time:.2f}s")

        for p in (0, 1):
            p_name = f"Player {p} ({spec0 if p == 0 else spec1})"
            p_str = str(p)
            timing = res['decision_seconds'].get(p_str, {})
            print(f"\n--- {p_name} Stats ---")
            mean_sec = timing.get('mean')
            total_sec = timing.get('total')
            max_sec = timing.get('max')
            if mean_sec is not None:
                print(f"Mean Latency     : {mean_sec * 1000.0:.1f}ms / turn (Max: {max_sec * 1000.0:.1f}ms)")
                print(f"Total Turn Time  : {total_sec:.2f}s")
            if res.get('search_stats') and p_str in res['search_stats']:
                s = res['search_stats'][p_str]
                print(f"Mean Search Depth: {s.get('mean_depth', 0.0):.1f}")
                print(f"Mean Nodes / Turn: {s.get('mean_nodes', 0.0):.1f}")
                print(f"Mean Cutoffs/Turn: {s.get('mean_cutoffs', 0.0):.1f}")

    winner = res['outcome']['winner']
    metrics = {
        0: res['decision_seconds'].get('0', {}),
        1: res['decision_seconds'].get('1', {}),
        'result': res,
    }
    return winner, res['length_plies'], metrics


def main():
    parser = argparse.ArgumentParser(description="Headless Match Runner for Modified Wallz (Thin Wrapper)")
    parser.add_argument("--p0", type=str, default="minimax", help="Player 0 agent spec or name")
    parser.add_argument("--p1", type=str, default="greedy", help="Player 1 agent spec or name")
    parser.add_argument("--board-size", type=int, default=9, help="Board size N (default: 9)")
    parser.add_argument("--depth", type=int, default=2, help="Minimax search depth (default: 2)")
    parser.add_argument("--time-limit", type=float, default=1.0, help="Per-turn time limit in seconds (default: 1.0)")
    parser.add_argument("--max-steps", type=int, default=200, help="Maximum steps before truncation (default: 200)")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")

    args = parser.parse_args()
    run_match(
        p0_type=args.p0,
        p1_type=args.p1,
        board_size=args.board_size,
        depth=args.depth,
        time_limit=args.time_limit,
        max_steps=args.max_steps,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
