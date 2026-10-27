import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_match import run_match

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Headless Match Runner for Modified Wallz")
    parser.add_argument("--p0", type=str, default="minimax", choices=["minimax", "greedy"], help="Player 0 agent")
    parser.add_argument("--p1", type=str, default="greedy", choices=["minimax", "greedy"], help="Player 1 agent")
    parser.add_argument("--board-size", type=int, default=9, help="Board size N (default: 9)")
    parser.add_argument("--depth", type=int, default=2, help="Minimax search depth (default: 2)")
    parser.add_argument("--time-limit", type=float, default=1.0, help="Per-turn time limit in seconds (default: 1.0)")
    parser.add_argument("--max-steps", type=int, default=200, help="Maximum steps before truncation (default: 200)")

    args = parser.parse_args()
    run_match(
        p0_type=args.p0,
        p1_type=args.p1,
        board_size=args.board_size,
        depth=args.depth,
        time_limit=args.time_limit,
        max_steps=args.max_steps
    )

