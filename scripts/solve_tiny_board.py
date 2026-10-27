import sys
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wallz_engine import WallzEngine
from agents.exact_solver import ExactSolver


def main():
    parser = argparse.ArgumentParser(
        description="Exact Retrograde Solver for Tractable Miniature Modified Wallz Games"
    )
    parser.add_argument(
        "--board-size", "-N", type=int, default=3, choices=[3, 5],
        help="Board size N (3 or 5, default: 3)"
    )
    parser.add_argument(
        "--max-walls", "-m", type=int, default=1,
        help="Max walls per player (default: 1, set 0 for pure pawn race)"
    )
    parser.add_argument(
        "--no-hop", action="store_true",
        help="Disable Modification B (One-time Hop)"
    )
    parser.add_argument(
        "--no-replace", action="store_true",
        help="Disable Modification A (Latest-wall repositioning)"
    )
    parser.add_argument(
        "--enable-block", action="store_true",
        help="Enable Modification C (Temporary cell blocking)"
    )
    parser.add_argument(
        "--block-duration", "-K", type=int, default=2,
        help="Cell block duration K moves if blocking enabled (default: 2)"
    )
    parser.add_argument(
        "--show-root-actions", action="store_true",
        help="Display game-theoretic evaluation for every legal root action"
    )

    args = parser.parse_args()

    rules = {
        'A': not args.no_replace,
        'B': not args.no_hop,
        'C': args.enable_block,
        'K': args.block_duration
    }

    print("=" * 70)
    print("MODIFIED WALLZ - EXACT RETROGRADE SOLVER (GROUND-TRUTH BENCHMARK)")
    print("=" * 70)
    print(f"Board Dimension:  {args.board_size}x{args.board_size}")
    print(f"Max Walls/Player: {args.max_walls}")
    print(f"Rules Config:     Mod A (Latest Wall Replace): {rules['A']}")
    print(f"                  Mod B (One-Time Hop):        {rules['B']}")
    print(f"                  Mod C (Temporary Block):     {rules['C']} (K={rules['K']})")
    print("-" * 70)

    # Sanity check on tractability
    if args.board_size == 5 and args.max_walls > 0:
        print("[WARNING] Solving N=5 with walls is computationally intensive and may exceed RAM limits.")
        print("          Recommended configuration for N=5 is --max-walls 0 (pawn race).")

    print("[1/3] Initializing Root Engine & Validating Rules...")
    engine = WallzEngine(board_size=args.board_size, rules=rules)
    engine.max_walls = args.max_walls
    engine.wall_inventory = {0: args.max_walls, 1: args.max_walls}

    root_key = engine._get_state_hashable_key()
    print(f"Root Player:      Player {engine.current_player}")
    print(f"Player Positions: P0={engine.pawns[0]}, P1={engine.pawns[1]}")
    print(f"Goal Lines:       P0={engine.goals[0]}, P1={engine.goals[1]}")

    print("\n[2/3] Executing Forward BFS & Retrograde Backward Propagation...")
    solver = ExactSolver(board_size=args.board_size, rules=rules, max_walls=args.max_walls)
    solver.solve(initial_engine=engine)

    stats = solver.get_stats()
    print("\n" + "=" * 70)
    print("COMPUTED GAME-THEORETIC OUTCOMES & COVERAGE")
    print("=" * 70)
    print(f"Reachable State Graph Size:  {stats['state_count']:,} states")
    print(f"Forward Graph Build Time:    {stats['build_time_sec']:.4f} s")
    print(f"Retrograde Solve Time:       {stats['solve_time_sec']:.4f} s")
    print(f"Total Computation Time:      {stats['total_time_sec']:.4f} s")
    print("-" * 70)
    print(f"Distribution of States (from Active Player Perspective):")
    pct_win = (stats['win_count'] / stats['state_count'] * 100) if stats['state_count'] else 0
    pct_loss = (stats['loss_count'] / stats['state_count'] * 100) if stats['state_count'] else 0
    pct_draw = (stats['draw_count'] / stats['state_count'] * 100) if stats['state_count'] else 0
    print(f"  * WIN  States: {stats['win_count']:,} ({pct_win:.2f}%)")
    print(f"  * LOSS States: {stats['loss_count']:,} ({pct_loss:.2f}%)")
    print(f"  * DRAW States: {stats['draw_count']:,} ({pct_draw:.2f}%)")
    print("=" * 70)

    # Root state evaluation
    sol = solver.solutions.get(root_key)
    if sol:
        status = sol['status']
        dist = sol['distance']
        best_act = sol['best_action']
        val = sol['value']
        print("\nROOT STATE GAME-THEORETIC EQUILIBRIUM:")
        print(f"  * Outcome for Player {engine.current_player}: {status}")
        print(f"  * Game-Theoretic Value:     {val:+.1f} (+1: Win, 0: Draw, -1: Loss)")
        print(f"  * Plies to Terminal Outcome: {dist if dist != float('inf') else 'Infinity (Cyclic Cycle)'}")
        print(f"  * Provably Optimal Move:    {best_act}")

        if args.show_root_actions:
            print("\nROOT LEGAL ACTIONS BREAKDOWN:")
            legal_actions = engine.get_legal_actions()
            for act in legal_actions:
                succ = engine.clone()
                succ.apply_action(act, update_clock=False)
                succ_key = succ._get_state_hashable_key()
                succ_sol = solver.solutions.get(succ_key)
                if succ_sol:
                    # succ_sol is from opponent's perspective
                    opp_status = succ_sol['status']
                    opp_dist = succ_sol['distance']
                    # Root player perspective
                    my_outcome = 'WIN' if opp_status == 'LOSS' else ('LOSS' if opp_status == 'WIN' else 'DRAW')
                    marker = " <-- OPTIMAL" if act == best_act else ""
                    act_str = f"{str(act):28s}"
                    print(f"  Action: {act_str} -> Yields {my_outcome:4s} (opp {opp_status:4s}, d={opp_dist}){marker}")
    else:
        print("\n[ERROR] Root state not found in solved solutions table.")


if __name__ == "__main__":
    main()
