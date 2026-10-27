import argparse
import time
from wallz_engine import WallzEngine
from agents.greedy import GreedyAgent
from agents.minimax import MinimaxAgent

def create_agent(agent_type, player_id, depth=2, time_limit=1.0):
    agent_type = agent_type.lower()
    if agent_type == 'minimax':
        return MinimaxAgent(player_id=player_id, depth=depth, time_limit=time_limit)
    elif agent_type == 'greedy':
        return GreedyAgent(player_id=player_id)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

def run_match(p0_type='minimax', p1_type='greedy', board_size=9, depth=2, time_limit=1.0, max_steps=200):
    engine = WallzEngine(board_size=board_size, time_limit_sec=600, rules={'A': True, 'B': True, 'C': True, 'K': 3})
    
    agent0 = create_agent(p0_type, player_id=0, depth=depth, time_limit=time_limit)
    agent1 = create_agent(p1_type, player_id=1, depth=depth, time_limit=time_limit)
    agents = {0: agent0, 1: agent1}
    agent_names = {0: f"{p0_type.capitalize()} (Player 0)", 1: f"{p1_type.capitalize()} (Player 1)"}

    print(f"--- Starting Match: {agent_names[0]} vs {agent_names[1]} ---")
    print(f"Configuration: Board {board_size}x{board_size} | Minimax Depth: {depth} | Time Limit: {time_limit}s/turn\n")

    # Match statistics tracking
    metrics = {
        0: {'latencies': [], 'nodes': [], 'cutoffs': [], 'depths': []},
        1: {'latencies': [], 'nodes': [], 'cutoffs': [], 'depths': []}
    }

    match_start = time.time()
    while engine.winner is None and not engine.is_draw and engine.turn_count < max_steps:
        current_p = engine.current_player
        agent = agents[current_p]

        t0 = time.time()
        action = agent.select_action(engine)
        elapsed = time.time() - t0

        if action is None:
            print(f"Player {current_p} has no legal actions available!")
            break

        metrics[current_p]['latencies'].append(elapsed)

        # Collect Minimax-specific metrics if applicable
        if isinstance(agent, MinimaxAgent):
            stats = agent.get_stats()
            metrics[current_p]['nodes'].append(stats['nodes_expanded'])
            metrics[current_p]['cutoffs'].append(stats['cutoffs'])
            metrics[current_p]['depths'].append(stats['last_search_depth'])
            extra_info = f" | Depth: {stats['last_search_depth']} | Nodes: {stats['nodes_expanded']:3d} | Cutoffs: {stats['cutoffs']:2d} | Time: {elapsed:.3f}s"
        else:
            extra_info = f" | Time: {elapsed:.3f}s"

        state, winner, is_draw = engine.apply_action(action)
        print(f"Turn {engine.turn_count:03d} | Player {current_p} executed {action[0]:7s} -> {str(action[1]):14s}{extra_info}")

    total_match_time = time.time() - match_start

    print("\n" + "=" * 50)
    print("MATCH SUMMARY & PERFORMANCE METRICS")
    print("=" * 50)
    if engine.winner is not None:
        print(f"Outcome         : WINNER -> Player {engine.winner} ({agent_names[engine.winner]})")
    elif engine.is_draw:
        print(f"Outcome         : DRAW -> 5-Fold State Repetition reached.")
    else:
        print(f"Outcome         : TRUNCATED -> Reached T_max = {max_steps} steps without a winner.")

    print(f"Total Plies     : {engine.turn_count}")
    print(f"Match Duration  : {total_match_time:.2f}s")

    for p in (0, 1):
        p_name = agent_names[p]
        latencies = metrics[p]['latencies']
        avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
        max_lat = max(latencies) if latencies else 0.0
        total_lat = sum(latencies)

        print(f"\n--- {p_name} Stats ---")
        print(f"Total Turns      : {len(latencies)}")
        print(f"Total Turn Time  : {total_lat:.2f}s")
        print(f"Mean Latency     : {avg_lat * 1000:.1f}ms / turn (Max: {max_lat * 1000:.1f}ms)")

        if metrics[p]['nodes']:
            nodes = metrics[p]['nodes']
            cutoffs = metrics[p]['cutoffs']
            depths = metrics[p]['depths']
            avg_nodes = sum(nodes) / len(nodes)
            avg_cutoffs = sum(cutoffs) / len(cutoffs)
            avg_depth = sum(depths) / len(depths)
            print(f"Mean Search Depth: {avg_depth:.1f}")
            print(f"Mean Nodes / Turn: {avg_nodes:.1f} (Total: {sum(nodes)})")
            print(f"Mean Cutoffs/Turn: {avg_cutoffs:.1f} (Total: {sum(cutoffs)})")

    return engine.winner, engine.turn_count, metrics

if __name__ == "__main__":
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
