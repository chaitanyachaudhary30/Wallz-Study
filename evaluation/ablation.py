"""
A/B/C factorial experiments: match ablations, exact-solver sweeps and agent-vs-exact
optimal-action agreement.

Design of the match ablation (paired, balanced):
  * the 8 cells A,B,C in {0,1} share board size N, block duration K and wall budget M;
  * each opening seed fixes a random pawn-MOVE opening that is legal in every cell;
  * every (cell, seed) is played twice with the two agents swapping seats.
Deterministic agents replay identically for a fixed opening, so independent evidence
comes from distinct opening seeds, not from repeating a match.
"""
import math
import random
from itertools import product

from evaluation.match_runner import MatchConfig, make_agent, opening_actions
from wallz_engine import WallzEngine
from evaluation.metrics import factorial_effects, agreement_rate
from evaluation.retrograde import solve_config
from evaluation.state_graph import GraphCaps, NONTERMINAL


def factorial_rules(K=3, M=10):
    """The 8 A/B/C rule dicts in a fixed order (A slowest, C fastest)."""
    return [{'A': bool(a), 'B': bool(b), 'C': bool(c), 'K': K, 'M': M}
            for a, b, c in product((0, 1), repeat=3)]


def ablation_configs(agent_x, agent_y, board_size=5, K=3, M=10, opening_seeds=range(10),
                     opening_plies=2, max_plies=200, move_time_limit=None, record_replay=True):
    configs = []
    for rules in factorial_rules(K, M):
        for seed in opening_seeds:
            for p0, p1 in ((agent_x, agent_y), (agent_y, agent_x)):
                configs.append(MatchConfig(board_size=board_size, rules=rules, p0=p0, p1=p1,
                                           max_plies=max_plies, opening_plies=opening_plies,
                                           opening_seed=seed, move_time_limit=move_time_limit,
                                           record_replay=record_replay))
    return configs


def first_distinct_openings(board_size, plies, max_openings, K=3, M=10, max_seed=10_000):
    """
    Scan seeds 0, 1, 2, ... and keep the first seed of each distinct opening (the action
    sequence). Returns [(seed, actions)] with at most ``max_openings`` entries. With
    deterministic agents, a repeated opening replays the same game, so only these seeds add
    evidence. Openings are pawn MOVEs only, so the same list holds in every A/B/C cell.
    """
    seen, out = set(), []
    for seed in range(max_seed):
        engine = WallzEngine(board_size=board_size, rules={'K': K, 'M': M})
        actions = tuple(opening_actions(engine, plies, seed))
        if actions not in seen:
            seen.add(actions)
            out.append((seed, actions))
            if len(out) >= max_openings:
                break
    return out


def solver_sweep(board_size, K=1, M=0, caps=None, progress=None, keep_solutions=True):
    """Exact solve of all 8 cells; returns (rows, effects on complexity measures, solutions).
    keep_solutions=False drops each graph after its summary, so peak memory is one cell's
    graph instead of all eight (solutions is then empty)."""
    rows, solutions = [], {}
    for rules in factorial_rules(K, M):
        sol = solve_config(board_size, rules, caps)
        row = sol.summary()
        row['config_label'] = MatchConfig(board_size=board_size, rules=rules).label()
        rows.append(row)
        if keep_solutions:
            solutions[row['config_label']] = sol
        del sol
        if progress:
            progress(row)
    effects = {}
    if all(r['complete'] for r in rows):
        effects = {
            'log10_states': factorial_effects({r['config_label']: math.log10(r['states']) for r in rows}),
            'mean_branching': factorial_effects({r['config_label']: r['mean_branching'] for r in rows}),
            'draw_fraction': factorial_effects({r['config_label']: r['value_counts']['DRAW'] / r['states']
                                                for r in rows}),
        }
    return rows, effects, solutions


def sample_nonterminal(solution, max_states=300, seed=0):
    """Uniform sample (without replacement) of reachable non-terminal nodes, sorted."""
    g = solution.graph
    nodes = [u for u in range(len(g.keys)) if g.terminal[u] is NONTERMINAL]
    rng = random.Random(seed)
    if len(nodes) > max_states:
        nodes = sorted(rng.sample(nodes, max_states))
    return nodes


def agreement(solution, agent_spec, max_states=300, seed=0, nodes=None):
    """
    Fraction of sampled solved non-terminal states where ``agent_spec`` picks a
    value-optimal action (any action preserving the exact value) and a rank-optimal one.
    Sampling is uniform over the reachable non-terminal states of a complete graph.
    Pass ``nodes`` (from sample_nonterminal) to score several agents on the same states.
    """
    g = solution.graph
    if not solution.exact:
        raise ValueError("Agreement needs a complete (exact) solution.")
    if nodes is None:
        nodes = sample_nonterminal(solution, max_states, seed)
    cfg = MatchConfig(board_size=g.board_size, rules=g.rules)
    agents = {p: make_agent(agent_spec, p, cfg)[0] for p in (0, 1)}
    records = []
    for u in nodes:
        engine = g.restore(g.keys[u])
        action = agents[g.to_move[u]].select_action(engine)
        records.append({
            'node': u,
            'status': solution.status(u),
            'value_optimal': action in solution.value_optimal_actions(u),
            'rank_optimal': action in solution.rank_optimal_actions(u),
        })
    return {'agent': agent_spec, 'config_label': cfg.label(), 'sampled_states': len(records),
            'population_states': sum(1 for t in g.terminal if t is NONTERMINAL),
            'rates': agreement_rate(records)}


def default_caps():
    return GraphCaps(max_states=1_000_000, max_edges=20_000_000, max_seconds=900.0)
