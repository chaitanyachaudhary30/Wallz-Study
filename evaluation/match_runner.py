"""
Reproducible match runner with persistent JSON logging and replay verification.

Outcome categories (``outcome['category']``) and how metrics use them:
    goal                   rule win  -> win / loss
    draw_repetition        rule draw -> draw (5-fold repetition, engine rule)
    draw_no_legal_actions  rule draw -> draw (side to move has no legal action)
    truncated              external ply cap reached -> reported separately, never a draw
    timeout                decision exceeded move_time_limit -> forfeit by the acting side
    illegal_action         agent returned an action not in get_legal_actions() -> forfeit
    no_action              agent returned None although legal actions existed -> forfeit
    agent_error            agent raised an exception -> forfeit
Forfeits name a winner but are counted separately from goal wins (see evaluation/metrics.py).

Matches are deterministic given (config, agents): transitions use
``apply_action(update_clock=False)`` and time budgets are enforced by the runner from
measured decision time, never by the engine's wall-clock chess clock.
"""
import json
import os
import random
import subprocess
import time
import traceback
from dataclasses import dataclass, field, asdict

from wallz_engine import WallzEngine
from evaluation.state_graph import state_id, full_rules

SCHEMA_VERSION = 1
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class AgentUnavailable(Exception):
    """Requested agent has no usable implementation in this checkout."""


def source_commit():
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'],
                                    cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip())
        return {'commit': commit, 'dirty': dirty}
    except (OSError, subprocess.CalledProcessError):
        return {'commit': None, 'dirty': None}


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

class RandomAgent:
    """Uniform over legal actions; seeded per match for reproducibility."""

    def __init__(self, player_id, seed=0):
        self.player_id = player_id
        self.rng = random.Random(seed)

    def select_action(self, engine):
        legal = engine.get_legal_actions(self.player_id)
        return self.rng.choice(legal) if legal else None


class ExactAgent:
    """Plays the first rank-optimal action from a retrograde Solution (tiny boards only)."""

    def __init__(self, player_id, solution):
        self.player_id = player_id
        self.solution = solution

    def select_action(self, engine):
        node = self.solution.graph.node_of(engine)
        if node is None:
            raise LookupError("State is outside the solved graph.")
        actions = self.solution.rank_optimal_actions(node)
        if not actions:
            raise LookupError("No solved optimal action for this state.")
        return actions[0]


def parse_agent_spec(spec):
    """'minimax:depth=2,time_limit=1.0' -> ('minimax', {'depth': 2, 'time_limit': 1.0})"""
    name, _, rest = spec.partition(':')
    params = {}
    for item in filter(None, rest.split(',')):
        k, _, v = item.partition('=')
        for cast in (int, float):
            try:
                v = cast(v)
                break
            except ValueError:
                pass
        params[k] = v
    return name.lower(), params


def make_agent(spec, player_id, config, solutions=None, match_seed=0):
    """
    Build an agent from a spec string. Returns (agent, identity dict).

    solutions: optional cache {config_label: Solution} used by the 'exact' agent.
    """
    name, params = parse_agent_spec(spec)
    identity = {'spec': spec, 'name': name, 'params': params}
    if name == 'greedy':
        from agents.greedy import GreedyAgent
        seed = params.get('seed')
        return GreedyAgent(player_id, seed=seed), identity
    if name == 'minimax':
        from agents.minimax import MinimaxAgent
        depth = params.get('depth', 2)
        # Without a time limit minimax is deterministic (pure depth-limited search).
        time_limit = params.get('time_limit')
        identity['params'] = {'depth': depth, 'time_limit': time_limit}
        return MinimaxAgent(player_id=player_id, depth=depth, time_limit=time_limit), identity
    if name == 'random':
        seed = params.get('seed', 0) * 1_000_003 + match_seed * 2 + player_id
        identity['params'] = {'seed': params.get('seed', 0), 'derived_seed': seed}
        return RandomAgent(player_id, seed), identity
    if name == 'exact':
        from evaluation.retrograde import solve_config
        from evaluation.state_graph import GraphCaps
        key = config.label()
        if solutions is None:
            solutions = {}
        if key not in solutions:
            caps = GraphCaps(max_states=params.get('max_states', 1_000_000),
                             max_seconds=params.get('max_seconds', 600.0))
            solutions[key] = solve_config(config.board_size, config.rules, caps)
        sol = solutions[key]
        if not sol.exact:
            raise AgentUnavailable(f"Exact solve of {key} hit cap '{sol.graph.stop_reason}'.")
        identity['params'] = {'states': len(sol.graph.keys), 'model': 'position game'}
        return ExactAgent(player_id, sol), identity
    if name == 'dqn':
        path = params.get('path')
        if not path:
            raise AgentUnavailable("DQN agent spec requires 'path' parameter, e.g. 'dqn:path=checkpoints/run1/model.pt'")
        try:
            from agents.dqn.agent import load_checkpoint
        except ImportError as e:
            raise AgentUnavailable(f"DQN dependencies not available: {e}")
        try:
            agent = load_checkpoint(
                checkpoint_path=path,
                player_id=player_id,
                expected_board_size=config.board_size,
                expected_rules=config.rules,
            )
        except Exception as e:
            raise AgentUnavailable(f"Failed to load DQN checkpoint from {path}: {e}")
        identity['codec_version'] = agent.codec_version
        identity['obs_version'] = agent.obs_version
        return agent, identity
    raise ValueError(f"Unknown agent spec {spec!r}")


# ---------------------------------------------------------------------------
# Match configuration and execution
# ---------------------------------------------------------------------------

@dataclass
class MatchConfig:
    board_size: int = 5
    rules: dict = field(default_factory=dict)
    p0: str = 'greedy'
    p1: str = 'greedy'
    max_plies: int = 200
    opening_plies: int = 0
    opening_seed: int = 0
    move_time_limit: float = None
    record_replay: bool = True

    def __post_init__(self):
        self.rules = full_rules(self.rules)

    def label(self):
        r = self.rules
        return f"N{self.board_size}_A{int(r['A'])}B{int(r['B'])}C{int(r['C'])}_K{r['K']}_M{r['M']}"

    def match_id(self):
        return (f"{self.label()}__{self.p0}__vs__{self.p1}__open{self.opening_plies}"
                f"s{self.opening_seed}__cap{self.max_plies}")


def opening_actions(engine, plies, seed):
    """
    Random pawn MOVEs only, so the same seed gives the same opening under every A/B/C
    configuration (MOVE legality does not depend on the mechanics before any wall,
    hop or block is used). Sorted before sampling for platform independence.
    """
    rng = random.Random(seed)
    played = []
    for _ in range(plies):
        if engine.is_terminal():
            break
        moves = sorted(a for a in engine.get_legal_actions() if a[0] == 'MOVE')
        if not moves:
            break
        action = rng.choice(moves)
        engine.apply_action(action, update_clock=False)
        played.append(action)
    return played


def action_to_json(action):
    return None if action is None else [action[0], list(action[1])]


def action_from_json(obj):
    return (obj[0], tuple(obj[1]))


def _forfeit(category, player, detail):
    return {'category': category, 'winner': 1 - player, 'loser': player, 'detail': detail}


def run_match(config, agents=None, identities=None, solutions=None):
    """
    Play one match and return a JSON-serialisable result dict.

    agents/identities may be supplied to reuse built agents; otherwise they are built
    from config.p0 / config.p1.
    """
    engine = WallzEngine(board_size=config.board_size, rules=config.rules)
    initial_id = state_id(engine._get_state_hashable_key())
    opening = opening_actions(engine, config.opening_plies, config.opening_seed)
    if agents is None:
        agents, identities = {}, {}
        for p, spec in ((0, config.p0), (1, config.p1)):
            agents[p], identities[p] = make_agent(spec, p, config, solutions, config.opening_seed)

    replay = [{'ply': i + 1, 'player': i % 2, 'source': 'opening', 'action': action_to_json(a)}
              for i, a in enumerate(opening)]
    times = {0: [], 1: []}
    legal_counts = {0: [], 1: []}
    nodes = {0: [], 1: []}
    depths = {0: [], 1: []}
    outcome = None

    while outcome is None:
        if engine.winner is not None:
            outcome = {'category': 'goal', 'winner': engine.winner, 'loser': 1 - engine.winner,
                       'detail': engine.terminal_reason}
            break
        if engine.is_draw:
            outcome = {'category': f"draw_{engine.terminal_reason}", 'winner': None, 'loser': None,
                       'detail': engine.terminal_reason}
            break
        if engine.turn_count >= config.max_plies:
            outcome = {'category': 'truncated', 'winner': None, 'loser': None,
                       'detail': f"external cap of {config.max_plies} plies (not a rule draw)"}
            break

        player = engine.current_player
        legal = engine.get_legal_actions()
        t0 = time.perf_counter()
        try:
            action = agents[player].select_action(engine)
        except Exception as exc:  # agent failures are recorded, not raised
            outcome = _forfeit('agent_error', player, ''.join(
                traceback.format_exception_only(type(exc), exc)).strip())
            break
        elapsed = time.perf_counter() - t0
        times[player].append(elapsed)
        legal_counts[player].append(len(legal))
        if hasattr(agents[player], 'get_stats'):
            st = agents[player].get_stats()
            nodes[player].append(st.get('nodes_expanded'))
            d_comp = st.get('depth_completed')
            if d_comp is not None:
                depths[player].append(d_comp)

        if config.move_time_limit is not None and elapsed > config.move_time_limit:
            outcome = _forfeit('timeout', player, f"{elapsed:.3f}s > {config.move_time_limit}s")
            break
        if action is None:
            outcome = _forfeit('no_action', player, f"{len(legal)} legal actions available")
            break
        if action not in legal:
            outcome = _forfeit('illegal_action', player, repr(action))
            break

        engine.apply_action(action, update_clock=False)
        if config.record_replay:
            replay.append({'ply': engine.turn_count, 'player': player, 'source': 'agent',
                           'action': action_to_json(action), 'legal_count': len(legal),
                           'decision_seconds': round(elapsed, 6),
                           'state_id': state_id(engine._get_state_hashable_key())})

    def tstats(xs):
        return {'n': len(xs), 'mean': (sum(xs) / len(xs)) if xs else None,
                'max': max(xs) if xs else None, 'total': sum(xs)}

    result = {
        'schema_version': SCHEMA_VERSION,
        'match_id': config.match_id(),
        'source': source_commit(),
        'config': asdict(config),
        'config_label': config.label(),
        'agents': {str(p): identities[p] for p in (0, 1)},
        'opening': {'seed': config.opening_seed, 'plies': config.opening_plies,
                    'actions': [action_to_json(a) for a in opening]},
        'initial_state_id': initial_id,
        'outcome': outcome,
        'length_plies': engine.turn_count,
        'final_state_id': state_id(engine._get_state_hashable_key()),
        'decision_seconds': {str(p): tstats(times[p]) for p in (0, 1)},
        'mean_legal_actions': {str(p): (sum(legal_counts[p]) / len(legal_counts[p]))
                               if legal_counts[p] else None for p in (0, 1)},
        'mean_nodes_expanded': {str(p): (sum(nodes[p]) / len(nodes[p])) if nodes[p] else None
                                for p in (0, 1)},
        'mean_depth_completed': {str(p): (sum(depths[p]) / len(depths[p])) if depths[p] else None
                                 for p in (0, 1)},
    }
    if config.record_replay:
        result['replay'] = replay
    return result


def verify_replay(result):
    """
    Re-simulate a logged match with the current engine. Returns (ok, message).
    Checks every logged action is legal, every post-action state id matches, and the
    final state / rule outcome agree.
    """
    if 'replay' not in result:
        return False, "result has no replay"
    cfg = result['config']
    engine = WallzEngine(board_size=cfg['board_size'], rules=cfg['rules'])
    if state_id(engine._get_state_hashable_key()) != result['initial_state_id']:
        return False, "initial state differs"
    for step in result['replay']:
        action = action_from_json(step['action'])
        if action not in engine.get_legal_actions():
            return False, f"ply {step['ply']}: {action} is not legal"
        engine.apply_action(action, update_clock=False)
        if 'state_id' in step and state_id(engine._get_state_hashable_key()) != step['state_id']:
            return False, f"ply {step['ply']}: state id mismatch"
    if state_id(engine._get_state_hashable_key()) != result['final_state_id']:
        return False, "final state differs"
    cat = result['outcome']['category']
    if cat == 'goal' and engine.winner != result['outcome']['winner']:
        return False, "winner differs"
    if cat.startswith('draw_') and not engine.is_draw:
        return False, "logged rule draw not reproduced"
    if cat in ('goal',) or cat.startswith('draw_'):
        return True, f"verified {len(result['replay'])} plies, outcome {cat}"
    if engine.is_terminal():
        return False, f"engine reached a terminal state but log says {cat}"
    return True, f"verified {len(result['replay'])} plies, outcome {cat} (non-rule ending)"


# ---------------------------------------------------------------------------
# Persistent batches
# ---------------------------------------------------------------------------

def run_batch(configs, out_path, keep_replay=False, progress=None):
    """
    Append one JSON line per match to ``out_path`` (JSONL). Matches whose match_id is
    already present are skipped, so an interrupted batch can be resumed.
    """
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding='utf-8') as fh:
            for line in fh:
                if line.strip():
                    done.add(json.loads(line)['match_id'])
    solutions = {}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    results = []
    with open(out_path, 'a', encoding='utf-8') as fh:
        for i, cfg in enumerate(configs):
            if cfg.match_id() in done:
                continue
            res = run_match(cfg, solutions=solutions)
            if not keep_replay:
                res.pop('replay', None)
            fh.write(json.dumps(res) + '\n')
            fh.flush()
            results.append(res)
            if progress:
                progress(i + 1, len(configs), res)
    return results


def load_results(path):
    with open(path, encoding='utf-8') as fh:
        return [json.loads(line) for line in fh if line.strip()]
