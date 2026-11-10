"""
Reachable position graph for Modified Wallz.

Nodes are engine state keys (``WallzEngine._get_state_hashable_key()``): pawn cells,
each player's wall list in placement order, wall inventories, latest-wall pointers,
hop/block availability, active block timers and the side to move. Board size and the
rules dict (A, B, C, K, M) are stored once as graph metadata.

Every transition is produced by the shared engine (``restore_state_key`` +
``apply_action(update_clock=False)``), so the graph never re-implements the rules.
Positions are history-free and clock-free: repetition counts, turn counters and chess
clocks are not part of a node. See docs/person3_evaluation.md for the exact game model.
"""
import hashlib
import time
from collections import deque
from dataclasses import dataclass

from wallz_engine import WallzEngine

# Terminal labels stored per node
NONTERMINAL = None
WIN_P0 = 'win_p0'
WIN_P1 = 'win_p1'
DRAW_NO_LEGAL_ACTIONS = 'draw_no_legal_actions'


def canonical_key(key, rules):
    """
    Position identity for solving (P3.2). Delegates to engine.position_key() if passed
    an engine instance; raw 14-tuples go through WallzEngine.canonicalize_key (single
    implementation shared with the engine's repetition detection).
    """
    if hasattr(key, 'position_key'):
        return key.position_key()
    if isinstance(key, tuple) and len(key) in (11, 13):
        return key  # Already canonical
    return WallzEngine.canonicalize_key(key, rules)


def state_id(key):
    """Short stable identifier for a state key (used in logs and replays)."""
    return hashlib.sha1(repr(key).encode('ascii')).hexdigest()[:16]


def full_rules(rules=None):
    """Engine defaults merged with overrides, so metadata always lists A, B, C, K, M."""
    return dict(WallzEngine(board_size=3, rules=rules).rules)


def terminal_label(engine):
    if engine.winner is not None:
        return WIN_P0 if engine.winner == 0 else WIN_P1
    if engine.is_draw:
        if engine.terminal_reason != 'no_legal_actions':
            # Repetition needs history; graph transitions always start from a fresh history.
            raise RuntimeError(f"Unexpected history-dependent terminal: {engine.terminal_reason}")
        return DRAW_NO_LEGAL_ACTIONS
    if not engine.has_legal_action():
        return DRAW_NO_LEGAL_ACTIONS
    return NONTERMINAL


@dataclass
class GraphCaps:
    """Enumeration limits. Caps are checked after each fully expanded node."""
    max_states: int = 1_000_000
    max_edges: int = 20_000_000
    max_seconds: float = 600.0


class StateGraph:
    """
    Forward-enumerated reachable position graph.

    Attributes (indexed by node id):
        keys[i]      engine state key of the node's representative position
        to_move[i]   side to move (0/1)
        terminal[i]  NONTERMINAL, WIN_P0, WIN_P1 or DRAW_NO_LEGAL_ACTIONS
        succ[i]      list of (action, child_id) for expanded nodes, [] for terminals,
                     None for nodes left unexpanded because a cap was hit
    """

    def __init__(self, board_size, rules=None, root_key=None, canonical=True):
        self.board_size = board_size
        self.rules = full_rules(rules)
        # canonical=True merges equivalent positions (see canonical_key); False keeps raw keys
        self.canonical = canonical
        if root_key is None:
            root_key = self._engine()._get_state_hashable_key()
        self.root_key = root_key
        self.keys = []
        self.index = {}
        self.to_move = []
        self.terminal = []
        self.succ = []
        self.complete = False
        self.stop_reason = None
        self.edge_count = 0
        self.build_seconds = 0.0
        # One shared tuple per distinct action: saves memory on large graphs, same values
        self._actions = {}

    def _engine(self):
        return WallzEngine(board_size=self.board_size, rules=self.rules)

    def restore(self, key):
        """Fresh engine positioned at ``key`` (fresh history, clocks unused)."""
        return self._engine().restore_state_key(key)

    def _identity(self, key):
        return canonical_key(key, self.rules) if self.canonical else key

    def _add(self, key, label):
        node = len(self.keys)
        self.keys.append(key)
        self.index[self._identity(key)] = node
        self.to_move.append(key[-1])
        self.terminal.append(label)
        self.succ.append(None)
        return node

    def build(self, caps=None):
        caps = caps or GraphCaps()
        t0 = time.perf_counter()
        root_engine = self.restore(self.root_key)
        self._add(self.root_key, terminal_label(root_engine))
        frontier = deque([0])
        self.stop_reason = None

        while frontier:
            node = frontier.popleft()
            if self.terminal[node] is not NONTERMINAL:
                self.succ[node] = []
                continue
            engine = self.restore(self.keys[node])
            edges = []
            for action in engine.get_legal_actions():
                action = self._actions.setdefault(action, action)
                child = engine.clone()
                child.apply_action(action, update_clock=False)
                child_key = child._get_state_hashable_key()
                child_id = self.index.get(self._identity(child_key))
                if child_id is None:
                    child_id = self._add(child_key, terminal_label(child))
                    frontier.append(child_id)
                edges.append((action, child_id))
            self.succ[node] = edges
            self.edge_count += len(edges)

            if len(self.keys) >= caps.max_states:
                self.stop_reason = 'max_states'
            elif self.edge_count >= caps.max_edges:
                self.stop_reason = 'max_edges'
            elif time.perf_counter() - t0 >= caps.max_seconds:
                self.stop_reason = 'max_seconds'
            if self.stop_reason and frontier:
                break

        # Terminal nodes need no expansion, so a frontier of terminals is still complete
        if all(self.terminal[n] is not NONTERMINAL for n in frontier):
            for n in frontier:
                self.succ[n] = []
            frontier.clear()
        self.complete = not frontier
        if self.complete:
            self.stop_reason = None
        self.build_seconds = time.perf_counter() - t0
        return self

    def node_of(self, engine_or_key):
        key = engine_or_key._get_state_hashable_key() if isinstance(engine_or_key, WallzEngine) else engine_or_key
        return self.index.get(self._identity(key))

    def stats(self):
        expanded = [s for s, t in zip(self.succ, self.terminal) if s is not None and t is NONTERMINAL]
        outdeg = [len(s) for s in expanded]
        return {
            'board_size': self.board_size,
            'rules': dict(self.rules),
            'root_state_id': state_id(self.root_key),
            'canonical_positions': self.canonical,
            'complete': self.complete,
            'stop_reason': self.stop_reason,
            'states': len(self.keys),
            'edges': self.edge_count,
            'expanded_nonterminal': len(expanded),
            'unexpanded': sum(1 for s in self.succ if s is None),
            'terminal_win_p0': self.terminal.count(WIN_P0),
            'terminal_win_p1': self.terminal.count(WIN_P1),
            'terminal_draw_no_legal_actions': self.terminal.count(DRAW_NO_LEGAL_ACTIONS),
            'mean_branching': (sum(outdeg) / len(outdeg)) if outdeg else 0.0,
            'max_branching': max(outdeg) if outdeg else 0,
            'build_seconds': round(self.build_seconds, 3),
        }
