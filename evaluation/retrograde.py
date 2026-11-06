"""
Cycle-aware retrograde (attractor) solver over a StateGraph.

Solved game ("position game"): alternating moves from the graph root, reaching the
goal row wins, a non-terminal position with no legal action is a draw (team rule),
and play that never terminates is a draw. Values are from the perspective of the
side to move: +1 WIN, -1 LOSS, 0 DRAW, None UNKNOWN (only possible when the graph is
incomplete). ``rank`` is the number of plies to the goal under optimal play (winner
minimises, loser maximises); it is exact on complete graphs and None for draws. On an
incomplete graph a rank is only an upper bound (a faster line may pass through an
unexpanded node), so rank-dependent queries return None unless the bound settles them.

Why this also answers the 5-fold-repetition game from a fresh history: a winner who
plays rank-decreasing moves never repeats a position, and a defender who stays inside
the draw set never lets the opponent reach a WIN state, so neither strategy is
affected by repetition. The 200-ply step cap is a different (finite-horizon) game;
see ``horizon_value``.
"""
import time
from collections import deque

from evaluation.state_graph import NONTERMINAL, WIN_P0, WIN_P1, DRAW_NO_LEGAL_ACTIONS

WIN, DRAW, LOSS = 1, 0, -1
STATUS = {WIN: 'WIN', DRAW: 'DRAW', LOSS: 'LOSS', None: 'UNKNOWN'}

MODEL = ("position game: history-free, clock-free; goal row wins; no legal action = draw; "
         "infinite play = draw; values from side-to-move perspective")


class Solution:
    def __init__(self, graph, value, rank, solve_seconds):
        self.graph = graph
        self.value = value
        self.rank = rank
        self.solve_seconds = solve_seconds
        self.exact = graph.complete

    # -- per-node queries ------------------------------------------------------
    def status(self, node):
        return STATUS[self.value[node]]

    def value_for(self, node, player):
        """Value from ``player``'s perspective."""
        v = self.value[node]
        if v is None:
            return None
        return v if self.graph.to_move[node] == player else -v

    def action_values(self, node):
        """[(action, value for the mover, child rank)] for every legal action of ``node``."""
        edges = self.graph.succ[node]
        if edges is None:
            return None
        out = []
        for action, child in edges:
            cv = self.value[child]
            out.append((action, None if cv is None else -cv, self.rank[child]))
        return out

    def value_optimal_actions(self, node):
        """All actions that preserve the node's game value (None if undecidable)."""
        av = self.action_values(node)
        if av == []:
            return []
        if av is None or self.value[node] is None or any(v is None for _, v, _ in av):
            return None
        return [a for a, v, _ in av if v == self.value[node]]

    def rank_optimal_actions(self, node):
        """
        Value-optimal actions that are also fastest (WIN) / slowest (LOSS). In WIN states
        only these guarantee progress: other value-preserving moves can cycle forever.
        None on incomplete graphs (ranks there are only upper bounds).
        """
        av = self.action_values(node)
        if not self.exact and av:
            return None
        v = self.value[node]
        if av == []:
            return []
        if av is None or v is None or any(cv is None for _, cv, _ in av):
            return None
        if v == DRAW:
            return [a for a, cv, _ in av if cv == DRAW]
        best = [(a, r) for a, cv, r in av if cv == v]
        target = min(r for _, r in best) if v == WIN else max(r for _, r in best)
        return [a for a, r in best if r == target]

    def horizon_value(self, node, plies_left):
        """Value when the game is declared drawn after ``plies_left`` more plies."""
        v = self.value[node]
        if v is None:
            return None
        if v != DRAW and self.rank[node] <= plies_left:
            return v
        if v == DRAW or self.exact:
            return DRAW
        return None  # incomplete graph: the true rank may still be <= plies_left

    def discounted_value(self, node, gamma):
        """
        Optimal value under the DQN reward convention: +1 to the mover on a winning
        move, 0 otherwise, negamax target r - gamma * V(next). Equals
        sign * gamma**(rank - 1) for decided non-terminal nodes and 0 at terminals/draws.
        """
        v = self.value[node]
        if v is None:
            return None
        if v == DRAW or self.rank[node] == 0:
            return 0.0
        if not self.exact:
            return None
        return v * gamma ** (self.rank[node] - 1)

    # -- summaries ---------------------------------------------------------------
    def root(self):
        return 0

    def counts(self):
        c = {'WIN': 0, 'LOSS': 0, 'DRAW': 0, 'UNKNOWN': 0}
        for v in self.value:
            c[STATUS[v]] += 1
        return c

    def summary(self):
        r = self.root()
        rank_opt = self.rank_optimal_actions(r)
        value_opt = self.value_optimal_actions(r)
        out = dict(self.graph.stats())
        out.update({
            'model': MODEL,
            'exact': self.exact,
            'value_counts': self.counts(),
            'root_to_move': self.graph.to_move[r],
            'root_status': self.status(r),
            'root_rank': self.rank[r],
            'root_rank_is_upper_bound': not self.exact and self.rank[r] is not None,
            'root_rank_optimal_actions': None if rank_opt is None else [list(a) for a in rank_opt],
            'root_value_optimal_actions': None if value_opt is None else [list(a) for a in value_opt],
            'solve_seconds': round(self.solve_seconds, 3),
        })
        return out


def solve(graph):
    """Retrograde attractor computation. Returns a Solution."""
    t0 = time.perf_counter()
    n = len(graph.keys)
    value = [None] * n
    rank = [None] * n
    children = [None] * n
    remaining = [0] * n
    preds = [[] for _ in range(n)]

    for u in range(n):
        edges = graph.succ[u]
        if edges is None or graph.terminal[u] is not NONTERMINAL:
            continue
        # Several actions may reach the same child; count each child once.
        kids = {c for _, c in edges}
        children[u] = kids
        remaining[u] = len(kids)
        for c in kids:
            preds[c].append(u)

    queue = deque()
    for u in range(n):
        label = graph.terminal[u]
        if label in (WIN_P0, WIN_P1):
            winner = 0 if label == WIN_P0 else 1
            value[u] = WIN if winner == graph.to_move[u] else LOSS
            rank[u] = 0
            queue.append(u)
        elif label == DRAW_NO_LEGAL_ACTIONS:
            value[u] = DRAW
            rank[u] = 0

    # FIFO order processes nodes in non-decreasing rank, which makes ranks exact.
    while queue:
        v = queue.popleft()
        for u in preds[v]:
            if value[u] is not None:
                continue
            if value[v] == LOSS:
                value[u] = WIN
                rank[u] = rank[v] + 1
                queue.append(u)
            elif value[v] == WIN:
                remaining[u] -= 1
                if remaining[u] == 0:
                    value[u] = LOSS
                    rank[u] = max(rank[c] for c in children[u]) + 1
                    queue.append(u)

    if graph.complete:
        for u in range(n):
            if value[u] is None:
                value[u] = DRAW
    return Solution(graph, value, rank, time.perf_counter() - t0)


def solve_config(board_size, rules=None, caps=None, root_key=None):
    from evaluation.state_graph import StateGraph
    graph = StateGraph(board_size, rules, root_key=root_key).build(caps)
    return solve(graph)
