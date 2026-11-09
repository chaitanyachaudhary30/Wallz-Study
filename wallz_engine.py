"""Modified Wallz Game Engine (Phase 3 Engine v2).

Implements complete game state, transition logic, legality gates,
repetition tracking via canonical position keys, O(1) edge-blocked geometry,
and staged action generation with shortest-path filtering.
"""
from __future__ import annotations

from collections import ChainMap, deque
import copy
import time
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from src.modified_wallz.board import WallzBoard
    from src.modified_wallz.config import WallzConfig
    from src.modified_wallz.pathfinding import (
        bfs_distance_and_path,
        has_path_to_goal,
        norm_edge,
    )
except ImportError:
    from modified_wallz.board import WallzBoard
    from modified_wallz.config import WallzConfig
    from modified_wallz.pathfinding import (
        bfs_distance_and_path,
        has_path_to_goal,
        norm_edge,
    )


class WallzEngine:
    _ACTION_TYPES = ('MOVE', 'HOP', 'WALL', 'REPLACE', 'BLOCK')

    @staticmethod
    def _is_int(x: Any) -> bool:
        return isinstance(x, int) and not isinstance(x, bool)

    def __init__(
        self,
        board_size: int | WallzConfig = 9,
        time_limit_sec: float = 600.0,
        rules: Optional[Dict[str, Any] | WallzConfig] = None,
        config: Optional[WallzConfig] = None,
    ):
        if isinstance(config, WallzConfig):
            self.config = config
        elif isinstance(board_size, WallzConfig):
            self.config = board_size
        elif isinstance(rules, WallzConfig):
            self.config = rules
        else:
            self.config = WallzConfig.from_rules(
                rules=rules,
                board_size=board_size,
                time_limit_sec=time_limit_sec,
            )

        self.board_size = self.config.board_size
        self.time_limit = self.config.time_limit_sec
        self.rules = dict(self.config.rules)
        self.max_plies = self.config.max_plies
        self.repetition_limit = self.config.repetition_limit

        self.board = WallzBoard(self.board_size)
        self.reset()

    def __deepcopy__(self, memo):
        return self.clone()

    def clone(self) -> WallzEngine:
        """Creates a fast, O(1) copy of the game engine for search simulation."""
        new_engine = WallzEngine.__new__(WallzEngine)
        new_engine.config = self.config
        new_engine.board_size = self.board_size
        new_engine.time_limit = self.time_limit
        new_engine.rules = dict(self.rules)
        new_engine.max_plies = self.max_plies
        new_engine.repetition_limit = self.repetition_limit

        new_engine.pawns = {0: self.pawns[0], 1: self.pawns[1]}
        new_engine.goals = {0: self.goals[0], 1: self.goals[1]}
        new_engine.walls = {0: list(self.walls[0]), 1: list(self.walls[1])}
        new_engine.wall_inventory = {0: self.wall_inventory[0], 1: self.wall_inventory[1]}
        new_engine.latest_wall = {0: self.latest_wall[0], 1: self.latest_wall[1]}
        new_engine.hop_available = {0: self.hop_available[0], 1: self.hop_available[1]}
        new_engine.block_available = {0: self.block_available[0], 1: self.block_available[1]}
        new_engine.blocked_cells = dict(self.blocked_cells)
        new_engine.time_remaining = {0: self.time_remaining[0], 1: self.time_remaining[1]}
        new_engine.last_turn_timestamp = self.last_turn_timestamp
        new_engine.current_player = self.current_player
        new_engine.turn_count = self.turn_count
        new_engine.winner = self.winner
        new_engine.is_draw = self.is_draw
        new_engine.is_truncated = self.is_truncated
        new_engine.terminal_reason = self.terminal_reason
        new_engine.max_walls = self.max_walls

        # Fast clone of board representation
        new_engine.board = self.board.clone()

        # Copy-on-write ChainMap history (P3.3)
        if isinstance(self.state_history, ChainMap):
            if self.state_history.maps and self.state_history.maps[0]:
                self.state_history.maps.insert(0, {})
            new_engine.state_history = ChainMap({}, *self.state_history.maps[1:])
            # Compaction beyond 32 maps
            if len(new_engine.state_history.maps) > 32:
                new_engine.state_history = ChainMap({}, dict(new_engine.state_history))
            if len(self.state_history.maps) > 32:
                self.state_history = ChainMap({}, dict(self.state_history))
        else:
            new_engine.state_history = ChainMap({}, dict(self.state_history))

        return new_engine

    @property
    def truncated(self) -> bool:
        return self.is_truncated

    def reset(self):
        mid = self.board_size // 2
        self.pawns = {0: (0, mid), 1: (self.board_size - 1, mid)}
        self.goals = {0: self.board_size - 1, 1: 0}

        self.max_walls = self.rules['M']
        self.wall_inventory = {0: self.max_walls, 1: self.max_walls}
        self.latest_wall = {0: None, 1: None}
        self.walls = {0: [], 1: []}

        self.hop_available = {0: self.rules['B'], 1: self.rules['B']}
        self.block_available = {0: self.rules['C'], 1: self.rules['C']}
        self.blocked_cells: Dict[Tuple[int, int], int] = {}

        self.time_remaining = {0: float(self.time_limit), 1: float(self.time_limit)}
        self.last_turn_timestamp = time.time()

        self.current_player = 0
        self.turn_count = 0
        self.winner = None
        self.is_draw = False
        self.is_truncated = False
        self.terminal_reason = None

        self.board.clear()
        self.state_history = ChainMap({}, {})
        self._record_state_history()

        return self.get_state()

    @staticmethod
    def canonicalize_key(raw_key: Tuple, rules: Dict[str, Any]) -> Tuple:
        """
        Canonical position identity (P3.2) of a raw key from _get_state_hashable_key().

        Merges what cannot influence future play:
        * the placement order of walls that can no longer move (all walls without Rule A,
          every wall except each player's latest one with Rule A);
        * wall ownership, but only while no hop can be played: a hop may only cross an
          *opponent's* wall (proposal LegalHop), so while either player still holds an
          unused hop (and Rule B is on) fixed walls are kept as (wall, owner) pairs.

        Shapes: 13-tuple with Rule A, 11-tuple without (see restore_state_key).
        """
        p0, p1, w0, w1, i0, i1, l0, l1, h0, h1, b0, b1, blocks, to_move = raw_key
        hop_relevant = bool(rules['B']) and bool(h0 or h1)
        if rules['A']:
            fixed0 = [w for w in w0 if w != l0]
            fixed1 = [w for w in w1 if w != l1]
        else:
            fixed0, fixed1 = list(w0), list(w1)
        if hop_relevant:
            fixed = frozenset([(w, 0) for w in fixed0] + [(w, 1) for w in fixed1])
        else:
            fixed = frozenset(fixed0 + fixed1)
        if rules['A']:
            return (p0, p1, fixed, l0, l1, i0, i1, h0, h1, b0, b1, blocks, to_move)
        return (p0, p1, fixed, i0, i1, h0, h1, b0, b1, blocks, to_move)

    def position_key(self) -> Tuple:
        """Canonical position identity; equals state_graph.canonical_key of the raw key."""
        return self.canonicalize_key(self._get_state_hashable_key(), self.rules)

    def _get_state_hashable_key(self) -> Tuple:
        """Generates raw hashable key tuple preserving walls in placement order."""
        walls_p0 = tuple(self.walls[0])
        walls_p1 = tuple(self.walls[1])
        blocks = tuple(sorted(self.blocked_cells.items()))
        return (
            self.pawns[0],
            self.pawns[1],
            walls_p0,
            walls_p1,
            self.wall_inventory[0],
            self.wall_inventory[1],
            self.latest_wall[0],
            self.latest_wall[1],
            self.hop_available[0],
            self.hop_available[1],
            self.block_available[0],
            self.block_available[1],
            blocks,
            self.current_player
        )

    def _record_state_history(self):
        pos_key = self.position_key()
        raw_key = self._get_state_hashable_key()

        if not isinstance(self.state_history, ChainMap):
            self.state_history = ChainMap({}, dict(self.state_history))

        count = self.state_history.get(pos_key, 0) + 1
        self.state_history[pos_key] = count
        if raw_key != pos_key:
            self.state_history[raw_key] = count

        # N-fold repetition draw rule (P3.2)
        if count >= self.repetition_limit and not self.is_terminal():
            self.is_draw = True
            self.terminal_reason = 'repetition'

    def is_terminal(self) -> bool:
        return self.winner is not None or self.is_draw

    @staticmethod
    def _split_fixed_walls(fixed) -> Tuple[List, List]:
        """Owner lists from a canonical fixed-wall set: (wall, owner) pairs keep ownership;
        plain walls (ownership irrelevant, see canonicalize_key) are all given to player 0."""
        w0, w1 = [], []
        for item in sorted(fixed, key=repr):
            if isinstance(item[0], tuple):
                (w0 if item[1] == 0 else w1).append(item[0])
            else:
                w0.append(item)
        return w0, w1

    def restore_state_key(self, key: Tuple) -> WallzEngine:
        """
        Loads a position produced by _get_state_hashable_key() or position_key()
        with a fresh history (count 1) and turn counter 0.
        """
        n = self.board_size

        if len(key) == 14:
            # Raw key
            (p0, p1, walls_p0, walls_p1, inv0, inv1, latest0, latest1,
             hop0, hop1, blk_av0, blk_av1, blocks, current_player) = key
            w0 = list(walls_p0)
            w1 = list(walls_p1)
        elif len(key) == 13:
            # Canonical key with Rule A
            (p0, p1, fixed, latest0, latest1, inv0, inv1,
             hop0, hop1, blk_av0, blk_av1, blocks, current_player) = key
            w0, w1 = self._split_fixed_walls(fixed)
            if latest0 is not None and latest0 not in w0:
                w0.append(latest0)
            if latest1 is not None and latest1 not in w1:
                w1.append(latest1)
        elif len(key) == 11:
            # Canonical key without Rule A
            (p0, p1, fixed, inv0, inv1,
             hop0, hop1, blk_av0, blk_av1, blocks, current_player) = key
            w0, w1 = self._split_fixed_walls(fixed)
            latest0, latest1 = None, None
        else:
            raise ValueError(f"Invalid key length {len(key)}.")

        cells = [p0, p1] + [cell for cell, _ in blocks]
        if not all(0 <= r < n and 0 <= c < n for r, c in cells) or not all(
                0 <= r < n - 1 and 0 <= c < n - 1 for r, c, _ in w0 + w1):
            raise ValueError(f"State key does not fit a {n}x{n} board.")

        self.pawns = {0: p0, 1: p1}
        self.walls = {0: w0, 1: w1}
        self.wall_inventory = {0: inv0, 1: inv1}
        self.latest_wall = {0: latest0, 1: latest1}
        self.hop_available = {0: hop0, 1: hop1}
        self.block_available = {0: blk_av0, 1: blk_av1}
        self.blocked_cells = dict(blocks)
        self.current_player = current_player
        self.turn_count = 0
        self.winner = None
        self.is_draw = False
        self.is_truncated = False
        self.terminal_reason = None

        self.board.rebuild(self.walls[0] + self.walls[1])

        for p in (0, 1):
            if self.pawns[p][0] == self.goals[p]:
                self.winner = p
                self.terminal_reason = 'goal'

        if self.winner is None and not self.has_legal_action(current_player):
            self.is_draw = True
            self.terminal_reason = 'no_legal_actions'

        # Store initial count for the restored key
        self.state_history = ChainMap({}, {key: 1})
        return self

    def get_state(self) -> Dict[str, Any]:
        return {
            'pawns': copy.deepcopy(self.pawns),
            'walls': copy.deepcopy(self.walls),
            'wall_inventory': copy.deepcopy(self.wall_inventory),
            'latest_wall': copy.deepcopy(self.latest_wall),
            'hop_available': copy.deepcopy(self.hop_available),
            'block_available': copy.deepcopy(self.block_available),
            'blocked_cells': copy.deepcopy(self.blocked_cells),
            'current_player': self.current_player,
            'turn_count': self.turn_count,
            'time_remaining': copy.deepcopy(self.time_remaining),
            'is_draw': self.is_draw,
            'winner': self.winner,
            'terminal_reason': self.terminal_reason,
            'truncated': self.is_truncated,
        }

    def update_clock(self):
        now = time.time()
        elapsed = now - self.last_turn_timestamp
        self.time_remaining[self.current_player] -= elapsed
        self.last_turn_timestamp = now

        if self.time_remaining[self.current_player] <= 0:
            self.time_remaining[self.current_player] = 0
            self.winner = 1 - self.current_player
            self.terminal_reason = 'timeout'

    def _sync_board(self):
        """Ensures self.board is synchronized with self.walls (handles direct test mutations)."""
        active_walls = set(self.walls[0] + self.walls[1])
        if self.board.walls != active_walls:
            self.board.rebuild(active_walls)

    def _get_all_active_walls(self) -> Set[Tuple[int, int, str]]:
        self._sync_board()
        return self.board.walls

    def _is_wall_blocking(self, cell1: Tuple[int, int], cell2: Tuple[int, int]) -> bool:
        """O(1) edge lookup via board grid (P3.4)."""
        self._sync_board()
        return self.board.is_edge_blocked(cell1, cell2)

    def shortest_path(self, player: int) -> float:
        """Computes shortest path distance to goal for player."""
        self._sync_board()
        dist, _, _ = bfs_distance_and_path(
            start=self.pawns[player],
            target_row=self.goals[player],
            board_size=self.board_size,
            board=self.board,
            blocked_cells=self.blocked_cells,
        )
        return dist

    def shortest_path_cells(self, player: int) -> List[Tuple[int, int]]:
        """Return the sequence of cells along a shortest path to goal for player."""
        self._sync_board()
        start = self.pawns[player]
        target_row = self.goals[player]
        board_size = self.board_size

        if start[0] == target_row:
            return [start]

        queue = deque([start])
        parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
        target_cell: Optional[Tuple[int, int]] = None

        while queue:
            curr = queue.popleft()
            r, c = curr
            if r == target_row:
                target_cell = curr
                break

            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < board_size and 0 <= nc < board_size:
                    nxt = (nr, nc)
                    if nxt not in parent and nxt not in self.blocked_cells:
                        if not self.board.is_edge_blocked(curr, nxt):
                            parent[nxt] = curr
                            queue.append(nxt)

        if target_cell is None:
            return []

        cells = []
        curr = target_cell
        while curr is not None:
            cells.append(curr)
            curr = parent[curr]
        cells.reverse()
        return cells

    def is_path_open(self, player: int) -> bool:
        self._sync_board()
        return has_path_to_goal(
            start=self.pawns[player],
            target_row=self.goals[player],
            board_size=self.board_size,
            board=self.board,
            blocked_cells=self.blocked_cells,
        )

    # -------------------------------------------------------------------------
    # Staged-Generation API (P3.6)
    # -------------------------------------------------------------------------

    def pawn_actions(self, player: Optional[int] = None) -> List[Tuple[str, Tuple[int, int]]]:
        """Generates legal pawn moves and hops."""
        if player is None:
            player = self.current_player
        if self.is_terminal():
            return []
        actions = []
        r, c = self.pawns[player]

        # Standard 1-step orthogonal moves
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            cell = (r + dr, c + dc)
            if self._is_valid_move(player, cell):
                actions.append(('MOVE', cell))

        # One-time hop across an opponent's wall
        if self.rules['B'] and self.hop_available[player]:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                cell = (r + dr, c + dc)
                if self._is_valid_hop(player, cell):
                    actions.append(('HOP', cell))

        return actions

    def wall_candidates(self, player: Optional[int] = None) -> List[Tuple[str, Tuple[int, int, str]]]:
        """Generates candidate wall placements and replacements satisfying geometry in O(1)."""
        if player is None:
            player = self.current_player
        if self.is_terminal():
            return []
        n = self.board_size
        candidates = []

        # New wall placement candidates
        if self.wall_inventory[player] > 0:
            for wr in range(n - 1):
                for wc in range(n - 1):
                    for orient in ('H', 'V'):
                        if self.board.can_place_wall(wr, wc, orient):
                            candidates.append(('WALL', (wr, wc, orient)))

        # Reusable wall replacement candidates (Rule A)
        if self.rules['A'] and self.latest_wall[player] is not None:
            latest = self.latest_wall[player]
            lr, lc, l_orient = latest
            self.board.remove_wall(lr, lc, l_orient)
            for wr in range(n - 1):
                for wc in range(n - 1):
                    for orient in ('H', 'V'):
                        new_w = (wr, wc, orient)
                        if new_w != latest and self.board.can_place_wall(wr, wc, orient):
                            candidates.append(('REPLACE', new_w))
            self.board.add_wall(lr, lc, l_orient)

        return candidates

    def block_candidates(self, player: Optional[int] = None) -> List[Tuple[str, Tuple[int, int]]]:
        """Generates candidate cell blocks satisfying cell occupancy in O(1)."""
        if player is None:
            player = self.current_player
        if self.is_terminal():
            return []
        if not (self.rules['C'] and self.block_available[player]):
            return []
        n = self.board_size
        p0, p1 = self.pawns[0], self.pawns[1]
        candidates = []
        for br in range(n):
            for bc in range(n):
                cell = (br, bc)
                if cell not in self.blocked_cells and cell != p0 and cell != p1:
                    candidates.append(('BLOCK', cell))
        return candidates

    def path_ok(self, action: Tuple[str, Any], player: Optional[int] = None) -> bool:
        """Verifies path preservation for a candidate action."""
        if player is None:
            player = self.current_player
        act_type, act_val = action

        if act_type == 'MOVE':
            return self._is_valid_move(player, act_val)
        elif act_type == 'HOP':
            return self._is_valid_hop(player, act_val)
        elif act_type == 'WALL':
            wr, wc, orient = act_val
            return self._is_valid_wall_placement(player, wr, wc, orient)
        elif act_type == 'REPLACE':
            return self._is_valid_wall_replacement(player, act_val)
        elif act_type == 'BLOCK':
            return self._is_valid_cell_block(act_val[0], act_val[1])
        return False

    def get_legal_actions(self, player: Optional[int] = None) -> List[Tuple[str, Any]]:
        """
        Returns all legal actions for player using staged generation and
        the exact shortest-path filter (P3.5 / P3.6).
        """
        self._sync_board()
        if player is None:
            player = self.current_player
        if self.is_terminal():
            return []

        # 1. Pawn actions (moves and hops)
        actions = self.pawn_actions(player)

        has_walls = (self.wall_inventory[player] > 0)
        has_replace = (self.rules['A'] and self.latest_wall[player] is not None)
        has_blocks = (self.rules['C'] and self.block_available[player])

        if not (has_walls or has_replace or has_blocks):
            return actions

        # 2. Compute shortest path edges and vertices once per player (P3.5)
        d0, edges0, vert0 = bfs_distance_and_path(
            start=self.pawns[0],
            target_row=self.goals[0],
            board_size=self.board_size,
            board=self.board,
            blocked_cells=self.blocked_cells,
        )
        d1, edges1, vert1 = bfs_distance_and_path(
            start=self.pawns[1],
            target_row=self.goals[1],
            board_size=self.board_size,
            board=self.board,
            blocked_cells=self.blocked_cells,
        )

        n = self.board_size

        # 3. WALL candidates with shortest-path filter
        if has_walls:
            for wr in range(n - 1):
                for wc in range(n - 1):
                    for orient in ('H', 'V'):
                        if self.board.can_place_wall(wr, wc, orient):
                            wall_edges = self.board.wall_edges(wr, wc, orient)
                            norm_e = {norm_edge(u, v) for u, v in wall_edges}

                            touches0 = bool(norm_e & edges0)
                            touches1 = bool(norm_e & edges1)

                            if not touches0 and not touches1:
                                # Candidate touches neither player's shortest path -> guaranteed valid
                                actions.append(('WALL', (wr, wc, orient)))
                            else:
                                # Test reachability for affected player(s) only
                                self.board.add_wall(wr, wc, orient)
                                ok = True
                                if touches0:
                                    ok = has_path_to_goal(self.pawns[0], self.goals[0], n, self.board, self.blocked_cells)
                                if ok and touches1:
                                    ok = has_path_to_goal(self.pawns[1], self.goals[1], n, self.board, self.blocked_cells)
                                self.board.remove_wall(wr, wc, orient)
                                if ok:
                                    actions.append(('WALL', (wr, wc, orient)))

        # 4. REPLACE candidates with shortest-path filter
        if has_replace:
            latest = self.latest_wall[player]
            lr, lc, l_orient = latest
            self.board.remove_wall(lr, lc, l_orient)

            # Recompute shortest paths without the removed wall if latest wall was on path
            latest_norm_e = {norm_edge(u, v) for u, v in self.board.wall_edges(lr, lc, l_orient)}
            if latest_norm_e & edges0:
                _, rep_edges0, _ = bfs_distance_and_path(
                    self.pawns[0], self.goals[0], n, self.board, self.blocked_cells
                )
            else:
                rep_edges0 = edges0

            if latest_norm_e & edges1:
                _, rep_edges1, _ = bfs_distance_and_path(
                    self.pawns[1], self.goals[1], n, self.board, self.blocked_cells
                )
            else:
                rep_edges1 = edges1

            for wr in range(n - 1):
                for wc in range(n - 1):
                    for orient in ('H', 'V'):
                        new_w = (wr, wc, orient)
                        if new_w != latest and self.board.can_place_wall(wr, wc, orient):
                            w_edges = self.board.wall_edges(wr, wc, orient)
                            norm_e = {norm_edge(u, v) for u, v in w_edges}
                            touches0 = bool(norm_e & rep_edges0)
                            touches1 = bool(norm_e & rep_edges1)

                            if not touches0 and not touches1:
                                actions.append(('REPLACE', new_w))
                            else:
                                self.board.add_wall(wr, wc, orient)
                                ok = True
                                if touches0:
                                    ok = has_path_to_goal(self.pawns[0], self.goals[0], n, self.board, self.blocked_cells)
                                if ok and touches1:
                                    ok = has_path_to_goal(self.pawns[1], self.goals[1], n, self.board, self.blocked_cells)
                                self.board.remove_wall(wr, wc, orient)
                                if ok:
                                    actions.append(('REPLACE', new_w))

            self.board.add_wall(lr, lc, l_orient)

        # 5. BLOCK candidates with shortest-path filter
        if has_blocks:
            p0, p1 = self.pawns[0], self.pawns[1]
            for br in range(n):
                for bc in range(n):
                    cell = (br, bc)
                    if cell not in self.blocked_cells and cell != p0 and cell != p1:
                        touches0 = (cell in vert0)
                        touches1 = (cell in vert1)

                        if not touches0 and not touches1:
                            # Cell is not on either shortest path -> guaranteed valid
                            actions.append(('BLOCK', cell))
                        else:
                            self.blocked_cells[cell] = self.rules['K']
                            ok = True
                            if touches0:
                                ok = has_path_to_goal(self.pawns[0], self.goals[0], n, self.board, self.blocked_cells)
                            if ok and touches1:
                                ok = has_path_to_goal(self.pawns[1], self.goals[1], n, self.board, self.blocked_cells)
                            del self.blocked_cells[cell]
                            if ok:
                                actions.append(('BLOCK', cell))

        return actions

    def has_legal_action(self, player: Optional[int] = None) -> bool:
        """Early-exit check for at least one legal action (cheap when a pawn move exists)."""
        if player is None:
            player = self.current_player
        if self.is_terminal():
            return False

        r, c = self.pawns[player]
        # Fast pawn move check
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            if self._is_valid_move(player, (r + dr, c + dc)):
                return True
        # Fast hop check
        if self.rules['B'] and self.hop_available[player]:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                if self._is_valid_hop(player, (r + dr, c + dc)):
                    return True

        n = self.board_size
        # Walls
        if self.wall_inventory[player] > 0:
            for wr in range(n - 1):
                for wc in range(n - 1):
                    for orient in ('H', 'V'):
                        if self._is_valid_wall_placement(player, wr, wc, orient):
                            return True
        # Replace
        if self.rules['A'] and self.latest_wall[player] is not None:
            for wr in range(n - 1):
                for wc in range(n - 1):
                    for orient in ('H', 'V'):
                        if self._is_valid_wall_replacement(player, (wr, wc, orient)):
                            return True
        # Block
        if self.rules['C'] and self.block_available[player]:
            for br in range(n):
                for bc in range(n):
                    if self._is_valid_cell_block(br, bc):
                        return True
        return False

    def _is_valid_move(self, player: int, cell: Tuple[int, int]) -> bool:
        r, c = self.pawns[player]
        nr, nc = cell
        if abs(nr - r) + abs(nc - c) != 1:
            return False
        if not (0 <= nr < self.board_size and 0 <= nc < self.board_size):
            return False
        if self.board.is_edge_blocked((r, c), (nr, nc)) or cell in self.blocked_cells:
            return False
        return cell != self.pawns[1 - player]

    def wall_owner_on_edge(self, cell1: Tuple[int, int], cell2: Tuple[int, int]) -> Optional[int]:
        """Owner (0 or 1) of the wall blocking the edge cell1-cell2, or None if the edge is open.
        Walls cannot overlap, so at most one wall blocks an edge."""
        edge = {cell1, cell2}
        for owner in (0, 1):
            for (wr, wc, wo) in self.walls[owner]:
                for a, b in self.board.wall_edges(wr, wc, wo):
                    if {a, b} == edge:
                        return owner
        return None

    def _is_valid_hop(self, player: int, cell: Tuple[int, int]) -> bool:
        """One-time hop (Modification B, proposal LegalHop): step to an orthogonally adjacent
        cell *across a wall placed by the opponent*. The landing cell must be on the board,
        not temporarily blocked and not the opponent's cell, and the mover must still have a
        path to its goal row from there."""
        if not (self.rules['B'] and self.hop_available[player]):
            return False
        r, c = self.pawns[player]
        nr, nc = cell
        if abs(nr - r) + abs(nc - c) != 1:
            return False
        if not (0 <= nr < self.board_size and 0 <= nc < self.board_size):
            return False
        if cell in self.blocked_cells or cell == self.pawns[1 - player]:
            return False
        self._sync_board()
        if not self.board.is_edge_blocked((r, c), cell):
            return False  # nothing to hop over: use MOVE
        if self.wall_owner_on_edge((r, c), cell) != 1 - player:
            return False  # only an opponent's wall may be bypassed
        old = self.pawns[player]
        self.pawns[player] = cell
        valid = self.is_path_open(player)
        self.pawns[player] = old
        return valid

    def _is_valid_wall_placement(self, player: int, r: int, c: int, orient: str) -> bool:
        self._sync_board()
        if self.wall_inventory[player] <= 0:
            return False
        if not self.board.can_place_wall(r, c, orient):
            return False

        n = self.board_size
        self.board.add_wall(r, c, orient)
        valid = (
            has_path_to_goal(self.pawns[0], self.goals[0], n, self.board, self.blocked_cells)
            and has_path_to_goal(self.pawns[1], self.goals[1], n, self.board, self.blocked_cells)
        )
        self.board.remove_wall(r, c, orient)
        return valid

    def _is_valid_wall_replacement(self, player: int, new_wall: Tuple[int, int, str], old_wall: Optional[Tuple[int, int, str]] = None) -> bool:
        self._sync_board()
        if not self.rules['A']:
            return False
        latest = self.latest_wall[player]
        if latest is None:
            return False
        if old_wall is not None and old_wall != latest:
            return False

        r, c, orient = new_wall
        if new_wall == latest:
            return False

        lr, lc, l_orient = latest
        self.board.remove_wall(lr, lc, l_orient)
        can_place = self.board.can_place_wall(r, c, orient)
        if not can_place:
            self.board.add_wall(lr, lc, l_orient)
            return False

        n = self.board_size
        self.board.add_wall(r, c, orient)
        valid = (
            has_path_to_goal(self.pawns[0], self.goals[0], n, self.board, self.blocked_cells)
            and has_path_to_goal(self.pawns[1], self.goals[1], n, self.board, self.blocked_cells)
        )
        self.board.remove_wall(r, c, orient)
        self.board.add_wall(lr, lc, l_orient)
        return valid

    def _is_valid_cell_block(self, br: int, bc: int) -> bool:
        self._sync_board()
        if not (0 <= br < self.board_size and 0 <= bc < self.board_size):
            return False
        if (br, bc) in self.blocked_cells or (br, bc) == self.pawns[0] or (br, bc) == self.pawns[1]:
            return False

        n = self.board_size
        self.blocked_cells[(br, bc)] = self.rules['K']
        valid = (
            has_path_to_goal(self.pawns[0], self.goals[0], n, self.board, self.blocked_cells)
            and has_path_to_goal(self.pawns[1], self.goals[1], n, self.board, self.blocked_cells)
        )
        del self.blocked_cells[(br, bc)]
        return valid

    def reposition_wall(self, player: int, new_wall: Tuple[int, int, str], old_wall: Optional[Tuple[int, int, str]] = None) -> bool:
        if not self._is_valid_wall_replacement(player, new_wall, old_wall=old_wall):
            return False
        if old_wall is None:
            old_wall = self.latest_wall[player]
        idx = self.walls[player].index(old_wall)
        self.walls[player][idx] = new_wall
        self.latest_wall[player] = new_wall

        self.board.remove_wall(old_wall[0], old_wall[1], old_wall[2])
        self.board.add_wall(new_wall[0], new_wall[1], new_wall[2])
        return True

    def _normalize_action(self, action: Any) -> Tuple[str, Any]:
        """Return a canonical (type, tuple) action or raise ValueError('Malformed action ...').
        Accepts tuples or lists (JSON). Performs no game-state checks."""
        if not isinstance(action, (tuple, list)) or isinstance(action, (str, bytes)):
            raise ValueError(f"Malformed action: expected 2-sequence (type, value), got {type(action).__name__}")
        if len(action) != 2:
            raise ValueError(f"Malformed action: expected 2-sequence (type, value), got length {len(action)}")

        act_type, act_val = action[0], action[1]
        if not isinstance(act_type, str) or act_type not in self._ACTION_TYPES:
            raise ValueError(f"Unknown action type {act_type!r}.")

        if act_type in ('MOVE', 'HOP', 'BLOCK'):
            if not isinstance(act_val, (tuple, list)) or isinstance(act_val, (str, bytes)):
                raise ValueError(f"Malformed {act_type} action value: expected 2-sequence, got {type(act_val).__name__}")
            if len(act_val) != 2:
                raise ValueError(f"Malformed {act_type} action value: expected (r, c), got length {len(act_val)}")
            r, c = act_val[0], act_val[1]
            if not (self._is_int(r) and self._is_int(c)):
                raise ValueError(f"Malformed {act_type} action coordinates: expected integers, got ({r!r}, {c!r})")
            return (act_type, (int(r), int(c)))

        elif act_type == 'WALL':
            if not isinstance(act_val, (tuple, list)) or isinstance(act_val, (str, bytes)):
                raise ValueError(f"Malformed WALL action value: expected 3-sequence, got {type(act_val).__name__}")
            if len(act_val) != 3:
                raise ValueError(f"Malformed WALL action value: expected (r, c, orient), got length {len(act_val)}")
            r, c, orient = act_val[0], act_val[1], act_val[2]
            if not (self._is_int(r) and self._is_int(c)):
                raise ValueError(f"Malformed WALL action coordinates: expected integers, got ({r!r}, {c!r})")
            if orient not in ('H', 'V'):
                raise ValueError(f"Malformed WALL action orientation: expected 'H' or 'V', got {orient!r}")
            return ('WALL', (int(r), int(c), str(orient)))

        elif act_type == 'REPLACE':
            if not isinstance(act_val, (tuple, list)) or isinstance(act_val, (str, bytes)):
                raise ValueError(f"Malformed REPLACE action value: expected wall triple or pair, got {type(act_val).__name__}")
            if len(act_val) == 2 and isinstance(act_val[0], (tuple, list)) and isinstance(act_val[1], (tuple, list)):
                w1, w2 = act_val[0], act_val[1]
                if (
                    len(w1) != 3
                    or not (self._is_int(w1[0]) and self._is_int(w1[1]) and w1[2] in ('H', 'V'))
                ):
                    raise ValueError(f"Malformed REPLACE old wall triple: {w1!r}")
                if (
                    len(w2) != 3
                    or not (self._is_int(w2[0]) and self._is_int(w2[1]) and w2[2] in ('H', 'V'))
                ):
                    raise ValueError(f"Malformed REPLACE new wall triple: {w2!r}")
                return ('REPLACE', ((int(w1[0]), int(w1[1]), str(w1[2])), (int(w2[0]), int(w2[1]), str(w2[2]))))
            elif (
                len(act_val) == 3
                and self._is_int(act_val[0])
                and self._is_int(act_val[1])
                and act_val[2] in ('H', 'V')
            ):
                return ('REPLACE', (int(act_val[0]), int(act_val[1]), str(act_val[2])))
            else:
                raise ValueError(f"Malformed REPLACE action value: expected wall triple or pair, got {act_val!r}")

        raise ValueError(f"Unknown action type {act_type!r}.")

    def _validate_action(self, action: Tuple[str, Any], player: int) -> None:
        """Raise ValueError if `action` is illegal for `player`. Must leave all state unchanged
        (the _is_valid_* helpers mutate temporarily and restore)."""
        act_type, act_val = action
        if act_type == 'MOVE':
            if not self._is_valid_move(player, act_val):
                raise ValueError(f"Illegal move to {act_val}.")
        elif act_type == 'HOP':
            if not self._is_valid_hop(player, act_val):
                raise ValueError(f"Illegal hop to {act_val}.")
        elif act_type == 'WALL':
            if self.wall_inventory[player] <= 0:
                raise ValueError(f"Player {player} has no walls remaining in inventory.")
            r, c, orient = act_val
            if not self._is_valid_wall_placement(player, r, c, orient):
                raise ValueError(f"Illegal wall placement at {act_val}.")
        elif act_type == 'REPLACE':
            if not self.rules['A']:
                raise ValueError("Reusable wall mechanic (Rule A) is disabled.")
            if self.latest_wall[player] is None:
                raise ValueError(f"Player {player} has not placed any wall yet to reposition.")

            if len(act_val) == 2 and isinstance(act_val[0], tuple) and isinstance(act_val[1], tuple):
                old_w, new_w = act_val
                if old_w != self.latest_wall[player]:
                    raise ValueError(f"Cannot reposition older wall {old_w}: only latest wall {self.latest_wall[player]} may be repositioned.")
            else:
                new_w = act_val
                old_w = self.latest_wall[player]

            if not self._is_valid_wall_replacement(player, new_w, old_wall=old_w):
                raise ValueError(f"Illegal wall replacement {old_w} -> {new_w}: collision or path preservation violated.")
        elif act_type == 'BLOCK':
            if not (self.rules['C'] and self.block_available[player]):
                raise ValueError(f"Player {player} has no cell block available.")
            if not self._is_valid_cell_block(act_val[0], act_val[1]):
                raise ValueError(f"Illegal cell block at {act_val}: path preservation violated.")
        else:
            raise ValueError(f"Unknown action type {act_type!r}.")

    def _mutate(self, action: Tuple[str, Any], player: int) -> None:
        """Apply a validated action: pawn/wall/inventory/hop/block changes only."""
        act_type, act_val = action
        if act_type == 'MOVE':
            self.pawns[player] = act_val
        elif act_type == 'HOP':
            self.pawns[player] = act_val
            self.hop_available[player] = False
        elif act_type == 'WALL':
            r, c, orient = act_val
            self.walls[player].append(act_val)
            self.latest_wall[player] = act_val
            self.wall_inventory[player] -= 1
            self.board.add_wall(r, c, orient)
        elif act_type == 'REPLACE':
            if len(act_val) == 2 and isinstance(act_val[0], tuple) and isinstance(act_val[1], tuple):
                old_w, new_w = act_val
            else:
                new_w = act_val
                old_w = self.latest_wall[player]
            idx = self.walls[player].index(old_w)
            self.walls[player][idx] = new_w
            self.latest_wall[player] = new_w
            self.board.remove_wall(old_w[0], old_w[1], old_w[2])
            self.board.add_wall(new_w[0], new_w[1], new_w[2])
        elif act_type == 'BLOCK':
            self.block_available[player] = False

    def apply_action(self, action: Tuple[str, Any], update_clock: bool = True):
        """Apply an action following strict order:
        terminal check -> normalise -> validate -> clock -> mutate -> goal -> timers -> turn -> truncation -> no-legal draw -> repetition.
        """
        if self.is_terminal():
            raise ValueError("Game is already over; no further actions may be applied.")
        action = self._normalize_action(action)
        player = self.current_player
        self._validate_action(action, player)
        if update_clock:
            self.update_clock()
            if self.winner is not None:
                return self.get_state(), self.winner, self.is_draw
        self._mutate(action, player)

        # Win condition check
        if self.pawns[player][0] == self.goals[player]:
            self.winner = player
            self.terminal_reason = 'goal'

        # Decrement block timers
        to_remove = []
        for cell in self.blocked_cells:
            self.blocked_cells[cell] -= 1
            if self.blocked_cells[cell] <= 0:
                to_remove.append(cell)
        for cell in to_remove:
            del self.blocked_cells[cell]
        if action[0] == 'BLOCK':
            self.blocked_cells[action[1]] = self.rules['K']

        # Turn transition
        self.current_player = 1 - self.current_player
        self.turn_count += 1
        if update_clock:
            self.last_turn_timestamp = time.time()

        # External ply limit truncation check (P3.7)
        if self.max_plies is not None and self.turn_count >= self.max_plies and not self.is_terminal():
            self.is_draw = True
            self.is_truncated = True
            self.terminal_reason = 'truncated'

        # Draw check on no legal actions (R3)
        if self.winner is None and not self.is_terminal() and not self.has_legal_action(self.current_player):
            self.is_draw = True
            self.terminal_reason = 'no_legal_actions'

        # Record repetition history
        self._record_state_history()

        return self.get_state(), self.winner, self.is_draw
