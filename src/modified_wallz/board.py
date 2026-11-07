"""Board Grid and Fast Edge-Blocked Geometry (P3.4 / C2).

Provides O(1) wall collision, overlap, crossing checks and edge-blocked lookups
using preallocated 2D grid arrays.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Set, Tuple

WallTuple = Tuple[int, int, str]
CellTuple = Tuple[int, int]


class WallzBoard:
    """Maintains edge-blocked arrays and centre grid for O(1) queries."""

    __slots__ = ('board_size', 'hblock', 'vblock', 'centre', 'walls')

    def __init__(self, board_size: int = 9):
        self.board_size = board_size
        n = board_size
        # hblock[r][c]: True if horizontal edge between (r, c) and (r+1, c) is blocked
        self.hblock: List[List[bool]] = [[False] * n for _ in range(n - 1)]
        # vblock[r][c]: True if vertical edge between (r, c) and (r, c+1) is blocked
        self.vblock: List[List[bool]] = [[False] * (n - 1) for _ in range(n)]
        # centre[r][c]: 'H', 'V', or None
        self.centre: List[List[Optional[str]]] = [[None] * (n - 1) for _ in range(n - 1)]
        self.walls: Set[WallTuple] = set()

    def clone(self) -> WallzBoard:
        """Fast clone of board state."""
        b = WallzBoard.__new__(WallzBoard)
        b.board_size = self.board_size
        b.hblock = [row[:] for row in self.hblock]
        b.vblock = [row[:] for row in self.vblock]
        b.centre = [row[:] for row in self.centre]
        b.walls = set(self.walls)
        return b

    def clear(self) -> None:
        """Clears all walls from the board."""
        n = self.board_size
        for r in range(n - 1):
            for c in range(n):
                self.hblock[r][c] = False
        for r in range(n):
            for c in range(n - 1):
                self.vblock[r][c] = False
        for r in range(n - 1):
            for c in range(n - 1):
                self.centre[r][c] = None
        self.walls.clear()

    def add_wall(self, r: int, c: int, orient: str) -> None:
        """Adds a wall to the board and updates blocked edges in O(1)."""
        w = (r, c, orient)
        self.walls.add(w)
        self.centre[r][c] = orient
        if orient == 'H':
            self.hblock[r][c] = True
            self.hblock[r][c + 1] = True
        else:  # 'V'
            self.vblock[r][c] = True
            self.vblock[r + 1][c] = True

    def remove_wall(self, r: int, c: int, orient: str) -> None:
        """Removes a wall from the board and updates blocked edges in O(1)."""
        w = (r, c, orient)
        self.walls.discard(w)
        self.centre[r][c] = None
        if orient == 'H':
            self.hblock[r][c] = False
            self.hblock[r][c + 1] = False
        else:  # 'V'
            self.vblock[r][c] = False
            self.vblock[r + 1][c] = False

    def is_edge_blocked(self, cell1: CellTuple, cell2: CellTuple) -> bool:
        """Returns True if a wall blocks the orthogonal edge between cell1 and cell2 in O(1)."""
        r1, c1 = cell1
        r2, c2 = cell2
        if r1 == r2:
            min_c = c1 if c1 < c2 else c2
            return self.vblock[r1][min_c]
        elif c1 == c2:
            min_r = r1 if r1 < r2 else r2
            return self.hblock[min_r][c1]
        return False

    def can_place_wall(self, r: int, c: int, orient: str) -> bool:
        """Checks bounds, crossing and overlapping in O(1). Does not check path preservation."""
        n_m1 = self.board_size - 1
        if not (0 <= r < n_m1 and 0 <= c < n_m1):
            return False
        if self.centre[r][c] is not None:
            return False  # Existing wall or cross at same centre
        if orient == 'H':
            if c > 0 and self.centre[r][c - 1] == 'H':
                return False
            if c < n_m1 - 1 and self.centre[r][c + 1] == 'H':
                return False
            return True
        elif orient == 'V':
            if r > 0 and self.centre[r - 1][c] == 'V':
                return False
            if r < n_m1 - 1 and self.centre[r + 1][c] == 'V':
                return False
            return True
        return False

    def wall_edges(self, r: int, c: int, orient: str) -> List[Tuple[CellTuple, CellTuple]]:
        """Returns the two orthogonal cell edges blocked by a wall at (r, c, orient)."""
        if orient == 'H':
            return [
                ((r, c), (r + 1, c)),
                ((r, c + 1), (r + 1, c + 1)),
            ]
        else:
            return [
                ((r, c), (r, c + 1)),
                ((r + 1, c), (r + 1, c + 1)),
            ]

    def rebuild(self, walls: Iterable[WallTuple]) -> None:
        """Rebuilds the board from an iterable of walls."""
        self.clear()
        for r, c, orient in walls:
            self.add_wall(r, c, orient)
