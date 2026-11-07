"""Shortest Path and Path Preservation Algorithms (P3.5 / C3).

Implements BFS distance, shortest path extraction (edges and vertices),
and the exact shortest-path filter for WALL, REPLACE and BLOCK legality.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Set, Tuple

CellTuple = Tuple[int, int]
UndirectedEdge = Tuple[CellTuple, CellTuple]


def norm_edge(u: CellTuple, v: CellTuple) -> UndirectedEdge:
    """Canonical undirected edge representation."""
    return (u, v) if u < v else (v, u)


def bfs_distance_and_path(
    start: CellTuple,
    target_row: int,
    board_size: int,
    board,
    blocked_cells: Dict[CellTuple, int] | Set[CellTuple],
) -> Tuple[float, Set[UndirectedEdge], Set[CellTuple]]:
    """
    Computes shortest path distance to target_row, plus the set of edges
    and vertices traversed along one shortest path.
    Under R7, all active blocked cells are treated as impassable vertices.
    """
    queue = deque([start])
    parent: Dict[CellTuple, Optional[CellTuple]] = {start: None}
    target_cell: Optional[CellTuple] = None

    while queue:
        curr = queue.popleft()
        r, c = curr
        if r == target_row:
            target_cell = curr
            break

        # Check 4 orthogonal neighbors
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < board_size and 0 <= nc < board_size:
                nxt = (nr, nc)
                if nxt not in parent and nxt not in blocked_cells:
                    if not board.is_edge_blocked(curr, nxt):
                        parent[nxt] = curr
                        queue.append(nxt)

    if target_cell is None:
        return float('inf'), set(), set()

    # Reconstruct shortest path edges and vertices
    edges: Set[UndirectedEdge] = set()
    vertices: Set[CellTuple] = set()
    curr = target_cell
    dist = 0

    while curr is not None:
        vertices.add(curr)
        p = parent[curr]
        if p is not None:
            edges.add(norm_edge(p, curr))
            dist += 1
        curr = p

    return float(dist), edges, vertices


def has_path_to_goal(
    start: CellTuple,
    target_row: int,
    board_size: int,
    board,
    blocked_cells: Dict[CellTuple, int] | Set[CellTuple],
) -> bool:
    """Checks reachability of target_row via early-exit BFS."""
    if start[0] == target_row:
        return True
    queue = deque([start])
    visited: Set[CellTuple] = {start}

    while queue:
        r, c = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < board_size and 0 <= nc < board_size:
                nxt = (nr, nc)
                if nxt not in visited and nxt not in blocked_cells:
                    if not board.is_edge_blocked((r, c), nxt):
                        if nr == target_row:
                            return True
                        visited.add(nxt)
                        queue.append(nxt)
    return False
