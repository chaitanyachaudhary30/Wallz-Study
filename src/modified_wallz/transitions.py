"""Transition helpers and pathfinding routines for Modified Wallz.

Re-exports core shortest path and reachability algorithms.
"""
from src.modified_wallz.pathfinding import (
    bfs_distance,
    bfs_distance_and_path,
    has_path_to_goal,
    norm_edge,
    shortest_path_edges,
)

__all__ = [
    'bfs_distance',
    'bfs_distance_and_path',
    'has_path_to_goal',
    'norm_edge',
    'shortest_path_edges',
]
