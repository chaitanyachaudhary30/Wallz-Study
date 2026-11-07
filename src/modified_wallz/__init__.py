from .config import WallzConfig
from .board import WallzBoard
from .pathfinding import bfs_distance_and_path, has_path_to_goal
from .actions import ActionCodec, CODEC_VERSION

try:
    from .environment import WallzEnv
except ImportError:
    WallzEnv = None


def __getattr__(name: str):
    if name == 'WallzEngine':
        from wallz_engine import WallzEngine
        return WallzEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ['WallzEngine', 'WallzConfig', 'WallzBoard', 'ActionCodec', 'CODEC_VERSION', 'WallzEnv']
