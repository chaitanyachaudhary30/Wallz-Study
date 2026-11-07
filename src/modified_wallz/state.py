"""State representation and board layout for Modified Wallz.

Note on Architecture (Decision 12):
The engine maintains state via the high-performance `WallzEngine` facade with
canonical `position_key()`, O(1) copy-on-write ChainMap history, and `restore_state_key()`,
avoiding redundant wrapper object allocations during search and graph exploration.
"""
from src.modified_wallz.board import WallzBoard

__all__ = ['WallzBoard']
