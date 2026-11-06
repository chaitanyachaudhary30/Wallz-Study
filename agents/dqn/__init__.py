"""DQN and Self-Play Package (Phase 1)."""
try:
    from agents.dqn.encoding import encode_observation, mirror_observation, OBS_VERSION
    from agents.dqn.network import QNetwork
    from agents.dqn.replay_buffer import ReplayBuffer
    from agents.dqn.trainer import DQNTrainer, DQNConfig
    from agents.dqn.self_play import SnapshotPool, SelfPlayManager
    from agents.dqn.agent import DQNAgent, save_checkpoint, load_checkpoint
except ImportError:
    pass

__all__ = [
    'encode_observation',
    'mirror_observation',
    'OBS_VERSION',
    'QNetwork',
    'ReplayBuffer',
    'DQNTrainer',
    'DQNConfig',
    'SnapshotPool',
    'SelfPlayManager',
    'DQNAgent',
    'save_checkpoint',
    'load_checkpoint',
]
