"""DQNAgent and Checkpoint Management (P1.9 / C6 / C9).

Implements:
- DQNAgent: Evaluation-ready agent wrapper conforming to Agent Protocol (C6).
  Deterministic lowest-ID argmax tie-breaking, masked action selection, stats() protocol.
  IMPORTANT: Does NOT define get_stats(), preventing collision with Minimax node count in runner.
- save_checkpoint / load_checkpoint: Standard checkpoint format (C9) with meta.json validation.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from typing import Optional, Union
import numpy as np

try:
    import torch
except ImportError:
    torch = None

from agents.dqn.encoding import encode_observation, OBS_VERSION
from agents.dqn.network import QNetwork
try:
    from modified_wallz.actions import ActionCodec, CODEC_VERSION
except ImportError:
    from src.modified_wallz.actions import ActionCodec, CODEC_VERSION


def get_git_commit() -> str:
    """Attempts to retrieve the current git commit short hash."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


class DQNAgent:
    """Evaluation agent powered by a trained DQN QNetwork."""

    def __init__(
        self,
        player_id: int,
        model: QNetwork,
        board_size: int,
        rules: dict,
        codec_version: int = CODEC_VERSION,
        obs_version: int = OBS_VERSION,
        device: str = "cpu",
    ):
        self.player_id = player_id
        self.model = model
        self.board_size = board_size
        self.rules = rules
        self.codec_version = codec_version
        self.obs_version = obs_version
        self.device = torch.device(device) if torch is not None else None

        if self.model is not None and self.device is not None:
            self.model.to(self.device)
            self.model.eval()

        self.codec = ActionCodec(board_size)
        self._inferences = 0

    def select_action(self, engine) -> Optional[tuple]:
        """Selects legal action greedily via QNetwork with lowest-ID tie-breaking."""
        if engine.board_size != self.board_size:
            raise ValueError(
                f"Board size mismatch: Agent configured for {self.board_size}, "
                f"engine is {engine.board_size}"
            )

        legal_mask = self.codec.legal_mask(engine, mover=self.player_id)
        legal_indices = [i for i, ok in enumerate(legal_mask) if ok]
        if not legal_indices:
            return None

        # Encode observation canonically for self.player_id
        obs = encode_observation(engine, mover=self.player_id)

        # Forward pass through Q-network
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_vals = self.model(x).squeeze(0).cpu().numpy()

        # Mask illegal actions with -inf
        masked_q = np.full_like(q_vals, -np.inf)
        masked_q[legal_indices] = q_vals[legal_indices]

        # Greedy choice with deterministic tie-breaking (lowest index)
        best_val = -np.inf
        best_id = legal_indices[0]
        for aid in legal_indices:
            val = masked_q[aid]
            if val > best_val:
                best_val = val
                best_id = aid

        self._inferences += 1
        return self.codec.decode(best_id, engine, mover=self.player_id)

    def stats(self) -> dict:
        """Returns inference statistics (NOT get_stats)."""
        return {
            'inferences': self._inferences,
            'board_size': self.board_size,
            'codec_version': self.codec_version,
            'obs_version': self.obs_version,
        }


def save_checkpoint(
    save_dir: Union[str, Path],
    model: QNetwork,
    meta: dict,
    filename: str = "model.pt",
) -> Path:
    """Saves torch model state_dict and meta.json metadata (C9 contract).

    Args:
        save_dir: Target directory path.
        model: Trained QNetwork.
        meta: Metadata dictionary containing {N, rules, codec_version, obs_version,
              gamma, double, dueling, commit, seed, step}.
        filename: Filename for the PyTorch weights (default: 'model.pt').

    Returns:
        Path to the saved weights file.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    weights_file = save_path / filename
    torch.save(model.state_dict(), weights_file)

    meta_payload = dict(meta)
    meta_payload.setdefault('codec_version', CODEC_VERSION)
    meta_payload.setdefault('obs_version', OBS_VERSION)
    if 'commit' not in meta_payload or not meta_payload['commit']:
        meta_payload['commit'] = get_git_commit()

    meta_file = save_path / "meta.json"
    with open(meta_file, 'w', encoding='utf-8') as f:
        json.dump(meta_payload, f, indent=2, sort_keys=True)

    return weights_file


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    player_id: int = 0,
    expected_board_size: Optional[int] = None,
    expected_rules: Optional[dict] = None,
    device: str = "cpu",
) -> DQNAgent:
    """Loads a DQNAgent from a checkpoint path and validates metadata (C9 contract).

    Args:
        checkpoint_path: Path to either the checkpoint directory or a specific .pt file.
        player_id: Target player seat (0 or 1).
        expected_board_size: If specified, validates match against meta['N'].
        expected_rules: If specified, validates match against meta['rules'].
        device: Device to load model onto ('cpu' or 'cuda').

    Returns:
        Configured DQNAgent with loaded weights in eval mode.
    """
    path = Path(checkpoint_path)
    if path.is_file():
        weights_file = path
        meta_file = path.parent / "meta.json"
    elif path.is_dir():
        # Look for model.pt, final.pt, or step_*.pt
        candidates = ["final.pt", "model.pt"]
        weights_file = None
        for c in candidates:
            if (path / c).exists():
                weights_file = path / c
                break
        if weights_file is None:
            pt_files = sorted(path.glob("*.pt"))
            if pt_files:
                weights_file = pt_files[-1]
            else:
                raise FileNotFoundError(f"No .pt weights file found in directory {path}")
        meta_file = path / "meta.json"
    else:
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")

    if not meta_file.exists():
        raise FileNotFoundError(f"Metadata file meta.json not found at {meta_file}")

    with open(meta_file, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    # 1. Validate Codec and Observation versions
    if meta.get('codec_version') != CODEC_VERSION:
        raise ValueError(
            f"Codec version mismatch: checkpoint has {meta.get('codec_version')}, "
            f"expected {CODEC_VERSION}"
        )
    if meta.get('obs_version') != OBS_VERSION:
        raise ValueError(
            f"Observation version mismatch: checkpoint has {meta.get('obs_version')}, "
            f"expected {OBS_VERSION}"
        )

    # 2. Validate Board Size
    n = meta.get('N')
    if expected_board_size is not None and n != expected_board_size:
        raise ValueError(
            f"Board size mismatch: checkpoint trained on N={n}, "
            f"expected N={expected_board_size}"
        )

    # 3. Validate Rules
    ckpt_rules = meta.get('rules', {})
    if expected_rules is not None:
        for key in ('A', 'B', 'C', 'K', 'M'):
            if key in expected_rules and key in ckpt_rules:
                if expected_rules[key] != ckpt_rules[key]:
                    raise ValueError(
                        f"Rules mismatch on key {key}: checkpoint has {ckpt_rules[key]}, "
                        f"expected {expected_rules[key]}"
                    )

    # 4. Reconstruct QNetwork
    num_actions = ActionCodec.get_num_actions(n)
    model = QNetwork(
        board_size=n,
        num_actions=num_actions,
        channels=meta.get('channels', 64),
        blocks=meta.get('blocks', 3),
        hidden=meta.get('hidden', 256),
        dueling=meta.get('dueling', False),
    )

    state_dict = torch.load(weights_file, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    return DQNAgent(
        player_id=player_id,
        model=model,
        board_size=n,
        rules=ckpt_rules,
        codec_version=meta.get('codec_version', CODEC_VERSION),
        obs_version=meta.get('obs_version', OBS_VERSION),
        device=device,
    )
