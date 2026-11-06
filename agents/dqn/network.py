"""Q-Network architecture for Modified Wallz DQN (P1.5).

Implements convolutional neural network with optional Dueling heads.
"""
from __future__ import annotations

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None


class QNetwork(nn.Module if nn is not None else object):
    """Convolutional Q-Network for Modified Wallz with optional Dueling streams."""

    def __init__(
        self,
        board_size: int,
        num_actions: int,
        channels: int = 64,
        blocks: int = 3,
        hidden: int = 256,
        dueling: bool = False,
    ):
        if nn is None:
            raise ImportError("PyTorch is required to instantiate QNetwork.")
        super().__init__()

        self.board_size = board_size
        self.num_actions = num_actions
        self.dueling = dueling
        self.channels = channels
        self.blocks = blocks
        self.hidden = hidden

        # 1. Feature extractor: 14 -> channels -> (blocks convs)
        conv_layers = [
            nn.Conv2d(14, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]
        for _ in range(blocks):
            conv_layers.extend([
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            ])
        self.conv = nn.Sequential(*conv_layers)

        # 2. Dense head
        flat_dim = channels * board_size * board_size
        self.fc = nn.Sequential(
            nn.Linear(flat_dim, hidden),
            nn.ReLU(inplace=True),
        )

        # 3. Output streams
        if dueling:
            self.v_head = nn.Linear(hidden, 1)
            self.a_head = nn.Linear(hidden, num_actions)
        else:
            self.out_head = nn.Linear(hidden, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (B, 14, N, N) representing canonical observations.

        Returns:
            Q-values of shape (B, num_actions).
        """
        h = self.conv(x)
        h = h.flatten(start_dim=1)
        feat = self.fc(h)

        if self.dueling:
            v = self.v_head(feat)  # (B, 1)
            a = self.a_head(feat)  # (B, num_actions)
            return v + a - a.mean(dim=-1, keepdim=True)
        else:
            return self.out_head(feat)
