"""Observation encoder for Modified Wallz (Observation v1).

Encodes game state into a 14-plane canonical observation tensor of shape (14, N, N).
All planes are encoded from the perspective of the active mover.
"""
from __future__ import annotations

OBS_VERSION = 1

try:
    import numpy as np
except ImportError:
    np = None


def encode_observation(engine, mover: int | None = None):
    """Encodes the current engine state into a canonical float32 tensor of shape (14, N, N).

    Channels:
        0: own pawn (1.0 at canonical mover pawn coordinates)
        1: opponent pawn (1.0 at canonical opponent pawn coordinates)
        2: H-wall centres (1.0 at canonical wall slot (wr_c, wc_c) for any active 'H' wall)
        3: V-wall centres (1.0 at canonical wall slot (wr_c, wc_c) for any active 'V' wall)
        4: own replaceable wall r_self (1.0 at canonical slot of mover's latest wall)
        5: opponent replaceable wall r_opp (1.0 at canonical slot of opponent's latest wall)
        6: block timer tau / K (value in [0.0, 1.0] at canonical cell coordinates)
        7: own wall inventory m_self / M (constant plane)
        8: opponent wall inventory m_opp / M (constant plane)
        9: own hop availability (constant plane: 1.0 if available else 0.0)
        10: opponent hop availability (constant plane: 1.0 if available else 0.0)
        11: own block availability (constant plane: 1.0 if available else 0.0)
        12: opponent block availability (constant plane: 1.0 if available else 0.0)
        13: constant boundary indicator plane (filled with 1.0)
    """
    if mover is None:
        mover = engine.current_player

    n = engine.board_size
    opp = 1 - mover

    # Initialize 14xNxN array (numpy if available, else nested list)
    if np is not None:
        planes = np.zeros((14, n, n), dtype=np.float32)
    else:
        planes = [[[0.0] * n for _ in range(n)] for _ in range(14)]

    def set_val(ch, r, c, val):
        planes[ch][r][c] = float(val)

    # 0 & 1: Pawn positions
    p_self_r, p_self_c = engine.pawns[mover]
    p_opp_r, p_opp_c = engine.pawns[opp]

    self_r_canon = p_self_r if mover == 0 else (n - 1 - p_self_r)
    opp_r_canon = p_opp_r if mover == 0 else (n - 1 - p_opp_r)

    set_val(0, self_r_canon, p_self_c, 1.0)
    set_val(1, opp_r_canon, p_opp_c, 1.0)

    # 2 & 3: Active walls (centres across both players)
    for p in (0, 1):
        for wr, wc, orient in engine.walls[p]:
            wr_canon = wr if mover == 0 else (n - 2 - wr)
            wc_canon = wc
            ch = 2 if orient == 'H' else 3
            set_val(ch, wr_canon, wc_canon, 1.0)

    # 4 & 5: Replaceable walls
    latest_self = engine.latest_wall[mover]
    if latest_self is not None:
        wr, wc, _ = latest_self
        wr_canon = wr if mover == 0 else (n - 2 - wr)
        set_val(4, wr_canon, wc, 1.0)

    latest_opp = engine.latest_wall[opp]
    if latest_opp is not None:
        wr, wc, _ = latest_opp
        wr_canon = wr if mover == 0 else (n - 2 - wr)
        set_val(5, wr_canon, wc, 1.0)

    # 6: Block timers (tau / K)
    k_dur = engine.rules.get('K', 3)
    for (br, bc), rem in engine.blocked_cells.items():
        br_canon = br if mover == 0 else (n - 1 - br)
        ratio = float(rem) / float(k_dur) if k_dur > 0 else 1.0
        set_val(6, br_canon, bc, ratio)

    # 7 & 8: Wall inventory (m / M)
    m_max = engine.rules.get('M', 10)
    self_inv_ratio = float(engine.wall_inventory[mover]) / float(m_max) if m_max > 0 else 0.0
    opp_inv_ratio = float(engine.wall_inventory[opp]) / float(m_max) if m_max > 0 else 0.0

    # 9..12: Hop and block flags
    h_self = 1.0 if engine.hop_available[mover] else 0.0
    h_opp = 1.0 if engine.hop_available[opp] else 0.0
    b_self = 1.0 if engine.block_available[mover] else 0.0
    b_opp = 1.0 if engine.block_available[opp] else 0.0

    # Fill scalar planes 7..13
    if np is not None:
        planes[7].fill(self_inv_ratio)
        planes[8].fill(opp_inv_ratio)
        planes[9].fill(h_self)
        planes[10].fill(h_opp)
        planes[11].fill(b_self)
        planes[12].fill(b_opp)
        planes[13].fill(1.0)
    else:
        for r in range(n):
            for c in range(n):
                planes[7][r][c] = self_inv_ratio
                planes[8][r][c] = opp_inv_ratio
                planes[9][r][c] = h_self
                planes[10][r][c] = h_opp
                planes[11][r][c] = b_self
                planes[12][r][c] = b_opp
                planes[13][r][c] = 1.0

    return planes


def mirror_observation(obs):
    """Mirrors an observation tensor of shape (14, N, N) horizontally.

    Cell planes (0, 1, 6) flip columns c -> N - 1 - c.
    Wall planes (2, 3, 4, 5) flip wall slot columns c -> N - 2 - c within [0, N-2],
    keeping column N - 1 zero.
    Constant planes (7..13) remain unchanged.
    """
    if np is not None and isinstance(obs, np.ndarray):
        m = obs.copy()
        n = obs.shape[-1]
        # Pawn positions & blocked cells: full horizontal flip
        for ch in (0, 1, 6):
            m[ch] = np.fliplr(m[ch])
        # Wall planes: flip columns 0..N-2, column N-1 stays zero
        for ch in (2, 3, 4, 5):
            m[ch, :n - 1, :n - 1] = np.fliplr(m[ch, :n - 1, :n - 1])
        return m
    else:
        n = len(obs[0])
        # Deep copy
        m = [[[obs[ch][r][c] for c in range(n)] for r in range(n)] for ch in range(14)]
        # Pawn positions & blocked cells: full horizontal flip
        for ch in (0, 1, 6):
            for r in range(n):
                m[ch][r] = list(reversed(m[ch][r]))
        # Wall planes: flip columns 0..N-2
        for ch in (2, 3, 4, 5):
            for r in range(n - 1):
                row_slice = list(reversed(m[ch][r][:n - 1]))
                for c in range(n - 1):
                    m[ch][r][c] = row_slice[c]
                m[ch][r][n - 1] = 0.0
        return m
