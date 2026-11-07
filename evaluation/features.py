"""Feature Extraction for State Evaluation and Feature Studies (P6.2 / X-5).

Extracts spatial, tempo, and resource features (f1-f6) from engine states:
    f1: Distance advantage H = d_opp - d_self
    f2: Tempo (+1 if player_id is to move, else -1)
    f3: Wall inventory advantage = m_self - m_opp
    f4: Hop availability advantage = h_self - h_opp
    f5: Block availability advantage = b_self - b_opp
    f6: Active replaceable wall advantage = [r_self != None] - [r_opp != None]
"""
from __future__ import annotations

import math
from typing import List
import numpy as np


FEATURE_NAMES = [
    'f1_distance_advantage',
    'f2_tempo',
    'f3_wall_inventory_adv',
    'f4_hop_availability_adv',
    'f5_block_availability_adv',
    'f6_replaceable_wall_adv',
]


def extract_features(engine, player_id: int) -> np.ndarray:
    """
    Extracts the 6 engineered features from engine relative to player_id.
    Returns float64 numpy array of shape (6,).
    """
    opp = 1 - player_id

    # f1: Distance advantage H = d_opp - d_self
    d_self = engine.shortest_path(player_id)
    d_opp = engine.shortest_path(opp)
    if math.isinf(d_self) or math.isinf(d_opp):
        f1 = 0.0
    else:
        f1 = float(d_opp - d_self)

    # f2: Tempo (+1 if player_id is to move, else -1)
    f2 = 1.0 if engine.current_player == player_id else -1.0

    # f3: Wall inventory advantage
    m_self = float(engine.wall_inventory[player_id])
    m_opp = float(engine.wall_inventory[opp])
    f3 = m_self - m_opp

    # f4: Hop availability advantage
    h_self = 1.0 if engine.hop_available[player_id] else 0.0
    h_opp = 1.0 if engine.hop_available[opp] else 0.0
    f4 = h_self - h_opp

    # f5: Block availability advantage
    b_self = 1.0 if engine.block_available[player_id] else 0.0
    b_opp = 1.0 if engine.block_available[opp] else 0.0
    f5 = b_self - b_opp

    # f6: Active replaceable wall advantage
    r_self = 1.0 if engine.latest_wall[player_id] is not None else 0.0
    r_opp = 1.0 if engine.latest_wall[opp] is not None else 0.0
    f6 = r_self - r_opp

    return np.array([f1, f2, f3, f4, f5, f6], dtype=np.float64)
