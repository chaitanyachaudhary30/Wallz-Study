"""Action encoding and decoding for Modified Wallz (ActionCodec v1).

Implements canonical action space representation, legal action masking,
and mirror symmetry permutations across odd board sizes.
"""
from __future__ import annotations

CODEC_VERSION = 1

# Canonical direction deltas (dr_canon, dc_canon): FWD, BACK, RIGHT, LEFT
# In canonical frame, moving FWD always advances towards row N - 1.
CANONICAL_DIRECTIONS = [
    (1, 0),   # 0: FWD
    (-1, 0),  # 1: BACK
    (0, 1),   # 2: RIGHT
    (0, -1),  # 3: LEFT
]

CANONICAL_HOP_DIRECTIONS = [
    (2, 0),   # 4: HOP FWD
    (-2, 0),  # 5: HOP BACK
    (0, 2),   # 6: HOP RIGHT
    (0, -2),  # 7: HOP LEFT
]


class ActionCodec:
    """ActionCodec v1 converting between engine action tuples and discrete integer IDs."""

    def __init__(self, board_size: int):
        self.board_size = board_size
        self.num_actions = self.get_num_actions(board_size)
        self._n_wall_slots = (board_size - 1) * (board_size - 1)
        self._wall_offset = 8
        self._replace_offset = 8 + 2 * self._n_wall_slots
        self._block_offset = 8 + 4 * self._n_wall_slots

    @staticmethod
    def get_num_actions(board_size: int) -> int:
        """Returns total distinct actions |A| = 8 + 4*(N-1)^2 + N^2."""
        return 8 + 4 * (board_size - 1) ** 2 + board_size ** 2

    def encode(self, action: tuple, engine, mover: int | None = None) -> int:
        """Encodes an engine action tuple into a canonical discrete integer action ID.

        Args:
            action: Action tuple: ('MOVE', (r, c)), ('HOP', (r, c)),
                                  ('WALL', (wr, wc, o)), ('REPLACE', (wr, wc, o)),
                                  or ('BLOCK', (br, bc)).
            engine: WallzEngine instance.
            mover: Player index (0 or 1). If None, engine.current_player is used.

        Returns:
            Integer action ID in range [0, num_actions - 1].
        """
        if mover is None:
            mover = engine.current_player

        act_type, act_val = action
        n = self.board_size

        if act_type in ('MOVE', 'HOP'):
            tr, tc = act_val
            pr, pc = engine.pawns[mover]
            dr = tr - pr
            dc = tc - pc

            # Transform delta to canonical mover frame
            dr_canon = dr if mover == 0 else -dr
            dc_canon = dc

            if act_type == 'MOVE':
                for idx, (cd_r, cd_c) in enumerate(CANONICAL_DIRECTIONS):
                    if dr_canon == cd_r and dc_canon == cd_c:
                        return idx
                raise ValueError(f"Invalid MOVE step delta ({dr}, {dc}) from {engine.pawns[mover]} to {act_val}")
            else:
                for idx, (cd_r, cd_c) in enumerate(CANONICAL_HOP_DIRECTIONS):
                    if dr_canon == cd_r and dc_canon == cd_c:
                        return 4 + idx
                raise ValueError(f"Invalid HOP step delta ({dr}, {dc}) from {engine.pawns[mover]} to {act_val}")

        elif act_type in ('WALL', 'REPLACE'):
            wr, wc, orient = act_val
            if not (0 <= wr < n - 1 and 0 <= wc < n - 1):
                raise ValueError(f"Wall slot coordinates out of bounds: {act_val}")
            if orient not in ('H', 'V'):
                raise ValueError(f"Invalid wall orientation: {orient}")

            # Vertical flip for player 1: wr_canon = (N - 2) - wr
            wr_canon = wr if mover == 0 else (n - 2 - wr)
            wc_canon = wc
            slot_idx = wr_canon * (n - 1) + wc_canon
            orient_idx = 0 if orient == 'H' else 1

            base = self._wall_offset if act_type == 'WALL' else self._replace_offset
            return base + 2 * slot_idx + orient_idx

        elif act_type == 'BLOCK':
            br, bc = act_val
            if not (0 <= br < n and 0 <= bc < n):
                raise ValueError(f"Block coordinates out of bounds: {act_val}")

            br_canon = br if mover == 0 else (n - 1 - br)
            bc_canon = bc
            return self._block_offset + br_canon * n + bc_canon

        else:
            raise ValueError(f"Unknown action type: {act_type}")

    def decode(self, action_id: int, engine, mover: int | None = None) -> tuple:
        """Decodes a canonical discrete integer action ID into an engine action tuple.

        Args:
            action_id: Integer action ID in range [0, num_actions - 1].
            engine: WallzEngine instance.
            mover: Player index (0 or 1). If None, engine.current_player is used.

        Returns:
            Action tuple suitable for engine.apply_action.
        """
        if not (0 <= action_id < self.num_actions):
            raise ValueError(f"action_id {action_id} out of bounds for board_size {self.board_size}")

        if mover is None:
            mover = engine.current_player

        n = self.board_size

        if action_id < 4:
            # MOVE
            cd_r, cd_c = CANONICAL_DIRECTIONS[action_id]
            dr = cd_r if mover == 0 else -cd_r
            dc = cd_c
            pr, pc = engine.pawns[mover]
            return ('MOVE', (pr + dr, pc + dc))

        elif action_id < 8:
            # HOP
            cd_r, cd_c = CANONICAL_HOP_DIRECTIONS[action_id - 4]
            dr = cd_r if mover == 0 else -cd_r
            dc = cd_c
            pr, pc = engine.pawns[mover]
            return ('HOP', (pr + dr, pc + dc))

        elif action_id < self._replace_offset:
            # WALL
            offset = action_id - self._wall_offset
            orient_idx = offset % 2
            slot_idx = offset // 2
            wr_canon = slot_idx // (n - 1)
            wc_canon = slot_idx % (n - 1)

            wr = wr_canon if mover == 0 else (n - 2 - wr_canon)
            wc = wc_canon
            orient = 'H' if orient_idx == 0 else 'V'
            return ('WALL', (wr, wc, orient))

        elif action_id < self._block_offset:
            # REPLACE
            offset = action_id - self._replace_offset
            orient_idx = offset % 2
            slot_idx = offset // 2
            wr_canon = slot_idx // (n - 1)
            wc_canon = slot_idx % (n - 1)

            wr = wr_canon if mover == 0 else (n - 2 - wr_canon)
            wc = wc_canon
            orient = 'H' if orient_idx == 0 else 'V'
            return ('REPLACE', (wr, wc, orient))

        else:
            # BLOCK
            offset = action_id - self._block_offset
            br_canon = offset // n
            bc_canon = offset % n
            br = br_canon if mover == 0 else (n - 1 - br_canon)
            bc = bc_canon
            return ('BLOCK', (br, bc))

    def legal_mask(self, engine, mover: int | None = None) -> list[bool]:
        """Returns boolean mask of length num_actions where index a is True iff action a is legal."""
        if mover is None:
            mover = engine.current_player

        mask = [False] * self.num_actions
        legal = engine.get_legal_actions(mover)
        for act in legal:
            aid = self.encode(act, engine, mover=mover)
            mask[aid] = True
        return mask

    def legal_ids(self, engine, mover: int | None = None) -> list[int]:
        """Returns list of discrete action IDs that are legally permissible in engine."""
        if mover is None:
            mover = engine.current_player

        legal = engine.get_legal_actions(mover)
        return [self.encode(act, engine, mover=mover) for act in legal]

    def mirror_permutation(self) -> list[int]:
        """Returns permutation P mapping action_id -> mirrored_action_id under horizontal flip.

        Only valid when board_size is odd (mirror symmetry holds).
        """
        n = self.board_size
        if n % 2 == 0:
            raise ValueError(f"Mirror permutation is only valid for odd board_size, got {n}")

        perm = list(range(self.num_actions))

        # 1. MOVE: RIGHT (2) <-> LEFT (3)
        perm[2] = 3
        perm[3] = 2

        # 2. HOP: HOP RIGHT (6) <-> HOP LEFT (7)
        perm[6] = 7
        perm[7] = 6

        # 3. WALL & REPLACE: wc -> N - 2 - wc
        for base in (self._wall_offset, self._replace_offset):
            for wr in range(n - 1):
                for wc in range(n - 1):
                    for orient_idx in (0, 1):
                        orig_slot = wr * (n - 1) + wc
                        orig_id = base + 2 * orig_slot + orient_idx

                        mirr_wc = n - 2 - wc
                        mirr_slot = wr * (n - 1) + mirr_wc
                        mirr_id = base + 2 * mirr_slot + orient_idx

                        perm[orig_id] = mirr_id

        # 4. BLOCK: bc -> N - 1 - bc
        for br in range(n):
            for bc in range(n):
                orig_id = self._block_offset + br * n + bc
                mirr_bc = n - 1 - bc
                mirr_id = self._block_offset + br * n + mirr_bc
                perm[orig_id] = mirr_id

        return perm
