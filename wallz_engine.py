from collections import deque
import copy
import time

class WallzEngine:
    def __init__(self, board_size=9, time_limit_sec=600, rules=None):
        self.board_size = board_size
        self.time_limit = time_limit_sec  # Default 10 mins (600s) per player
        
        # Game Mechanics Configuration
        self.rules = {
            'A': True,   # Reusable walls (only the most recently placed wall can be repositioned)
            'B': True,   # One-time hop
            'C': True,   # Temporary cell block
            'K': 3,      # Block duration: cell stays impassable for the next K plies
            'M': 10      # Initial wall inventory per player
        }
        if rules:
            self.rules.update(rules)
        if self.rules['K'] < 1:
            raise ValueError("Block duration K must be >= 1.")
            
        self.reset()

    def __deepcopy__(self, memo):
        return self.clone()

    def clone(self):
        """Creates a fast, independent copy of the game engine for search simulation."""
        new_engine = WallzEngine.__new__(WallzEngine)
        new_engine.board_size = self.board_size
        new_engine.time_limit = self.time_limit
        new_engine.rules = dict(self.rules)
        new_engine.pawns = {0: self.pawns[0], 1: self.pawns[1]}
        new_engine.goals = {0: self.goals[0], 1: self.goals[1]}
        new_engine.walls = {0: list(self.walls[0]), 1: list(self.walls[1])}
        new_engine.wall_inventory = {0: self.wall_inventory[0], 1: self.wall_inventory[1]}
        new_engine.latest_wall = {0: self.latest_wall[0], 1: self.latest_wall[1]}
        new_engine.hop_available = {0: self.hop_available[0], 1: self.hop_available[1]}
        new_engine.block_available = {0: self.block_available[0], 1: self.block_available[1]}
        new_engine.blocked_cells = dict(self.blocked_cells)
        new_engine.time_remaining = {0: self.time_remaining[0], 1: self.time_remaining[1]}
        new_engine.last_turn_timestamp = self.last_turn_timestamp
        new_engine.current_player = self.current_player
        new_engine.turn_count = self.turn_count
        new_engine.winner = self.winner
        new_engine.is_draw = self.is_draw
        new_engine.terminal_reason = self.terminal_reason
        new_engine.max_walls = self.max_walls
        new_engine.state_history = dict(self.state_history)
        return new_engine

    def reset(self):
        mid = self.board_size // 2
        self.pawns = {0: (0, mid), 1: (self.board_size - 1, mid)}
        self.goals = {0: self.board_size - 1, 1: 0}
        
        # Walls & Inventory: M walls per player (rules["M"]); only latest wall is repositionable
        self.max_walls = self.rules['M']
        self.wall_inventory = {0: self.max_walls, 1: self.max_walls}
        self.latest_wall = {0: None, 1: None}
        self.walls = {0: [], 1: []}
        
        # Abilities & Timers
        self.hop_available = {0: self.rules['B'], 1: self.rules['B']}
        self.block_available = {0: self.rules['C'], 1: self.rules['C']}
        self.blocked_cells = {} # {(r, c): remaining_turns}
        
        # Per-player remaining time in seconds
        self.time_remaining = {0: float(self.time_limit), 1: float(self.time_limit)}
        self.last_turn_timestamp = time.time()
        
        self.current_player = 0
        self.turn_count = 0
        self.winner = None
        self.is_draw = False
        # 'goal', 'timeout', 'repetition' or 'no_legal_actions' once the game is over
        self.terminal_reason = None

        # History tracking for N-fold repetition draw rule
        self.state_history = {}
        self._record_state_history()
        
        return self.get_state()

    def _get_state_hashable_key(self):
        """Generates a unique tuple representing complete current state for repetition checks."""
        walls_p0 = tuple(self.walls[0])
        walls_p1 = tuple(self.walls[1])
        blocks = tuple(sorted(self.blocked_cells.items()))
        return (
            self.pawns[0],
            self.pawns[1],
            walls_p0,
            walls_p1,
            self.wall_inventory[0],
            self.wall_inventory[1],
            self.latest_wall[0],
            self.latest_wall[1],
            self.hop_available[0],
            self.hop_available[1],
            self.block_available[0],
            self.block_available[1],
            blocks,
            self.current_player
        )

    def _record_state_history(self):
        key = self._get_state_hashable_key()
        self.state_history[key] = self.state_history.get(key, 0) + 1
        # 5-fold repetition draw rule
        if self.state_history[key] >= 5 and not self.is_terminal():
            self.is_draw = True
            self.terminal_reason = 'repetition'

    def is_terminal(self):
        return self.winner is not None or self.is_draw

    def restore_state_key(self, key):
        """
        Loads a position produced by _get_state_hashable_key() with a fresh history
        (count 1 for this position) and turn counter 0. The key holds everything that
        affects future legal play; board size and rules come from this engine. Clocks
        are left untouched because positions are clock-free.
        """
        (p0, p1, walls_p0, walls_p1, inv0, inv1, latest0, latest1,
         hop0, hop1, blk_av0, blk_av1, blocks, current_player) = key
        n = self.board_size
        cells = [p0, p1] + [cell for cell, _ in blocks]
        if not all(0 <= r < n and 0 <= c < n for r, c in cells) or not all(
                0 <= r < n - 1 and 0 <= c < n - 1 for r, c, _ in walls_p0 + walls_p1):
            raise ValueError(f"State key does not fit a {n}x{n} board.")
        self.pawns = {0: p0, 1: p1}
        self.walls = {0: list(walls_p0), 1: list(walls_p1)}
        self.wall_inventory = {0: inv0, 1: inv1}
        self.latest_wall = {0: latest0, 1: latest1}
        self.hop_available = {0: hop0, 1: hop1}
        self.block_available = {0: blk_av0, 1: blk_av1}
        self.blocked_cells = dict(blocks)
        self.current_player = current_player
        self.turn_count = 0
        self.winner = None
        self.is_draw = False
        self.terminal_reason = None
        for p in (0, 1):
            if self.pawns[p][0] == self.goals[p]:
                self.winner = p
                self.terminal_reason = 'goal'
        if self.winner is None and not self.has_legal_action(current_player):
            self.is_draw = True
            self.terminal_reason = 'no_legal_actions'
        self.state_history = {key: 1}
        return self

    def get_state(self):
        return {
            'pawns': copy.deepcopy(self.pawns),
            'walls': copy.deepcopy(self.walls),
            'wall_inventory': copy.deepcopy(self.wall_inventory),
            'latest_wall': copy.deepcopy(self.latest_wall),
            'hop_available': copy.deepcopy(self.hop_available),
            'block_available': copy.deepcopy(self.block_available),
            'blocked_cells': copy.deepcopy(self.blocked_cells),
            'current_player': self.current_player,
            'turn_count': self.turn_count,
            'time_remaining': copy.deepcopy(self.time_remaining),
            'is_draw': self.is_draw,
            'winner': self.winner,
            'terminal_reason': self.terminal_reason
        }

    def update_clock(self):
        """Subtracts elapsed time for active player; flags loss on time if depleted."""
        now = time.time()
        elapsed = now - self.last_turn_timestamp
        self.time_remaining[self.current_player] -= elapsed
        self.last_turn_timestamp = now
        
        if self.time_remaining[self.current_player] <= 0:
            self.time_remaining[self.current_player] = 0
            self.winner = 1 - self.current_player # Opponent wins on time
            self.terminal_reason = 'timeout'

    def _get_all_active_walls(self):
        return set(self.walls[0] + self.walls[1])

    def shortest_path(self, player):
        start = self.pawns[player]
        target_row = self.goals[player]
        queue = deque([(start[0], start[1], 0)])
        visited = {start}
        
        while queue:
            r, c, dist = queue.popleft()
            if r == target_row:
                return dist
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                    if (nr, nc) not in visited and not self._is_wall_blocking((r, c), (nr, nc)) and (nr, nc) not in self.blocked_cells:
                        visited.add((nr, nc))
                        queue.append((nr, nc, dist + 1))
        return float('inf')

    def _is_wall_blocking(self, cell1, cell2):
        r1, c1 = cell1
        r2, c2 = cell2
        active = self._get_all_active_walls()
        if r1 == r2:
            min_c = min(c1, c2)
            if (r1, min_c, 'V') in active or (r1 - 1, min_c, 'V') in active:
                return True
        elif c1 == c2:
            min_r = min(r1, r2)
            if (min_r, c1, 'H') in active or (min_r, c1 - 1, 'H') in active:
                return True
        return False

    def is_path_open(self, player):
        return self.shortest_path(player) < float('inf')

    def get_legal_actions(self, player=None):
        if player is None:
            player = self.current_player
        if self.is_terminal():
            return []
        actions = []
        r, c = self.pawns[player]

        # 1. Standard Moves
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if self._is_valid_move(player, (r + dr, c + dc)):
                actions.append(('MOVE', (r + dr, c + dc)))

        # 2. One-time Hop
        for dr, dc in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
            if self._is_valid_hop(player, (r + dr, c + dc)):
                actions.append(('HOP', (r + dr, c + dc)))

        # 3. New Wall Placement (consumes 1 wall from inventory)
        if self.wall_inventory[player] > 0:
            for wr in range(self.board_size - 1):
                for wc in range(self.board_size - 1):
                    for orient in ['H', 'V']:
                        if self._is_valid_wall_placement(player, wr, wc, orient):
                            actions.append(('WALL', (wr, wc, orient)))

        # 4. Reusable Wall Repositioning (Rule A: only latest wall can be moved; does not consume inventory)
        if self.rules['A'] and self.latest_wall[player] is not None:
            for wr in range(self.board_size - 1):
                for wc in range(self.board_size - 1):
                    for orient in ['H', 'V']:
                        if self._is_valid_wall_replacement(player, (wr, wc, orient)):
                            actions.append(('REPLACE', (wr, wc, orient)))

        # 5. Temporary Cell Block (preserves path to goal for both players)
        if self.rules['C'] and self.block_available[player]:
            for br in range(self.board_size):
                for bc in range(self.board_size):
                    if self._is_valid_cell_block(br, bc):
                        actions.append(('BLOCK', (br, bc)))

        return actions

    def has_legal_action(self, player=None):
        """Early-exit check for at least one legal action (cheap when a pawn move exists)."""
        if player is None:
            player = self.current_player
        if self.is_terminal():
            return False
        r, c = self.pawns[player]
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if self._is_valid_move(player, (r + dr, c + dc)):
                return True
        for dr, dc in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
            if self._is_valid_hop(player, (r + dr, c + dc)):
                return True
        n = self.board_size
        if self.wall_inventory[player] > 0:
            for wr in range(n - 1):
                for wc in range(n - 1):
                    for orient in ('H', 'V'):
                        if self._is_valid_wall_placement(player, wr, wc, orient):
                            return True
        if self.rules['A'] and self.latest_wall[player] is not None:
            for wr in range(n - 1):
                for wc in range(n - 1):
                    for orient in ('H', 'V'):
                        if self._is_valid_wall_replacement(player, (wr, wc, orient)):
                            return True
        if self.rules['C'] and self.block_available[player]:
            for br in range(n):
                for bc in range(n):
                    if self._is_valid_cell_block(br, bc):
                        return True
        return False

    def _is_valid_move(self, player, cell):
        r, c = self.pawns[player]
        nr, nc = cell
        if abs(nr - r) + abs(nc - c) != 1:
            return False
        if not (0 <= nr < self.board_size and 0 <= nc < self.board_size):
            return False
        if self._is_wall_blocking((r, c), (nr, nc)) or cell in self.blocked_cells:
            return False
        return cell != self.pawns[1 - player]

    def _is_valid_hop(self, player, cell):
        """2-cell orthogonal jump over anything; must land on a free cell and keep a goal path."""
        if not (self.rules['B'] and self.hop_available[player]):
            return False
        r, c = self.pawns[player]
        nr, nc = cell
        if (abs(nr - r), abs(nc - c)) not in ((2, 0), (0, 2)):
            return False
        if not (0 <= nr < self.board_size and 0 <= nc < self.board_size):
            return False
        if cell in self.blocked_cells or cell == self.pawns[1 - player]:
            return False
        # Path preservation: a hop must not strand the pawn in a sealed pocket
        old = self.pawns[player]
        self.pawns[player] = cell
        valid = self.is_path_open(player)
        self.pawns[player] = old
        return valid

    def _is_valid_wall_placement(self, player, r, c, orient):
        if self.wall_inventory[player] <= 0:
            return False
        if not (0 <= r < self.board_size - 1 and 0 <= c < self.board_size - 1):
            return False
        if orient not in ('H', 'V'):
            return False

        active = self._get_all_active_walls()
        if (r, c, orient) in active:
            return False
        if orient == 'H' and ((r, c, 'V') in active or (r, c - 1, 'H') in active or (r, c + 1, 'H') in active):
            return False
        if orient == 'V' and ((r, c, 'H') in active or (r - 1, c, 'V') in active or (r + 1, c, 'V') in active):
            return False

        # Tentatively apply wall and verify path preservation
        self.walls[player].append((r, c, orient))
        valid = self.is_path_open(0) and self.is_path_open(1)
        self.walls[player].pop()
        return valid

    def _is_valid_wall_replacement(self, player, new_wall, old_wall=None):
        """
        Validates repositioning the player's most recently placed wall.
        Repositioning older walls is strictly illegal.
        """
        if not self.rules['A']:
            return False
        latest = self.latest_wall[player]
        if latest is None:
            return False
        if old_wall is not None and old_wall != latest:
            # Rejection of repositioning an older wall
            return False

        r, c, orient = new_wall
        if not (0 <= r < self.board_size - 1 and 0 <= c < self.board_size - 1):
            return False
        if orient not in ('H', 'V'):
            return False
        if new_wall == latest:
            return False

        # Remaining active walls excluding the wall being repositioned
        all_active = self._get_all_active_walls()
        active = all_active - {latest}

        if new_wall in active:
            return False
        if orient == 'H' and ((r, c, 'V') in active or (r, c - 1, 'H') in active or (r, c + 1, 'H') in active):
            return False
        if orient == 'V' and ((r, c, 'H') in active or (r - 1, c, 'V') in active or (r + 1, c, 'V') in active):
            return False

        # Tentatively replace in player's wall list and verify path preservation
        idx = self.walls[player].index(latest)
        self.walls[player][idx] = new_wall
        valid = self.is_path_open(0) and self.is_path_open(1)
        self.walls[player][idx] = latest
        return valid

    def _is_valid_cell_block(self, br, bc):
        """Validates temporary cell block, ensuring path preservation for both players."""
        if not (0 <= br < self.board_size and 0 <= bc < self.board_size):
            return False
        if (br, bc) in self.blocked_cells or (br, bc) == self.pawns[0] or (br, bc) == self.pawns[1]:
            return False

        self.blocked_cells[(br, bc)] = self.rules['K']
        valid = self.is_path_open(0) and self.is_path_open(1)
        del self.blocked_cells[(br, bc)]
        return valid

    def reposition_wall(self, player, new_wall, old_wall=None):
        """
        Direct API to reposition player's most recently placed wall.
        Returns True if successful, False if invalid (preserving original state).
        """
        if not self._is_valid_wall_replacement(player, new_wall, old_wall=old_wall):
            return False
        if old_wall is None:
            old_wall = self.latest_wall[player]
        idx = self.walls[player].index(old_wall)
        self.walls[player][idx] = new_wall
        self.latest_wall[player] = new_wall
        return True

    def apply_action(self, action, update_clock=True):
        if self.is_terminal():
            raise ValueError("Game is already over; no further actions may be applied.")
        if update_clock:
            self.update_clock()
            if self.winner is not None:
                return self.get_state(), self.winner, self.is_draw

        player = self.current_player
        act_type, act_val = action
        
        if act_type == 'MOVE':
            if not self._is_valid_move(player, act_val):
                raise ValueError(f"Illegal move to {act_val}.")
            self.pawns[player] = act_val
        elif act_type == 'HOP':
            if not self._is_valid_hop(player, act_val):
                raise ValueError(f"Illegal hop to {act_val}.")
            self.pawns[player] = act_val
            self.hop_available[player] = False
        elif act_type == 'WALL':
            if self.wall_inventory[player] <= 0:
                raise ValueError(f"Player {player} has no walls remaining in inventory.")
            r, c, orient = act_val
            if not self._is_valid_wall_placement(player, r, c, orient):
                raise ValueError(f"Illegal wall placement at {act_val}.")
            self.walls[player].append(act_val)
            self.latest_wall[player] = act_val
            self.wall_inventory[player] -= 1
        elif act_type == 'REPLACE':
            if not self.rules['A']:
                raise ValueError("Reusable wall mechanic (Rule A) is disabled.")
            if self.latest_wall[player] is None:
                raise ValueError(f"Player {player} has not placed any wall yet to reposition.")

            # Support both act_val = new_wall or act_val = (old_wall, new_wall)
            if len(act_val) == 2 and isinstance(act_val[0], tuple) and isinstance(act_val[1], tuple):
                old_w, new_w = act_val
                if old_w != self.latest_wall[player]:
                    raise ValueError(f"Cannot reposition older wall {old_w}: only latest wall {self.latest_wall[player]} may be repositioned.")
            else:
                new_w = act_val
                old_w = self.latest_wall[player]

            if not self._is_valid_wall_replacement(player, new_w, old_wall=old_w):
                raise ValueError(f"Illegal wall replacement {old_w} -> {new_w}: collision or path preservation violated.")

            idx = self.walls[player].index(old_w)
            self.walls[player][idx] = new_w
            self.latest_wall[player] = new_w
            # wall_inventory[player] remains unchanged
        elif act_type == 'BLOCK':
            if not (self.rules['C'] and self.block_available[player]):
                raise ValueError(f"Player {player} has no cell block available.")
            if not self._is_valid_cell_block(act_val[0], act_val[1]):
                raise ValueError(f"Illegal cell block at {act_val}: path preservation violated.")
            self.block_available[player] = False
        else:
            raise ValueError(f"Unknown action type {act_type!r}.")

        # Win condition check
        if self.pawns[player][0] == self.goals[player]:
            self.winner = player
            self.terminal_reason = 'goal'

        # Decrement existing block timers; a block placed this ply starts at K, so the
        # cell is impassable for exactly the next K plies (proposal eq. temporary_block_decay)
        to_remove = []
        for cell in self.blocked_cells:
            self.blocked_cells[cell] -= 1
            if self.blocked_cells[cell] <= 0:
                to_remove.append(cell)
        for cell in to_remove:
            del self.blocked_cells[cell]
        if act_type == 'BLOCK':
            self.blocked_cells[act_val] = self.rules['K']

        # Turn transition
        self.current_player = 1 - self.current_player
        self.turn_count += 1
        if update_clock:
            self.last_turn_timestamp = time.time()

        # Team rule: a non-terminal position with no legal action for the side to move is a draw
        if self.winner is None and not self.has_legal_action(self.current_player):
            self.is_draw = True
            self.terminal_reason = 'no_legal_actions'

        # Check repetition rule
        self._record_state_history()
        
        return self.get_state(), self.winner, self.is_draw