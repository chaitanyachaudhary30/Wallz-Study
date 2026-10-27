from collections import deque
import copy
import time
from wallz_engine import WallzEngine

class ExactSolver:
    """
    Tier-3 Exact Retrograde Solver for Tractable Miniature Modified Wallz Configurations.
    
    Computes exact game-theoretic values (WIN / LOSS / DRAW) and optimal policies
    via forward reachable state graph generation and backward retrograde propagation
    (game-graph value iteration for cyclic zero-sum games).
    
    Guarantees:
    - Exactness: Every reachable state in the solved configuration is classified into
      its true game-theoretic equilibrium value (WIN, LOSS, or DRAW) without heuristics.
    - Optimal Play:
        * WIN states minimize distance-to-win (fastest forced win).
        * LOSS states maximize distance-to-loss (longest delay of defeat).
        * DRAW states preserve non-losing cycles against all opponent moves.
    - Complete Domain Bounding: Explicitly tracks the solved domain and distinguishes
      solved states from unvisited or out-of-domain states.
    """

    def __init__(self, board_size=3, rules=None, max_walls=None):
        """
        Args:
            board_size (int): Board dimension N (tractable for N=3, or bounded N=5).
            rules (dict, optional): Rules configuration dictionary.
            max_walls (int, optional): Caps wall inventory per player for tractability.
                                       If None, uses engine default (10).
        """
        self.board_size = board_size
        self.rules = {
            'A': True,
            'B': True,
            'C': True,
            'K': 3
        }
        if rules:
            self.rules.update(rules)
        self.max_walls = max_walls

        # State storage and graph
        # key -> (status, distance, best_action)
        self.solutions = {}  # key -> dict: {'status': WIN|LOSS|DRAW, 'distance': int, 'best_action': tuple, 'value': float}
        self.state_count = 0
        self.win_count = 0
        self.loss_count = 0
        self.draw_count = 0
        self.build_time = 0.0
        self.solve_time = 0.0
        self.is_compiled = False

    def solve(self, initial_engine=None):
        """
        Executes complete forward reachability enumeration and retrograde analysis.
        
        Args:
            initial_engine (WallzEngine, optional): Custom starting state. If None,
                                                   creates a default initial engine.
        Returns:
            self: The compiled solver instance.
        """
        t_start = time.time()

        if initial_engine is not None:
            engine = initial_engine.clone()
        else:
            engine = WallzEngine(board_size=self.board_size, rules=self.rules)
            if self.max_walls is not None:
                engine.max_walls = self.max_walls
                engine.wall_inventory = {0: self.max_walls, 1: self.max_walls}

        # 1. Forward Reachable State Graph Construction
        states = {}        # key -> engine instance
        adj = {}           # key -> list of (action, next_key)
        rev_adj = {}       # key -> list of (action, prev_key)
        deg = {}           # key -> int (unresolved legal actions count)

        start_key = engine._get_state_hashable_key()
        states[start_key] = engine
        adj[start_key] = []
        rev_adj[start_key] = []

        queue = deque([start_key])

        while queue:
            curr_key = queue.popleft()
            curr_eng = states[curr_key]

            # Terminal states and draw states have no forward actions explored
            if curr_eng.winner is not None or curr_eng.is_draw:
                continue

            legal_actions = curr_eng.get_legal_actions()
            deg[curr_key] = len(legal_actions)

            for action in legal_actions:
                nxt = curr_eng.clone()
                nxt.apply_action(action, update_clock=False)
                nxt_key = nxt._get_state_hashable_key()

                if nxt_key not in states:
                    states[nxt_key] = nxt
                    adj[nxt_key] = []
                    rev_adj[nxt_key] = []
                    queue.append(nxt_key)

                adj[curr_key].append((action, nxt_key))
                rev_adj[nxt_key].append((action, curr_key))

        t_graph = time.time()
        self.build_time = t_graph - t_start

        # 2. Retrograde Analysis (Backward Induction with Cycle Handling)
        status = {}       # key -> 'WIN', 'LOSS', 'DRAW' (from perspective of active player)
        distance = {}     # key -> int (distance to terminal outcome in plies)
        best_action = {}  # key -> action tuple

        q = deque()

        # Seed the queue with terminal states
        for k, eng in states.items():
            if eng.winner is not None:
                if eng.winner == eng.current_player:
                    status[k] = 'WIN'
                else:
                    status[k] = 'LOSS'
                distance[k] = 0
                best_action[k] = None
                q.append(k)
            elif eng.is_draw or (k in deg and deg[k] == 0):
                status[k] = 'DRAW'
                distance[k] = 0
                best_action[k] = None
                q.append(k)

        # Propagate values backward along incoming edges
        while q:
            v = q.popleft()
            v_status = status[v]
            v_dist = distance[v]

            for a, u in rev_adj[v]:
                if u in status:
                    continue

                # In u, the active player chooses a move.
                # If moving to v gives the opponent a LOSS, the active player in u WINS!
                if v_status == 'LOSS':
                    status[u] = 'WIN'
                    distance[u] = v_dist + 1
                    best_action[u] = a
                    q.append(u)
                elif v_status == 'WIN':
                    # Moving to v gives opponent a WIN, so action 'a' is a losing candidate for u.
                    deg[u] -= 1
                    # If ALL candidate actions lead to an opponent win, u is a forced LOSS
                    if deg[u] == 0:
                        status[u] = 'LOSS'
                        # In a forced loss, optimal defense delays loss longest
                        max_d = 0
                        delay_action = None
                        for act_succ, succ in adj[u]:
                            d_succ = distance.get(succ, 0)
                            if d_succ >= max_d:
                                max_d = d_succ
                                delay_action = act_succ
                        distance[u] = max_d + 1
                        best_action[u] = delay_action
                        q.append(u)

        # 3. Draw and Cycle Resolution
        # Any state that cannot force a win, and where the opponent cannot force a win,
        # is a theoretical DRAW under optimal mutual play.
        for k in states:
            if k not in status:
                status[k] = 'DRAW'
                distance[k] = float('inf')

                # An optimal move in a draw state is any move that maintains a draw
                # (i.e. does not transition to a state where the next player can WIN)
                draw_action = None
                for a, succ in adj[k]:
                    if status.get(succ) != 'WIN':
                        draw_action = a
                        break
                if draw_action is None and adj[k]:
                    draw_action = adj[k][0][0]
                best_action[k] = draw_action

        # 4. Compile Solutions Table
        self.solutions.clear()
        self.win_count = 0
        self.loss_count = 0
        self.draw_count = 0

        for k, eng in states.items():
            st = status[k]
            dist = distance[k]
            act = best_action.get(k)

            # Numeric value from perspective of active player: +1.0 for WIN, 0.0 for DRAW, -1.0 for LOSS
            val = 1.0 if st == 'WIN' else (-1.0 if st == 'LOSS' else 0.0)

            self.solutions[k] = {
                'status': st,
                'distance': dist,
                'best_action': act,
                'value': val,
                'active_player': eng.current_player
            }

            if st == 'WIN':
                self.win_count += 1
            elif st == 'LOSS':
                self.loss_count += 1
            else:
                self.draw_count += 1

        t_end = time.time()
        self.solve_time = t_end - t_graph
        self.state_count = len(self.solutions)
        self.is_compiled = True

        return self

    def is_solved(self, engine_or_key):
        """Checks whether the given engine state is in the solved domain."""
        key = engine_or_key._get_state_hashable_key() if isinstance(engine_or_key, WallzEngine) else engine_or_key
        return key in self.solutions

    def get_state_outcome(self, engine_or_key):
        """
        Returns (status, distance) from the perspective of the active player.
        status: 'WIN', 'LOSS', or 'DRAW'.
        distance: Plies to terminal outcome (inf for DRAW).
        """
        key = engine_or_key._get_state_hashable_key() if isinstance(engine_or_key, WallzEngine) else engine_or_key
        if key not in self.solutions:
            return None, None
        sol = self.solutions[key]
        return sol['status'], sol['distance']

    def get_value(self, engine_or_key, player=None):
        """
        Returns exact game-theoretic value (+1.0, 0.0, -1.0).
        If player is specified (0 or 1), evaluates from that player's perspective.
        If player is None, evaluates from active player's perspective.
        """
        key = engine_or_key._get_state_hashable_key() if isinstance(engine_or_key, WallzEngine) else engine_or_key
        if key not in self.solutions:
            return None
        sol = self.solutions[key]
        raw_val = sol['value']
        active_p = sol['active_player']

        if player is None or player == active_p:
            return raw_val
        else:
            return -raw_val

    def get_optimal_action(self, engine):
        """
        Returns the provably optimal action for the current active player.
        Returns None if state is outside the solved domain or terminal.
        """
        if not isinstance(engine, WallzEngine):
            raise TypeError("Expected WallzEngine instance.")
        key = engine._get_state_hashable_key()
        if key not in self.solutions:
            return None
        return self.solutions[key]['best_action']

    def get_stats(self):
        """Returns solver complexity, metrics, and coverage summary."""
        return {
            'board_size': self.board_size,
            'rules': dict(self.rules),
            'max_walls': self.max_walls,
            'state_count': self.state_count,
            'win_count': self.win_count,
            'loss_count': self.loss_count,
            'draw_count': self.draw_count,
            'build_time_sec': self.build_time,
            'solve_time_sec': self.solve_time,
            'total_time_sec': self.build_time + self.solve_time,
            'is_compiled': self.is_compiled,
        }


class ExactSolverAgent:
    """
    Tier-3 Exact Ground-Truth Baseline Agent.
    
    Plays provably optimal actions derived from the ExactSolver retrograde table.
    Conforms to the standard agent interface: select_action(engine).
    """

    def __init__(self, player_id, solver=None, board_size=3, rules=None, max_walls=None):
        """
        Args:
            player_id (int): 0 or 1, player controlled by this agent.
            solver (ExactSolver, optional): Pre-compiled solver instance.
            board_size (int): Board dimension if compiling on demand.
            rules (dict, optional): Rules dict if compiling on demand.
            max_walls (int, optional): Wall cap if compiling on demand.
        """
        self.player_id = player_id
        if solver is not None:
            self.solver = solver
        else:
            self.solver = ExactSolver(board_size=board_size, rules=rules, max_walls=max_walls)
            self.solver.solve()

    def select_action(self, engine):
        """
        Selects the exact optimal action from the solved retrograde solution.
        If the state is outside the solved domain, returns a legal fallback action.
        """
        if not self.solver.is_solved(engine):
            # Out of domain: return first legal action if available
            legal = engine.get_legal_actions(self.player_id)
            return legal[0] if legal else None

        opt_action = self.solver.get_optimal_action(engine)
        if opt_action is not None:
            return opt_action

        # Fallback for terminal states or unassigned actions
        legal = engine.get_legal_actions(self.player_id)
        return legal[0] if legal else None

