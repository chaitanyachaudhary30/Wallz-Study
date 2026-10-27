import copy
import math
import time

class SearchTimeout(Exception):
    """Raised when the search exceeds the configured per-turn time limit."""
    pass

class MinimaxAgent:
    """
    Tier-2 Minimax Adversarial Agent with Alpha-Beta Pruning.
    
    Operates on the Modified Wallz game engine using:
    - Depth-limited minimax tree search (1 depth = 1 ply)
    - Alpha-Beta pruning
    - Spatial difference leaf evaluation: U_i(s) = tanh(0.2 * (d_opp(s) - d_self(s)))
    - Deterministic 4-tier move ordering:
        1. Immediate winning moves
        2. Favorable spatial heuristic values
        3. Pawn moves (MOVE / HOP) over walls and blocks
        4. Deterministic tie-breaking by original action index
    - Configurable per-turn time limit via Iterative Deepening with guaranteed legal fallback
    """

    WIN_SCORE = 1000.0  # Terminal win constant (> 1.0, dominating tanh in (-1, 1))

    def __init__(self, player_id, depth=2, time_limit=None, use_alpha_beta=True, order_moves=True):
        """
        Args:
            player_id (int): 0 or 1, the player controlled by this agent.
            depth (int): Maximum lookahead search depth in plies (>= 1).
            time_limit (float or None): Maximum search time per turn in seconds.
            use_alpha_beta (bool): Whether to enable Alpha-Beta pruning (default True).
            order_moves (bool): Whether to order moves before exploration (default True).
        """
        self.player_id = player_id
        self.depth = max(1, depth)
        self.time_limit = time_limit
        self.use_alpha_beta = use_alpha_beta
        self.order_moves = order_moves

        # Performance & evaluation metrics tracking
        self.nodes_expanded = 0
        self.cutoffs = 0
        self.last_search_depth = 0
        self.last_decision_time = 0.0
        self.last_action_value = 0.0

    def get_stats(self):
        """Returns performance metrics from the most recent decision."""
        return {
            'nodes_expanded': self.nodes_expanded,
            'cutoffs': self.cutoffs,
            'last_search_depth': self.last_search_depth,
            'last_decision_time': self.last_decision_time,
            'last_action_value': self.last_action_value,
        }

    def evaluate(self, engine, depth=0, ply=0):
        """
        Leaf evaluation function strictly adhering to the project formalization.
        
        For terminal states:
            - Win for self.player_id: +(WIN_SCORE - ply) (prefers faster wins)
            - Loss for self.player_id: -(WIN_SCORE - ply) (delays unavoidable losses)
            - Draw (repetition / truncation): 0.0
        
        For non-terminal states at search horizon:
            H_i(s) = d_{-i}(s) - d_i(s)
            U_i(s) = tanh(0.2 * H_i(s))
        """
        # Terminal state precedence
        if engine.winner is not None:
            if engine.winner == self.player_id:
                return self.WIN_SCORE - ply
            else:
                return -self.WIN_SCORE + ply

        if engine.is_draw:
            return 0.0

        # Non-terminal spatial heuristic evaluation
        d_self = engine.shortest_path(self.player_id)
        d_opp = engine.shortest_path(1 - self.player_id)
        
        h_val = d_opp - d_self
        return math.tanh(0.2 * h_val)

    def order_actions(self, engine, legal_actions, active_player):
        """
        Orders actions deterministically to maximize Alpha-Beta pruning efficiency.
        
        Prioritization:
            1. Immediate winning actions (MOVE / HOP landing on goal row)
            2. Actions with favorable spatial heuristic values (pawn progress advancing d_self)
            3. Pawn moves (HOP over MOVE, advancing over sideways/retreating)
            4. Wall placements (WALL)
            5. Cell blocks (BLOCK)
            6. Deterministic tie-breaker (-original_idx)
        """
        target_goal = engine.goals[active_player]
        cur_r = engine.pawns[active_player][0]
        scored_actions = []

        for idx, action in enumerate(legal_actions):
            act_type, act_val = action

            # 1. Immediate winning action check
            if act_type in ('MOVE', 'HOP') and act_val[0] == target_goal:
                score = (3, 0, 0, -idx)
            elif act_type in ('MOVE', 'HOP'):
                # Spatial progress directly improves H_p = d_opp - d_self
                progress = abs(cur_r - target_goal) - abs(act_val[0] - target_goal)
                hop_bonus = 2 if act_type == 'HOP' else 1
                score = (2, hop_bonus, progress, -idx)
            elif act_type in ('WALL', 'REPLACE'):
                score = (1, 0, 0, -idx)
            else:  # BLOCK
                score = (0, 0, 0, -idx)

            scored_actions.append((score, action))

        scored_actions.sort(key=lambda item: item[0], reverse=True)
        return [action for _, action in scored_actions]

    def select_action(self, engine):
        """
        Selects the best legal action for the agent.
        
        If time_limit is configured, uses iterative deepening from depth 1 to self.depth.
        Guarantees a legal fallback action if the time limit expires.
        """
        start_time = time.time()
        self.nodes_expanded = 0
        self.cutoffs = 0

        legal_actions = engine.get_legal_actions(self.player_id)
        if not legal_actions:
            self.last_decision_time = time.time() - start_time
            return None

        # Trivial optimization: only one legal action available
        if len(legal_actions) == 1:
            self.last_search_depth = 1
            self.last_decision_time = time.time() - start_time
            self.last_action_value = 0.0
            return legal_actions[0]

        # Establish guaranteed legal fallback action
        fallback_action = legal_actions[0]
        for a in legal_actions:
            if a[0] in ('MOVE', 'HOP') and a[1][0] == engine.goals[self.player_id]:
                fallback_action = a
                break

        deadline = (start_time + self.time_limit) if self.time_limit is not None else None

        best_action = fallback_action
        best_val = -float('inf')
        deepest_completed = 0

        # Iterative deepening if time_limit is set, else direct search to self.depth
        max_d = self.depth
        depth_range = range(1, max_d + 1) if self.time_limit is not None else [max_d]

        for current_d in depth_range:
            if deadline is not None and time.time() >= deadline:
                break
            try:
                action, val = self._search_root(engine, current_d, deadline)
                if action is not None:
                    best_action = action
                    best_val = val
                    deepest_completed = current_d
                    
                # If a forced win is found at root, stop deeper searches
                if best_val >= self.WIN_SCORE - 100:
                    break
            except SearchTimeout:
                break

        self.last_search_depth = deepest_completed if deepest_completed > 0 else 1
        self.last_decision_time = time.time() - start_time
        self.last_action_value = best_val

        return best_action

    def _search_root(self, engine, depth, deadline):
        """Root search step returning (best_action, best_value)."""
        legal_actions = engine.get_legal_actions(self.player_id)
        if not legal_actions:
            return None, -self.WIN_SCORE

        if self.order_moves:
            actions = self.order_actions(engine, legal_actions, self.player_id)
        else:
            actions = legal_actions

        best_action = actions[0]
        alpha = -float('inf')
        beta = float('inf')
        max_eval = -float('inf')

        for action in actions:
            if deadline is not None and time.time() >= deadline:
                raise SearchTimeout()

            self.nodes_expanded += 1
            sim = engine.clone() if hasattr(engine, 'clone') else copy.deepcopy(engine)
            sim.apply_action(action, update_clock=False)

            eval_score = self._minimax(
                sim,
                depth=depth - 1,
                alpha=alpha,
                beta=beta,
                is_maximizing=False,
                ply=1,
                deadline=deadline
            )

            if eval_score > max_eval:
                max_eval = eval_score
                best_action = action

            if self.use_alpha_beta:
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    self.cutoffs += 1
                    break

        return best_action, max_eval

    def _minimax(self, engine, depth, alpha, beta, is_maximizing, ply, deadline):
        """
        Recursive Minimax search with Alpha-Beta pruning.
        
        Args:
            engine: Game engine instance at current state.
            depth: Remaining depth in plies.
            alpha: Current best score for maximizing player.
            beta: Current best score for minimizing player.
            is_maximizing: True if current player is the root player (self.player_id).
            ply: Steps taken from root node.
            deadline: Wall-clock time threshold for timeout.
        """
        if deadline is not None and time.time() >= deadline:
            raise SearchTimeout()

        # Terminal state check
        if engine.winner is not None or engine.is_draw:
            return self.evaluate(engine, depth, ply)

        # Depth horizon reached
        if depth == 0:
            return self.evaluate(engine, 0, ply)

        current_player = engine.current_player
        legal_actions = engine.get_legal_actions(current_player)

        # If a player has no legal actions, they lose
        if not legal_actions:
            return -self.WIN_SCORE + ply if is_maximizing else self.WIN_SCORE - ply

        if self.order_moves:
            actions = self.order_actions(engine, legal_actions, current_player)
        else:
            actions = legal_actions

        if is_maximizing:
            max_eval = -float('inf')
            for action in actions:
                if deadline is not None and time.time() >= deadline:
                    raise SearchTimeout()

                self.nodes_expanded += 1
                sim = engine.clone() if hasattr(engine, 'clone') else copy.deepcopy(engine)
                sim.apply_action(action, update_clock=False)

                next_is_max = (sim.current_player == self.player_id)
                eval_score = self._minimax(
                    sim,
                    depth=depth - 1,
                    alpha=alpha,
                    beta=beta,
                    is_maximizing=next_is_max,
                    ply=ply + 1,
                    deadline=deadline
                )

                if eval_score > max_eval:
                    max_eval = eval_score

                if self.use_alpha_beta:
                    alpha = max(alpha, eval_score)
                    if beta <= alpha:
                        self.cutoffs += 1
                        break

            return max_eval
        else:
            min_eval = float('inf')
            for action in actions:
                if deadline is not None and time.time() >= deadline:
                    raise SearchTimeout()

                self.nodes_expanded += 1
                sim = engine.clone() if hasattr(engine, 'clone') else copy.deepcopy(engine)
                sim.apply_action(action, update_clock=False)

                next_is_max = (sim.current_player == self.player_id)
                eval_score = self._minimax(
                    sim,
                    depth=depth - 1,
                    alpha=alpha,
                    beta=beta,
                    is_maximizing=next_is_max,
                    ply=ply + 1,
                    deadline=deadline
                )

                if eval_score < min_eval:
                    min_eval = eval_score

                if self.use_alpha_beta:
                    beta = min(beta, eval_score)
                    if beta <= alpha:
                        self.cutoffs += 1
                        break

            return min_eval
