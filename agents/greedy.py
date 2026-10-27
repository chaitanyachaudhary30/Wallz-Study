import copy

class GreedyAgent:
    """
    Tier-1 Spatial Greedy Baseline Agent.
    
    Evaluates all legal actions with 1-ply lookahead using lexicographic priority:
    1. Immediate winning move
    2. Draw avoidance
    3. Maximize shortest path distance advantage (d_opp - d_self)
    4. Maximize opponent goal distance (d_opp)
    5. Minimize own goal distance (-d_self)
    6. Action index tie-breaker (-idx)
    """

    def __init__(self, player_id):
        self.player_id = player_id

    def select_action(self, engine):
        legal_actions = engine.get_legal_actions(self.player_id)
        if not legal_actions:
            return None

        best_action = None
        best_score = None

        for idx, action in enumerate(legal_actions):
            # Fast clone if available, fallback to deepcopy
            sim_engine = engine.clone() if hasattr(engine, 'clone') else copy.deepcopy(engine)
            _, winner, is_draw = sim_engine.apply_action(action, update_clock=False)

            # Priority 1: Immediate winning move
            if winner == self.player_id:
                return action

            # Priority 2: Avoid moves that lead to a draw if winning is possible
            if is_draw:
                continue

            # Calculate path distances after action
            d_self = sim_engine.shortest_path(self.player_id)
            d_opp = sim_engine.shortest_path(1 - self.player_id)
            d_adv = d_opp - d_self

            # Lexicographic evaluation tuple:
            # (1. Distance Advantage, 2. Max Opponent Distance, 3. Min Own Distance, 4. Action Index Tie-Breaker)
            score = (
                d_adv,    # Maximize distance advantage
                d_opp,    # Maximize opponent goal distance
                -d_self,  # Minimize own goal distance
                -idx      # Deterministic tie-breaking by lower action index
            )

            if best_score is None or score > best_score:
                best_score = score
                best_action = action

        # Fallback if all moves lead to a draw or none scored
        if best_action is None and legal_actions:
            best_action = legal_actions[0]

        return best_action

