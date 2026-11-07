"""Agent decision explanation engine (P09 / Contract C-EXPL).

Generates rich, model-accurate explanations ("how the models think") for:
- Random: uniform probability distribution over legal actions.
- Greedy: objective H = d_opp - d_self, score_actions rows, tie-break chain.
- Minimax: depth searched, nodes expanded, cutoffs, full-window root move values,
  chosen action features.
- DQN: Q-values for all legal actions via QNetwork forward pass under torch.no_grad(),
  top 5 actions, network metadata.
- Exact: game-theoretic status (WIN/LOSS/DRAW), rank (distance to win), discounted value,
  per-action child values.
"""
from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from evaluation.match_runner import parse_agent_spec
from web.backend.serialize import action_to_json


def explain_and_choose(agent: Any, family: str, engine: Any, spec: Optional[str] = None) -> Tuple[Optional[Tuple[str, Any]], Dict[str, Any]]:
    """Executes agent decision and computes a C-EXPL explanation envelope.

    Args:
        agent: Agent instance (RandomAgent, GreedyAgent, MinimaxAgent, DQNAgent, ExactAgent).
        family: Agent family string ('random', 'greedy', 'minimax', 'dqn', 'exact').
        engine: WallzEngine instance at current state.
        spec: Agent spec string (e.g. 'minimax:depth=2', 'dqn:path=...'); used for labels
            and, for DQN, to locate the checkpoint's meta.json.

    Returns:
        (chosen_action, explanation_dict)
    """
    mover = engine.current_player
    spec = spec or getattr(agent, "spec", family)
    agent_name = getattr(agent, "name", None) or _display_name(family, spec)

    t0 = time.time()

    if family == "random":
        chosen, details, overlay, summary = _explain_random(agent, engine, mover)
    elif family == "greedy":
        chosen, details, overlay, summary = _explain_greedy(agent, engine, mover)
    elif family == "minimax":
        chosen, details, overlay, summary = _explain_minimax(agent, engine, mover)
    elif family == "dqn":
        chosen, details, overlay, summary = _explain_dqn(agent, engine, mover, spec)
    elif family == "exact":
        chosen, details, overlay, summary = _explain_exact(agent, engine, mover)
    else:
        # Fallback for unknown / generic agent
        chosen = agent.select_action(engine) if agent is not None else None
        details = {}
        overlay = []
        summary = f"Selected action {chosen}."

    elapsed = time.time() - t0

    explanation = {
        "agent": agent_name,
        "spec": spec,
        "player": mover,
        "chosen": action_to_json(chosen) if chosen is not None else None,
        "decision_seconds": round(elapsed, 4),
        "summary": summary,
        "details": details,
        "overlay": overlay,
    }

    return chosen, explanation


def _display_name(family: str, spec: str) -> str:
    names = {"random": "Random", "greedy": "Greedy", "minimax": "Minimax", "dqn": "DQN", "exact": "Exact solver"}
    return names.get(family, family.capitalize())


def _dqn_meta(spec: Optional[str]) -> Dict[str, Any]:
    """Network metadata read from the checkpoint's meta.json (never invented)."""
    import json
    import os
    if not spec:
        return {}
    _, params = parse_agent_spec(spec)
    path = params.get("path")
    if not path:
        return {}
    meta_path = path if os.path.isdir(path) else os.path.dirname(path)
    meta_path = os.path.join(meta_path, "meta.json")
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, ValueError):
        return {}
    keep = ("N", "blocks", "channels", "hidden", "gamma", "double", "dueling", "step", "seed", "rules")
    out = {k: meta[k] for k in keep if k in meta}
    base = os.path.basename(str(path))
    if base.startswith("step_") and base.endswith(".pt"):
        out["checkpoint"] = base
        out["step"] = int(base[5:-3])
    return out


# -----------------------------------------------------------------------------
# Family Explanations
# -----------------------------------------------------------------------------

def _explain_random(agent: Any, engine: Any, mover: int) -> Tuple[Optional[Tuple[str, Any]], Dict[str, Any], List[Dict[str, Any]], str]:
    legal = engine.get_legal_actions(mover)
    chosen = agent.select_action(engine) if legal else None
    n = len(legal)
    prob = round(1.0 / n, 4) if n > 0 else 0.0

    details = {
        "n_legal": n,
        "probability": prob,
    }
    pct = round(prob * 100.0, 1)
    overlay = [
        {"action": action_to_json(a), "value": prob, "label": f"{pct}%"}
        for a in legal
    ]
    summary = f"Chose randomly from {n} legal actions with equal probability ({pct}% each)."
    return chosen, details, overlay, summary


def _explain_greedy(agent: Any, engine: Any, mover: int) -> Tuple[Optional[Tuple[str, Any]], Dict[str, Any], List[Dict[str, Any]], str]:
    rows = agent.score_actions(engine)
    if not rows:
        return None, {"objective": "H = d_opp − d_self", "rows": [], "immediate_win": False}, [], "No legal moves."

    has_win = any(r.get("immediate_win") for r in rows)
    if has_win:
        chosen = next(r["action"] for r in rows if r.get("immediate_win"))
    else:
        chosen = max(rows, key=lambda r: r["score"])["action"]

    # Top 25 rows for details
    sorted_rows = sorted(rows, key=lambda r: r["score"], reverse=True)
    serialized_rows = []
    for r in sorted_rows[:25]:
        sr = dict(r)
        sr["action"] = action_to_json(sr["action"])
        # Format infinite H values cleanly for JSON
        if sr["H"] == float("inf"):
            sr["H"] = None
        elif sr["H"] == -float("inf"):
            sr["H"] = None
        elif sr["H"] is not None:
            sr["H"] = round(float(sr["H"]), 2)
        # score tuple to list
        if isinstance(sr.get("score"), tuple):
            sr["score"] = [s if s != float("inf") and s != -float("inf") else str(s) for s in sr["score"]]
        serialized_rows.append(sr)

    overlay = []
    for r in rows:
        act_json = action_to_json(r["action"])
        if r.get("immediate_win") or r["H"] == float("inf"):
            val = None
            label = "win"
        elif r.get("outcome") == "loss" or r["H"] == -float("inf"):
            val = None
            label = "loss"
        elif r.get("outcome") == "draw":
            val = 0.0
            label = "draw"
        else:
            val = float(r["H"])
            label = f"H={int(val) if val.is_integer() else val:+.1f}"
        overlay.append({"action": act_json, "value": val, "label": label})

    # If score_actions cut short at immediate win, ensure overlay includes all legal actions
    scored_actions = {r["action"] for r in rows}
    for act in engine.get_legal_actions(mover):
        if act not in scored_actions:
            overlay.append({"action": action_to_json(act), "value": None, "label": "unanalysed"})

    chosen_row = next((r for r in rows if r["action"] == chosen), rows[0])
    if chosen_row.get("immediate_win"):
        summary = f"Picked immediate winning move {chosen[0]} {chosen[1]}."
    else:
        h_val = chosen_row["H"]
        if isinstance(h_val, (int, float)) and not math.isinf(h_val):
            h_str = f"{int(h_val):+d}" if float(h_val).is_integer() else f"{h_val:+.2f}"
        else:
            h_str = str(h_val)
        summary = f"Picked the move that maximises my distance advantage H = d_opp − d_self (H = {h_str})."
        if chosen_row.get("outcome") == "draw":
            summary = "No move improves on a draw: picked a move that ends the game in a rule draw (scored H = 0)."

    details = {
        "objective": "H = d_opp − d_self",
        "rows": serialized_rows,
        "tie_break_chain": ["H", "category priority MOVE>REPLACE>HOP>WALL>BLOCK", "−d_self", "tie key"],
        "immediate_win": has_win,
    }

    return chosen, details, overlay, summary


def _explain_minimax(agent: Any, engine: Any, mover: int) -> Tuple[Optional[Tuple[str, Any]], Dict[str, Any], List[Dict[str, Any]], str]:
    chosen = agent.select_action(engine)
    stats = agent.get_stats()

    depth_completed = stats.get("depth_completed", agent.depth)
    nodes_expanded = stats.get("nodes_expanded", 0)
    cutoffs = stats.get("cutoffs", 0)
    root_value = stats.get("last_action_value", 0.0)

    legal = engine.get_legal_actions(mover)
    if not legal:
        return None, {}, [], "No legal moves."

    # Full-window search of root children
    # Bound root analysis budget on large boards / high legal action count
    if engine.board_size >= 7 or len(legal) > 40:
        ordered = agent.order_actions(engine, legal, mover)
        analyzed_set = set(ordered[:30])
    else:
        analyzed_set = set(legal)

    root_analysis = []
    overlay = []
    chosen_child_engine = None

    # The extra full-window analysis must not change the agent's future decisions:
    # _negamax updates the history heuristic and node counters, so snapshot and restore.
    saved_history = dict(agent.history) if hasattr(agent, "history") else None
    saved_counters = (getattr(agent, "nodes_expanded", 0), getattr(agent, "cutoffs", 0))

    for act in legal:
        act_json = action_to_json(act)
        if act in analyzed_set:
            sim = engine.clone()
            sim.apply_action(act, update_clock=False)
            if act == chosen:
                chosen_child_engine = sim

            # Negamax search at depth_completed - 1 with full window (-inf, inf)
            if depth_completed <= 1:
                val = -agent.evaluate(sim, depth=0, ply=1)
            else:
                val = -agent._negamax(
                    sim,
                    depth=depth_completed - 1,
                    alpha=-float("inf"),
                    beta=float("inf"),
                    ply=1,
                    deadline=None,
                    extensions=0,
                )

            # Format label
            if val >= agent.WIN_SCORE - 100:
                plies_to_win = int(round(agent.WIN_SCORE - val))
                label = f"win in {plies_to_win}"
            elif val <= -agent.WIN_SCORE + 100:
                plies_to_loss = int(round(agent.WIN_SCORE + val))
                label = f"loss in {plies_to_loss}"
            else:
                label = f"{val:+.2f}"

            val_float = round(float(val), 2)
            root_analysis.append({"action": act_json, "value": val_float, "kind": "exact"})
            overlay.append({"action": act_json, "value": val_float, "label": label})
        else:
            root_analysis.append({"action": act_json, "value": None, "kind": "not analysed"})
            overlay.append({"action": act_json, "value": None, "label": "not analysed"})

    if saved_history is not None:
        agent.history.clear()
        agent.history.update(saved_history)
    agent.nodes_expanded, agent.cutoffs = saved_counters

    # Extract features for chosen child
    features_chosen = {}
    if chosen_child_engine is not None:
        try:
            from evaluation.features import extract_features, FEATURE_NAMES
            f_vec = extract_features(chosen_child_engine, mover)
            features_chosen = {name: round(float(v), 3) for name, v in zip(FEATURE_NAMES, f_vec)}
        except Exception:
            features_chosen = {}

    chosen_act_name = chosen[0] if chosen else "None"
    chosen_act_val = chosen[1] if chosen else ""
    summary = (
        f"Searched {nodes_expanded:,} positions to depth {depth_completed} ({cutoffs} cutoffs); "
        f"best line starts with {chosen_act_name} {chosen_act_val}, value {root_value:+.2f}."
    )

    details = {
        "depth_requested": agent.depth,
        "depth_completed": depth_completed,
        "nodes_expanded": nodes_expanded,
        "cutoffs": cutoffs,
        "decision_seconds": stats.get("last_decision_time", 0.0),
        "root_value": round(float(root_value), 2) if not math.isinf(root_value) else str(root_value),
        "pv_first_move": action_to_json(chosen) if chosen else None,
        "root": root_analysis,
        "features_chosen": features_chosen,
        "eval_note": "Evaluated with Negamax full-window root search",
    }

    return chosen, details, overlay, summary


def _explain_dqn(agent: Any, engine: Any, mover: int, spec: Optional[str] = None) -> Tuple[Optional[Tuple[str, Any]], Dict[str, Any], List[Dict[str, Any]], str]:
    import torch
    from agents.dqn.encoding import encode_observation

    legal_mask = agent.codec.legal_mask(engine, mover=mover)
    legal_indices = [i for i, ok in enumerate(legal_mask) if ok]
    if not legal_indices:
        return None, {}, [], "No legal moves."

    obs = encode_observation(engine, mover=mover)
    with torch.no_grad():
        x = torch.as_tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
        q_vals = agent.model(x).squeeze(0).cpu().numpy()

    # Greedy choice with deterministic tie-breaking matching DQNAgent.select_action
    best_val = -np.inf
    best_id = legal_indices[0]
    for aid in legal_indices:
        val = q_vals[aid]
        if val > best_val:
            best_val = val
            best_id = aid

    chosen = agent.codec.decode(best_id, engine, mover=mover)

    # Decode all legal actions and pair with Q-values
    action_q_pairs = []
    overlay = []
    for aid in legal_indices:
        act = agent.codec.decode(aid, engine, mover=mover)
        q_val = round(float(q_vals[aid]), 4)
        act_json = action_to_json(act)
        action_q_pairs.append((act_json, q_val))
        overlay.append({"action": act_json, "value": q_val, "label": f"Q={q_val:+.3f}"})

    # Sort descending for top 5
    action_q_pairs.sort(key=lambda item: item[1], reverse=True)
    top5 = [{"action": a, "q": q} for a, q in action_q_pairs[:5]]

    q_chosen = float(q_vals[best_id])
    q_max = float(max(q_vals[aid] for aid in legal_indices))
    q_min = float(min(q_vals[aid] for aid in legal_indices))

    network_meta = _dqn_meta(spec)

    summary = (
        f"Network estimates this position at Q = {q_chosen:+.2f} for the chosen move "
        f"(max over {len(legal_indices)} legal actions)."
    )

    details = {
        "q_chosen": round(q_chosen, 4),
        "q_max": round(q_max, 4),
        "q_min": round(q_min, 4),
        "value_estimate": round(q_max, 4),
        "n_legal": len(legal_indices),
        "top5": top5,
        "network": network_meta,
        "note": "Q ≈ discounted return; +1 win, 0 draw, −1 loss",
    }

    return chosen, details, overlay, summary


def _explain_exact(agent: Any, engine: Any, mover: int) -> Tuple[Optional[Tuple[str, Any]], Dict[str, Any], List[Dict[str, Any]], str]:
    solution = agent.solution
    node = solution.graph.node_of(engine)
    if node is None:
        raise LookupError("State is outside the solved graph.")

    status = solution.status(node)
    raw_rank = solution.rank[node]
    rank = int(raw_rank) if raw_rank is not None else None
    draw_type = solution.draw_type(node)
    disc_val = solution.discounted_value(node, gamma=0.99)
    states_in_graph = len(solution.graph.keys)

    action_values = solution.action_values(node) or []
    chosen = agent.select_action(engine)

    overlay = []
    for act, val, child_rank in action_values:
        act_json = action_to_json(act)
        v = float(val) if val is not None else None
        # child_rank counts plies from the child; this move adds one more ply
        plies = child_rank + 1 if child_rank is not None else None
        if v == 1.0:
            label = f"WIN in {plies}" if plies is not None else "WIN"
        elif v == 0.0:
            label = "DRAW"
        elif v == -1.0:
            label = f"LOSS in {plies}" if plies is not None else "LOSS"
        else:
            label = "?"
        overlay.append({"action": act_json, "value": v, "label": label})

    if status == "WIN":
        summary = f"Solver: this position is a WIN in {rank} plies; chosen move keeps the win."
    elif status == "DRAW":
        summary = "Solver: this position is a DRAW; chosen move preserves the draw."
    else:
        rank_str = f" in {rank} plies" if rank is not None else ""
        summary = f"Solver: this position is a LOSS{rank_str}."

    details = {
        "status": status,
        "rank": rank,
        "draw_type": draw_type,
        "discounted_value": round(float(disc_val), 4) if disc_val is not None else None,
        "states_in_graph": states_in_graph,
    }

    return chosen, details, overlay, summary
