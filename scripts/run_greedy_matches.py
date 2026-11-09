#!/usr/bin/env python3
"""Greedy match campaign runner, statistical aggregator, and table generator (P07).

Executes reproducible evaluation matches of Greedy against Random, Minimax (depth 2),
and seeded Greedy self-play across N3, N5, and N9 boards.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.match_runner import MatchConfig, run_batch, verify_replay
from evaluation.metrics import outcome_for, wilson


BOARD_CONFIGS = {
    "N3_solver": {"board_size": 3, "rules": {"A": True, "B": True, "C": True, "K": 1, "M": 1}},
    "N5_final": {"board_size": 5, "rules": {"A": True, "B": True, "C": True, "K": 3, "M": 10}},
    "N9_final": {"board_size": 9, "rules": {"A": True, "B": True, "C": True, "K": 3, "M": 10}},
}


def get_git_commit() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "unknown"


def build_configs(
    quick: bool = False,
    seeds: Optional[List[int]] = None,
) -> Dict[Tuple[str, str], List[MatchConfig]]:
    """Builds MatchConfig lists grouped by (pairing_name, board_name)."""
    if seeds is None:
        seeds = [42, 43, 44, 45, 46]

    if quick:
        selected_boards = ["N3_solver", "N5_final"]
        selected_seeds = [seeds[0]]
        opening_counts = {"N3_solver": 2, "N5_final": 2}
    else:
        selected_boards = ["N3_solver", "N5_final", "N9_final"]
        selected_seeds = seeds
        opening_counts = {"N3_solver": 10, "N5_final": 10, "N9_final": 4}

    batches: Dict[Tuple[str, str], List[MatchConfig]] = {}

    pairings = ["greedy_vs_random", "greedy_vs_minimax2", "greedy_vs_greedy_self"]

    for b_name in selected_boards:
        b_info = BOARD_CONFIGS[b_name]
        n_openings = opening_counts[b_name]

        for p_name in pairings:
            configs = []
            for s in selected_seeds:
                for op_idx in range(n_openings):
                    for greedy_colour in (0, 1):
                        if p_name == "greedy_vs_random":
                            p0 = "greedy" if greedy_colour == 0 else f"random:seed={s}"
                            p1 = f"random:seed={s}" if greedy_colour == 0 else "greedy"
                        elif p_name == "greedy_vs_minimax2":
                            p0 = "greedy" if greedy_colour == 0 else "minimax:depth=2"
                            p1 = "minimax:depth=2" if greedy_colour == 0 else "greedy"
                        elif p_name == "greedy_vs_greedy_self":
                            p0 = f"greedy:seed={s}" if greedy_colour == 0 else f"greedy:seed={s+1000}"
                            p1 = f"greedy:seed={s+1000}" if greedy_colour == 0 else f"greedy:seed={s}"
                        else:
                            raise ValueError(f"Unknown pairing {p_name}")

                        cfg = MatchConfig(
                            board_size=b_info["board_size"],
                            rules=b_info["rules"],
                            p0=p0,
                            p1=p1,
                            max_plies=200,
                            opening_plies=2,
                            opening_seed=s * 100 + op_idx,
                            record_replay=True,
                        )
                        configs.append(cfg)

            batches[(p_name, b_name)] = configs

    return batches


def summarise(results: List[Dict[str, Any]], pairing_name: str, board_name: str) -> Dict[str, Any]:
    """Computes summary metrics for a batch of matches from greedy's perspective."""
    total_games = len(results)
    if total_games == 0:
        return {}

    wins = 0
    draws = 0
    losses = 0
    plies_list = []
    greedy_time_total = 0.0
    greedy_moves = 0
    categories: Dict[str, int] = {}
    colour_stats = {0: {"games": 0, "wins": 0, "draws": 0, "losses": 0}, 1: {"games": 0, "wins": 0, "draws": 0, "losses": 0}}
    seed_wins: Dict[int, List[int]] = {}

    for res in results:
        # Determine which player index was the focused greedy agent
        cfg = res["config"]
        p0_spec = cfg.get("p0", "")
        p1_spec = cfg.get("p1", "")

        if p0_spec.startswith("greedy") and not p1_spec.startswith("greedy"):
            greedy_p = 0
        elif p1_spec.startswith("greedy") and not p0_spec.startswith("greedy"):
            greedy_p = 1
        elif p0_spec.startswith("greedy") and p1_spec.startswith("greedy"):
            # Self-play: focus is p0 (seed s)
            greedy_p = 0
        else:
            greedy_p = 0

        # Outcome from greedy's perspective
        res_outcome = outcome_for(res, greedy_p)
        if res_outcome == "win" or res_outcome == "forfeit_win":
            wins += 1
            colour_stats[greedy_p]["wins"] += 1
            is_win = 1
        elif res_outcome == "draw" or res_outcome == "truncated":
            draws += 1
            colour_stats[greedy_p]["draws"] += 1
            is_win = 0
        else:
            losses += 1
            colour_stats[greedy_p]["losses"] += 1
            is_win = 0

        colour_stats[greedy_p]["games"] += 1

        # Seed tracking
        seed = cfg.get("opening_seed", 0) // 100
        seed_wins.setdefault(seed, []).append(is_win)

        # Plies & timing
        plies = res.get("total_plies", len(res.get("replay", [])))
        plies_list.append(plies)

        cat = res["outcome"]["category"]
        categories[cat] = categories.get(cat, 0) + 1

        # run_match records per-player {"n", "mean", "max", "total"} under decision_seconds
        # (the old lookup of metrics.move_times did not exist, so every mean was 0.0 ms)
        ds = (res.get("decision_seconds") or {}).get(str(greedy_p)) or {}
        if ds.get("n"):
            greedy_time_total += ds.get("total", 0.0)
            greedy_moves += ds["n"]

    win_rate = wins / total_games
    draw_rate = draws / total_games
    w_ci_low, w_ci_high = wilson(wins, total_games)
    d_ci_low, d_ci_high = wilson(draws, total_games)

    mean_plies = sum(plies_list) / total_games
    var_plies = sum((x - mean_plies) ** 2 for x in plies_list) / max(1, total_games - 1)
    sd_plies = math.sqrt(var_plies)

    mean_time = greedy_time_total / max(1, greedy_moves)

    # Per-seed win rates
    seed_rates = {s: sum(arr) / len(arr) for s, arr in seed_wins.items()}
    mean_seed_wr = sum(seed_rates.values()) / max(1, len(seed_rates))
    var_seed_wr = sum((r - mean_seed_wr) ** 2 for r in seed_rates.values()) / max(1, len(seed_rates) - 1) if len(seed_rates) > 1 else 0.0
    sd_seed_wr = math.sqrt(var_seed_wr)

    return {
        "pairing": pairing_name,
        "board": board_name,
        "games": total_games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": round(win_rate, 4),
        "win_rate_ci": [round(w_ci_low, 4), round(w_ci_high, 4)],
        "draw_rate": round(draw_rate, 4),
        "draw_rate_ci": [round(d_ci_low, 4), round(d_ci_high, 4)],
        "mean_plies": round(mean_plies, 2),
        "sd_plies": round(sd_plies, 2),
        "categories": categories,
        "mean_greedy_decision_sec": round(mean_time, 5),
        "by_colour": colour_stats,
        "per_seed_rates": {str(k): round(v, 4) for k, v in seed_rates.items()},
        "seed_mean_sd": [round(mean_seed_wr, 4), round(sd_seed_wr, 4)],
    }


def make_greedy_table(summary_data: Dict[str, Any]) -> Tuple[str, str]:
    """Generates Markdown and LaTeX table representations for Greedy campaign results."""
    records = summary_data.get("campaign_results", [])
    if not records and isinstance(summary_data, list):
        records = summary_data

    headers_md = [
        "Pairing", "Board", "Games", "W-D-L", "Win Rate [95% CI]", "Draw Rate", "Plies (Mean±SD)", "Time/Move"
    ]
    lines_md = [
        "| " + " | ".join(headers_md) + " |",
        "| " + " | ".join(["---"] * len(headers_md)) + " |",
    ]

    tex_rows = []

    for r in records:
        pairing_disp = r["pairing"].replace("_", " ").title()
        board_disp = r["board"]
        games = str(r["games"])
        wdl = f"{r['wins']}-{r['draws']}-{r['losses']}"
        wr_ci = f"{r['win_rate']*100:.1f}% [{r['win_rate_ci'][0]*100:.1f}, {r['win_rate_ci'][1]*100:.1f}]"
        dr = f"{r['draw_rate']*100:.1f}%"
        plies_sd = f"{r['mean_plies']:.1f} ± {r['sd_plies']:.1f}"
        t_move = f"{r['mean_greedy_decision_sec']*1000:.1f} ms"

        lines_md.append(f"| {pairing_disp} | {board_disp} | {games} | {wdl} | {wr_ci} | {dr} | {plies_sd} | {t_move} |")

        tex_rows.append(
            f"{pairing_disp} & {board_disp} & {games} & {wdl} & {r['win_rate']*100:.1f}\\% [{r['win_rate_ci'][0]*100:.1f}, {r['win_rate_ci'][1]*100:.1f}] & {dr} & {plies_sd} & {t_move} \\\\"
        )

    md_table = "\n".join(lines_md)

    tex_table = "\n".join([
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Greedy Baseline Match Campaign Performance Across Configs}",
        r"\label{tab:greedy_matches}",
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        r"Pairing & Board & Games & W-D-L & Win Rate [95\% CI] & Draw Rate & Plies (Mean$\pm$SD) & Time/Move \\",
        r"\midrule",
        *tex_rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return md_table, tex_table


def main():
    parser = argparse.ArgumentParser(description="Run full greedy baseline match campaign.")
    parser.add_argument("--quick", action="store_true", help="Quick smoke run on N3 and N5")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46], help="Seeds to evaluate")
    parser.add_argument("--out-dir", type=str, default="experiments/results/greedy", help="Output directory")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    table_dir = "experiments/tables"
    os.makedirs(table_dir, exist_ok=True)

    print(f"Building match configs (quick={args.quick}, seeds={args.seeds})...")
    batches = build_configs(quick=args.quick, seeds=args.seeds)
    total_matches = sum(len(cfgs) for cfgs in batches.values())
    print(f"Total matches scheduled: {total_matches} across {len(batches)} batches.")

    campaign_summary = []
    all_results = []

    for (p_name, b_name), cfgs in batches.items():
        jsonl_path = os.path.join(args.out_dir, f"{p_name}_{b_name}.jsonl")
        print(f"Running batch {p_name} on {b_name} ({len(cfgs)} games) -> {jsonl_path}...")
        results = run_batch(cfgs, jsonl_path, keep_replay=True)
        # Read all results from file (including previously computed)
        batch_results = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    batch_results.append(json.loads(line))

        # Verify replays
        for r in batch_results:
            ok, msg = verify_replay(r)
            if not ok:
                print(f"Error: replay verification failed for {r.get('match_id')}: {msg}", file=sys.stderr)
                sys.exit(1)

        summary_row = summarise(batch_results, p_name, b_name)
        campaign_summary.append(summary_row)
        all_results.extend(batch_results)

    summary_payload = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "commit": get_git_commit(),
        "args": vars(args),
        "campaign_results": campaign_summary,
    }

    summary_file = os.path.join(args.out_dir, "summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)
    print(f"Wrote summary to {summary_file}")

    md_tbl, tex_tbl = make_greedy_table(summary_payload)
    with open(os.path.join(table_dir, "table_greedy_matches.md"), "w", encoding="utf-8") as f:
        f.write(md_tbl + "\n")
    with open(os.path.join(table_dir, "table_greedy_matches.tex"), "w", encoding="utf-8") as f:
        f.write(tex_tbl + "\n")
    print("Wrote table_greedy_matches.md and table_greedy_matches.tex")


if __name__ == "__main__":
    main()
