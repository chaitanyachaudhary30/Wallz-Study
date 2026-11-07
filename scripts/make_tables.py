"""Report Table Generator from JSON/JSONL Experiment Logs (P7.5).

Generates markdown and LaTeX publication tables for:
1. Table X-6: Optimal Action Agreement Rates (Exact retrograde vs Greedy, Minimax, DQN)
2. Table X-9: Agent Strength & Match Benchmark Performance (Wilson CIs)
3. Table X-10: 2^3 Factorial Mechanic Ablation (A/B/C cell rates & Holm-corrected effects)
4. Table X-17: Rule R3 Stalemate vs Repetition Draw Breakdown

Usage:
    python scripts/make_tables.py --output-dir experiments/tables/
    python scripts/make_tables.py --ablation experiments/results/person3_pilot/ablation_N5_minimax2_vs_greedy/summary.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.metrics import holm_bonferroni, wilson


def format_ci(ci: Optional[Tuple[float, float]], pct: bool = True) -> str:
    if ci is None or ci[0] is None or ci[1] is None:
        return "-"
    scale = 100.0 if pct else 1.0
    unit = "%" if pct else ""
    return f"[{ci[0] * scale:.1f}{unit}, {ci[1] * scale:.1f}{unit}]"


def make_agreement_table(agreement_data: Dict[str, Any]) -> Tuple[str, str]:
    """Generates Table X-6: Agreement rates across agents and cells."""
    rows = agreement_data.get('rows', [])
    if not rows:
        return "", ""

    # Markdown
    md = [
        "### Table X-6: Optimal Action Agreement Rates with Exact Retrograde Solutions\n",
        "| Configuration | Agent | Sample Size (N) | Value-Optimal Rate (95% CI) | Rank-Optimal Rate |",
        "| :--- | :--- | :---: | :---: | :---: |",
    ]
    # LaTeX
    tex = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Optimal Action Agreement Rates with Exact Retrograde Solutions (Gate X-6)}",
        r"\label{tab:gate_x6_agreement}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Configuration & Agent & Sample $N$ & Value-Optimal Rate (95\% CI) & Rank-Optimal Rate \\",
        r"\midrule",
    ]

    for row in rows:
        cfg = row.get('config_label', '-')
        agent = row.get('agent', '-')
        rates = row.get('rates', {}).get('all', {})
        n = rates.get('n', row.get('sampled_states', 0))
        val_rate = rates.get('value_optimal_rate')
        val_ci = rates.get('value_optimal_ci95')
        rank_rate = rates.get('rank_optimal_rate')

        val_str = f"{val_rate * 100:.1f}% {format_ci(val_ci)}" if val_rate is not None else "-"
        rank_str = f"{rank_rate * 100:.1f}%" if rank_rate is not None else "-"

        md.append(f"| `{cfg}` | `{agent}` | {n} | {val_str} | {rank_str} |")
        tex.append(f"{cfg.replace('_', r'\_')} & {agent.replace('_', r'\_')} & {n} & {val_str} & {rank_str} \\\\")

    tex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(md), "\n".join(tex)


def make_ablation_cell_table(ablation_data: Dict[str, Any]) -> Tuple[str, str]:
    """Generates Table X-10a: Ablation Cell Summaries."""
    cells = ablation_data.get('cells', {})
    design = ablation_data.get('design', {})
    agent_x = design.get('agent_x', 'Focus Agent')
    agent_y = design.get('agent_y', 'Opponent')

    md = [
        f"### Table X-10a: 2^3 Factorial Ablation Cell Performance ({agent_x} vs {agent_y})\n",
        "| Cell | Games | Win Rate (95% Wilson CI) | Truncation Rate | Mean Plies | Mean Actions | Mean Latency (ms) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    tex = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{Factorial Mechanic Cell Performance: {agent_x} vs {agent_y}}}",
        r"\label{tab:ablation_cells}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Cell & Games & Win Rate (95\% CI) & Trunc. & Length (plies) & Legal Actions & Latency (ms) \\",
        r"\midrule",
    ]

    for label, c in sorted(cells.items()):
        n = c.get('n_games', 0)
        win_rate = c.get('x_goal_win_rate', 0.0)
        win_ci = c.get('x_goal_win_rate_ci95')
        trunc_rate = c.get('truncation_rate', 0.0)
        length = c.get('mean_length_plies', 0.0)
        actions = c.get('mean_legal_actions', 0.0)
        latency = c.get('x_mean_decision_ms', 0.0)

        win_str = f"{win_rate * 100:.1f}% {format_ci(win_ci)}"
        trunc_str = f"{trunc_rate * 100:.1f}%"
        len_str = f"{length:.1f}" if length is not None else "-"
        act_str = f"{actions:.1f}" if actions is not None else "-"
        lat_str = f"{latency:.1f}" if latency is not None else "-"

        md.append(f"| `{label}` | {n} | {win_str} | {trunc_str} | {len_str} | {act_str} | {lat_str} |")
        tex.append(f"{label.replace('_', r'\_')} & {n} & {win_str} & {trunc_str} & {len_str} & {act_str} & {lat_str} \\\\")

    tex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(md), "\n".join(tex)


def make_ablation_effects_table(ablation_data: Dict[str, Any]) -> Tuple[str, str]:
    """Generates Table X-10b: Factorial Effects with Bootstrap CIs and Holm correction."""
    effects_by_resp = ablation_data.get('effects', {})
    if not effects_by_resp:
        return "", ""

    # Primary response is x_goal_win
    win_eff = effects_by_resp.get('x_goal_win', {}).get('effects', {})
    len_eff = effects_by_resp.get('length_plies', {}).get('effects', {})

    md = [
        "### Table X-10b: 2^3 Factorial Effects with Holm-Bonferroni Significance\n",
        "| Factor / Interaction | Win Rate Effect (95% CI) | Win Rate Holm Sig. | Length Effect (plies) | Length 95% CI |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    tex = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Factorial Main Effects and Interactions (95\% Paired Bootstrap CI, Holm-Adjusted)}",
        r"\label{tab:factorial_effects}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Effect & Win Rate $\Delta$ (95\% CI) & Holm Sig. & Length $\Delta$ (plies) & Length 95\% CI \\",
        r"\midrule",
    ]

    for eff_name in ['A', 'B', 'C', 'AB', 'AC', 'BC', 'ABC']:
        w = win_eff.get(eff_name, {})
        l = len_eff.get(eff_name, {})

        w_est = w.get('estimate')
        w_ci = w.get('ci95')
        w_sig = w.get('significant_holm', False)
        if w_sig is None:
            w_sig = (w_ci[0] > 0 or w_ci[1] < 0) if w_ci else False

        l_est = l.get('estimate')
        l_ci = l.get('ci95')

        w_str = f"{w_est * 100:+.2f}% {format_ci(w_ci)}" if w_est is not None else "-"
        sig_str = "Yes (p < 0.05)" if w_sig else "No"
        l_str = f"{l_est:+.2f}" if l_est is not None else "-"
        l_ci_str = format_ci(l_ci, pct=False) if l_ci else "-"

        md.append(f"| **{eff_name}** | {w_str} | {sig_str} | {l_str} | {l_ci_str} |")
        tex.append(f"\\textbf{{{eff_name}}} & {w_str} & {sig_str} & {l_str} & {l_ci_str} \\\\")

    tex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(md), "\n".join(tex)


def make_stalemate_table(ablation_data: Dict[str, Any]) -> Tuple[str, str]:
    """Generates Table X-17: Rule R3 Stalemate vs Repetition Draw Analysis."""
    cells = ablation_data.get('cells', {})
    if not cells:
        return "", ""

    md = [
        "### Table X-17: Rule R3 Stalemate vs Repetition Draw Analysis\n",
        "| Cell | Total Games | Total Draws | 5-Fold Repetition | R3 Stalemate (No Actions) | Stalemate Rate (95% CI) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    tex = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Terminal Draw Breakdown: 5-Fold Repetition vs Rule R3 Stalemate (Gate X-17)}",
        r"\label{tab:draw_breakdown}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Cell & Total Games & Total Draws & Repetition Draws & Stalemate (R3) & Stalemate Rate (95\% CI) \\",
        r"\midrule",
    ]

    total_games = 0
    total_draws = 0
    total_reps = 0
    total_stales = 0

    for label, c in sorted(cells.items()):
        n = c.get('n_games', 0)
        counts = c.get('counts', {})
        draws = counts.get('draw', 0)
        breakdown = c.get('draw_breakdown', {})
        stales = breakdown.get('stalemate', 0)
        reps = breakdown.get('repetition', draws - stales)

        stale_rate = c.get('stalemate_rate', stales / n if n else 0.0)
        stale_ci = c.get('stalemate_rate_ci95', wilson(stales, n))
        stale_ci_str = format_ci(stale_ci)

        total_games += n
        total_draws += draws
        total_reps += reps
        total_stales += stales

        md.append(f"| `{label}` | {n} | {draws} | {reps} | {stales} | {stale_rate * 100:.1f}% {stale_ci_str} |")
        tex.append(f"{label.replace('_', r'\_')} & {n} & {draws} & {reps} & {stales} & {stale_rate * 100:.1f}\\% {stale_ci_str} \\\\")

    # Overall summary row
    overall_stale_rate = total_stales / total_games if total_games else 0.0
    overall_ci = wilson(total_stales, total_games)
    overall_str = f"{overall_stale_rate * 100:.1f}% {format_ci(overall_ci)}"

    md.append(f"| **TOTAL** | **{total_games}** | **{total_draws}** | **{total_reps}** | **{total_stales}** | **{overall_str}** |")
    tex.extend([
        r"\midrule",
        f"\\textbf{{TOTAL}} & {total_games} & {total_draws} & {total_reps} & {total_stales} & {overall_str} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(md), "\n".join(tex)


def make_benchmark_table(benchmark_data: Dict[str, Any]) -> Tuple[str, str]:
    """Generates Table X-9: Agent Strength Benchmarks."""
    results = benchmark_data.get('results', {})
    dqn_spec = benchmark_data.get('dqn_spec', 'DQN Agent')
    board_size = benchmark_data.get('board_size', '?')

    md = [
        f"### Table X-9: Agent Match Benchmarks ({board_size}x{board_size})\n",
        "| Agent | Opponent | Total Matches | Win Rate (95% Wilson CI) | Draw Rate | Loss Rate | Mean Plies |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    tex = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{Agent Match Benchmarks on {board_size}x{board_size} Board (Gate X-9)}}",
        r"\label{tab:gate_x9_benchmarks}",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"Agent & Opponent & Matches & Win Rate (95\% CI) & Draw Rate & Loss Rate & Mean Plies \\",
        r"\midrule",
    ]

    for opp_key, r in results.items():
        opp = r.get('opponent', opp_key)
        n = r.get('total_matches', 0)
        win_rate = r.get('win_rate', 0.0)
        win_ci = r.get('win_rate_wilson_ci95')
        draw_rate = r.get('draw_rate', 0.0)
        loss_rate = r.get('loss_rate', 0.0)
        plies = r.get('mean_plies', 0.0)

        win_str = f"{win_rate * 100:.1f}% {format_ci(win_ci)}"
        draw_str = f"{draw_rate * 100:.1f}%"
        loss_str = f"{loss_rate * 100:.1f}%"
        plies_str = f"{plies:.1f}"

        md.append(f"| `{dqn_spec}` | `{opp}` | {n} | {win_str} | {draw_str} | {loss_str} | {plies_str} |")
        tex.append(f"{dqn_spec.replace('_', r'\_')} & {opp.replace('_', r'\_')} & {n} & {win_str} & {draw_str} & {loss_str} & {plies_str} \\\\")

    tex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(md), "\n".join(tex)


def make_solve_table(solve_data: Dict[str, Any]) -> Tuple[str, str]:
    """Generates Table X-4: Retrograde solve state space and root outcomes across cells (P8.2/P8.3)."""
    cells = solve_data.get('cells', [])
    if not cells:
        return "", ""

    md = [
        "### Exact Retrograde Solve State Space & Equilibrium Values\n",
        "| Cell Config | States | Edges | Complete | Root Status | Root Rank | Draws (Stalemate / Cycle) | Time (s) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    tex = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Exact Retrograde Solved State Space and Game-Theoretic Equilibria}",
        r"\label{tab:exact_solves}",
        r"\begin{tabular}{lccccccr}",
        r"\toprule",
        r"Cell & States & Edges & Complete & Root & Rank & Draws (Stalemate / Cycle) & Time (s) \\",
        r"\midrule",
    ]

    for c in cells:
        rules = c.get('rules', {})
        n = c.get('board_size', '?')
        abc = f"A{int(rules.get('A', False))}B{int(rules.get('B', False))}C{int(rules.get('C', False))}"
        k = rules.get('K', '?')
        m = rules.get('M', '?')
        label = f"N{n}_{abc}_K{k}_M{m}"

        states = c.get('states', 0)
        edges = c.get('edges', 0)
        complete = "Yes" if c.get('complete', False) else "No"
        root_st = c.get('root_status', '-')
        root_rank = c.get('root_rank', '-')
        rank_str = str(root_rank) if root_rank is not None else "$\\infty$"

        dc = c.get('draw_counts', {})
        stale = dc.get('no_legal_actions', 0)
        cycle = dc.get('cycle', 0)
        draw_str = f"{stale} / {cycle}"

        sec = c.get('build_seconds', 0.0) + c.get('solve_seconds', 0.0)

        md.append(f"| `{label}` | {states:,} | {edges:,} | {complete} | **{root_st}** | {rank_str} | {draw_str} | {sec:.2f} |")
        tex.append(f"{label.replace('_', r'\_')} & {states:,} & {edges:,} & {complete} & \\textbf{{{root_st}}} & {rank_str} & {draw_str} & {sec:.2f} \\\\")

    tex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(md), "\n".join(tex)


def generate_all_tables(output_dir: str = "experiments/tables/"):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    generated = []

    # 1. Ablation tables
    ablation_summary_paths = [
        Path("experiments/results/person3_pilot/ablation_N5_minimax2_vs_greedy/summary.json"),
        Path("experiments/results/ablation_5x5_k3/summary.json"),
    ]
    for p in ablation_summary_paths:
        if p.exists():
            with open(p, 'r', encoding='utf-8') as f:
                data = json.load(f)
            md_cells, tex_cells = make_ablation_cell_table(data)
            md_eff, tex_eff = make_ablation_effects_table(data)
            md_draw, tex_draw = make_stalemate_table(data)

            (out_path / "table_x10a_ablation_cells.md").write_text(md_cells, encoding='utf-8')
            (out_path / "table_x10a_ablation_cells.tex").write_text(tex_cells, encoding='utf-8')
            (out_path / "table_x10b_factorial_effects.md").write_text(md_eff, encoding='utf-8')
            (out_path / "table_x10b_factorial_effects.tex").write_text(tex_eff, encoding='utf-8')
            (out_path / "table_x17_draw_breakdown.md").write_text(md_draw, encoding='utf-8')
            (out_path / "table_x17_draw_breakdown.tex").write_text(tex_draw, encoding='utf-8')
            generated.extend([
                "table_x10a_ablation_cells",
                "table_x10b_factorial_effects",
                "table_x17_draw_breakdown"
            ])
            break

    # 2. Agreement table
    agreement_paths = [
        Path("experiments/results/person3_pilot/agreement_N3_K1_M1.json"),
        Path("experiments/results/agreement/agreement_3x3.json"),
        Path("experiments/results/dqn/agreement_N4.json"),
    ]
    for p in agreement_paths:
        if p.exists():
            with open(p, 'r', encoding='utf-8') as f:
                data = json.load(f)
            md_agr, tex_agr = make_agreement_table(data)
            stem = f"table_x6_agreement_{p.stem}"
            (out_path / f"{stem}.md").write_text(md_agr, encoding='utf-8')
            (out_path / f"{stem}.tex").write_text(tex_agr, encoding='utf-8')
            generated.append(stem)

    # 3. Benchmark table
    bench_paths = [
        Path("experiments/results/dqn/benchmark_N3.json"),
        Path("experiments/results/dqn/benchmark_N4.json"),
    ]
    for p in bench_paths:
        if p.exists():
            with open(p, 'r', encoding='utf-8') as f:
                data = json.load(f)
            md_bench, tex_bench = make_benchmark_table(data)
            stem = f"table_x9_benchmark_{p.stem}"
            (out_path / f"{stem}.md").write_text(md_bench, encoding='utf-8')
            (out_path / f"{stem}.tex").write_text(tex_bench, encoding='utf-8')
            generated.append(stem)

    # 4. Solves tables (P8.2 / P8.3)
    solves_dir = Path("experiments/results/solves")
    if solves_dir.exists():
        for p in sorted(solves_dir.glob("solve_*.json")):
            with open(p, 'r', encoding='utf-8') as f:
                data = json.load(f)
            md_sol, tex_sol = make_solve_table(data)
            stem = f"table_x4_{p.stem}"
            (out_path / f"{stem}.md").write_text(md_sol, encoding='utf-8')
            (out_path / f"{stem}.tex").write_text(tex_sol, encoding='utf-8')
            generated.append(stem)

    print("============================================================")
    print(f"Generated {len(generated)} report tables in {out_path}:")
    for g in generated:
        print(f"  - {g}.md / {g}.tex")
    print("============================================================")
    return generated


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate Report Tables from Experiment Data.")
    parser.add_argument("--output-dir", type=str, default="experiments/tables/")
    parser.add_argument("--ablation", type=str, default=None)
    parser.add_argument("--agreement", type=str, default=None)
    parser.add_argument("--benchmark", type=str, default=None)
    args = parser.parse_args()

    if args.ablation or args.agreement or args.benchmark:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        if args.ablation:
            with open(args.ablation, 'r', encoding='utf-8') as f:
                d = json.load(f)
            md, tex = make_ablation_cell_table(d)
            print(md)
        if args.agreement:
            with open(args.agreement, 'r', encoding='utf-8') as f:
                d = json.load(f)
            md, tex = make_agreement_table(d)
            print(md)
        if args.benchmark:
            with open(args.benchmark, 'r', encoding='utf-8') as f:
                d = json.load(f)
            md, tex = make_benchmark_table(d)
            print(md)
    else:
        generate_all_tables(args.output_dir)
