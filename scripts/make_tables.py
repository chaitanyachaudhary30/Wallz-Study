"""Report Table Generator from JSON/JSONL Experiment Logs (P7.5).

Generates markdown and LaTeX publication tables for:
1. Table X-6: Optimal Action Agreement Rates (Exact retrograde vs Greedy, Minimax, DQN)
2. Table X-9: Agent Strength & Match Benchmark Performance (Wilson CIs)
3. Table X-10: 2^3 Factorial Mechanic Ablation (A/B/C cell rates & Holm-corrected effects)
4. Table X-17: Rule R3 Stalemate vs Repetition Draw Breakdown

Reads only the final result folders (experiments/results/final_eval, final_minimax,
final_minimax_hop, final_dqn). Older folders are read only with --include-stale, and
those tables are titled "STALE baseline". The 4-ply factorial ablation is the primary
result; the 2- and 6-ply runs are labelled sensitivity checks. Writes .md and .tex files
plus manifest.json (inputs per table, commit) to the output directory.

Usage:
    python scripts/make_tables.py                      # -> experiments/results/final_eval/tables/
    python scripts/make_tables.py --include-stale
    python scripts/make_tables.py --ablation experiments/results/final_eval/ablation_N5_K3_M10_open4/summary.json
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
        if row.get('applicable') is False:
            continue  # e.g. a DQN checkpoint outside the rules it was trained for
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
        "| Cell | Games | Win Rate (95% Wilson CI) | Draw Rate | Truncation Rate | Mean Plies | Mean Actions | Mean Latency (ms) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    tex = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{Factorial Mechanic Cell Performance: {agent_x} vs {agent_y}}}",
        r"\label{tab:ablation_cells}",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"Cell & Games & Win Rate (95\% CI) & Draw & Trunc. & Length (plies) & Legal Actions & Latency (ms) \\",
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
        draw_rate = c.get('draw_rate')
        draw_str = f"{draw_rate * 100:.1f}%" if draw_rate is not None else "-"
        len_str = f"{length:.1f}" if length is not None else "-"
        act_str = f"{actions:.1f}" if actions is not None else "-"
        lat_str = f"{latency:.1f}" if latency is not None else "-"

        md.append(f"| `{label}` | {n} | {win_str} | {draw_str} | {trunc_str} | {len_str} | {act_str} | {lat_str} |")
        tex.append(f"{label.replace('_', r'\_')} & {n} & {win_str} & {draw_str} & {trunc_str} & {len_str} & {act_str} & {lat_str} \\\\")

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


def make_greedy_table(summary_data: dict) -> tuple[str, str]:
    """Delegates greedy table rendering to scripts.run_greedy_matches."""
    from scripts.run_greedy_matches import make_greedy_table as _make
    return _make(summary_data)



# ---------------------------------------------------------------------------
# Inputs. Only the final result folders are read by default. Older folders are
# read only with --include-stale, and every table made from them is titled
# "STALE baseline" (see experiments/results/STALE.md for why each one is stale).
# ---------------------------------------------------------------------------

R = Path("experiments/results")
FINAL_EVAL, FINAL_MINIMAX, FINAL_MINIMAX_HOP, FINAL_DQN = (
    R / "final_eval", R / "final_minimax", R / "final_minimax_hop", R / "final_dqn")
# The 4-ply ablation is the primary result, fixed beforehand (shortest opening length
# with at least 30 distinct openings); 2- and 6-ply runs are sensitivity checks only.
ABLATIONS = [
    ("primary (4-ply openings, 40 distinct)", FINAL_EVAL / "ablation_N5_K3_M10_open4" / "summary.json", "primary"),
    ("sensitivity (2-ply openings, 9 distinct; pilot comparison)",
     FINAL_EVAL / "ablation_N5_K3_M10_open2_pilot_comparison" / "summary.json", "sens_open2"),
    ("sensitivity (6-ply openings, 40 distinct)", FINAL_EVAL / "ablation_N5_K3_M10_open6" / "summary.json",
     "sens_open6"),
]
STALE_INPUTS = {
    "ablation": R / "person3_pilot" / "ablation_N5_minimax2_vs_greedy" / "summary.json",
    "agreement": R / "person3_pilot" / "agreement_N3_K1_M1.json",
    "greedy": R / "greedy" / "summary.json",
}


def _label(md: str, tex: str, label: str) -> Tuple[str, str]:
    """Put a label in front of a table's markdown heading and LaTeX caption."""
    return (md.replace("### ", f"### {label}: ", 1),
            tex.replace("\\caption{", "\\caption{" + label.replace('_', r'\_') + ": ", 1))


def _load(p: Path):
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)


def make_multi_seed_table(data: Dict[str, Any]) -> Tuple[str, str]:
    """DQN 3x3 across seeds: mean, SD, 95% t-interval per opponent and rate."""
    agg = data.get('aggregate', {})
    md = ["### DQN 3x3 multi-seed study (seeds " + ", ".join(map(str, agg.get('seeds', []))) + ")\n",
          "| Opponent | Metric | Mean | SD | 95% t-interval | Min | Max | Pooled (95% Wilson) |",
          "| :--- | :--- | ---: | ---: | :---: | ---: | ---: | :---: |"]
    tex = [r"\begin{table}[htbp]", r"\centering", r"\caption{DQN 3x3 Multi-Seed Study}",
           r"\label{tab:multi_seed}", r"\begin{tabular}{llrrcrrc}", r"\toprule",
           r"Opponent & Metric & Mean & SD & 95\% t-interval & Min & Max & Pooled (Wilson) \\", r"\midrule"]

    def line(opp, name, s, pooled=None):
        if not s or s.get('mean') is None:
            return
        ci = format_ci(s.get('ci95_t'), pct=False) if s.get('ci95_t') else "-"
        sd = f"{s['sd']:.3f}" if s.get('sd') is not None else "-"
        pl = "-"
        if pooled is not None:
            pl = f"{pooled[0]:.3f} {format_ci(pooled[1], pct=False)}"
        row = [opp, name, f"{s['mean']:.3f}", sd, ci, f"{s['min']:.3f}", f"{s['max']:.3f}", pl]
        md.append("| " + " | ".join(row) + " |")
        tex.append(" & ".join(x.replace('_', r'\_') for x in row) + r" \\")

    line('-', 'sign agreement', agg.get('sign_agreement'))
    line('-', 'value MAE', agg.get('mae'))
    for opp, b in agg.get('vs', {}).items():
        p = b.get('pooled', {})
        for m in ('win_rate', 'draw_rate', 'truncation_rate'):
            line(opp, m.replace('_', ' '), b.get(m), (p.get(m), p.get(m + '_ci95')))
        line(opp, 'mean length (plies)', b.get('mean_length_plies'))
    tex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(md), "\n".join(tex)


def make_vs_greedy_table(paths: List[Path]) -> Tuple[str, str]:
    """Minimax-vs-Greedy match summaries from final_minimax/ and final_minimax_hop/."""
    md = ["### Minimax vs Greedy (final engine)\n",
          "| Run | Minimax spec | N | Opening plies | Games | Win rate (95% Wilson) | Draw rate | Truncation rate | Mean plies |",
          "| :--- | :--- | :---: | :---: | ---: | :---: | ---: | ---: | ---: |"]
    tex = [r"\begin{table}[htbp]", r"\centering", r"\caption{Minimax vs Greedy (final engine)}",
           r"\label{tab:minimax_vs_greedy}", r"\begin{tabular}{llccrcrrr}", r"\toprule",
           r"Run & Spec & N & Open. & Games & Win (95\% CI) & Draw & Trunc. & Plies \\", r"\midrule"]
    for p in paths:
        d = _load(p)
        o = d.get('results', {}).get('overall', {})
        if 'minimax_win_rate' not in o:
            continue
        c = d.get('config', {})
        row = [f"{p.parent.name}/{p.stem}", c.get('minimax_spec', '-'), str(c.get('board_size', '-')),
               str(c.get('opening_plies', '-')), str(o.get('matches', '-')),
               f"{o['minimax_win_rate'] * 100:.1f}% {format_ci(o.get('minimax_win_rate_wilson95'))}",
               f"{o.get('draw_rate', 0) * 100:.1f}%", f"{o.get('truncation_rate', 0) * 100:.1f}%",
               f"{o.get('mean_length_plies_incl_opening', 0):.1f}"]
        md.append("| " + " | ".join(f"`{x}`" if i < 2 else x for i, x in enumerate(row)) + " |")
        tex.append(" & ".join(x.replace('_', r'\_').replace('%', r'\%') for x in row) + r" \\")
    tex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(md), "\n".join(tex)


def make_dqn_aggregate_table(data: Dict[str, Any]) -> Tuple[str, str]:
    """final_dqn/aggregate_3x3_seeds.json: mean / min / max over the five 3x3 seeds."""
    md = ["### DQN 3x3, five seeds (Member 2 evaluation)\n",
          "| Metric | Mean | Min | Max |", "| :--- | ---: | ---: | ---: |"]
    tex = [r"\begin{table}[htbp]", r"\centering", r"\caption{DQN 3x3, five seeds}", r"\label{tab:dqn_seeds}",
           r"\begin{tabular}{lrrr}", r"\toprule", r"Metric & Mean & Min & Max \\", r"\midrule"]
    for k, v in data.items():
        if isinstance(v, dict) and {'mean', 'min', 'max'} <= set(v):
            row = [k, f"{v['mean']:.4f}", f"{v['min']:.4f}", f"{v['max']:.4f}"]
            md.append(f"| `{k}` | {row[1]} | {row[2]} | {row[3]} |")
            tex.append(" & ".join(x.replace('_', r'\_') for x in row) + r" \\")
    tex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(md), "\n".join(tex)


def generate_all_tables(output_dir: str = "experiments/results/final_eval/tables/", include_stale: bool = False):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    generated, inputs = [], []

    def write(stem, md, tex, src):
        if not md:
            return
        (out_path / f"{stem}.md").write_text(md + "\n", encoding='utf-8')
        (out_path / f"{stem}.tex").write_text(tex + "\n", encoding='utf-8')
        generated.append(stem)
        inputs.append({'table': stem, 'inputs': [str(s).replace('\\', '/') for s in src]})

    # 1. Exact solves
    for p in sorted(FINAL_EVAL.glob("solve_*.json")):
        md, tex = make_solve_table(_load(p))
        write(f"solve_{p.stem[6:]}", *_label(md, tex, p.stem), [p])

    # 2. Factorial ablation: primary first, sensitivity checks labelled as such
    for label, p, tag in ABLATIONS:
        if not p.exists():
            continue
        d = _load(p)
        write(f"ablation_{tag}_cells", *_label(*make_ablation_cell_table(d), label), [p])
        write(f"ablation_{tag}_effects", *_label(*make_ablation_effects_table(d), label), [p])
        write(f"ablation_{tag}_draws", *_label(*make_stalemate_table(d), label), [p])

    # 3. Agent-vs-exact agreement (all agents on the same sampled states)
    p = FINAL_EVAL / "agreement_N3_K1.json"
    if p.exists():
        write("agreement_N3_K1", *make_agreement_table(_load(p)), [p])

    # 4. DQN multi-seed study
    p = FINAL_EVAL / "multi_seed_3x3.json"
    if p.exists():
        write("multi_seed_3x3", *make_multi_seed_table(_load(p)), [p])

    # 5. Minimax vs Greedy
    mm = sorted(FINAL_MINIMAX.glob("*_summary.json")) + sorted(FINAL_MINIMAX_HOP.glob("*_summary.json"))
    if mm:
        write("minimax_vs_greedy", *make_vs_greedy_table(mm), mm)

    # 6. DQN evaluation by Member 2
    p = FINAL_DQN / "aggregate_3x3_seeds.json"
    if p.exists():
        write("dqn_3x3_seeds", *make_dqn_aggregate_table(_load(p)), [p])

    # 7. Stale baselines, only on request
    if include_stale:
        lab = "STALE baseline (pre-freeze rules or agents; see experiments/results/STALE.md)"
        s = STALE_INPUTS
        if s['ablation'].exists():
            write("stale_pilot_ablation_effects", *_label(*make_ablation_effects_table(_load(s['ablation'])), lab),
                  [s['ablation']])
        if s['agreement'].exists():
            write("stale_pilot_agreement", *_label(*make_agreement_table(_load(s['agreement'])), lab),
                  [s['agreement']])
        if s['greedy'].exists():
            write("stale_greedy_matches", *_label(*make_greedy_table(_load(s['greedy'])), lab), [s['greedy']])

    from evaluation.match_runner import run_metadata
    with open(out_path / "manifest.json", 'w', encoding='utf-8') as f:
        json.dump({'run': run_metadata(), 'include_stale': include_stale, 'tables': inputs}, f, indent=2)
    print(f"Generated {len(generated)} tables in {out_path}: {', '.join(generated)}")
    return generated


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate report tables from the final result folders.")
    parser.add_argument("--output-dir", type=str, default="experiments/results/final_eval/tables/")
    parser.add_argument("--seed", type=int, default=42, help="kept for old commands; tables use no randomness")
    parser.add_argument("--include-stale", action="store_true",
                        help="also render old (pre-freeze) results, each table titled 'STALE baseline'")
    parser.add_argument("--ablation", type=str, default=None)
    parser.add_argument("--agreement", type=str, default=None)
    parser.add_argument("--benchmark", type=str, default=None)
    args = parser.parse_args()

    if args.ablation or args.agreement or args.benchmark:
        if args.ablation:
            print(make_ablation_cell_table(_load(Path(args.ablation)))[0])
        if args.agreement:
            print(make_agreement_table(_load(Path(args.agreement)))[0])
        if args.benchmark:
            print(make_benchmark_table(_load(Path(args.benchmark)))[0])
    else:
        generate_all_tables(args.output_dir, include_stale=args.include_stale)
