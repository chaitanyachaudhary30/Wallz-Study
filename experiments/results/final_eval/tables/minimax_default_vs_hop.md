### New contribution (Wallz-Study): Minimax default vs hop-aware scorer vs Greedy (paired, distinct games)

| Pairing | Games | Default win % | Hop win % | Difference | Paired bootstrap 95% | Default / hop win % as P0 | as P1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `paired_d1_default_vs_hop` | 100 | 45 | 50 | +5 | [-1, +11] | 50 / 60 | 40 / 40 |
| `paired_d2_default_vs_hop` | 100 | 39 | 50 | +11 | [+1, +21] | 40 / 62 | 38 / 38 |
| `paired_d3_default_vs_hop` | 100 | 45 | 50 | +5 | [-5, +15] | 44 / 60 | 46 / 40 |

Head to head at depth 2 (`vs_minimax_N5_hop_d2_vs_default_d2_summary.json`): hop scorer 54, default 44, draws 2, truncated 0 of 100 distinct games; hop win rate 54.0 [44.3, 63.4]. New contribution (Wallz-Study).
