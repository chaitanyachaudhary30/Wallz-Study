# Person 3 — Match Runner, Exact Solver and Evaluation

**Owner**: Chaitanya Chaudhary (Person 3) · **Branch**: `chaitanya/person3-evaluation`
**Scope**: reproducible match runner and logs, reachable-state graph, exact tiny-board
solver, evaluation/ablation tools, replay export. Minimax (Person 1), the shared engine and
greedy baseline (Person 2) and DQN self-play (Person 4) are used through their existing
interfaces, not re-implemented.

---

## 1. Module map

| File | Purpose |
| :--- | :--- |
| `evaluation/state_graph.py` | Reachable position graph built **only** through the shared engine; caps on states / edges / seconds. |
| `evaluation/retrograde.py` | Cycle-aware retrograde (attractor) solver: values, exact ranks, value-/rank-optimal action sets, horizon and discounted values. |
| `evaluation/match_runner.py` | Agent registry, `MatchConfig`, `run_match`, outcome taxonomy, JSONL batches with resume, `verify_replay`. |
| `evaluation/metrics.py` | Outcome accounting, Wilson intervals, 2³ factorial effects, paired bootstrap over opening seeds, agreement rates. |
| `evaluation/ablation.py` | The 8 A/B/C cells, paired ablation design, solver sweeps, agent-vs-exact agreement. |
| `evaluation/replay_export.py` | Converts a logged match into the browser demo's action objects, with a compatibility verdict. |
| `scripts/run_experiments.py` | CLI: `match`, `ablation`, `solve`, `agreement`, `verify-replay`, `export-demo`. |
| `tests/test_engine_prerequisites.py`, `tests/test_retrograde.py`, `tests/test_match_runner.py` | 55 tests for the engine fixes, solver and runner. |
| `experiments/results/person3_pilot/` | Small JSON outputs of the pilots in §6 (full per-ply logs stay in the git-ignored `experiments/logs/`). |

Existing tools are kept: `run_match.py` (console runner) and `agents/exact_solver.py` +
`scripts/solve_tiny_board.py` (the earlier solver; `tests/test_retrograde.py` checks that both
solvers give identical WIN/LOSS/DRAW labels on every state of two 3×3 configurations).

---

## 2. Engine prerequisites fixed in `wallz_engine.py`

Each fix has a regression test in `tests/test_engine_prerequisites.py`.

| Issue found | Fix | Source of the rule |
| :--- | :--- | :--- |
| Block timer was decremented on the placement ply, so a block lasted K−1 plies (K=1 vanished at once). | New block is inserted after the decay step: impassable for exactly the next **K** plies. | Proposal eq. *temporary_block_decay* (τ<sub>t+1</sub>(c*) = K); the demo also locks 3 plies for K=3. |
| Wall budget fixed at 10. | `rules['M']` (default 10). | Needed for tiny-board solving. |
| A HOP could land in a sealed pocket with no goal path. | HOP checks path preservation. | `docs/game_rules.md`: every action must preserve a path. |
| `apply_action` accepted any MOVE/HOP target (pawns could teleport). | MOVE/HOP validated; unknown action types rejected. | — (three old tests relied on illegal moves and were corrected). |
| Legal actions were generated in finished games; actions could be applied after the end. | `get_legal_actions()` returns `[]` and `apply_action` raises once terminal. | — |
| Side to move could have **no legal action** (reachable on tiny boards). | Such a position is a **draw** (`terminal_reason = 'no_legal_actions'`). | Team decision, 6 Nov 2026. |
| No record of *why* a game ended. | `terminal_reason` ∈ {`goal`, `timeout`, `repetition`, `no_legal_actions`}. | — |
| Solver needs exact restore without history/clock. | `restore_state_key(key)` (fresh history) and `has_legal_action()`. | — |

**Convention made explicit (open for the team).** Path-preservation checks for WALL,
REPLACE and BLOCK treat every currently active temporary block as impassable. That includes a
block that expires on this same ply. Walls therefore also treat temporary blocks as obstacles.
The proposal's PathExists(s<sub>t+1</sub>) would ignore a block expiring on this ply, so the engine is
conservative by at most one ply. It was left as implemented on `main`.

Checked and left unchanged: history is recorded after the side to move switches (already fixed
on `main`); HOP needs no obstacle to jump over (current `docs/game_rules.md`); REPLACE moves only
the latest wall (current team rule, replacing the older FIFO design).

---

## 3. The exact game that is solved

**Node** = engine state key: both pawn cells, each player's wall list *in placement order*,
inventories, latest-wall pointers, hop/block availability, active block cells with remaining
timers, side to move. Board size N and rules (A, B, C, K, M) are graph metadata. The key is the
complete future-relevant state: `restore_state_key` reproduces identical legal actions and
successors (tested on ~10k random positions by an independent review).

**Canonical positions.** The engine key keeps each player's walls in placement order, but
later play depends only on the union of walls, the inventories and, with rule A, each
player's latest wall. `canonical_key` therefore merges keys that differ only in the order
or owner of the other walls, and drops the latest pointer when A is off. Without this,
state counts are inflated in the A = 0 cells (N=3, M=1, A=0: 1953 raw keys vs 1341
positions), which would bias A's measured effect. A test checks that every raw key merged
into a node has the same exact value and rank under the independent reference. Use
`StateGraph(..., canonical=False)` for raw keys.

**Position game** (what `retrograde.solve` answers):

* players alternate from the root; reaching the goal row wins immediately;
* a non-terminal position whose mover has no legal action is a draw;
* play that never ends is a draw;
* no clocks, no turn counter, no repetition history.

Values are from the **side to move**'s perspective: WIN = +1, DRAW = 0, LOSS = −1.
`value_for(node, player)` converts perspective. `rank` = plies to the goal under optimal play
(winner minimises, loser maximises); draws have no rank.

**Relation to the engine's real rules**

* *5-fold repetition.* From a position with a fresh history (e.g. the initial position) the value is
  the same as in the position game. If the position game says WIN, the winner can always move to a
  child of rank one lower; ranks strictly decrease, so no position ever repeats and repetition never
  intervenes. If it says DRAW, each side can always move to another DRAW node, so the opponent never
  reaches the goal; any infinite play in a finite graph repeats a position 5 times and is a
  repetition draw. With a *non-empty* history the solver's value is an optimistic bound: a WIN may
  become a repetition draw if its winning line passes positions already seen four times.
* *Step limit (T<sub>max</sub> = 200 in `docs/game_rules.md`).* This is a finite-horizon game. Its exact value with h
  plies left is `horizon_value(node, h)` = value if rank ≤ h, else DRAW. The runner treats the cap as
  **truncation**, not as a proven draw.
* *Clocks.* Not modelled. The runner can enforce a per-decision time budget instead (§4).

**Algorithm** (`evaluation/retrograde.py`, `solve`):

1. Seed goal terminals (rank 0): WIN if the side to move has already won, else LOSS.
   No-legal-action terminals are DRAW and are not propagated.
2. For every expanded non-terminal node, count its *distinct* children (`remaining[u]`).
   Two actions that reach the same child count once, and the predecessor list holds each parent once.
3. FIFO queue: when a LOSS node v is popped, each unresolved parent becomes WIN with rank(v)+1. When a WIN node is popped, each
   unresolved parent's counter drops; at zero the parent is LOSS with rank = max child rank + 1.
   FIFO order handles nodes in non-decreasing rank, so ranks are exact.
4. Complete graph: every unresolved node is DRAW (neither side can force the goal).
   **Incomplete graph** (a cap was hit): unresolved nodes are UNKNOWN. A WIN needs only one proven LOSS child and a LOSS needs all children of a fully expanded node, so every WIN/LOSS label is still sound.

**Optimal actions.** `value_optimal_actions` = every action that keeps the node's value (used for
agreement scores: any optimal action is accepted). `rank_optimal_actions` = the fastest win /
slowest loss subset. In WIN nodes only these guarantee progress; a value-keeping move
can otherwise cycle forever.

**Value conventions for comparing against learned values.**
`discounted_value(node, γ)` is the optimal value under the DQN convention in the proposal
(reward +1 to the mover on a winning move, 0 otherwise, target r − γ·V(next)):
±γ<sup>rank−1</sup> for decided nodes, 0 for draws and terminals. Exact ±1/0 values (γ = 1) must not be
compared with γ = 0.99 Q-values directly. Under discounting only rank-optimal actions are
greedy-optimal. Both identities are tested against direct value iteration.

**Caps.** `GraphCaps(max_states, max_edges, max_seconds)` is checked after each fully expanded
node. A capped run reports `complete: false`, `stop_reason` and UNKNOWN counts; it is never
reported as exact or drawn. Its WIN/LOSS labels are sound, but a rank is only an **upper
bound** (a faster line may pass through an unexpanded node). So `rank_optimal_actions` and
`discounted_value` return `None`, and `horizon_value` answers only when the bound settles it.
Summaries mark this with `root_rank_is_upper_bound`.

---

## 4. Match runner

```python
from evaluation.match_runner import MatchConfig, run_match, verify_replay
cfg = MatchConfig(board_size=5, rules={'A': True, 'B': True, 'C': True, 'K': 3, 'M': 10},
                  p0='minimax:depth=2', p1='greedy', max_plies=200,
                  opening_plies=2, opening_seed=7, move_time_limit=None)
result = run_match(cfg)          # JSON-serialisable dict
ok, msg = verify_replay(result)  # re-simulates every ply with the current engine
```

**Agent specs**: `greedy`, `minimax:depth=D[,time_limit=S]` (no time limit means deterministic),
`random:seed=S`, `exact` (tiny boards, rank-optimal play from a complete solve), `dqn` → raises
`AgentUnavailable` because `agents/dqn/*.py` are empty scaffolds (no encoder, action-id map or
checkpoint format yet).

**Openings**: `opening_plies` random pawn MOVEs from `random.Random(opening_seed)`. A MOVE's
legality does not depend on A/B/C before any wall/hop/block is used, so a seed yields the
**same opening in all 8 cells** (tested), which makes the ablation paired.

**Outcome taxonomy and metrics**

| `outcome.category` | Meaning | Counted as |
| :--- | :--- | :--- |
| `goal` | pawn reached goal row | win / loss |
| `draw_repetition` | 5-fold repetition (engine rule) | draw |
| `draw_no_legal_actions` | mover has no legal action | draw |
| `truncated` | runner's `max_plies` cap reached | **truncated** (separate; not a draw) |
| `timeout` | decision took longer than `move_time_limit` | forfeit by the mover |
| `illegal_action` | returned action not in `get_legal_actions()` | forfeit |
| `no_action` | returned `None` although actions existed | forfeit |
| `agent_error` | `select_action` raised | forfeit (message logged) |

Forfeits are reported as `forfeit_win` / `forfeit_loss`, never merged with goal wins. Transitions
use `apply_action(update_clock=False)`; time is measured by the runner with `perf_counter`.

**Result schema** (`schema_version` 1, one JSON object per match / one line in JSONL):
`match_id`, `source{commit, dirty}`, `config{board_size, rules{A,B,C,K,M}, p0, p1, max_plies,
opening_plies, opening_seed, move_time_limit}`, `config_label` (e.g. `N5_A1B0C1_K3_M10`),
`agents{"0","1": {spec, name, params}}`, `opening{seed, plies, actions}`, `initial_state_id`,
`outcome{category, winner, loser, detail}`, `length_plies`, `final_state_id`,
`decision_seconds{"0","1": {n, mean, max, total}}`, `mean_legal_actions{"0","1"}`,
`mean_nodes_expanded{"0","1"}` (minimax), and `replay`: a list of
`{ply, player, source: opening|agent, action: [type, args], legal_count, decision_seconds, state_id}`.
`state_id` = first 16 hex characters of SHA-1 of the state key's `repr`.

`run_batch(configs, path)` appends JSONL and skips `match_id`s already present (resumable).

---

## 5. Commands

All commands run from the repository root (Python ≥ 3.10, standard library only).

```bash
python -m unittest discover tests                       # full suite

# one logged match + replay check, then demo export
python scripts/run_experiments.py match --p0 minimax:depth=2 --p1 greedy -N 5 --abc 111 \
       --opening-plies 2 --opening-seed 3 --out m.json
python scripts/run_experiments.py verify-replay m.json
python scripts/run_experiments.py export-demo m.json --out m_demo.json

# exact solves (all 8 A/B/C cells) and agent agreement
python scripts/run_experiments.py solve -N 3 -K 1 -M 0 --abc all --out experiments/results/person3_pilot/solve_N3_K1_M0.json
python scripts/run_experiments.py solve -N 3 -K 1 -M 1 --abc all --out experiments/results/person3_pilot/solve_N3_K1_M1.json
python scripts/run_experiments.py solve -N 4 -K 1 -M 0 --abc all --out experiments/results/person3_pilot/solve_N4_K1_M0.json
python scripts/run_experiments.py solve -N 4 -K 1 -M 1 --abc all --max-states 400000 --max-seconds 900 --out experiments/results/person3_pilot/solve_N4_K1_M1_capped.json
python scripts/run_experiments.py agreement -N 3 -K 1 -M 0 --agents greedy minimax:depth=2 \
       --out experiments/results/person3_pilot/agreement_N3_K1_M0.json
python scripts/run_experiments.py agreement -N 3 -K 1 -M 1 --agents greedy minimax:depth=2 \
       --out experiments/results/person3_pilot/agreement_N3_K1_M1.json

# paired 2^3 match ablation (writes experiments/logs/<name>/matches.jsonl + <out-dir>/summary.json)
python scripts/run_experiments.py ablation --x minimax:depth=2 --y greedy -N 5 -K 3 -M 10 \
       --seeds 10 --opening-plies 2 --max-plies 200 \
       --out-dir experiments/results/person3_pilot/ablation_N5_minimax2_vs_greedy
python scripts/run_experiments.py verify-replay experiments/logs/ablation_N5_minimax2_vs_greedy/matches.jsonl
```

---

## 6. Pilot results

All files in `experiments/results/person3_pilot/` were produced from commit `7e2d6da` with a clean tree (recorded in each file's `source`), by the commands in §5. Exact solves use K = 1. "Value" is from the side to move at the root (player 0).

### 6.1 Exact solves (reachable canonical positions)

| Cell | N=3, M=0 | root | N=3, M=1 | root | N=4, M=0 | root | N=4, M=1 (cap 400k) | root |
| :--- | ---: | :--- | ---: | :--- | ---: | :--- | ---: | :--- |
| A0B0C0 | 45 | WIN in 3 | 1,341 | WIN in 7 | 176 | LOSS in 6 | 27,860 | LOSS in 10 |
| A0B0C1 | 642 | WIN in 5 | 16,696 | WIN in 9 | 4,512 | LOSS in 8 | capped | UNKNOWN |
| A0B1C0 | 186 | LOSS in 2 | 5,730 | LOSS in 4 | 712 | WIN in 3 | 114,488 | WIN in 5 |
| A0B1C1 | 2,592 | LOSS in 4 | 68,292 | LOSS in 6 | 18,080 | WIN in 5 | capped | UNKNOWN |
| A1B0C0 | 45 | WIN in 3 | 3,861 | DRAW | 176 | LOSS in 6 | 98,504 | DRAW |
| A1B0C1 | 642 | WIN in 5 | 46,886 | DRAW | 4,512 | LOSS in 8 | capped | UNKNOWN |
| A1B1C0 | 186 | LOSS in 2 | 16,554 | DRAW | 712 | WIN in 3 | capped | UNKNOWN |
| A1B1C1 | 2,592 | LOSS in 4 | 192,072 | DRAW | 18,080 | WIN in 5 | capped | UNKNOWN |

The largest complete solve is N=3, M=1 with all mechanics: 192,072 positions and 1,335,672 edges, solved in 165 s. N=4, M=1 is the measured feasibility limit. Every cell with C on, and also A1B1C0, exceeded 400k positions. Those cells report UNKNOWN, and their partial WIN/LOSS labels are sound. Their only DRAW labels are the no-legal-action terminals. With M = 0, rule A has nothing to act on, so A0 and A1 cells are identical, which serves as a built-in sanity check.

Factorial effects (exact, no sampling error) on log10(positions), mean branching and draw fraction:

| Effect | N=3 M=1 log10 states | N=3 M=1 branching | N=3 M=1 draw fraction | N=4 M=0 log10 states | N=4 M=0 branching |
| :--- | ---: | ---: | ---: | ---: | ---: |
| A | +0.45 | +2.24 | +0.25 | 0 | 0 |
| B | +0.62 | +0.46 | −0.23 | +0.61 | +0.91 |
| C | +1.08 | +1.88 | +0.04 | +1.41 | +6.33 |
| AB | 0.00 | +0.01 | −0.13 | 0 | 0 |
| AC | −0.01 | −0.19 | +0.01 | 0 | 0 |
| BC | −0.01 | −0.03 | −0.02 | 0.00 | −0.03 |
| ABC | 0.00 | 0.00 | −0.01 | 0 | 0 |

**Reading.**
- State counts are almost exactly multiplicative across mechanics: the interactions on the log scale are about 0. C multiplies the state space by about 12× (N=3) or 26× (N=4), B by about 4×, and A by about 3× (M=1).
- Game *values*, by contrast, show strong interactions. On N=3, M=1, A turns every cell into a draw. Without A, B flips the root from a first-player win to a loss; with A on, B no longer changes the root value.
- The AB interaction on the draw fraction (−0.13) is the same pattern measured over all positions.

### 6.2 Agent vs. exact optimal actions (N=3, K=1; 300 sampled non-terminal positions per cell, or all if fewer)

The table gives the share of positions where the agent's move preserves the exact value (any optimal move is accepted):

| Cell | greedy M=0 | minimax d2 M=0 | greedy M=1 | minimax d2 M=1 |
| :--- | ---: | ---: | ---: | ---: |
| A0B0C0 | 0.88 | 1.00 | 0.91 | 0.98 |
| A0B0C1 | 0.76 | 0.99 | 0.91 | 0.95 |
| A0B1C0 | 0.95 | 1.00 | 0.95 | 0.99 |
| A0B1C1 | 0.89 | 1.00 | 0.93 | 0.97 |
| A1B0C0 | 0.88 | 1.00 | 0.93 | 0.98 |
| A1B0C1 | 0.76 | 0.99 | 0.89 | 0.97 |
| A1B1C0 | 0.95 | 1.00 | 0.93 | 0.99 |
| A1B1C1 | 0.89 | 1.00 | 0.90 | 0.99 |

Greedy is weakest in exact-DRAW positions (0.38–0.91 per cell). Its rank-optimality in LOSS positions, i.e. delaying a forced loss, ranges from 0.59 to 1.00. Depth-2 minimax keeps the value in at least 0.95 of positions. But with A and B on at M=1 it is rank-optimal in only 0.88–0.90: it keeps winning positions won without always taking the fastest win. Per-status rates and Wilson intervals are in `agreement_*.json`.

### 6.3 Match ablation (N=5, K=3, M=10, minimax depth 2 vs greedy)

The design is paired. There are 10 opening seeds, each 2 random pawn plies. Each seed is played with both seatings in all 8 cells, giving 160 matches (20 per cell). Matches use a 200-ply cap and no time limit. All 160 replays re-verified. Outcomes were 140 goal wins, 13 repetition draws, 7 no-legal-action draws, 0 truncations and 0 forfeits. A first run made before the review fixes gave identical outcomes.

| Cell | minimax W-L-D | win rate (95% Wilson) | mean plies | legal actions / ply |
| :--- | :---: | :---: | ---: | ---: |
| A0B0C0 | 13-5-2 | 0.65 (0.43–0.82) | 31.2 | 13.4 |
| A0B0C1 | 12-5-3 | 0.60 (0.39–0.78) | 36.0 | 23.0 |
| A0B1C0 | 11-7-2 | 0.55 (0.34–0.74) | 26.4 | 16.5 |
| A0B1C1 | 9-10-1 | 0.45 (0.26–0.66) | 28.8 | 27.1 |
| A1B0C0 | 9-9-2 | 0.45 (0.26–0.66) | 41.5 | 18.6 |
| A1B0C1 | 10-8-2 | 0.50 (0.30–0.70) | 44.9 | 28.9 |
| A1B1C0 | 13-2-5 | 0.65 (0.43–0.82) | 37.3 | 18.4 |
| A1B1C1 | 10-7-3 | 0.50 (0.30–0.70) | 36.8 | 32.2 |

Factorial effects with paired-bootstrap 95% intervals over opening seeds:

| Effect | minimax goal-win rate | game length (plies) | legal actions / ply |
| :--- | :---: | :---: | :---: |
| A | −0.04 (−0.26, 0.19) | **+9.5 (5.5, 13.3)** | **+4.5 (2.6, 6.1)** |
| B | −0.01 (−0.15, 0.13) | **−6.1 (−13.2, −0.6)** | +2.6 (−1.7, 7.6) |
| C | −0.06 (−0.19, 0.08) | **+2.5 (0.3, 4.5)** | **+11.1 (9.4, 12.9)** |
| AB | **+0.11 (0.03, 0.20)** | −0.1 (−3.7, 3.6) | −1.0 (−2.4, 0.2) |
| AC | +0.01 (−0.04, 0.06) | −1.1 (−5.0, 2.2) | +1.0 (−0.5, 2.5) |
| BC | −0.06 (−0.15, 0.03) | −1.6 (−4.3, 1.6) | +1.1 (−0.5, 2.6) |
| ABC | −0.04 (−0.21, 0.11) | −0.3 (−2.2, 2.4) | +0.6 (−0.5, 1.8) |

**Reading.**
- Depth-2 minimax is not clearly stronger than greedy on 5×5: with 10 seeds, no main effect on win rate is distinguishable from zero.
- The single AB interval that excludes zero is one of seven tests, so treat it as a hypothesis for the full run, not a finding.
- The mechanics clearly change game *dynamics*: C mainly widens the action space, A lengthens games, and B shortens them.
- Decision times are recorded in `summary.json` but not interpreted here: they were measured while a solver job shared the machine, so they depend on load.

These are pilots. A full study needs more seeds (the proposal suggested 500 matches per pairing), deeper minimax and the settings the team agrees on.

---

## 7. Replay export and the browser demo

`export-demo` writes the demo's own action objects (`js/game.js`, `executeAITurn`):
`{type:'MOVE',x,y,isWallHop}`, `{type:'WALL',gx,gy,orientation}`, `{type:'BLOCK',x,y}`, plus
`player` (1 or 2). Mapping: engine player p → demo player p+1, cell (r, c) → (x=c, y=N−1−r),
wall (r, c, o) → (gx=c, gy=N−2−r, o). `tests/test_match_runner.py` checks the wall mapping
edge by edge against a Python transcription of the demo's `isEdgeBlocked`.

The demo (commit `77a566c`) has **no replay loader** and different rules: fixed 9×9, FIFO wall
recycling and no REPLACE, a one-cell "wall hop" instead of the engine's two-cell HOP, and
Quoridor-style pawn jumps. The export therefore reports `demo_compatible` and lists every
incompatibility. Compatible games are 9×9, with no HOP or REPLACE, and K = 3 when blocks are on.
**Remaining dependency (demo owner):** add a loader that, for each event, sets `currentPlayer` and
runs the same branch as `executeAITurn` for that player. Nothing was pushed to the demo repository.

---

## 8. Limitations

* Exact results hold only for the solved tiny configurations and the position-game model of §3.
  They do not establish optimality on 5×5 or 9×9 boards; larger-board agents are compared with
  greedy/minimax by matches.
* Enumeration is pure Python and single-threaded. Feasibility limits are the measured ones in §6, not assumed.
* Ablation pilots use deterministic agents. Independent evidence comes from distinct opening
  seeds (balanced sides), and a 10-seed pilot is too small for firm conclusions.
* DQN evaluation is wired (`dqn` spec, `discounted_value`, agreement) but blocked until Person 4
  defines the encoder, action ids and checkpoint format.
