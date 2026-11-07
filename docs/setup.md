# Setup & Reproducibility Guide — Modified Wallz

This guide provides complete instructions to set up the environment, install dependencies, run the test suite, and reproduce every experimental result, benchmark, publication table, and trained MARL checkpoint in **Modified Wallz**.

---

## 1. System Requirements

- **Operating System**: Linux (Ubuntu 20.04+, Debian 11+, Fedora 38+), macOS (12+), or Windows 10/11 with WSL2.
- **Python Version**: Python 3.10, 3.11, 3.12, 3.13, or 3.14.
- **Hardware Requirements**:
  - Minimum: 4 GB RAM, dual-core CPU.
  - Recommended: 16 GB RAM, 4+ CPU cores (for large state-graph solving and factorial ablations).
  - GPU: Optional. All models, baselines, and solvers run efficiently on standard CPU architectures.

---

## 2. Installation & Environment Setup

### Step 1: Clone Repository
```bash
git clone https://github.com/vedantb7/FAI_project.git
cd FAI_project
```

### Step 2: Create and Activate Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows (cmd): .venv\Scripts\activate.bat
```

### Step 3: Upgrade Packaging Tools
```bash
pip install --upgrade pip setuptools wheel
```

### Step 4: Install Dependencies
Install all core dependencies and install the package in editable development mode:
```bash
pip install -r requirements.txt
pip install -e .
```

> [!TIP]
> **Optional GPU Support**: To install CUDA-accelerated PyTorch instead of CPU:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

---

## 3. Immediate Verification

Confirm that the environment is correctly set up by running the test suite and engine performance benchmark:

### Run Full Test Suite (160 tests)
```bash
python3 -m unittest discover -s tests
```
*Expected Result*: `Ran 160 tests in ~20s ... OK (0 failures, 0 errors)`.

### Run Engine Throughput Benchmark
```bash
python3 scripts/benchmark_engine_performance.py --seed 42
```
*Expected Result*:
```text
=== Running Wallz Engine Performance Benchmarks ===
9x9 Start Legal Actions: mean = ~1.5 ms (Target <= 3.0 ms)   [PASS]
9x9 Midgame Legal Actions: mean = ~2.0 ms (Target <= 4.0 ms) [PASS]
5x5 Start Legal Actions: mean = ~0.25 ms (Target <= 1.0 ms)  [PASS]
Engine Clone Latency: ~3.0 us (Target <= 50.0 us)             [PASS]
All performance thresholds PASSED!
```

---

## 4. Seeding & Determinism Conventions

Reproducibility is enforced through strict, centralized pseudo-random number generator (PRNG) seeding across all modules:
- **Default Seed**: All CLI scripts, match runners, and experiment drivers default to `seed = 42`.
- **Seeded Libraries**: Python `random.seed(S)`, NumPy `np.random.seed(S)`, and PyTorch `torch.manual_seed(S)` (plus `torch.cuda.manual_seed_all(S)`) are synchronized simultaneously.
- **Match Openings**: Matches use `opening_seed` to generate identical pawn-move starting layouts across all factorial mechanics cells before agents take control.
- **Greedy Tie-Breaking**: `GreedyAgent` supports seeded randomized tie-breaking via `greedy:seed=S` while maintaining deterministic lexicographical fallback.

---

## 5. End-to-End Experiment & Artifact Reproduction

### 1. Generating Publication Report Tables (14 Tables)
To regenerate all 14 LaTeX (`.tex`) and Markdown (`.md`) tables in `experiments/tables/`:
```bash
python3 scripts/make_tables.py --seed 42
```
*Output Artifacts*:
- `table_x6_agreement_*.md` / `.tex`: Optimal Action Agreement rates (Gate X-6).
- `table_x9_benchmark_*.md` / `.tex`: Head-to-head match win rates with Wilson 95% CIs (Gate X-9).
- `table_x10a_ablation_cells.md` / `.tex`: $2^3$ factorial mechanics ablation cells.
- `table_x10b_factorial_effects.md` / `.tex`: Main and interaction effects with Holm-Bonferroni correction.
- `table_x17_draw_breakdown.md` / `.tex`: Stalemate vs repetition draw taxonomy.
- `table_x4_solve_*.md` / `.tex`: Exact state space and game-theoretic value distributions.

### 2. Miniature Board Retrograde Solves (Gate X-4 / Phase 8)
Solve all 8 factorial rule combinations for $3 \times 3$ boards ($K=1, M=0$):
```bash
python3 scripts/run_experiments.py solve -N 3 -K 1 -M 0 --abc all --out experiments/results/solves/solve_N3_K1_M0.json --seed 42
```
Solve higher block durations ($K=2$ and $K=3$):
```bash
python3 scripts/run_experiments.py solve -N 3 -K 2 -M 0 --abc all --out experiments/results/solves/solve_N3_K2_M0.json --seed 42
python3 scripts/run_experiments.py solve -N 3 -K 3 -M 0 --abc all --out experiments/results/solves/solve_N3_K3_M0.json --seed 42
```

### 3. Feature Study & Logistic Regression AUC Benchmark (Gate X-5)
Evaluate the predictive power of heuristic features against complete retrograde solutions:
```bash
python3 scripts/feature_study.py --board-size 3 --seed 42 --output experiments/results/features/x5_feature_study.json
```
*Gate X-5 Criterion*: Both Model 1 ($H$ alone) and Model 2 (All 6 features) must exceed 70.0% AUC.

### 4. Optimal Action Agreement Benchmark (Gate X-6)
Measure decision agreement rates of Greedy, Minimax (depth 1 & 2), and DQN against exact retrograde solutions:
```bash
python3 scripts/run_experiments.py --config configs/experiments/agreement_3x3.yaml
```

### 5. Factorial Mechanic Ablation (Gate X-10 & Gate X-17)
Run paired 160-game tournament ablation on a $5 \times 5$ board ($K=3, M=10$):
```bash
python3 scripts/run_experiments.py --config configs/experiments/ablation_5x5_k3.yaml
```

### 6. MARL DQN Training & Validation (Gate X-16)
Train a Double-DQN self-play policy on a $3 \times 3$ board:
```bash
python3 scripts/train_dqn.py --config configs/training.yaml --board-size 3 --seed 42
```
Validate the released $3 \times 3$ checkpoint against ground-truth retrograde values (Gate X-16):
```bash
python3 scripts/validate_dqn.py --checkpoint checkpoints/release/dqn_3x3/model.pt --seed 42
```
*Gate X-16 Criterion*: Value sign agreement on decided non-terminal states must exceed 90.0%.

---

## 6. Running Interactive and Tournament Matches

Run head-to-head matches using `run_match.py`:

```bash
# Minimax (depth 2) vs Greedy on default 9x9 board with seed 42
python3 run_match.py --p0 minimax --p1 greedy --depth 2 --seed 42

# DQN vs Minimax (depth 2) on 3x3 board
python3 run_match.py --p0 dqn:path=checkpoints/release/dqn_3x3/model.pt --p1 minimax:depth=2 --board-size 3 --seed 42

# Greedy vs Greedy with custom rule toggles (A=1, B=0, C=1) on 5x5 board
python3 run_match.py --p0 greedy --p1 greedy --board-size 5 --abc 101 --seed 42
```

---

## 7. Troubleshooting & FAQ

### 1. `ModuleNotFoundError: No module named 'modified_wallz'`
Ensure the virtual environment is active and the package is installed in editable mode:
```bash
pip install -e .
```

### 2. PyTorch GPU / CUDA Detection
Check whether PyTorch detects your GPU:
```bash
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```
If CUDA is available, `DQNTrainer` automatically selects the GPU device when `--device cuda` is specified.

### 3. Verification of Replays
To verify the integrity of all transitions recorded in any match JSONL log:
```bash
python3 scripts/run_experiments.py verify-replay experiments/logs/ablation_N5_minimax2_vs_greedy/matches.jsonl
```
This confirms that every move in the replay sequence produces legal transitions matching state hashes and final outcomes.
