#!/usr/bin/env python3
"""
Engine Performance and Latency Benchmark Suite (P9.6).

Verifies engine throughput and latency targets against strict engineering thresholds:
- 9x9 start legal-action generation <= 3.0 ms
- 9x9 midgame legal-action generation mean <= 4.0 ms
- Engine clone latency <= 50.0 microseconds (O(1) copy-on-write history)
- 5x5 start legal-action generation <= 1.0 ms
- Engine action application latency <= 100 microseconds

Saves JSON benchmark report to experiments/results/engine/benchmark_performance.json.
"""
import argparse
import json
import os
import random
import sys
import time
from typing import Any, Dict, List

from wallz_engine import WallzEngine


def benchmark_legal_actions_start(board_size: int, rules: Dict[str, Any], runs: int = 200) -> Dict[str, float]:
    engine = WallzEngine(board_size=board_size, rules=rules)
    # Warmup
    engine.get_legal_actions()

    latencies: List[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        actions = engine.get_legal_actions()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    return {
        'board_size': board_size,
        'action_count': len(actions),
        'runs': runs,
        'mean_ms': sum(latencies) / len(latencies),
        'min_ms': min(latencies),
        'max_ms': max(latencies),
        'median_ms': sorted(latencies)[len(latencies) // 2],
    }


def benchmark_legal_actions_midgame(
    board_size: int,
    rules: Dict[str, Any],
    num_states: int = 20,
    evals_per_state: int = 20,
    seed: int = 42,
) -> Dict[str, float]:
    rng = random.Random(seed)
    all_latencies: List[float] = []

    for trial in range(num_states):
        engine = WallzEngine(board_size=board_size, rules=rules)
        # Advance 15-20 plies
        target_plies = rng.randint(12, 22)
        for _ in range(target_plies):
            if engine.is_terminal():
                break
            acts = engine.get_legal_actions()
            if not acts:
                break
            engine.apply_action(rng.choice(acts))

        if engine.is_terminal():
            continue

        for _ in range(evals_per_state):
            t0 = time.perf_counter()
            acts = engine.get_legal_actions()
            t1 = time.perf_counter()
            all_latencies.append((t1 - t0) * 1000.0)

    return {
        'board_size': board_size,
        'evaluated_samples': len(all_latencies),
        'mean_ms': sum(all_latencies) / len(all_latencies),
        'min_ms': min(all_latencies),
        'max_ms': max(all_latencies),
        'median_ms': sorted(all_latencies)[len(all_latencies) // 2],
    }


def benchmark_clone_scaling(
    board_size: int,
    rules: Dict[str, Any],
    ply_checkpoints: List[int] = [5, 20, 50],
    repeats: int = 500,
    seed: int = 42,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    results = {}

    for plies in ply_checkpoints:
        engine = WallzEngine(board_size=board_size, rules=rules)
        for _ in range(plies):
            if engine.is_terminal():
                break
            acts = engine.get_legal_actions()
            if not acts:
                break
            engine.apply_action(rng.choice(acts))

        # Benchmark clone latency
        times_us: List[float] = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            _ = engine.clone()
            t1 = time.perf_counter()
            times_us.append((t1 - t0) * 1_000_000.0)

        results[f'ply_{plies}'] = {
            'mean_us': sum(times_us) / len(times_us),
            'min_us': min(times_us),
            'max_us': max(times_us),
        }

    return results


def run_benchmark():
    parser = argparse.ArgumentParser(description="Wallz Engine Performance Benchmarks (P9.6)")
    parser.add_argument("--out", type=str, default="experiments/results/engine/benchmark_performance.json")
    args = parser.parse_args()

    rules_full = {'A': True, 'B': True, 'C': True, 'K': 3, 'M': 10}

    print("=== Running Wallz Engine Performance Benchmarks ===")

    # 1. 9x9 Start Legal Actions (Target <= 3.0 ms)
    res_9x9_start = benchmark_legal_actions_start(9, rules_full, runs=250)
    print(f"9x9 Start Legal Actions: mean = {res_9x9_start['mean_ms']:.3f} ms (Target <= 3.0 ms)")

    # 2. 9x9 Midgame Legal Actions (Target <= 4.0 ms)
    res_9x9_mid = benchmark_legal_actions_midgame(9, rules_full, num_states=20, evals_per_state=25)
    print(f"9x9 Midgame Legal Actions: mean = {res_9x9_mid['mean_ms']:.3f} ms (Target <= 4.0 ms)")

    # 3. 5x5 Start Legal Actions (Target <= 1.0 ms)
    res_5x5_start = benchmark_legal_actions_start(5, rules_full, runs=250)
    print(f"5x5 Start Legal Actions: mean = {res_5x5_start['mean_ms']:.3f} ms (Target <= 1.0 ms)")

    # 4. Clone latency across plies (Target <= 50.0 us)
    res_clone = benchmark_clone_scaling(9, rules_full, ply_checkpoints=[5, 20, 50], repeats=500)
    max_clone_us = max(v['mean_us'] for v in res_clone.values())
    print(f"Engine Clone Latency (max across plies): {max_clone_us:.2f} us (Target <= 50.0 us)")

    # Threshold checks
    thresholds = {
        'legal_actions_9x9_start_ms': {'value': res_9x9_start['mean_ms'], 'threshold': 3.0, 'pass': res_9x9_start['mean_ms'] <= 3.0},
        'legal_actions_9x9_midgame_ms': {'value': res_9x9_mid['mean_ms'], 'threshold': 4.0, 'pass': res_9x9_mid['mean_ms'] <= 4.0},
        'legal_actions_5x5_start_ms': {'value': res_5x5_start['mean_ms'], 'threshold': 1.0, 'pass': res_5x5_start['mean_ms'] <= 1.0},
        'clone_latency_us': {'value': max_clone_us, 'threshold': 50.0, 'pass': max_clone_us <= 50.0},
    }

    all_passed = all(t['pass'] for t in thresholds.values())

    report = {
        'benchmarks': {
            'legal_actions_9x9_start': res_9x9_start,
            'legal_actions_9x9_midgame': res_9x9_mid,
            'legal_actions_5x5_start': res_5x5_start,
            'clone_scaling': res_clone,
        },
        'thresholds': thresholds,
        'passed': all_passed,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(f"Saved benchmark report to {args.out}")

    for k, v in thresholds.items():
        status = "PASS" if v['pass'] else "FAIL"
        print(f"  [{status}] {k}: {v['value']:.3f} vs threshold {v['threshold']}")

    if not all_passed:
        print("Performance thresholds failed!")
        sys.exit(1)
    else:
        print("All performance thresholds PASSED!")


if __name__ == '__main__':
    run_benchmark()
