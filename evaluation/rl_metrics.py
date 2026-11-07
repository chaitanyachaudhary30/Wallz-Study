"""RL Evaluation Metrics (P2.1).

Implements:
- Interquartile Mean (IQM): 25% trimmed mean across seeds/evaluations following
  Agarwal et al. (2021) "Deep Reinforcement Learning at the Edge of the Statistical Precipice".
- Bootstrap Confidence Intervals: Non-parametric percentile bootstrap for point estimates.
- Summary aggregators combining IQM with confidence bounds.
"""
from __future__ import annotations

import math
from typing import Callable, Optional, Sequence, Tuple, Union
import numpy as np


def interquartile_mean(data: Sequence[float] | np.ndarray, proportiontocut: float = 0.25) -> float:
    """Computes the Interquartile Mean (IQM) or trimmed mean of the input data.

    The IQM trims the lowest and highest 25% of scores and calculates the mean of the
    remaining middle 50% of the distribution.

    Args:
        data: Sequence of numerical scores.
        proportiontocut: Fraction of values to trim from EACH end (default 0.25 for IQM).

    Returns:
        The trimmed mean as a float.
    """
    arr = np.asarray(data, dtype=np.float64)
    if arr.size == 0:
        return float('nan')
    if arr.size <= 2:
        return float(np.mean(arr))

    sorted_arr = np.sort(arr)
    n = len(sorted_arr)

    # Standard trimmed mean calculation with fractional weight interpolation
    low_idx = n * proportiontocut
    high_idx = n * (1.0 - proportiontocut)

    i_low = int(math.floor(low_idx))
    i_high = int(math.ceil(high_idx))

    weights = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if i < i_low or i >= i_high:
            weights[i] = 0.0
        elif i == i_low:
            weights[i] = min(1.0, (i_low + 1.0) - low_idx)
        elif i == i_high - 1:
            weights[i] = min(1.0, high_idx - (i_high - 1.0))
        else:
            weights[i] = 1.0

    total_weight = np.sum(weights)
    if total_weight == 0:
        return float(np.mean(sorted_arr))

    return float(np.sum(sorted_arr * weights) / total_weight)


def bootstrap_ci(
    data: Sequence[float] | np.ndarray,
    statistic_fn: Callable[[np.ndarray], float] = np.mean,
    num_bootstraps: int = 2000,
    confidence_level: float = 0.95,
    random_state: Optional[Union[int, np.random.Generator, np.random.RandomState]] = None,
) -> Tuple[float, float]:
    """Computes non-parametric percentile bootstrap confidence interval for a statistic.

    Args:
        data: 1D array or sequence of values.
        statistic_fn: Function mapping 1D sample array to scalar statistic (default: np.mean).
        num_bootstraps: Number of bootstrap resamples (default: 2000).
        confidence_level: Nominal confidence level in (0, 1) (default: 0.95).
        random_state: Seed or numpy RNG instance.

    Returns:
        (lower_bound, upper_bound) tuple of floats.
    """
    arr = np.asarray(data, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return float('nan'), float('nan')
    if n == 1:
        val = float(statistic_fn(arr))
        return val, val

    if isinstance(random_state, np.random.Generator):
        rng = random_state
    elif isinstance(random_state, np.random.RandomState):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)

    # Generate bootstrap replicates
    boot_indices = rng.integers(0, n, size=(num_bootstraps, n))
    boot_samples = arr[boot_indices]

    boot_stats = np.empty(num_bootstraps, dtype=np.float64)
    for b in range(num_bootstraps):
        boot_stats[b] = statistic_fn(boot_samples[b])

    alpha = 1.0 - confidence_level
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    lower_bound = float(np.percentile(boot_stats, lower_pct))
    upper_bound = float(np.percentile(boot_stats, upper_pct))

    return lower_bound, upper_bound


def iqm_with_ci(
    data: Sequence[float] | np.ndarray,
    num_bootstraps: int = 2000,
    confidence_level: float = 0.95,
    random_state: Optional[int] = 42,
) -> dict:
    """Computes IQM point estimate along with bootstrap confidence intervals.

    Returns:
        dict with keys {'iqm', 'ci_lower', 'ci_upper', 'mean', 'median', 'std', 'n'}.
    """
    arr = np.asarray(data, dtype=np.float64)
    if arr.size == 0:
        return {
            'iqm': float('nan'),
            'ci_lower': float('nan'),
            'ci_upper': float('nan'),
            'mean': float('nan'),
            'median': float('nan'),
            'std': float('nan'),
            'n': 0,
        }

    iqm_val = interquartile_mean(arr)
    ci_low, ci_high = bootstrap_ci(
        arr,
        statistic_fn=interquartile_mean,
        num_bootstraps=num_bootstraps,
        confidence_level=confidence_level,
        random_state=random_state,
    )

    return {
        'iqm': round(iqm_val, 4),
        'ci_lower': round(ci_low, 4),
        'ci_upper': round(ci_high, 4),
        'mean': round(float(np.mean(arr)), 4),
        'median': round(float(np.median(arr)), 4),
        'std': round(float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0, 4),
        'n': len(arr),
    }
