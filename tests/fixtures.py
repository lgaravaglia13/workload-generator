"""Deterministic shared datasets for the test suite.

Every dataset is built once (lru_cache) from a fixed seed; tests must never
mutate the returned arrays in place. Import this module before the project
modules so the repo root is importable regardless of the invocation directory.
"""
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

UINT64_MAX = int(np.iinfo(np.uint64).max)


@lru_cache(maxsize=None)
def uniform_dense() -> np.ndarray:
    """Contiguous dense keys arange(0, 100_000): no gaps, k_min == 0."""
    return np.arange(0, 100_000, dtype=np.uint64)


@lru_cache(maxsize=None)
def uniform_sparse() -> np.ndarray:
    """100k unique keys drawn uniformly over [0, 10^7]: ~1% density, large gaps."""
    rng = np.random.default_rng(2024)
    return np.sort(rng.choice(10_000_001, size=100_000, replace=False)).astype(np.uint64)


@lru_cache(maxsize=None)
def zipfian() -> np.ndarray:
    """Inverse-CDF Zipf keys over [0, 10^7] (theta=1.2), deduplicated, sorted.

    Dense near the origin (k_min == 0) with a long sparse tail.
    """
    rng = np.random.default_rng(2012)
    weights = np.arange(1, 10_000_002, dtype=np.float64) ** (-1.2)
    cdf = np.cumsum(weights)
    cdf /= cdf[-1]
    draws = rng.random(1_000_000)
    return np.unique(np.searchsorted(cdf, draws)).astype(np.uint64)


@lru_cache(maxsize=None)
def gaussian() -> np.ndarray:
    """Rounded normal keys (mu=5e6, sigma=1e6), deduplicated, sorted: bell density."""
    rng = np.random.default_rng(2008)
    draws = np.maximum(np.rint(rng.normal(5e6, 1e6, size=200_000)), 0)
    return np.unique(draws).astype(np.uint64)


@lru_cache(maxsize=None)
def clustered_two_hot() -> np.ndarray:
    """Two dense 50k-wide clusters near 1e6 and 9e6 (30k keys each), empty middle."""
    rng = np.random.default_rng(2016)
    lower_cluster = 1_000_000 + rng.choice(50_000, size=30_000, replace=False)
    upper_cluster = 9_000_000 + rng.choice(50_000, size=30_000, replace=False)
    return np.unique(np.concatenate([lower_cluster, upper_cluster])).astype(np.uint64)


@lru_cache(maxsize=None)
def tiny_single_low() -> np.ndarray:
    """Single-key dataset anchored at 5: prepend headroom is exactly 5."""
    return np.array([5], dtype=np.uint64)


@lru_cache(maxsize=None)
def tiny_single_high() -> np.ndarray:
    """Single-key dataset anchored at UINT64_MAX - 5: append headroom is exactly 5."""
    return np.array([UINT64_MAX - 5], dtype=np.uint64)


@lru_cache(maxsize=None)
def tiny_two_keys() -> np.ndarray:
    """Two-key dataset for minimal non-degenerate boundary behaviour."""
    return np.array([1_000, 2_000], dtype=np.uint64)


@lru_cache(maxsize=None)
def trace_base_keys() -> np.ndarray:
    """1000 sorted synthetic keys in [1000, 10^6): the baseline trace dataset."""
    rng = np.random.default_rng(42)
    return np.sort(
        rng.choice(np.arange(1000, 1_000_000, dtype=np.uint64), size=1000, replace=False)
    )


def strictly_sorted(arr) -> bool:
    """Strict monotonicity check shared by state assertions."""
    arr = np.asarray(arr)
    return len(arr) < 2 or bool(np.all(arr[:-1] < arr[1:]))
