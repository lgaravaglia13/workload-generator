"""Distribution kernels (gaussian, zipf, uniform) for the sampling engine."""
from dataclasses import dataclass
from functools import lru_cache
from typing import Union

import numpy as np

DEFAULT_THETA = 1.2

# Above this size, switch from exact discrete CDF to continuous inversion O(1/d_max).
_EXACT_CDF_LIMIT = 1 << 22


@dataclass(frozen=True)
class ZipfConfig:
    """Zipfian kernel: weights and step distances decay as (1 + d)^-theta."""

    theta: float = DEFAULT_THETA


@dataclass(frozen=True)
class GaussianConfig:
    """Gaussian kernel. The scale is always derived from the window/radius
    (sigma = radius / 3), so the configuration carries no parameters."""


@dataclass(frozen=True)
class UniformConfig:
    """Uniform draws over the full window/radius; no parameters."""


DistributionConfig = Union[ZipfConfig, GaussianConfig, UniformConfig]


def from_token(token: str) -> DistributionConfig:
    """Returns the default DistributionConfig for a workload-DSL distribution token."""
    d = token.lower()
    if d in ("g", "gaussian"):
        return GaussianConfig()
    if d in ("z", "zipf"):
        return ZipfConfig()
    if d in ("r", "random", "uniform"):
        return UniformConfig()
    raise ValueError(f"unknown distribution token: {token!r}")


@lru_cache(maxsize=16)
def _powerlaw_cdf(d_max: int, theta: float) -> np.ndarray:
    """Truncated discrete CDF over the integer support [1, d_max]. Cached
    because batches of any size redraw the same unchanged (d_max, theta)
    pair and the CDF is a pure function of the pair. The returned array is
    shared across calls and must never be mutated."""
    support = np.arange(1, d_max + 1, dtype=np.float64)
    cdf = np.cumsum((1.0 + support) ** (-theta))
    return cdf / cdf[-1]


def _draw_powerlaw_distances(
    size: int,
    d_max: int,
    theta: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Integer distances in [1, d_max] with P(d) proportional to (1 + d)^-theta."""
    if d_max <= 1:
        return np.ones(size, dtype=np.int64)

    u = rng.uniform(0.0, 1.0, size=size)
    if d_max <= _EXACT_CDF_LIMIT:
        cdf = _powerlaw_cdf(d_max, theta)
        # side='right' keeps u=0 on the first support point; u < 1 never past
        # the end; support position i corresponds to the distance i + 1.
        return (np.searchsorted(cdf, u, side="right") + 1).astype(np.int64)

    # Truncated continuous CDF on [1, d_max]: x = CDF^-1(u) in closed form.
    a = 2.0 ** (1.0 - theta)
    b = (1.0 + d_max) ** (1.0 - theta)
    if abs(1.0 - theta) < 1e-12:
        # theta == 1: the CDF is logarithmic and the generic form blows up.
        x = 2.0 * ((1.0 + d_max) / 2.0) ** u - 1.0
    else:
        x = (a + u * (b - a)) ** (1.0 / (1.0 - theta)) - 1.0
    return np.clip(np.floor(x).astype(np.int64), 1, d_max)


def draw_offsets(
    size: int,
    radius: float,
    config: DistributionConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draws signed key-space offsets around a pivot. Gaussian and uniform
    return continuous draws (the caller rounds and clamps them to the
    dataset); zipf returns discrete power-law distances with random signs:
    spatial clustering around the pivot, not access frequency."""
    if isinstance(config, GaussianConfig):
        return rng.normal(loc=0.0, scale=max(1.0, radius / 3.0), size=size)
    if isinstance(config, ZipfConfig):
        d_max = max(1, int(np.floor(radius)))
        distances = _draw_powerlaw_distances(size, d_max, config.theta, rng)
        signs = rng.choice(np.array([-1, 1], dtype=np.int64), size=size)
        return signs * distances
    if isinstance(config, UniformConfig):
        return rng.uniform(low=-radius, high=radius, size=size)
    raise TypeError(f"unsupported distribution config: {type(config).__name__}")


def draw_steps(
    count: int,
    radius: float,
    config: DistributionConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draws strictly positive integer steps (>= 1) scaled by the local
    key-gap radius: cumulative sums of these produce monotone boundary keys."""
    if isinstance(config, GaussianConfig):
        scale = max(1.0, radius / 3.0)
        steps = np.round(np.abs(rng.normal(loc=0.0, scale=scale, size=count)))
        return np.maximum(steps, 1.0).astype(np.int64)
    d_max = max(1, int(np.floor(radius)))
    if isinstance(config, ZipfConfig):
        return _draw_powerlaw_distances(count, d_max, config.theta, rng)
    if isinstance(config, UniformConfig):
        return rng.integers(1, d_max + 1, size=count).astype(np.int64)
    raise TypeError(f"unsupported distribution config: {type(config).__name__}")
