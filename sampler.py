"""Vectorized key sampling engine.

Controls dataset geometry (percentile pivots, rank windows, boundary anchors)
and sampling algorithms (A-Res selection, capacity-bounded allocation for new keys).
"""
import logging
from functools import lru_cache
from typing import Optional

import numpy as np

from distributions import (
    DEFAULT_THETA,
    DistributionConfig,
    GaussianConfig,
    UniformConfig,
    ZipfConfig,
    draw_offsets,
    draw_steps,
)

RANK_SPREAD_RATIO = 0.025
UINT64_MAX = int(np.iinfo(np.uint64).max)
# Safe clip bound for signed offsets derived from a uint64 domain.
INT64_CLIP = 1 << 62
# Threshold above which rejection-deduplication is faster than full permutation.
DISTINCT_OFFSET_CHOICE_LIMIT = 1 << 20
# Allowance factor on median hostable gap when bounding drawable spans.
SPAN_ALLOWANCE = 4.0

_DRIFT_LOGGER = logging.getLogger("sampler.drift")


def _window_radius(n: int, rank_spread: float) -> int:
    """Calculates rank window radius from dataset size and spread ratio."""
    if rank_spread <= 0.0:
        raise ValueError(f"rank_spread must be positive, got {rank_spread}.")
    return max(1, int(n * rank_spread))


def _percentile_pivot(n: int, percentile: float) -> int:
    """Calculates floor-based percentile rank, clamped to valid indices."""
    return int(np.clip(np.floor((n - 1) * (percentile / 100.0)), 0, n - 1))


def _rank_window(n: int, pivot: int, rank_spread: float = RANK_SPREAD_RATIO) -> tuple:
    """Rank window [low, high) around the pivot: 2R+1 wide when interior."""
    rank_radius = _window_radius(n, rank_spread)
    low = max(0, pivot - rank_radius)
    high = min(n, pivot + rank_radius + 1)
    return low, high


def _pivot_and_radius(active_keys: np.ndarray, percentile: float,
                      rank_spread: float = RANK_SPREAD_RATIO) -> tuple:
    """Pivot key and key-space radius at the percentile rank pivot. The radius
    is the window key span rescaled per rank step to adapt to local density."""
    n = len(active_keys)
    pivot_idx = _percentile_pivot(n, percentile)
    pivot = int(active_keys[pivot_idx])

    rank_radius = _window_radius(n, rank_spread)
    low_idx = max(0, pivot_idx - rank_radius)
    high_idx = min(n - 1, pivot_idx + rank_radius)
    delta_idx = max(1, high_idx - low_idx)
    key_span = float(int(active_keys[high_idx]) - int(active_keys[low_idx]))
    key_radius = max(1.0, (key_span / float(delta_idx)) * float(rank_radius))
    return pivot, key_radius


def _to_uint64_offsets(base: int, offsets: np.ndarray) -> np.ndarray:
    """Converts signed offsets around `base` to uint64 keys with same-sign
    arithmetic: numpy promotes uint64 + int64 to float64, which silently loses
    precision above 2^53 (and would wrap a negative offset)."""
    pos_mask = offsets >= 0
    candidates = np.empty(len(offsets), dtype=np.uint64)
    candidates[pos_mask] = np.uint64(base) + offsets[pos_mask].astype(np.uint64)
    candidates[~pos_mask] = np.uint64(base) - (-offsets[~pos_mask]).astype(np.uint64)
    return candidates


def _ares_select(weights: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    """A-Res weighted selection without replacement: keys tau_i = -ln(u_i)/w_i.
    Returns indices of the `count` smallest taus, ordered by ascending tau_i (selection order)."""
    tau = rng.uniform(0.0, 1.0, size=len(weights))
    np.log(tau, out=tau)
    np.negative(tau, out=tau)
    tau /= weights
    if count >= len(tau):
        return np.argsort(tau)
    top = np.argpartition(tau, count)[:count]
    return top[np.argsort(tau[top])]


# Direct evaluation threshold: large untargeted windows avoid pinning huge tables.
_WEIGHT_TABLE_MAX_RADIUS = 1 << 22


@lru_cache(maxsize=4)
def _gaussian_weight_table(radius: int) -> np.ndarray:
    d = np.arange(radius + 1, dtype=np.float64)
    sigma = max(1.0, radius / 3.0)
    return np.exp(-(d**2) / (2.0 * sigma * sigma))


@lru_cache(maxsize=4)
def _zipf_weight_table(theta: float, radius: int) -> np.ndarray:
    d = np.arange(radius + 1, dtype=np.float64)
    return (1.0 + d) ** (-theta)


def _position_weights(config, distances: np.ndarray, rank_radius: int) -> Optional[np.ndarray]:
    """Kernel weights at non-negative integer distances from the pivot (0 <= distances <= rank_radius)."""
    if isinstance(config, UniformConfig):
        return None
    if isinstance(config, GaussianConfig):
        if rank_radius <= _WEIGHT_TABLE_MAX_RADIUS:
            return _gaussian_weight_table(rank_radius)[distances]
        sigma = max(1.0, rank_radius / 3.0)
        return np.exp(-(distances.astype(np.float64) ** 2) / (2.0 * sigma * sigma))
    if isinstance(config, ZipfConfig):
        if rank_radius <= _WEIGHT_TABLE_MAX_RADIUS:
            return _zipf_weight_table(config.theta, rank_radius)[distances]
        return (1.0 + distances.astype(np.float64)) ** (-config.theta)
    raise TypeError(f"unsupported distribution config: {type(config).__name__}")


def _window_weights(config, low: int, high: int, pivot: int,
                    rank_radius: int) -> Optional[np.ndarray]:
    """Kernel weights over rank window [low, high) using cached table slices."""
    if isinstance(config, UniformConfig):
        return None
    if rank_radius <= _WEIGHT_TABLE_MAX_RADIUS:
        if isinstance(config, GaussianConfig):
            table = _gaussian_weight_table(rank_radius)
        elif isinstance(config, ZipfConfig):
            table = _zipf_weight_table(config.theta, rank_radius)
        else:
            raise TypeError(f"unsupported distribution config: {type(config).__name__}")
        return np.concatenate([table[pivot - low : 0 : -1], table[: high - pivot]])
    distances = np.abs(np.arange(low, high, dtype=np.int64) - pivot)
    return _position_weights(config, distances, rank_radius)


def _boundary_anchor(active_keys: np.ndarray, percentile: float,
                     rank_spread: float = RANK_SPREAD_RATIO) -> tuple:
    """Anchor key, gap scale and uint64 headroom for prepend/append generation."""
    n = len(active_keys)
    rank_radius = _window_radius(n, rank_spread)
    if percentile <= 0.0:
        anchor = int(active_keys[0])
        high = min(n - 1, rank_radius)
        span = int(active_keys[high]) - anchor
        return anchor, max(1.0, span / float(max(1, high))), anchor
    anchor = int(active_keys[-1])
    low = max(0, n - 1 - rank_radius)
    span = anchor - int(active_keys[low])
    radius = max(1.0, span / float(max(1, n - 1 - low)))
    return anchor, radius, UINT64_MAX - anchor


def _generate_boundary_keys(
    active_keys: np.ndarray,
    count: int,
    percentile: float,
    distribution: DistributionConfig,
    rng: np.random.Generator,
    rank_spread: float = RANK_SPREAD_RATIO,
) -> np.ndarray:
    """Prepend (percentile <= 0) / append (percentile >= 100) keys from
    cumulative gaps off the anchor."""
    anchor, radius, headroom = _boundary_anchor(active_keys, percentile, rank_spread)
    cumulative_steps = np.cumsum(draw_steps(count, radius, distribution, rng))

    if percentile <= 0.0:
        if int(cumulative_steps[-1]) > headroom:
            raise RuntimeError(
                "Prepend underflow: generated keys fall below uint64 minimum (0)."
            )
        signed = -cumulative_steps
    else:
        if int(cumulative_steps[-1]) > headroom:
            raise RuntimeError("Append overflow: generated keys exceed UINT64_MAX.")
        signed = cumulative_steps
    return _to_uint64_offsets(anchor, signed)


def _sample_distinct_offsets(capacity: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Samples `k` distinct uint64 values uniformly from [1, capacity] without replacement."""
    if k == capacity:
        return np.arange(1, k + 1, dtype=np.uint64)
    if k == 1:
        return rng.integers(1, capacity + 1, size=1, dtype=np.uint64)
    if capacity <= DISTINCT_OFFSET_CHOICE_LIMIT:
        return rng.choice(capacity, size=k, replace=False).astype(np.uint64) + 1
    # For large gaps, rejection deduplication avoids an O(capacity) array.
    picked = np.empty(0, dtype=np.uint64)
    while len(picked) < k:
        draws = rng.integers(1, capacity + 1, size=2 * (k - len(picked)), dtype=np.uint64)
        merged = np.concatenate([picked, draws])
        # Preserve draw order so slicing [:k] does not bias toward smaller values.
        _, first_positions = np.unique(merged, return_index=True)
        picked = merged[np.sort(first_positions)][:k]
    return picked


def _span_slot(gap: int, span_cap: int, offset: int) -> Optional[int]:
    """1-based span slot of a gap-relative offset, or None when outside the local span cap."""
    if offset < 1 or offset >= gap:
        return None
    if gap - 1 <= 2 * span_cap:
        return offset
    if offset <= span_cap:
        return offset
    if offset >= gap - span_cap:
        return offset - gap + 2 * span_cap + 1
    return None


def _span_offsets(gaps: np.ndarray, span_cap: int, slots: np.ndarray) -> np.ndarray:
    """Maps 1-based span slots to gap-relative uint64 offsets. uint64 arithmetic
    prevents overflow on wide domains (gaps >= 2^63)."""
    gaps_u64 = np.asarray(gaps, dtype=np.uint64)
    slots_u64 = np.asarray(slots, dtype=np.uint64)
    cap_u64 = np.uint64(span_cap)
    regular = (gaps_u64 - np.uint64(1)) <= np.uint64(2) * cap_u64
    left = slots_u64 <= cap_u64
    return np.where(
        regular | left,
        slots_u64,
        gaps_u64 - np.uint64(2) * cap_u64 + slots_u64 - np.uint64(1),
    )


def _categorical_counts(p: np.ndarray, m: int, rng: np.random.Generator) -> np.ndarray:
    """Histogram of m Categorical(p) draws in O(m log W) time."""
    cdf = np.cumsum(p)
    # Scale uniforms into [0, cdf[-1]) and clamp to prevent rounding onto cdf[-1].
    u = rng.random(m) * cdf[-1]
    draws = np.searchsorted(cdf, u, side="right")
    np.minimum(draws, len(p) - 1, out=draws)
    return np.bincount(draws, minlength=len(p))


def _allocate_gap_quotas(
    capacities: np.ndarray,
    weights: Optional[np.ndarray],
    count: int,
    rng: np.random.Generator,
    has_zero_cap: bool = False,
) -> tuple[np.ndarray, int]:
    """Stage 1 of in-domain key generation: allocates per-gap integer quotas
    k_j <= capacities summing to `count` via multinomial water-filling.

    Returns `(quotas, spilled)` where `spilled` counts keys redirected from
    saturated gaps."""
    # Cap per-gap capacity at count (also prevents int64 overflow in rem.sum()).
    rem = np.minimum(capacities, count)
    total = int(rem.sum())
    if total < count:
        raise RuntimeError(
            f"Domain saturated: window holds {total} free keys, need {count}."
        )
    if total == count:
        return rem, count

    # First round satisfies unsaturated batches directly.
    w = np.ones(len(rem), dtype=np.float64) if weights is None else weights
    if has_zero_cap:
        if w is weights:
            w = w.copy()
        w[rem == 0] = 0.0
    p = w / w.sum()
    taken = np.minimum(_categorical_counts(p, count, rng), rem)
    needed = count - int(taken.sum())
    if needed == 0:
        return taken, 0

    quotas = taken
    rem -= taken
    if w is weights:
        w = w.copy()
    spilled = needed
    while needed > 0:
        w[rem == 0] = 0.0
        p = w / w.sum()
        taken = np.minimum(_categorical_counts(p, needed, rng), rem)
        quotas += taken
        rem -= taken
        needed -= int(taken.sum())
        spilled += needed
    return quotas, spilled


def _log_allocation_drift(
    percentile: float,
    quotas: np.ndarray,
    capacities: np.ndarray,
    unhostable_mass: float,
    spilled: int,
) -> None:
    """Logs structural allocation drift: unhostable kernel mass on dead positions
    and spill redirected from saturated gaps."""
    batch = int(quotas.sum())
    if batch == 0:
        return
    saturated = int(((quotas == capacities) & (capacities > 0)).sum())
    _DRIFT_LOGGER.debug(
        "insert allocation drift at percentile %.2f: unhostable_mass=%.4f, "
        "spill=%.4f (%d/%d keys redirected), %d/%d gaps saturated",
        percentile, unhostable_mass, spilled / batch, spilled, batch,
        saturated, len(quotas),
    )


def _generate_rank_placed_keys(
    active_keys: np.ndarray,
    count: int,
    percentile: float,
    distribution: DistributionConfig,
    rng: np.random.Generator,
    exclude: Optional[np.ndarray] = None,
    geometry: Optional[tuple] = None,
    rank_spread: float = RANK_SPREAD_RATIO,
) -> np.ndarray:
    """In-domain key generation via two-stage capacity-bounded allocation.

    Stage 1 draws per-gap quotas k_j <= c_j by kernel-weighted multinomial
    water-filling. Stage 2 draws k_j distinct keys per gap inside its local
    drawable span (bounded by SPAN_ALLOWANCE * median hostable gap).

    `exclude` forbids a specific key (single-record update guard).
    `geometry=(n, pivot, low, high, base)` overrides the window derived from
    `len(active_keys)` for sliced update refills.
    """
    if geometry is None:
        n = len(active_keys)
        pivot = _percentile_pivot(n, percentile)
        low, high = _rank_window(n, pivot, rank_spread)
        # Materialize the window once; base = low - 1 covers left-neighbor lookups.
        base = max(0, low - 1)
        active_keys = active_keys[base:high]
    else:
        n, pivot, low, high, base = geometry
    rank_radius = _window_radius(n, rank_spread)

    # Position j hosts keys inside (active[j-1], active[j]); j = 0 has no left anchor.
    j_lo = max(low, 1)
    hostable_all = False
    if high > j_lo:
        start, stop = j_lo - base, high - base
        # uint64 arithmetic prevents negative wrap for gaps >= 2^63.
        all_gaps = active_keys[start:stop] - active_keys[start - 1 : stop - 1]
        hostable_mask = all_gaps >= 2
        if bool(hostable_mask.all()):
            hostable_all = True
            hostable_rows = None
            hostable_positions = None
            hostable_gaps = all_gaps
        else:
            hostable_rows = np.nonzero(hostable_mask)[0]
            hostable_positions = hostable_rows + j_lo
            hostable_gaps = all_gaps[hostable_rows]
    else:
        hostable_rows = np.empty(0, dtype=np.int64)
        hostable_positions = np.empty(0, dtype=np.int64)
        hostable_gaps = np.empty(0, dtype=np.uint64)

    # Bounded drawable span: median hostable gap scales the drawable span to
    # local density while capping outlier voids. INT64_CLIP avoids overflow.
    span_cap = (
        int(min(SPAN_ALLOWANCE * float(np.median(hostable_gaps)), INT64_CLIP))
        if len(hostable_gaps)
        else 0
    )
    capacities = np.minimum(
        hostable_gaps - np.uint64(1),
        np.uint64(min(2 * span_cap, INT64_CLIP)),
    ).astype(np.int64)

    forbidden = None
    if exclude is not None and len(exclude):
        slice_idx = int(np.searchsorted(active_keys, np.uint64(exclude[0])))
        pos = slice_idx + base
        if j_lo <= pos < high:
            if hostable_all:
                row, found = pos - j_lo, True
            else:
                row = int(np.searchsorted(hostable_positions, pos))
                found = row < len(hostable_rows) and hostable_positions[row] == pos
            if found:
                gap_left = int(active_keys[slice_idx - 1])
                slot = _span_slot(int(hostable_gaps[row]), span_cap, int(exclude[0]) - gap_left)
                if slot is not None:
                    capacities[row] -= 1
                    forbidden = (row, slot)

    # Weights sit on hostable ranks, not window-relative indices.
    if hostable_all and pivot >= j_lo:
        hostable_weights = _window_weights(distribution, j_lo, high, pivot, rank_radius)
    elif hostable_all:
        # When pivot == 0 and j_lo == 1, pivot lies outside [j_lo, high), so _window_weights's slice bounds do not apply.
        distances = np.arange(j_lo - pivot, high - pivot, dtype=np.int64)
        hostable_weights = _position_weights(distribution, distances, rank_radius)
    else:
        distances = np.abs(hostable_positions - pivot)
        hostable_weights = _position_weights(distribution, distances, rank_radius)

    quotas, spilled = _allocate_gap_quotas(
        capacities, hostable_weights, count, rng,
        has_zero_cap=forbidden is not None and capacities[forbidden[0]] == 0,
    )
    if _DRIFT_LOGGER.isEnabledFor(logging.DEBUG):
        if hostable_weights is None:
            unhostable_mass = (high - low - len(quotas)) / (high - low)
        else:
            full_weights = _window_weights(
                distribution, low, high, pivot, rank_radius
            )
            unhostable_mass = 1.0 - hostable_weights.sum() / full_weights.sum()
        _log_allocation_drift(percentile, quotas, capacities, unhostable_mass, spilled)

    # Stage 2: distinct span slots per gap. Quota-1 gaps vectorize; multi-key gaps loop.
    def left_records(rows):
        if hostable_all:
            rows = rows + j_lo
        else:
            rows = hostable_positions[rows]
        return active_keys[rows - base - 1]

    keys_parts = []

    forbidden_draw = forbidden is not None and quotas[forbidden[0]] == 1
    single = quotas == 1
    if forbidden_draw:
        single[forbidden[0]] = False
    single_rows = np.nonzero(single)[0]
    if len(single_rows):
        s = capacities[single_rows]
        slots = rng.integers(1, s + 1, size=len(single_rows), dtype=np.int64)
        offs = _span_offsets(hostable_gaps[single_rows], span_cap, slots)
        keys_parts.append(left_records(single_rows) + offs)

    if forbidden_draw:
        row, f_slot = forbidden
        s = capacities[row]
        slot = int(rng.integers(1, s + 2, dtype=np.int64))
        while slot == f_slot:
            slot = int(rng.integers(1, s + 2, dtype=np.int64))
        offs = _span_offsets(hostable_gaps[row : row + 1], span_cap, np.array([slot]))
        keys_parts.append(left_records(row) + offs)

    for row in np.nonzero(quotas > 1)[0]:
        k = int(quotas[row])
        s = int(capacities[row])
        if forbidden is not None and row == forbidden[0]:
            # s already excludes the forbidden slot; draw from [1, s] and shift slots >= f_slot.
            raw = _sample_distinct_offsets(s, k, rng)
            slots = np.where(raw < f_slot, raw, raw + np.uint64(1))
        else:
            slots = _sample_distinct_offsets(s, k, rng)
        offs = _span_offsets(hostable_gaps[row : row + 1], span_cap, slots)
        keys_parts.append(left_records(row) + offs)

    return np.concatenate(keys_parts)


def sample_new_keys(
    active_keys: np.ndarray,
    count: int,
    percentile: float,
    distribution: DistributionConfig,
    rng: np.random.Generator,
    rank_spread: float = RANK_SPREAD_RATIO,
) -> np.ndarray:
    """Samples `count` distinct keys not present in `active_keys`.

    percentile <= 0 prepends below k_min; percentile >= 100 appends above k_max;
    in-domain percentiles allocate keys across window gaps via two-stage allocation.
    """
    if count <= 0:
        return np.empty(0, dtype=np.uint64)

    if percentile <= 0.0 or percentile >= 100.0:
        return _generate_boundary_keys(
            active_keys, count, percentile, distribution, rng, rank_spread
        )

    return _generate_rank_placed_keys(
        active_keys, count, percentile, distribution, rng, rank_spread=rank_spread
    )


def sample_insert_keys(
    active_keys: np.ndarray,
    count: int,
    percentile: float,
    distribution: DistributionConfig,
    is_sorted: bool,
    rng: Optional[np.random.Generator] = None,
    rank_spread: float = RANK_SPREAD_RATIO,
) -> np.ndarray:
    """Samples `count` distinct insert keys around `percentile`."""
    if count <= 0:
        return np.empty(0, dtype=np.uint64)

    if rng is None:
        rng = np.random.default_rng()

    result = sample_new_keys(
        active_keys=active_keys,
        count=count,
        percentile=percentile,
        distribution=distribution,
        rng=rng,
        rank_spread=rank_spread,
    )
    if is_sorted:
        return np.sort(result)
    return rng.permutation(result)


def sample_existing_indices(
    n: int,
    count: int,
    percentile: float,
    distribution: DistributionConfig,
    is_sorted: bool,
    rng: Optional[np.random.Generator] = None,
    rank_spread: float = RANK_SPREAD_RATIO,
) -> np.ndarray:
    """Samples `count` distinct indices around `percentile` over the rank window."""
    if count <= 0:
        return np.empty(0, dtype=np.int64)

    if rng is None:
        rng = np.random.default_rng()

    count = min(count, n)
    if count == n:
        arr = np.arange(n, dtype=np.int64)
        return arr if is_sorted else rng.permutation(arr)

    pivot = _percentile_pivot(n, percentile)
    low, high = _rank_window(n, pivot, rank_spread)
    if high - low < count:
        raise RuntimeError(
            f"Domain saturated: unable to sample {count} unique indices from window [{low}, {high})."
        )

    if isinstance(distribution, UniformConfig):
        chosen = rng.choice(high - low, size=count, replace=False) + low
    else:
        if not isinstance(distribution, (ZipfConfig, GaussianConfig)):
            raise TypeError(
                f"unsupported distribution config: {type(distribution).__name__}"
            )
        weights = _window_weights(distribution, low, high, pivot,
                                  _window_radius(n, rank_spread))
        chosen = _ares_select(weights, count, rng) + low
    return np.sort(chosen) if is_sorted else chosen


def sample_existing_keys(
    active_keys: np.ndarray,
    count: int,
    percentile: float,
    distribution: DistributionConfig,
    is_sorted: bool,
    rng: Optional[np.random.Generator] = None,
    rank_spread: float = RANK_SPREAD_RATIO,
) -> np.ndarray:
    """Samples `count` distinct keys from `active_keys` around `percentile`."""
    indices = sample_existing_indices(
        n=len(active_keys),
        count=count,
        percentile=percentile,
        distribution=distribution,
        is_sorted=is_sorted,
        rng=rng,
        rank_spread=rank_spread,
    )
    return active_keys[indices]


def sample_point_queries(
    active_keys: np.ndarray,
    count: int,
    percentile: Optional[float],
    distribution: DistributionConfig = ZipfConfig(),
    rng: Optional[np.random.Generator] = None,
    rank_spread: float = RANK_SPREAD_RATIO,
) -> np.ndarray:
    """Point-read stream over `active_keys`: draws `count` keys with replacement
    weighted around `percentile` (or full range when percentile=None)."""
    if count <= 0:
        return np.empty(0, dtype=np.uint64)

    if rng is None:
        rng = np.random.default_rng()

    n = len(active_keys)
    if percentile is None:
        low, high, pivot = 0, n, (n - 1) // 2
    else:
        pivot = _percentile_pivot(n, percentile)
        low, high = _rank_window(n, pivot, rank_spread)

    # Scale kernel to window span so untargeted reads cover the dataset.
    radius = max(pivot - low, high - 1 - pivot, 1)
    weights = _window_weights(distribution, low, high, pivot, radius)
    p = None if weights is None else weights / weights.sum()

    ranks = rng.choice(high - low, size=count, replace=True, p=p)
    return active_keys[ranks + low]


def sample_knn_query_points(
    active_keys: np.ndarray,
    count: int,
    percentile: Optional[float],
    distribution: DistributionConfig,
    rng: Optional[np.random.Generator] = None,
    rank_spread: float = RANK_SPREAD_RATIO,
) -> np.ndarray:
    """Samples `count` query points in [k_min, k_max] around `percentile`
    (or full range when percentile=None). Out-of-bounds kernel draws are
    clamped to [k_min, k_max]."""
    if count <= 0:
        return np.empty(0, dtype=np.uint64)

    if rng is None:
        rng = np.random.default_rng()

    k_min, k_max = int(active_keys[0]), int(active_keys[-1])
    if percentile is None:
        pivot = int(active_keys[(len(active_keys) - 1) // 2])
        key_radius = (k_max - k_min) / 2.0
    else:
        pivot, key_radius = _pivot_and_radius(active_keys, percentile, rank_spread)

    raw = draw_offsets(count, key_radius, distribution, rng)
    # Clamp signed offsets to stay within domain bounds and INT64_CLIP.
    lo = max(k_min - pivot, -INT64_CLIP)
    hi = min(k_max - pivot, INT64_CLIP)
    offsets = np.clip(np.round(raw), lo, hi).astype(np.int64)
    return _to_uint64_offsets(pivot, offsets)


def sample_update_pairs(
    active_keys: np.ndarray,
    count: int,
    percentile: float,
    distribution: DistributionConfig,
    is_sorted: bool,
    rng: Optional[np.random.Generator] = None,
    rank_spread: float = RANK_SPREAD_RATIO,
) -> tuple:
    """Samples `count` distinct (k_old, k_new) pairs for an update operation.

    - k_old keys are sampled around `percentile`.
    - k_new keys are refilled into the post-drain dataset around the same
      percentile. Vacated slots are eligible refill targets; for count == 1,
      the vacated slot is excluded to prevent a no-op update.
    - Fails fast if the drain empties the active set.
    """
    if count <= 0:
        return np.empty(0, dtype=np.uint64), np.empty(0, dtype=np.uint64)

    if rng is None:
        rng = np.random.default_rng()

    count = min(count, len(active_keys))
    boundary = percentile <= 0.0 or percentile >= 100.0

    old_indices = sample_existing_indices(
        n=len(active_keys),
        count=count,
        percentile=percentile,
        distribution=distribution,
        is_sorted=is_sorted,
        rng=rng,
        rank_spread=rank_spread,
    )
    old_keys = active_keys[old_indices]

    if boundary:
        new_keys = _generate_boundary_keys(
            active_keys, count, percentile, distribution, rng, rank_spread
        )
        if not is_sorted:
            new_keys = rng.permutation(new_keys)
    else:
        n_remaining = len(active_keys) - count
        if n_remaining == 0:
            raise RuntimeError(
                "Full-drain update: no survivors remain to anchor refill keys; "
                "declare a smaller batch."
            )
        else:
            # Post-drain window is covered by slicing the pre-drain window extended by `count`.
            pivot = _percentile_pivot(n_remaining, percentile)
            low, high = _rank_window(n_remaining, pivot, rank_spread)
            slice_lo = max(0, low - count - 1)
            slice_hi = min(len(active_keys), high + count)
            window = active_keys[slice_lo:slice_hi]
            idx = np.searchsorted(window, old_keys)
            valid = idx < len(window)
            idx = idx[valid]
            idx = idx[window[idx] == old_keys[valid]]
            window = np.delete(window, idx)
            new_keys = _generate_rank_placed_keys(
                window, count, percentile, distribution, rng,
                exclude=old_keys if count == 1 else None,
                geometry=(n_remaining, pivot, low, high, slice_lo),
                rank_spread=rank_spread,
            )

    if is_sorted:
        old_keys = np.sort(old_keys)
        new_keys = np.sort(new_keys)
    elif not boundary:
        new_keys = rng.permutation(new_keys)

    return old_keys, new_keys
