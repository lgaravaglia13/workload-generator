"""Tests for the sampling engine public API contract."""
import inspect
import re
import sys
import unittest
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fixtures
import numpy as np

from distributions import DEFAULT_THETA, GaussianConfig, UniformConfig, ZipfConfig
from sampler import (
    RANK_SPREAD_RATIO,
    sample_existing_indices,
    sample_existing_keys,
    sample_insert_keys,
    sample_knn_query_points,
    sample_new_keys,
    sample_point_queries,
    sample_update_pairs,
)

SEED = 42
UINT64_MAX = int(np.iinfo(np.uint64).max)

COUNT_SMALL = 10
COUNT_MEDIUM = 40
COUNT_LARGE = 100

DISTRIBUTIONS = (GaussianConfig(), ZipfConfig(), UniformConfig())
INDOMAIN_PERCENTILES = (10.0, 50.0, 90.0)
LOCALIZATION_PERCENTILES = (1.0, 50.0, 99.0)

PREPEND_SAFE_FIXTURES = (fixtures.gaussian, fixtures.clustered_two_hot)
INDOMAIN_INSERT_FIXTURES = (
    fixtures.uniform_sparse,
    fixtures.zipfian,
    fixtures.gaussian,
    fixtures.clustered_two_hot,
)
APPEND_FIXTURES = (
    fixtures.uniform_dense,
    fixtures.uniform_sparse,
    fixtures.zipfian,
    fixtures.gaussian,
    fixtures.clustered_two_hot,
    fixtures.tiny_two_keys,
)

MSG_PREPEND_UNDERFLOW = "Prepend underflow: generated keys fall below uint64 minimum (0)."
MSG_APPEND_OVERFLOW = "Append overflow: generated keys exceed UINT64_MAX."


class TestExistingIndices(unittest.TestCase):
    def test_zero_count_returns_empty_int64(self):
        rng = np.random.default_rng(SEED)

        empty = sample_existing_indices(100, 0, 50, GaussianConfig(), True, rng)
        self.assertEqual(empty.dtype, np.int64)
        self.assertEqual(empty.size, 0)

    def test_full_count_returns_entire_index_set(self):
        n = 1000
        rng = np.random.default_rng(SEED)

        sorted_idx = sample_existing_indices(n, n, 50.0, GaussianConfig(), True, rng)
        np.testing.assert_array_equal(sorted_idx, np.arange(n, dtype=np.int64))

        clamped = sample_existing_indices(n, n + 500, 50.0, GaussianConfig(), True, rng)
        np.testing.assert_array_equal(clamped, np.arange(n, dtype=np.int64))

        unsorted_idx = sample_existing_indices(n, n, 50.0, GaussianConfig(), False, rng)
        np.testing.assert_array_equal(np.sort(unsorted_idx), np.arange(n, dtype=np.int64))

    def test_indices_are_distinct_and_inside_the_rank_window(self):
        n = 100_000
        for dist in DISTRIBUTIONS:
            for percentile in (0.0, 25.0, 50.0, 100.0):
                with self.subTest(dist=type(dist).__name__, percentile=percentile):
                    pivot = int(np.clip(np.floor((n - 1) * percentile / 100.0), 0, n - 1))
                    radius = max(1, int(n * RANK_SPREAD_RATIO))
                    low = max(0, pivot - radius)
                    high = min(n, pivot + radius + 1)

                    indices = sample_existing_indices(
                        n, COUNT_MEDIUM, percentile, dist, is_sorted=True,
                        rng=np.random.default_rng(SEED),
                    )

                    self.assertEqual(indices.dtype, np.int64)
                    self.assertEqual(len(indices), COUNT_MEDIUM)
                    self.assertEqual(len(np.unique(indices)), COUNT_MEDIUM)
                    self.assertGreaterEqual(int(indices.min()), low)
                    self.assertLess(int(indices.max()), high)

    def test_unsorted_emission_is_a_permutation_of_the_sorted_multiset(self):
        n = 5_000
        for dist in DISTRIBUTIONS:
            with self.subTest(dist=type(dist).__name__):
                sorted_idx = sample_existing_indices(
                    n, COUNT_MEDIUM, 50.0, dist, True, np.random.default_rng(SEED)
                )
                unsorted_idx = sample_existing_indices(
                    n, COUNT_MEDIUM, 50.0, dist, False, np.random.default_rng(SEED)
                )

                np.testing.assert_array_equal(np.sort(unsorted_idx), sorted_idx)
                self.assertTrue(np.all(np.diff(sorted_idx) > 0))
                self.assertFalse(np.array_equal(unsorted_idx, sorted(unsorted_idx)))

    def test_median_aligns_with_the_percentile_pivot(self):
        n = 1_000
        for dist in DISTRIBUTIONS:
            for pct in (10, 25, 50, 75, 90):
                with self.subTest(dist=type(dist).__name__, pct=pct):
                    indices = sample_existing_indices(
                        n, 50, pct, dist, True, np.random.default_rng(SEED)
                    )
                    expected_pivot = int(np.floor((n - 1) * (pct / 100.0)))
                    rel_error = abs(float(np.median(indices)) - expected_pivot) / n
                    self.assertLessEqual(rel_error, 0.05)

    def test_window_saturation_fails_fast_with_exact_message(self):
        n = 2_000
        count = 300
        pivot = int(np.floor((n - 1) * 0.5))
        radius = max(1, int(n * RANK_SPREAD_RATIO))
        low = max(0, pivot - radius)
        high = min(n, pivot + radius + 1)

        for dist in DISTRIBUTIONS:
            with self.subTest(dist=type(dist).__name__):
                with self.assertRaises(RuntimeError) as ctx:
                    sample_existing_indices(
                        n, count, 50.0, dist, is_sorted=True,
                        rng=np.random.default_rng(SEED),
                    )
                self.assertEqual(
                    str(ctx.exception),
                    f"Domain saturated: unable to sample {count} unique indices "
                    f"from window [{low}, {high}).",
                )

    def test_large_scale_selection_stays_window_bounded(self):
        # Window-based selection must scale without full index space allocation.
        n = 10_000_000
        indices = sample_existing_indices(
            n, 100, 50, GaussianConfig(), True, np.random.default_rng(SEED)
        )

        self.assertEqual(len(indices), 100)
        self.assertEqual(len(np.unique(indices)), 100)
        self.assertTrue(np.all(indices >= 0))
        self.assertTrue(np.all(indices < n))


class TestExistingKeys(unittest.TestCase):
    def test_sampled_keys_are_sorted_uint64_members_of_the_active_set(self):
        keys = fixtures.uniform_sparse()

        sampled = sample_existing_keys(
            keys, COUNT_MEDIUM, 40.0, GaussianConfig(), True, np.random.default_rng(SEED)
        )

        self.assertEqual(sampled.dtype, np.uint64)
        self.assertEqual(len(sampled), COUNT_MEDIUM)
        self.assertTrue(np.all(np.isin(sampled, keys)))
        self.assertTrue(fixtures.strictly_sorted(sampled))


class TestInDomainInserts(unittest.TestCase):
    def test_batches_are_exact_unique_and_collision_free(self):
        for keys_fn in INDOMAIN_INSERT_FIXTURES:
            keys = keys_fn()
            for dist in DISTRIBUTIONS:
                for percentile in INDOMAIN_PERCENTILES:
                    with self.subTest(
                        fixture=len(keys), dist=type(dist).__name__, percentile=percentile
                    ):
                        new_keys = sample_insert_keys(
                            keys, COUNT_MEDIUM, percentile, dist,
                            is_sorted=True,
                            rng=np.random.default_rng(SEED),
                        )

                        self.assertEqual(len(new_keys), COUNT_MEDIUM)
                        self.assertEqual(len(np.unique(new_keys)), COUNT_MEDIUM)
                        self.assertFalse(np.any(np.isin(new_keys, keys)))

    def test_keys_stay_inside_the_declared_rank_window(self):
        # In-domain inserts must fall inside the rank window.
        for keys in (fixtures.uniform_sparse(), fixtures.gaussian(), fixtures.clustered_two_hot()):
            for dist in (UniformConfig(), ZipfConfig(), GaussianConfig()):
                for percentile in LOCALIZATION_PERCENTILES:
                    with self.subTest(
                        fixture=len(keys), dist=type(dist).__name__, percentile=percentile
                    ):
                        n = len(keys)
                        radius = max(1, int(n * RANK_SPREAD_RATIO))
                        pivot_rank = int(np.clip(np.floor((n - 1) * percentile / 100.0), 0, n - 1))
                        low = max(0, pivot_rank - radius)
                        high = min(n, pivot_rank + radius + 1)

                        new_keys = sample_insert_keys(
                            keys, COUNT_MEDIUM, percentile, dist,
                            is_sorted=True, rng=np.random.default_rng(SEED),
                        )

                        ranks = np.searchsorted(keys, new_keys)
                        self.assertTrue(np.all(ranks >= low))
                        self.assertTrue(np.all(ranks < high))

    def test_keys_stay_inside_the_domain_at_edge_percentiles(self):
        for keys in (fixtures.uniform_sparse(), fixtures.clustered_two_hot()):
            for percentile in (1.0, 99.0):
                with self.subTest(fixture=len(keys), percentile=percentile):
                    new_keys = sample_insert_keys(
                        keys, COUNT_MEDIUM, percentile, GaussianConfig(),
                        is_sorted=True, rng=np.random.default_rng(SEED),
                    )
                    self.assertTrue(np.all(new_keys >= keys[0]))
                    self.assertTrue(np.all(new_keys <= keys[-1]))

    def test_dispersion_follows_the_zipf_gaussian_uniform_hierarchy(self):
        # Median absolute rank distance confirms zipf < gaussian < uniform spread.
        keys = fixtures.trace_base_keys()
        pivot = int(np.clip(np.floor((len(keys) - 1) * 0.5), 0, len(keys) - 1))

        def median_abs_rank_distance(config, seeds):
            medians = []
            for seed in seeds:
                sampled = sample_insert_keys(
                    keys, 200, 50, config, False, np.random.default_rng(seed)
                )
                ranks = np.searchsorted(keys, sampled)
                medians.append(float(np.median(np.abs(ranks - pivot))))
            return float(np.mean(medians))

        med_z = median_abs_rank_distance(ZipfConfig(), range(1, 6))
        med_g = median_abs_rank_distance(GaussianConfig(), range(6, 11))
        med_u = median_abs_rank_distance(UniformConfig(), range(11, 16))
        self.assertLess(med_z, med_g)
        self.assertLess(med_g, med_u)

    def test_rank_window_is_immune_to_far_away_outliers(self):
        normal_keys = np.arange(100, 11_000, step=10, dtype=np.uint64)
        outlier_keys = np.append(normal_keys, np.uint64(10**15))
        outlier_keys.sort()

        sampled = sample_insert_keys(
            outlier_keys, 50, 50, GaussianConfig(), True, np.random.default_rng(SEED)
        )

        self.assertLess(int(sampled.max()), 20_000)
        self.assertGreater(int(sampled.min()), 1_000)

    def test_inserts_work_in_the_upper_uint64_key_space(self):
        base_offset = np.uint64(10_000_000_000_000_000_000)
        rng = np.random.default_rng(SEED)
        active_keys = np.sort(
            base_offset + rng.choice(np.arange(0, 100_000, dtype=np.uint64), size=1000, replace=False)
        )

        for dist in DISTRIBUTIONS:
            with self.subTest(dist=type(dist).__name__):
                new_keys = sample_insert_keys(
                    active_keys, 50, 50, dist, is_sorted=True, rng=rng
                )

                self.assertEqual(len(new_keys), 50)
                self.assertEqual(len(np.unique(new_keys)), 50)
                self.assertEqual(new_keys.dtype, np.uint64)
                self.assertTrue(np.all(new_keys > np.uint64(2**63 - 1)))
                self.assertTrue(np.all(new_keys >= active_keys[0]))
                self.assertTrue(np.all(new_keys <= active_keys[-1]))
                self.assertTrue(np.isin(new_keys, active_keys).sum() == 0)

    def test_sparse_dataset_fills_the_batch(self):
        keys = fixtures.trace_base_keys()
        new_keys = sample_insert_keys(
            keys, 50, 50.0, GaussianConfig(), is_sorted=True,
            rng=np.random.default_rng(SEED),
        )

        self.assertEqual(len(new_keys), 50)
        self.assertEqual(len(np.unique(new_keys)), 50)

    def test_dense_domain_insert_saturates_with_exact_message(self):
        with self.assertRaises(RuntimeError) as ctx:
            sample_insert_keys(
                fixtures.uniform_dense(), COUNT_MEDIUM, 50.0, ZipfConfig(),
                is_sorted=True,
                rng=np.random.default_rng(SEED),
            )

        self.assertEqual(
            str(ctx.exception),
            f"Domain saturated: window holds 0 free keys, need {COUNT_MEDIUM}.",
        )

    def test_saturation_is_reachable_inside_a_sparse_dataset(self):
        sparse_part = np.arange(0, 5_000, 500)
        dense_part = np.arange(5_000, 7_000)
        active_keys = np.concatenate([sparse_part, dense_part]).astype(np.uint64)

        with self.assertRaises(RuntimeError) as ctx:
            sample_insert_keys(
                active_keys, 150, 99.0, ZipfConfig(),
                is_sorted=True,
                rng=np.random.default_rng(SEED),
            )

        self.assertEqual(
            str(ctx.exception),
            "Domain saturated: window holds 0 free keys, need 150.",
        )


class TestBoundaryInserts(unittest.TestCase):
    def test_prepend_keys_are_strictly_below_k_min(self):
        for keys in PREPEND_SAFE_FIXTURES:
            for dist in DISTRIBUTIONS:
                with self.subTest(fixture=len(keys()), dist=type(dist).__name__):
                    active_keys = keys()
                    k_min = int(active_keys[0])

                    new_keys = sample_insert_keys(
                        active_keys, COUNT_MEDIUM, 0.0, dist,
                        is_sorted=True, rng=np.random.default_rng(SEED),
                    )

                    self.assertEqual(len(new_keys), COUNT_MEDIUM)
                    self.assertEqual(new_keys.dtype, np.uint64)
                    self.assertEqual(len(np.unique(new_keys)), COUNT_MEDIUM)
                    self.assertTrue(np.all(new_keys < np.uint64(k_min)))

    def test_prepend_on_single_key_anchor_yields_exact_unit_gaps(self):
        new_keys = sample_insert_keys(
            fixtures.tiny_single_low(), 3, 0.0, UniformConfig(),
            is_sorted=True, rng=np.random.default_rng(SEED),
        )

        np.testing.assert_array_equal(new_keys, np.array([2, 3, 4], dtype=np.uint64))

    def test_prepend_underflow_guard_raises_with_exact_message(self):
        cases = [
            (fixtures.tiny_single_low(), COUNT_LARGE),  # headroom 5 < batch 100
            (fixtures.uniform_dense(), 1),               # k_min == 0: no headroom
            (fixtures.zipfian(), 1),
        ]
        for keys, count in cases:
            with self.subTest(fixture=len(keys), count=count):
                with self.assertRaises(RuntimeError) as ctx:
                    sample_insert_keys(
                        keys, count, 0.0, ZipfConfig(),
                        is_sorted=True, rng=np.random.default_rng(SEED),
                    )
                self.assertEqual(str(ctx.exception), MSG_PREPEND_UNDERFLOW)

    def test_append_keys_are_strictly_above_k_max(self):
        for keys in APPEND_FIXTURES:
            for dist in DISTRIBUTIONS:
                with self.subTest(fixture=len(keys()), dist=type(dist).__name__):
                    active_keys = keys()
                    k_max = int(active_keys[-1])

                    new_keys = sample_insert_keys(
                        active_keys, COUNT_MEDIUM, 100.0, dist,
                        is_sorted=True, rng=np.random.default_rng(SEED),
                    )

                    self.assertEqual(len(new_keys), COUNT_MEDIUM)
                    self.assertEqual(new_keys.dtype, np.uint64)
                    self.assertEqual(len(np.unique(new_keys)), COUNT_MEDIUM)
                    self.assertTrue(np.all(new_keys > np.uint64(k_max)))
                    self.assertTrue(np.all(new_keys <= np.uint64(UINT64_MAX)))

    def test_append_on_single_key_anchor_yields_exact_unit_gaps(self):
        new_keys = sample_insert_keys(
            fixtures.tiny_single_high(), 3, 100.0, UniformConfig(),
            is_sorted=True, rng=np.random.default_rng(SEED),
        )

        np.testing.assert_array_equal(
            new_keys,
            np.array([UINT64_MAX - 4, UINT64_MAX - 3, UINT64_MAX - 2], dtype=np.uint64),
        )

    def test_append_on_dense_dataset_yields_consecutive_keys(self):
        keys = fixtures.uniform_dense()
        k_max = int(keys[-1])

        new_keys = sample_insert_keys(
            keys, COUNT_LARGE, 100.0, ZipfConfig(),
            is_sorted=True, rng=np.random.default_rng(SEED),
        )

        np.testing.assert_array_equal(
            new_keys, np.arange(k_max + 1, k_max + 1 + COUNT_LARGE, dtype=np.uint64)
        )

    def test_append_overflow_guard_raises_with_exact_message(self):
        with self.assertRaises(RuntimeError) as ctx:
            sample_insert_keys(
                fixtures.tiny_single_high(), COUNT_LARGE, 100.0, ZipfConfig(),
                is_sorted=True, rng=np.random.default_rng(SEED),
            )

        self.assertEqual(str(ctx.exception), MSG_APPEND_OVERFLOW)

    def test_boundary_sorted_emission_is_ascending(self):
        # prepend headroom is validated only for PREPEND_SAFE_FIXTURES;
        # append headroom (UINT64_MAX - k_max) is effectively unbounded
        for keys, percentile in [
            (keys, 0.0) for keys in PREPEND_SAFE_FIXTURES
        ] + [
            (keys, 100.0) for keys in APPEND_FIXTURES
        ]:
            with self.subTest(fixture=len(keys()), percentile=percentile):
                new_keys = sample_insert_keys(
                    keys(), COUNT_LARGE, percentile, ZipfConfig(),
                    is_sorted=True, rng=np.random.default_rng(SEED),
                )
                self.assertTrue(fixtures.strictly_sorted(new_keys))

    def test_boundary_unsorted_emission_permutes_the_same_multiset(self):
        for keys in PREPEND_SAFE_FIXTURES:
            for percentile in (0.0, 100.0):
                with self.subTest(fixture=len(keys()), percentile=percentile):
                    sorted_keys = sample_insert_keys(
                        keys(), COUNT_MEDIUM, percentile, ZipfConfig(),
                        is_sorted=True, rng=np.random.default_rng(SEED),
                    )
                    unsorted_keys = sample_insert_keys(
                        keys(), COUNT_MEDIUM, percentile, ZipfConfig(),
                        is_sorted=False, rng=np.random.default_rng(SEED),
                    )

                    self.assertEqual(len(unsorted_keys), COUNT_MEDIUM)
                    np.testing.assert_array_equal(np.sort(unsorted_keys), sorted_keys)

    def test_boundary_keys_never_collide_with_active_keys(self):
        for keys in PREPEND_SAFE_FIXTURES:
            for dist in DISTRIBUTIONS:
                for percentile in (0.0, 100.0):
                    with self.subTest(
                        fixture=len(keys()), dist=type(dist).__name__, percentile=percentile
                    ):
                        active_keys = keys()
                        new_keys = sample_insert_keys(
                            active_keys, COUNT_MEDIUM, percentile, dist,
                            is_sorted=True, rng=np.random.default_rng(SEED),
                        )

                        self.assertEqual(len(np.unique(new_keys)), COUNT_MEDIUM)
                        self.assertFalse(np.any(np.isin(new_keys, active_keys)))


class TestUpdatePairs(unittest.TestCase):
    def test_in_domain_contract(self):
        for keys in INDOMAIN_INSERT_FIXTURES:
            for dist in DISTRIBUTIONS:
                with self.subTest(fixture=len(keys()), dist=type(dist).__name__):
                    active_keys = keys()
                    old_keys, new_keys = sample_update_pairs(
                        active_keys, COUNT_MEDIUM, 50.0, dist,
                        is_sorted=True, rng=np.random.default_rng(SEED),
                    )

                    self.assertEqual(len(old_keys), COUNT_MEDIUM)
                    self.assertEqual(len(new_keys), COUNT_MEDIUM)
                    self.assertEqual(old_keys.dtype, np.uint64)
                    self.assertEqual(new_keys.dtype, np.uint64)

                    self.assertTrue(np.all(np.isin(old_keys, active_keys)))
                    survivors = np.setdiff1d(active_keys, old_keys)
                    self.assertFalse(np.any(np.isin(new_keys, survivors)))
                    self.assertEqual(len(np.unique(new_keys)), COUNT_MEDIUM)
                    self.assertTrue(fixtures.strictly_sorted(old_keys))
                    self.assertTrue(fixtures.strictly_sorted(new_keys))

    def test_boundary_modes_drain_the_edge_and_move_keys_outside(self):
        for keys in PREPEND_SAFE_FIXTURES:
            for percentile in (0.0, 100.0):
                with self.subTest(fixture=len(keys()), percentile=percentile):
                    active_keys = keys()
                    n = len(active_keys)
                    radius = max(1, int(n * RANK_SPREAD_RATIO))
                    if percentile <= 0.0:
                        drain_low, drain_high = 0, min(n, radius + 1)
                    else:
                        drain_low, drain_high = max(0, n - 1 - radius), n

                    old_keys, new_keys = sample_update_pairs(
                        active_keys, COUNT_SMALL, percentile, ZipfConfig(),
                        is_sorted=True, rng=np.random.default_rng(SEED),
                    )

                    self.assertEqual(len(old_keys), COUNT_SMALL)
                    self.assertEqual(len(new_keys), COUNT_SMALL)
                    old_ranks = np.searchsorted(active_keys, old_keys)
                    self.assertGreaterEqual(int(old_ranks.min()), drain_low)
                    self.assertLess(int(old_ranks.max()), drain_high)
                    if percentile <= 0.0:
                        self.assertTrue(np.all(new_keys < active_keys[0]))
                    else:
                        self.assertTrue(np.all(new_keys > active_keys[-1]))
                    self.assertFalse(np.any(old_keys == new_keys))

    def test_refill_reuses_vacated_slots_in_a_fully_dense_key_space(self):
        # Vacated slots are the only collision-free refill locations in a dense domain.
        dense_keys = np.arange(1000, 1200, dtype=np.uint64)

        old_keys, new_keys = sample_update_pairs(
            dense_keys, 10, 50, UniformConfig(), False, np.random.default_rng(SEED)
        )

        dense_set = set(int(k) for k in dense_keys)
        old_set = set(int(k) for k in old_keys)
        new_set = set(int(k) for k in new_keys)
        self.assertTrue(old_set.issubset(dense_set))
        self.assertTrue(new_set.issubset(dense_set))
        self.assertTrue(new_set.isdisjoint(dense_set - old_set))

    def test_count_one_in_a_fully_dense_space_saturates(self):
        active_keys = np.arange(1_000, 1_100, dtype=np.uint64)

        with self.assertRaises(RuntimeError) as ctx:
            sample_update_pairs(
                active_keys, 1, 50.0, UniformConfig(),
                is_sorted=True,
                rng=np.random.default_rng(SEED),
            )

        self.assertEqual(
            str(ctx.exception),
            "Domain saturated: window holds 0 free keys, need 1.",
        )

    def test_full_drain_fails_fast_instead_of_no_op(self):
        # Draining all keys leaves no survivors to anchor an in-domain refill.
        small_keys = np.array(
            [10, 1000, 20000, 50000, 100000, 200000, 400000, 600000, 800000, 999999],
            dtype=np.uint64,
        )

        for count in (10, 50):
            with self.subTest(count=count):
                with self.assertRaises(RuntimeError) as ctx:
                    sample_update_pairs(
                        small_keys, count, 50, UniformConfig(), True, np.random.default_rng(SEED)
                    )
                self.assertIn("Full-drain update", str(ctx.exception))

    def test_single_key_in_domain_update_fails_fast(self):
        with self.assertRaises(RuntimeError) as ctx:
            sample_update_pairs(
                np.array([500], dtype=np.uint64), 1, 50.0, ZipfConfig(),
                is_sorted=True, rng=np.random.default_rng(SEED),
            )
        self.assertIn("Full-drain update", str(ctx.exception))


class TestPointQueries(unittest.TestCase):
    def test_reads_hit_only_active_keys(self):
        for keys in (fixtures.uniform_sparse(), fixtures.clustered_two_hot()):
            for dist in DISTRIBUTIONS:
                for percentile in (None, 10.0, 50.0, 90.0):
                    with self.subTest(
                        fixture=len(keys), dist=type(dist).__name__, percentile=percentile
                    ):
                        reads = sample_point_queries(
                            keys, 500, percentile, distribution=dist,
                            rng=np.random.default_rng(SEED),
                        )

                        self.assertEqual(len(reads), 500)
                        self.assertTrue(np.all(np.isin(reads, keys)))

    def test_reads_stay_inside_the_rank_window_key_range(self):
        keys = fixtures.uniform_sparse()
        n = len(keys)
        percentile = 25.0
        pivot = int(np.clip(np.floor((n - 1) * percentile / 100.0), 0, n - 1))
        radius = max(1, int(n * RANK_SPREAD_RATIO))
        low = max(0, pivot - radius)
        high = min(n, pivot + radius + 1)

        for dist in DISTRIBUTIONS:
            with self.subTest(dist=type(dist).__name__):
                reads = sample_point_queries(
                    keys, 1_000, percentile, distribution=dist,
                    rng=np.random.default_rng(SEED),
                )
                self.assertTrue(np.all(reads >= keys[low]))
                self.assertTrue(np.all(reads <= keys[high - 1]))

    def test_reads_allow_duplicates_beyond_the_window_size(self):
        # Repeatable hot-spot stream: far more reads than window slots
        active_keys = np.arange(1_000, dtype=np.uint64)

        reads = sample_point_queries(
            active_keys, 5_000, 50.0, distribution=ZipfConfig(),
            rng=np.random.default_rng(SEED),
        )

        self.assertEqual(len(reads), 5_000)
        self.assertLess(len(np.unique(reads)), 5_000)

    def test_untargeted_kernel_reads_cover_the_full_active_range(self):
        # percentile=None scales the kernel to the full-range span rather than rank_spread.
        keys = fixtures.uniform_sparse()
        reads = sample_point_queries(
            keys, 5_000, None, distribution=GaussianConfig(),
            rng=np.random.default_rng(SEED),
        )

        frac = np.searchsorted(keys, reads) / len(keys)
        self.assertGreater(float(np.mean((frac < 0.45) | (frac > 0.55))), 0.5)
        self.assertLess(float(frac.min()), 0.25)
        self.assertGreater(float(frac.max()), 0.75)


class TestKnnQueryPoints(unittest.TestCase):
    def test_points_stay_inside_the_active_domain(self):
        for keys in (fixtures.uniform_sparse(), fixtures.gaussian(), fixtures.clustered_two_hot()):
            for dist in DISTRIBUTIONS:
                for percentile in (None, 0.0, 10.0, 50.0, 90.0, 100.0):
                    with self.subTest(
                        fixture=len(keys), dist=type(dist).__name__, percentile=percentile
                    ):
                        pts = sample_knn_query_points(
                            keys, 500, percentile, dist,
                            rng=np.random.default_rng(SEED),
                        )

                        self.assertEqual(len(pts), 500)
                        self.assertEqual(pts.dtype, np.uint64)
                        self.assertTrue(np.all(pts >= keys[0]))
                        self.assertTrue(np.all(pts <= keys[-1]))

    def test_non_positive_count_returns_empty_typed_stream(self):
        pts = sample_knn_query_points(
            fixtures.uniform_sparse(), 0, 50.0, ZipfConfig(),
            rng=np.random.default_rng(SEED),
        )
        self.assertEqual(pts.dtype, np.uint64)
        self.assertEqual(pts.size, 0)

    def test_full_span_uint64_domain_stays_in_domain_without_cast_warnings(self):
        # Large uint64 range must avoid float-to-int overflow warnings.
        keys = np.array([0, 5, UINT64_MAX - 4], dtype=np.uint64)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pts = sample_knn_query_points(
                keys, 1_000, 100.0, GaussianConfig(),
                rng=np.random.default_rng(SEED),
            )

        self.assertEqual(len(pts), 1_000)
        self.assertEqual([str(w.message) for w in caught], [])
        self.assertTrue(np.all(pts >= keys[0]))
        self.assertTrue(np.all(pts <= keys[-1]))
        self.assertGreater(len(np.unique(pts)), 100)

    def test_same_seed_reproduces_the_same_points(self):
        keys = fixtures.gaussian()
        first = sample_knn_query_points(
            keys, 300, 75.0, GaussianConfig(), rng=np.random.default_rng(SEED)
        )
        second = sample_knn_query_points(
            keys, 300, 75.0, GaussianConfig(), rng=np.random.default_rng(SEED)
        )
        self.assertTrue(np.array_equal(first, second))


class TestInsertDriftLogging(unittest.TestCase):
    @staticmethod
    def _two_free_blocks_off_pivot() -> np.ndarray:
        return np.setdiff1d(
            np.arange(500, 600, dtype=np.uint64),
            np.concatenate([np.arange(550, 560), np.arange(570, 580)]),
        )

    def _drift_record(self, keys, count, distribution, rank_spread=RANK_SPREAD_RATIO):
        with self.assertLogs("sampler.drift", level="DEBUG") as captured:
            batch = sample_insert_keys(
                keys, count, 50.0, distribution,
                is_sorted=True, rng=np.random.default_rng(SEED),
                rank_spread=rank_spread,
            )
        self.assertEqual(len(captured.output), 1, msg=captured.output)
        match = re.search(
            r"insert allocation drift at percentile 50\.00: "
            r"unhostable_mass=(\d+\.\d+), spill=(\d+\.\d+) "
            r"\((\d+)/(\d+) keys redirected\), (\d+)/(\d+) gaps saturated",
            captured.output[0],
        )
        self.assertIsNotNone(match, msg=captured.output[0])
        return len(batch), {
            "unhostable_mass": float(match.group(1)),
            "spill": float(match.group(2)),
            "redirected": int(match.group(3)),
            "batch": int(match.group(4)),
            "saturated": int(match.group(5)),
            "gaps": int(match.group(6)),
        }

    def test_forced_allocation_reports_full_spill_and_unhostable_mass(self):
        # Batch == window capacity: the allocation is forced, the kernel is
        # never consulted, so the whole batch counts as redirected and
        # only the two free blocks stay hostable out of the window.
        batch, drift = self._drift_record(
            self._two_free_blocks_off_pivot(), 20, ZipfConfig(), rank_spread=0.35
        )
        self.assertEqual(batch, 20)
        self.assertEqual(drift["batch"], 20)
        self.assertEqual(drift["redirected"], 20)
        self.assertEqual(drift["spill"], 1.0)
        self.assertGreater(drift["unhostable_mass"], 0.5)
        self.assertEqual(drift["saturated"], 2)
        self.assertEqual(drift["gaps"], 2)

    def test_hot_gap_saturation_reports_fractional_spill(self):
        # A 2-wide gap at the pivot (capacity 1) and an 11-wide gap at the
        # window edge: the gaussian concentrates the draws on the pivot
        # gap, whose overflow must spill to the roomy gap. The spill is
        # fractional (batch < total capacity, so not forced) and the pivot
        # gap ends saturated at its single slot.
        keys = np.arange(1000, 1401, dtype=np.uint64)
        keys[200:] += 1   # pivot position: 2-wide gap, capacity 1
        keys[210:] += 10  # window edge: 11-wide gap, the only spill target
        batch, drift = self._drift_record(keys, 4, GaussianConfig())
        self.assertEqual(batch, 4)
        self.assertEqual(drift["batch"], 4)
        self.assertGreater(drift["unhostable_mass"], 0.5)
        self.assertGreater(drift["spill"], 0.0)
        self.assertLess(drift["spill"], 1.0)
        self.assertEqual(drift["saturated"], 1)
        self.assertEqual(drift["gaps"], 2)

    def test_fully_hostable_undersubscribed_window_reports_zero_drift(self):
        keys = np.arange(0, 2000, 10, dtype=np.uint64)
        for distribution in (UniformConfig(), ZipfConfig()):
            with self.subTest(distribution=type(distribution).__name__):
                batch, drift = self._drift_record(keys, 5, distribution)
                self.assertEqual(batch, 5)
                self.assertEqual(drift["batch"], 5)
                self.assertEqual(drift["unhostable_mass"], 0.0)
                self.assertEqual(drift["spill"], 0.0)
                self.assertEqual(drift["redirected"], 0)
                self.assertEqual(drift["saturated"], 0)
                self.assertEqual(drift["gaps"], 11)


class TestDegenerateInputs(unittest.TestCase):
    def test_non_positive_count_returns_empty_typed_streams(self):
        keys = fixtures.uniform_sparse()
        for count in (0, -3):
            with self.subTest(count=count):
                rng = np.random.default_rng(SEED)

                indices = sample_existing_indices(
                    len(keys), count, 50.0, ZipfConfig(), True, rng
                )
                existing = sample_existing_keys(keys, count, 50.0, ZipfConfig(), True, rng)
                batches = [
                    sample_insert_keys(keys, count, p, ZipfConfig(), True, rng)
                    for p in (0.0, 50.0, 100.0)
                ]
                generated = sample_new_keys(keys, count, 50.0, ZipfConfig(), rng)
                drained, refilled = sample_update_pairs(
                    keys, count, 50.0, ZipfConfig(), True, rng
                )
                reads = sample_point_queries(
                    keys, count, 50.0, distribution=ZipfConfig(), rng=rng
                )

                self.assertEqual(indices.dtype, np.int64)
                self.assertEqual(indices.size, 0)
                self.assertEqual(existing.size, 0)
                for batch in batches:
                    self.assertEqual(batch.dtype, np.uint64)
                    self.assertEqual(batch.size, 0)
                self.assertEqual(generated.size, 0)
                self.assertEqual(drained.size, 0)
                self.assertEqual(refilled.size, 0)
                self.assertEqual(reads.size, 0)

    def test_output_dtypes_are_uint64_keys_and_int64_indices(self):
        active_keys = np.arange(1_000, 101_000, 10, dtype=np.uint64)
        rng = np.random.default_rng(SEED)

        results = [
            sample_insert_keys(active_keys, COUNT_SMALL, 50.0, ZipfConfig(), True, rng),
            sample_insert_keys(active_keys, COUNT_SMALL, 0.0, ZipfConfig(), True, rng),
            sample_insert_keys(active_keys, COUNT_SMALL, 100.0, ZipfConfig(), True, rng),
            sample_existing_keys(active_keys, COUNT_SMALL, 50.0, ZipfConfig(), True, rng),
            sample_new_keys(active_keys, COUNT_SMALL, 50.0, ZipfConfig(), rng),
            sample_update_pairs(active_keys, COUNT_SMALL, 50.0, ZipfConfig(), True, rng)[0],
            sample_update_pairs(active_keys, COUNT_SMALL, 50.0, ZipfConfig(), True, rng)[1],
            sample_point_queries(active_keys, COUNT_SMALL, 50.0, distribution=ZipfConfig(), rng=rng),
        ]
        for result in results:
            self.assertEqual(result.dtype, np.uint64)

        indices = sample_existing_indices(
            len(active_keys), COUNT_SMALL, 50.0, ZipfConfig(), True, rng
        )
        self.assertEqual(indices.dtype, np.int64)


class TestApiContract(unittest.TestCase):
    def test_custom_zipf_theta_is_accepted_by_every_public_function(self):
        active_keys = np.arange(0, 200_000, 20, dtype=np.uint64)

        for theta in (0.5, 2.5):
            with self.subTest(theta=theta):
                rng = np.random.default_rng(SEED)
                results = [
                    sample_existing_indices(
                        1_000, COUNT_SMALL, 50.0, ZipfConfig(theta), True, rng
                    ),
                    sample_existing_keys(
                        active_keys, COUNT_SMALL, 50.0, ZipfConfig(theta), True, rng
                    ),
                    sample_insert_keys(
                        active_keys, COUNT_SMALL, 50.0, ZipfConfig(theta), True, rng
                    ),
                    sample_new_keys(active_keys, COUNT_SMALL, 50.0, ZipfConfig(theta), rng),
                    sample_update_pairs(
                        active_keys, COUNT_SMALL, 50.0, ZipfConfig(theta), True, rng
                    )[0],
                    sample_update_pairs(
                        active_keys, COUNT_SMALL, 50.0, ZipfConfig(theta), True, rng
                    )[1],
                    sample_point_queries(
                        active_keys, COUNT_SMALL, 50.0,
                        distribution=ZipfConfig(theta), rng=rng,
                    ),
                ]
                for result in results:
                    self.assertEqual(len(result), COUNT_SMALL)

    def test_signatures_and_constants_match_the_documented_contract(self):
        distribution_functions = [
            sample_existing_indices,
            sample_existing_keys,
            sample_insert_keys,
            sample_new_keys,
            sample_update_pairs,
            sample_point_queries,
        ]

        for function in distribution_functions:
            params = inspect.signature(function).parameters
            self.assertIn("distribution", params, msg=function.__name__)
            self.assertNotIn("theta", params, msg=function.__name__)

        spread_functions = [
            sample_insert_keys,
            sample_new_keys,
            sample_existing_indices,
            sample_existing_keys,
            sample_point_queries,
            sample_knn_query_points,
            sample_update_pairs,
        ]
        for function in spread_functions:
            self.assertEqual(
                inspect.signature(function).parameters["rank_spread"].default,
                RANK_SPREAD_RATIO,
                msg=function.__name__,
            )
            self.assertNotIn(
                "max_retries", inspect.signature(function).parameters, msg=function.__name__
            )

        self.assertEqual(ZipfConfig().theta, DEFAULT_THETA)
        self.assertEqual(DEFAULT_THETA, 1.2)
        self.assertEqual(RANK_SPREAD_RATIO, 0.025)


class TestDeterminism(unittest.TestCase):
    def test_same_seed_reproduces_every_output_exactly(self):
        def run_schedule():
            rng = np.random.default_rng(123)
            keys = fixtures.gaussian()
            return [
                sample_insert_keys(keys, 50, 50.0, GaussianConfig(), True, rng),
                sample_existing_keys(keys, 50, 50.0, ZipfConfig(), True, rng),
                sample_point_queries(keys, 1_000, 50.0, distribution=ZipfConfig(), rng=rng),
                sample_update_pairs(keys, 50, 50.0, GaussianConfig(), True, rng)[0],
                sample_update_pairs(keys, 50, 50.0, GaussianConfig(), True, rng)[1],
                sample_insert_keys(keys, 50, 100.0, ZipfConfig(), True, rng),
                sample_insert_keys(keys, 50, 0.0, ZipfConfig(), True, rng),
            ]

        first_run = run_schedule()
        second_run = run_schedule()

        for first, second in zip(first_run, second_run):
            np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main()
