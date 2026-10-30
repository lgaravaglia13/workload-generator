"""In-domain key sampler tests for deduplication, saturation, and capacity allocation."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fixtures
import numpy as np

from distributions import UniformConfig, ZipfConfig
from sampler import RANK_SPREAD_RATIO, sample_insert_keys, sample_new_keys

UINT64_MAX = int(np.iinfo(np.uint64).max)

SEED = 42

BAND_SIZE = 2_000
FREE_SLOTS = np.arange(995, 1_005, dtype=np.uint64)
VACATED_SLOTS = np.arange(1_005, 1_025, dtype=np.uint64)


def dense_band_with_free_slots() -> np.ndarray:
    """Dense band whose only hostable gaps lie inside the rank window."""
    return np.setdiff1d(np.arange(BAND_SIZE, dtype=np.uint64), FREE_SLOTS)


class TestSamplerDeduplication(unittest.TestCase):
    def test_zipf_fill_collects_every_free_slot(self):
        active = dense_band_with_free_slots()

        keys = sample_insert_keys(
            active, len(FREE_SLOTS), 50.0, ZipfConfig(), True,
            rng=np.random.default_rng(SEED),
        )

        self.assertEqual(len(keys), len(FREE_SLOTS))
        self.assertEqual(set(int(k) for k in keys), set(int(k) for k in FREE_SLOTS))

    def test_truncation_returns_exact_count_when_free_slots_abound(self):
        active = dense_band_with_free_slots()
        requested = 4

        keys = sample_insert_keys(
            active, requested, 50.0, ZipfConfig(), False,
            rng=np.random.default_rng(SEED),
        )

        self.assertEqual(len(keys), requested)
        self.assertEqual(len(np.unique(keys)), requested)
        self.assertTrue(set(int(k) for k in keys).issubset(set(int(k) for k in FREE_SLOTS)))

    def test_keys_collected_across_gaps_are_globally_distinct(self):
        active = dense_band_with_free_slots()
        count = 25

        drained = np.setdiff1d(active, VACATED_SLOTS)
        keys = sample_new_keys(
            drained, count, 50.0, ZipfConfig(),
            rng=np.random.default_rng(SEED),
        )

        self.assertEqual(len(keys), count)
        self.assertEqual(len(np.unique(keys)), count)
        legal = set(int(k) for k in FREE_SLOTS) | set(int(k) for k in VACATED_SLOTS)
        self.assertTrue(set(int(k) for k in keys).issubset(legal))

    def test_fully_dense_band_saturates_with_exact_message(self):
        active = np.arange(BAND_SIZE, dtype=np.uint64)

        with self.assertRaises(RuntimeError) as ctx:
            sample_insert_keys(
                active, 3, 50.0, ZipfConfig(), True,
                rng=np.random.default_rng(SEED),
            )

        self.assertEqual(
            str(ctx.exception),
            "Domain saturated: window holds 0 free keys, need 3.",
        )

    def test_single_key_batch_deduplicates_to_one(self):
        active = dense_band_with_free_slots()

        keys = sample_insert_keys(
            active, 1, 50.0, ZipfConfig(), True, rng=np.random.default_rng(SEED)
        )

        self.assertEqual(len(keys), 1)
        self.assertIn(int(keys[0]), set(int(k) for k in FREE_SLOTS))

    def test_zero_count_returns_empty_array(self):
        active = dense_band_with_free_slots()

        keys = sample_insert_keys(
            active, 0, 50.0, ZipfConfig(), True, rng=np.random.default_rng(SEED)
        )

        self.assertEqual(keys.size, 0)
        self.assertEqual(keys.dtype, np.uint64)


class TestTwoStageAllocation(unittest.TestCase):
    def test_walkthrough_window_allocates_and_spills_exactly(self):
        mini = np.array([100, 101, 104, 106, 117, 118], dtype=np.uint64)

        keys = sample_insert_keys(
            mini, 5, 50.0, ZipfConfig(), True,
            rng=np.random.default_rng(SEED), rank_spread=0.5,
        )

        self.assertEqual(len(keys), 5)
        self.assertEqual(len(np.unique(keys)), 5)
        legal = {102, 103, 105} | set(range(107, 117))
        self.assertTrue(set(int(k) for k in keys).issubset(legal))
        self.assertLessEqual(int((keys == np.uint64(105)).sum()), 1)

    def test_forced_fill_takes_every_free_slot(self):
        keys = np.array([0, 2, 4], dtype=np.uint64)

        batch = sample_insert_keys(
            keys, 2, 50.0, UniformConfig(), True,
            rng=np.random.default_rng(SEED), rank_spread=0.9,
        )

        self.assertEqual(sorted(int(k) for k in batch), [1, 3])

    def test_multi_key_gap_draws_distinct_slots(self):
        keys = np.array([0, 1000], dtype=np.uint64)

        batch = sample_insert_keys(
            keys, 10, 50.0, UniformConfig(), True,
            rng=np.random.default_rng(SEED), rank_spread=0.9,
        )

        self.assertEqual(len(batch), 10)
        self.assertEqual(len(np.unique(batch)), 10)
        self.assertTrue(np.all(batch > np.uint64(0)))
        self.assertTrue(np.all(batch < np.uint64(1000)))


class TestWindowGeometryEdgeCases(unittest.TestCase):
    def test_edge_percentile_weights_align_with_the_pivot(self):
        keys = np.arange(0, 200_000, 50, dtype=np.uint64)
        pivot = int(np.floor((len(keys) - 1) * 0.01))
        self.assertEqual(pivot, 39)

        batch = sample_insert_keys(
            keys, 300, 1.0, ZipfConfig(), True, rng=np.random.default_rng(SEED)
        )
        ranks = np.searchsorted(keys, batch)

        at_pivot = int((ranks == pivot).sum())
        at_next = int((ranks == pivot + 1).sum())
        at_prev = int((ranks == pivot - 1).sum())
        self.assertGreater(at_pivot, at_next)
        self.assertGreater(at_pivot, at_prev)

    def test_gap_wider_than_int64_stays_drawable(self):
        wide = np.array([0, 10, UINT64_MAX], dtype=np.uint64)

        batch = sample_insert_keys(
            wide, 5, 50.0, ZipfConfig(), True,
            rng=np.random.default_rng(SEED), rank_spread=0.9,
        )

        self.assertEqual(len(batch), 5)
        self.assertEqual(len(np.unique(batch)), 5)
        self.assertTrue(np.all(batch > wide[0]))
        self.assertTrue(np.all(batch < wide[-1]))

    def test_semi_dense_window_is_not_choked_by_the_span_cap(self):
        gaps = [2] * 10 + [20] * 4 + [2] * 16
        keys = (100 + np.concatenate([[0], np.cumsum(gaps)])).astype(np.uint64)

        batch = sample_insert_keys(
            keys, 40, 50.0, ZipfConfig(), True,
            rng=np.random.default_rng(SEED), rank_spread=0.35,
        )

        self.assertEqual(len(batch), 40)
        self.assertEqual(len(np.unique(batch)), 40)
        rank = np.searchsorted(keys, batch)
        left = keys[rank - 1].astype(np.int64)
        right = keys[rank].astype(np.int64)
        key = batch.astype(np.int64)
        self.assertTrue(np.all(np.minimum(key - left, right - key) <= 8))


class TestLocalPlacementCap(unittest.TestCase):
    def test_inter_cluster_gap_never_sprays(self):
        keys = fixtures.clustered_two_hot()

        batch = sample_insert_keys(
            keys, 100, 50.0, ZipfConfig(), True,
            rng=np.random.default_rng(42),
        )

        in_gap = batch[(batch > np.uint64(1_050_000)) & (batch < np.uint64(9_000_000))]
        self.assertGreater(len(in_gap), 0)
        distance = np.minimum(
            in_gap.astype(np.int64) - 1_050_000, 9_000_000 - in_gap.astype(np.int64)
        )
        self.assertLess(int(distance.max()), 1000)

    def test_batch_granularity_does_not_change_the_reachable_span(self):
        keys = fixtures.clustered_two_hot()
        rng = np.random.default_rng(7)

        active = keys.copy()
        worst = 0
        for _ in range(10):
            batch = sample_insert_keys(active, 10, 50.0, ZipfConfig(), True, rng=rng)
            in_gap = batch[
                (batch > np.uint64(1_050_001)) & (batch < np.uint64(8_999_997))
            ]
            if len(in_gap):
                distance = np.minimum(
                    in_gap.astype(np.int64) - 1_050_000,
                    9_000_000 - in_gap.astype(np.int64),
                )
                worst = max(worst, int(distance.max()))
            active = np.sort(np.union1d(active, batch))

        self.assertLess(worst, 1000)

    def test_regular_gaps_keep_their_full_extent(self):
        keys = fixtures.uniform_sparse()

        batch = sample_insert_keys(
            keys, 200, 50.0, UniformConfig(), True,
            rng=np.random.default_rng(SEED),
        )

        ranks = np.searchsorted(keys, batch)
        pivot = int(np.clip(np.floor((len(keys) - 1) * 0.5), 0, len(keys) - 1))
        radius = max(1, int(len(keys) * RANK_SPREAD_RATIO))
        self.assertTrue(np.all(np.abs(ranks - pivot) <= radius + 1))


if __name__ == "__main__":
    unittest.main()
