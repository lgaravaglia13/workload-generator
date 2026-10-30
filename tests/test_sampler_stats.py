"""Statistical goodness-of-fit tests for the key sampling engine.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fixtures
import numpy as np
from scipy.stats import chi2

from distributions import GaussianConfig, ZipfConfig
from sampler import RANK_SPREAD_RATIO, sample_existing_indices, sample_insert_keys, sample_point_queries

# Fixed per-test seeds: every assertion is a deterministic realization.
SEED_ARES_ZIPF = 777
SEED_ARES_GAUSS = 1001
SEED_GAP_CHANNEL = 55
SEED_POINT_QUERIES = 31
SEED_THETA_SWEEP = 97

# Significance floor for every chi-square test.
PVALUE_FLOOR = 0.01
# Chi-square bins with expected count below this are pooled into one tail bin.
MIN_EXPECTED = 5.0

# A-Res first-selection protocol: W ~= 250 window, B = 50, 2000 batches.
ARES_N = 5_000
ARES_COUNT = 50
ARES_BATCHES = 2_000
ARES_PERCENTILE = 50.0

# Coarse distance bins for the pooled-inclusion monotonicity check.
INCLUSION_BIN_EDGES = [0, 1, 6, 16, 41, 101]

# Gaussian no-boundary-spike edge bound: positions at >= 80% of the rank
# radius form the "edge"; observed inclusions must stay within EDGE_SLACK of
# the first-order inclusion expectation sum(B * w_i / sum(w)). A clipping
# spike would inflate edge weights to the pivot weight and blow this bound.
EDGE_FRACTION = 0.8
EDGE_SLACK = 3.0

# Boundary gap channel: append batches on fixtures whose headroom
# (UINT64_MAX - k_max) is effectively unbounded, so the overflow guard can
# never condition the gap stream.
GAP_CHANNEL_FIXTURES = (fixtures.uniform_sparse, fixtures.zipfian, fixtures.gaussian)
GAP_THETAS = (0.5, 1.2, 2.5)
GAP_BATCHES = 1_000
GAP_COUNT = 50

# Point-query zipf frequencies: 10,000 reads over 1,000 active keys.
POINT_N = 1_000
POINT_READS = 10_000
POINT_PERCENTILE = 50.0

# Theta sweep: fraction of reads inside the near-pivot 1% of the rank radius.
SWEEP_N = 10_000
SWEEP_READS = 20_000
SWEEP_THETAS = (0.5, 1.2, 2.5)
NEAR_PIVOT_FRACTION = 0.01


def pooled_chi_square(observed: np.ndarray, expected: np.ndarray) -> tuple:
    """Chi-square statistic, df and p-value with sub-5 expected bins pooled
    into a single tail bin (expected vectors are non-increasing)."""
    small_bins = expected < MIN_EXPECTED
    if np.any(small_bins):
        cut = int(np.argmax(small_bins))
        observed = np.append(observed[:cut], observed[cut:].sum())
        expected = np.append(expected[:cut], expected[cut:].sum())
    statistic = float(np.sum((observed - expected) ** 2 / expected))
    degrees_of_freedom = len(observed) - 1
    return statistic, degrees_of_freedom, float(chi2.sf(statistic, degrees_of_freedom))


class TestAresFirstSelection(unittest.TestCase):
    def _collect_first_selections_and_inclusions(self, config):
        n = ARES_N
        pivot = int(np.clip(np.floor((n - 1) * ARES_PERCENTILE / 100.0), 0, n - 1))
        radius = max(1, int(n * RANK_SPREAD_RATIO))
        low = max(0, pivot - radius)
        high = min(n, pivot + radius + 1)
        distances = np.abs(np.arange(low, high) - pivot)
        if isinstance(config, GaussianConfig):
            sigma = max(1.0, radius / 3.0)
            weights = np.exp(-(distances.astype(np.float64) ** 2) / (2 * sigma**2))
        else:
            weights = (1.0 + distances) ** (-config.theta)
        weight_sum = float(weights.sum())

        rng = np.random.default_rng(SEED_ARES_ZIPF if isinstance(config, ZipfConfig) else SEED_ARES_GAUSS)
        first_distances = []
        inclusion_counts = np.zeros(high - low, dtype=np.int64)
        for _ in range(ARES_BATCHES):
            batch = sample_existing_indices(
                n, ARES_COUNT, ARES_PERCENTILE, config, is_sorted=False, rng=rng
            )
            self.assertEqual(len(batch), ARES_COUNT)
            first_distances.append(abs(int(batch[0]) - pivot))
            inclusion_counts[np.asarray(batch, dtype=np.int64) - low] += 1

        if isinstance(config, GaussianConfig):
            sigma = max(1.0, radius / 3.0)
            distance_weights = np.exp(-(np.arange(int(distances.max()) + 1) ** 2) / (2 * sigma**2))
        else:
            distance_weights = (1.0 + np.arange(int(distances.max()) + 1)) ** (-config.theta)
        return (
            np.asarray(first_distances),
            inclusion_counts,
            distances,
            weights,
            weight_sum,
            distance_weights,
            radius,
        )

    def _assert_first_selection_matches_kernel(self, config):
        (
            first_distances,
            _,
            _,
            _,
            weight_sum,
            distance_weights,
            _,
        ) = self._collect_first_selections_and_inclusions(config)

        max_distance = len(distance_weights) - 1
        # distance d pools the two symmetric window positions; d = 0 is single
        multiplicity = np.where(np.arange(max_distance + 1) == 0, 1.0, 2.0)
        observed = np.bincount(first_distances, minlength=max_distance + 1).astype(float)
        expected = ARES_BATCHES * multiplicity * distance_weights / weight_sum
        self.assertGreaterEqual(expected[0], MIN_EXPECTED)

        statistic, df, p_value = pooled_chi_square(observed, expected)
        self.assertGreater(
            p_value,
            PVALUE_FLOOR,
            msg=f"{type(config).__name__} first-selection chi-square p={p_value:.4f} "
            f"(chi2={statistic:.2f}, df={df}); result[0] must follow the "
            "kernel weights exactly",
        )

    def test_zipf_first_selections_follow_the_pl_weights_exactly(self):
        self._assert_first_selection_matches_kernel(ZipfConfig())

    def test_gaussian_first_selections_follow_the_kernel_weights_exactly(self):
        self._assert_first_selection_matches_kernel(GaussianConfig())

    def test_zipf_inclusion_frequency_is_non_increasing_in_distance(self):
        _, inclusion_counts, distances, _, _, _, _ = self._collect_first_selections_and_inclusions(
            ZipfConfig()
        )

        # Coarse bins: without replacement the exact inclusion frequency is
        # not B*w/sum(w), but it must remain monotone in the kernel weight.
        frequencies = []
        for i in range(len(INCLUSION_BIN_EDGES) - 1):
            bin_mask = (distances >= INCLUSION_BIN_EDGES[i]) & (
                distances < INCLUSION_BIN_EDGES[i + 1]
            )
            frequencies.append(float(inclusion_counts[bin_mask].sum()) / int(bin_mask.sum()))
        for i in range(len(frequencies) - 1):
            self.assertGreaterEqual(
                frequencies[i],
                frequencies[i + 1],
                msg=f"inclusion frequency must be non-increasing in distance: "
                f"bin {i} -> {frequencies[i]:.4f} vs bin {i + 1} -> {frequencies[i + 1]:.4f}",
            )

    def test_gaussian_window_edge_is_never_inflated(self):
        (
            _,
            inclusion_counts,
            distances,
            weights,
            weight_sum,
            _,
            _,
        ) = self._collect_first_selections_and_inclusions(GaussianConfig())

        # Edge positions (>= 80% of the rank radius) must not exceed a loose
        # multiple of their nominal first-order expectation: a clipping
        # implementation would assign them the maximum window weight.
        edge_mask = distances >= EDGE_FRACTION * max(1, int(ARES_N * RANK_SPREAD_RATIO))
        observed_edge = float(inclusion_counts[edge_mask].sum())
        expected_edge = ARES_BATCHES * float(np.sum(ARES_COUNT * weights[edge_mask] / weight_sum))
        self.assertLessEqual(
            observed_edge,
            EDGE_SLACK * expected_edge,
            msg=f"edge inclusions {observed_edge:.0f} exceed {EDGE_SLACK}x the nominal "
            f"expectation {expected_edge:.0f}: boundary clipping spike",
        )


class TestBoundaryGapChannel(unittest.TestCase):
    def test_append_gap_histograms_follow_the_powerlaw_exactly(self):
        for keys_fn in GAP_CHANNEL_FIXTURES:
            for theta in GAP_THETAS:
                with self.subTest(fixture=keys_fn.__name__, theta=theta):
                    keys = keys_fn()
                    n = len(keys)
                    radius = max(1, int(n * RANK_SPREAD_RATIO))
                    low_index = max(0, n - 1 - radius)
                    boundary_radius = max(
                        1.0, (int(keys[-1]) - int(keys[low_index])) / (n - 1 - low_index)
                    )
                    support_max = int(np.floor(boundary_radius))
                    self.assertGreaterEqual(support_max, 1)

                    rng = np.random.default_rng(SEED_GAP_CHANNEL)
                    gaps = []
                    for _ in range(GAP_BATCHES):
                        batch = sample_insert_keys(
                            keys, GAP_COUNT, 100.0, ZipfConfig(theta=theta),
                            is_sorted=True, rng=rng,
                        )
                        self.assertEqual(len(batch), GAP_COUNT)
                        batch_gaps = np.diff(batch.astype(np.int64))
                        self.assertTrue(np.all(batch_gaps >= 1))
                        self.assertTrue(np.all(batch_gaps <= support_max))
                        gaps.append(batch_gaps)
                    gaps = np.concatenate(gaps)

                    support = np.arange(1, support_max + 1)
                    probabilities = (1.0 + support) ** (-theta)
                    probabilities /= probabilities.sum()
                    observed = np.bincount(gaps, minlength=support_max + 1)[1:].astype(float)
                    expected = len(gaps) * probabilities
                    self.assertGreaterEqual(expected[0], MIN_EXPECTED)

                    statistic, df, p_value = pooled_chi_square(observed, expected)
                    self.assertGreater(
                        p_value,
                        PVALUE_FLOOR,
                        msg=f"theta={theta} gap-channel chi-square p={p_value:.4f} "
                        f"(chi2={statistic:.2f}, df={df}); boundary gaps must be "
                        f"i.i.d. draws from P(d) ~ (1+d)^-{theta} over [1, {support_max}]",
                    )


class TestPointQueryFrequencies(unittest.TestCase):
    def test_read_histogram_matches_the_normalized_zipf_weights(self):
        active_keys = np.arange(POINT_N, dtype=np.uint64)
        pivot = int(np.clip(np.floor((POINT_N - 1) * POINT_PERCENTILE / 100.0), 0, POINT_N - 1))
        radius = max(1, int(POINT_N * RANK_SPREAD_RATIO))
        low = max(0, pivot - radius)
        high = min(POINT_N, pivot + radius + 1)
        distances = np.abs(np.arange(low, high) - pivot)
        weights = (1.0 + distances) ** (-1.2)
        probabilities = weights / weights.sum()

        reads = sample_point_queries(
            active_keys, POINT_READS, POINT_PERCENTILE,
            distribution=ZipfConfig(), rng=np.random.default_rng(SEED_POINT_QUERIES),
        )

        self.assertEqual(len(reads), POINT_READS)
        observed = np.bincount(reads.astype(np.int64), minlength=POINT_N)[low:high].astype(float)
        expected = POINT_READS * probabilities
        self.assertGreaterEqual(expected[0], MIN_EXPECTED)

        statistic, df, p_value = pooled_chi_square(observed, expected)
        self.assertGreater(
            p_value,
            PVALUE_FLOOR,
            msg=f"point-query chi-square p={p_value:.4f} "
            f"(chi2={statistic:.2f}, df={df}); reads must follow the "
            "normalized zipf window weights exactly",
        )

        read_counts = np.bincount(reads.astype(np.int64), minlength=POINT_N)
        self.assertEqual(
            int(read_counts.argmax()),
            pivot,
            msg="the pivot key must be the hottest point-query target",
        )

    def test_larger_theta_concentrates_reads_strictly_tighter(self):
        active_keys = np.arange(SWEEP_N, dtype=np.uint64)
        percentile = 50.0
        pivot = int(np.clip(np.floor((SWEEP_N - 1) * percentile / 100.0), 0, SWEEP_N - 1))
        radius = max(1, int(SWEEP_N * RANK_SPREAD_RATIO))
        near_pivot_ranks = max(1, int(NEAR_PIVOT_FRACTION * radius))

        concentrations = []
        for theta in SWEEP_THETAS:
            reads = sample_point_queries(
                active_keys, SWEEP_READS, percentile,
                distribution=ZipfConfig(theta=theta),
                rng=np.random.default_rng(SEED_THETA_SWEEP),
            )
            self.assertEqual(len(reads), SWEEP_READS)
            near_pivot = np.abs(reads.astype(np.int64) - pivot) <= near_pivot_ranks
            concentrations.append(float(np.mean(near_pivot)))

        for i in range(len(concentrations) - 1):
            self.assertLess(
                concentrations[i],
                concentrations[i + 1],
                msg=f"near-pivot concentration must increase with theta: "
                f"theta={SWEEP_THETAS[i]} -> {concentrations[i]:.4f}, "
                f"theta={SWEEP_THETAS[i + 1]} -> {concentrations[i + 1]:.4f}",
            )


if __name__ == "__main__":
    unittest.main()
