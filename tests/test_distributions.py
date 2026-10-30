"""Contract tests for distribution kernels."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fixtures 
import numpy as np

from distributions import (
    DEFAULT_THETA,
    GaussianConfig,
    UniformConfig,
    ZipfConfig,
    _draw_powerlaw_distances,
    draw_offsets,
    draw_steps,
    from_token,
)

SEED = 42


class TestFromToken(unittest.TestCase):
    def test_valid_tokens_map_to_default_configs(self):
        cases = [
            ("g", GaussianConfig),
            ("gaussian", GaussianConfig),
            ("z", ZipfConfig),
            ("zipf", ZipfConfig),
            ("r", UniformConfig),
            ("random", UniformConfig),
            ("uniform", UniformConfig),
        ]
        for token, expected in cases:
            with self.subTest(token=token):
                self.assertIsInstance(from_token(token), expected)

    def test_tokens_are_case_insensitive(self):
        for token in ("G", "Zipf", "RANDOM"):
            with self.subTest(token=token):
                self.assertIsInstance(from_token(token), (GaussianConfig, ZipfConfig, UniformConfig))

    def test_unknown_token_raises_value_error(self):
        with self.assertRaises(ValueError):
            from_token("exponential")


class TestDrawOffsets(unittest.TestCase):
    def test_gaussian_offsets_are_continuous_with_radius_over_three_scale(self):
        rng = np.random.default_rng(SEED)
        radius = 300.0

        offsets = draw_offsets(20_000, radius, GaussianConfig(), rng)

        self.assertEqual(offsets.dtype, np.float64)
        self.assertLess(abs(float(offsets.mean())), 5.0)
        self.assertAlmostEqual(float(offsets.std()), radius / 3.0, delta=5.0)

    def test_zipf_offsets_are_signed_integers_with_dominant_unit_distance(self):
        rng = np.random.default_rng(SEED)
        radius = 500.0

        offsets = draw_offsets(20_000, radius, ZipfConfig(), rng)

        self.assertEqual(offsets.dtype, np.int64)
        self.assertTrue(np.all(np.abs(offsets) >= 1))
        self.assertTrue(np.all(np.abs(offsets) <= int(radius)))
        self.assertEqual(int(np.bincount(np.abs(offsets)).argmax()), 1)
        self.assertGreater(int((offsets > 0).sum()), 0)
        self.assertGreater(int((offsets < 0).sum()), 0)

    def test_uniform_offsets_are_bounded_by_radius(self):
        rng = np.random.default_rng(SEED)
        radius = 250.0

        offsets = draw_offsets(20_000, radius, UniformConfig(), rng)

        self.assertEqual(offsets.dtype, np.float64)
        self.assertTrue(np.all(np.abs(offsets) <= radius))

    def test_unsupported_config_raises_type_error(self):
        rng = np.random.default_rng(SEED)
        with self.assertRaises(TypeError):
            draw_offsets(1, 100.0, object(), rng)


class TestDrawSteps(unittest.TestCase):
    def test_all_kernels_return_positive_integers(self):
        rng = np.random.default_rng(SEED)
        for config in (GaussianConfig(), ZipfConfig(), UniformConfig()):
            with self.subTest(config=type(config).__name__):
                steps = draw_steps(1_000, 500.0, config, rng)
                self.assertEqual(steps.dtype, np.int64)
                self.assertTrue(np.all(steps >= 1))

    def test_zipf_and_uniform_steps_stay_below_floor_of_radius(self):
        rng = np.random.default_rng(SEED)
        radius = 50.0
        for config in (ZipfConfig(), UniformConfig()):
            with self.subTest(config=type(config).__name__):
                steps = draw_steps(100_000, radius, config, rng)
                self.assertTrue(np.all(steps >= 1))
                self.assertTrue(np.all(steps <= int(radius)))
                self.assertEqual(int(steps.max()), int(radius))

    def test_unit_radius_collapses_to_unit_steps(self):
        rng = np.random.default_rng(SEED)
        for config in (ZipfConfig(), UniformConfig()):
            with self.subTest(config=type(config).__name__):
                steps = draw_steps(100, 1.0, config, rng)
                np.testing.assert_array_equal(steps, np.ones(100, dtype=np.int64))

    def test_gaussian_steps_never_return_zero(self):
        rng = np.random.default_rng(SEED)
        steps = draw_steps(100_000, 3.0, GaussianConfig(), rng)
        self.assertTrue(np.all(steps >= 1))

    def test_unsupported_config_raises_type_error(self):
        rng = np.random.default_rng(SEED)
        with self.assertRaises(TypeError):
            draw_steps(1, 100.0, object(), rng)


class TestConfigDefaults(unittest.TestCase):
    def test_zipf_default_theta_matches_module_constant(self):
        self.assertEqual(ZipfConfig().theta, DEFAULT_THETA)


class TestLargeSupportPowerLaw(unittest.TestCase):
    """Tests continuous power-law inversion for large supports (D_MAX > 2^22)."""

    D_MAX = 10**12

    def test_bounded_in_support_and_reproducible(self):
        rng = np.random.default_rng(3)
        draws = _draw_powerlaw_distances(1_000, self.D_MAX, 1.2, rng)
        self.assertTrue(bool(np.all((draws >= 1) & (draws <= self.D_MAX))))
        again = _draw_powerlaw_distances(1_000, self.D_MAX, 1.2, np.random.default_rng(3))
        self.assertTrue(np.array_equal(draws, again))

    def test_higher_theta_concentrates_mass_near_one(self):
        near = _draw_powerlaw_distances(1_000, self.D_MAX, 2.0, np.random.default_rng(5))
        far = _draw_powerlaw_distances(1_000, self.D_MAX, 0.8, np.random.default_rng(5))
        self.assertLess(np.median(near), np.median(far))

    def test_theta_one_uses_the_logarithmic_inversion(self):
        draws = _draw_powerlaw_distances(2_000, self.D_MAX, 1.0, np.random.default_rng(11))
        self.assertTrue(bool(np.all((draws >= 1) & (draws <= self.D_MAX))))
        self.assertGreater(float(np.median(draws)), 1_000)
        self.assertLess(float(np.median(draws)), 10_000_000)
        again = _draw_powerlaw_distances(2_000, self.D_MAX, 1.0, np.random.default_rng(11))
        self.assertTrue(np.array_equal(draws, again))


if __name__ == "__main__":
    unittest.main()
