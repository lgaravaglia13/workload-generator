"""Integration tests driving active-key state across phases."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fixtures
import numpy as np

from distributions import GaussianConfig, ZipfConfig
from sampler import (
    sample_existing_keys,
    sample_insert_keys,
    sample_point_queries,
    sample_update_pairs,
)

SEED_WRITE_READ = 11
SEED_EVOLUTION = 22
SEED_CARDINALITY = 33
SEED_PHASE_MIX = 44
SEED_SATURATION = 55

INSERT_COUNT = 200
DELETE_COUNT = 50
UPDATE_COUNT = 50
READ_COUNT = 1_000
BOUNDARY_COUNT = 100
TAIL_INSERT_COUNT = 50
PREPEND_COUNT = 50

UPDATE_ITERATIONS = 20

SATURATION_DELETE_COUNT = 100
SATURATION_INSERT_COUNT = 200

INDOMAIN_FIXTURES = (fixtures.uniform_sparse, fixtures.zipfian, fixtures.clustered_two_hot)
PREPEND_SAFE_FIXTURES = (fixtures.gaussian, fixtures.clustered_two_hot)


class TestPhaseSequences(unittest.TestCase):
    def test_write_phase_then_read_phase_against_the_updated_state(self):
        for keys_fn in INDOMAIN_FIXTURES:
            with self.subTest(fixture=keys_fn.__name__):
                base_keys = keys_fn()
                initial_size = len(base_keys)
                rng = np.random.default_rng(SEED_WRITE_READ)

                inserted = sample_insert_keys(
                    base_keys, INSERT_COUNT, 50.0, GaussianConfig(),
                    is_sorted=True, rng=rng,
                )
                active_keys = np.union1d(base_keys, inserted)

                self.assertEqual(len(active_keys), initial_size + INSERT_COUNT)
                self.assertTrue(fixtures.strictly_sorted(active_keys))

                reads = sample_point_queries(
                    active_keys, READ_COUNT, 50.0,
                    distribution=ZipfConfig(), rng=rng,
                )

                self.assertEqual(len(reads), READ_COUNT)
                self.assertTrue(np.all(np.isin(reads, active_keys)))

    def test_state_evolution_across_insert_delete_update(self):
        for keys_fn in INDOMAIN_FIXTURES:
            with self.subTest(fixture=keys_fn.__name__):
                base_keys = keys_fn()
                initial_size = len(base_keys)
                rng = np.random.default_rng(SEED_EVOLUTION)
                active_keys = base_keys

                inserted = sample_insert_keys(
                    active_keys, 100, 50.0, GaussianConfig(), is_sorted=True, rng=rng
                )
                active_keys = np.union1d(active_keys, inserted)
                self.assertEqual(len(active_keys), initial_size + 100)

                deleted = sample_existing_keys(
                    active_keys, DELETE_COUNT, 30.0, ZipfConfig(), is_sorted=True, rng=rng
                )
                active_keys = np.setdiff1d(active_keys, deleted)
                self.assertEqual(len(active_keys), initial_size + 100 - DELETE_COUNT)
                self.assertFalse(np.any(np.isin(deleted, active_keys)))

                old_keys, new_keys = sample_update_pairs(
                    active_keys, UPDATE_COUNT, 70.0, GaussianConfig(),
                    is_sorted=True, rng=rng,
                )
                survivors = np.setdiff1d(active_keys, old_keys)
                self.assertTrue(np.all(np.isin(old_keys, active_keys)))
                self.assertFalse(np.any(np.isin(new_keys, survivors)))
                active_keys = np.union1d(survivors, new_keys)

                self.assertEqual(len(active_keys), initial_size + 100 - DELETE_COUNT)
                self.assertTrue(fixtures.strictly_sorted(active_keys))

    def test_updates_preserve_cardinality_and_extremes_across_phases(self):
        for keys_fn in (fixtures.zipfian, fixtures.gaussian):
            with self.subTest(fixture=keys_fn.__name__):
                active_keys = keys_fn()
                initial_size = len(active_keys)
                initial_k_min = int(active_keys[0])
                initial_k_max = int(active_keys[-1])
                rng = np.random.default_rng(SEED_CARDINALITY)

                for _ in range(UPDATE_ITERATIONS):
                    old_keys, new_keys = sample_update_pairs(
                        active_keys, UPDATE_COUNT, 50.0, GaussianConfig(),
                        is_sorted=True, rng=rng,
                    )
                    self.assertEqual(len(old_keys), UPDATE_COUNT)
                    self.assertEqual(len(new_keys), UPDATE_COUNT)
                    active_keys = np.union1d(np.setdiff1d(active_keys, old_keys), new_keys)

                    self.assertEqual(len(active_keys), initial_size)
                    self.assertTrue(fixtures.strictly_sorted(active_keys))

                self.assertEqual(int(active_keys[0]), initial_k_min)
                self.assertEqual(int(active_keys[-1]), initial_k_max)

    def test_boundary_to_indomain_phase_mix(self):
        for keys_fn in PREPEND_SAFE_FIXTURES:
            with self.subTest(fixture=keys_fn.__name__):
                base_keys = keys_fn()
                initial_size = len(base_keys)
                original_k_min = int(base_keys[0])
                original_k_max = int(base_keys[-1])
                rng = np.random.default_rng(SEED_PHASE_MIX)
                active_keys = base_keys

                appended = sample_insert_keys(
                    active_keys, BOUNDARY_COUNT, 100.0, ZipfConfig(),
                    is_sorted=True, rng=rng,
                )
                self.assertTrue(np.all(appended > np.uint64(original_k_max)))
                active_keys = np.union1d(active_keys, appended)

                tail_inserted = sample_insert_keys(
                    active_keys, TAIL_INSERT_COUNT, 99.0, ZipfConfig(),
                    is_sorted=True, rng=rng,
                )
                self.assertTrue(np.all(tail_inserted >= active_keys[0]))
                self.assertTrue(np.all(tail_inserted <= active_keys[-1]))
                self.assertFalse(np.any(np.isin(tail_inserted, active_keys)))
                active_keys = np.union1d(active_keys, tail_inserted)

                prepended = sample_insert_keys(
                    active_keys, PREPEND_COUNT, 0.0, ZipfConfig(),
                    is_sorted=True, rng=rng,
                )
                self.assertTrue(np.all(prepended < np.uint64(original_k_min)))
                active_keys = np.union1d(active_keys, prepended)

                self.assertEqual(
                    len(active_keys),
                    initial_size + BOUNDARY_COUNT + TAIL_INSERT_COUNT + PREPEND_COUNT,
                )
                self.assertTrue(fixtures.strictly_sorted(active_keys))

    def test_saturated_dense_schedule_fails_fast_at_the_saturated_phase(self):
        uniform_dense = fixtures.uniform_dense()
        rng = np.random.default_rng(SEED_SATURATION)

        drained = sample_existing_keys(
            uniform_dense, SATURATION_DELETE_COUNT, 50.0, ZipfConfig(),
            is_sorted=True, rng=rng,
        )
        active_after_delete = np.setdiff1d(uniform_dense, drained)

        self.assertEqual(len(active_after_delete), len(uniform_dense) - SATURATION_DELETE_COUNT)
        self.assertTrue(fixtures.strictly_sorted(active_after_delete))

        with self.assertRaises(RuntimeError) as ctx:
            sample_insert_keys(
                active_after_delete, SATURATION_INSERT_COUNT, 50.0, ZipfConfig(),
                is_sorted=True, rng=rng,
            )

        self.assertRegex(
            str(ctx.exception),
            rf"Domain saturated: window holds \d+ free keys, need {SATURATION_INSERT_COUNT}\.",
        )
        self.assertEqual(len(active_after_delete), len(uniform_dense) - SATURATION_DELETE_COUNT)

    def test_same_seed_reproduces_the_whole_schedule(self):
        gaussian_keys = fixtures.gaussian()

        def run_schedule():
            rng = np.random.default_rng(123)
            return {
                "insert": sample_insert_keys(
                    gaussian_keys, UPDATE_COUNT, 50.0, GaussianConfig(), is_sorted=True, rng=rng
                ),
                "existing": sample_existing_keys(
                    gaussian_keys, UPDATE_COUNT, 50.0, ZipfConfig(), is_sorted=True, rng=rng
                ),
                "point_reads": sample_point_queries(
                    gaussian_keys, READ_COUNT, 50.0, distribution=ZipfConfig(), rng=rng
                ),
                "update_old": sample_update_pairs(
                    gaussian_keys, UPDATE_COUNT, 50.0, GaussianConfig(), is_sorted=True, rng=rng
                )[0],
                "update_new": sample_update_pairs(
                    gaussian_keys, UPDATE_COUNT, 50.0, GaussianConfig(), is_sorted=True, rng=rng
                )[1],
                "append": sample_insert_keys(
                    gaussian_keys, UPDATE_COUNT, 100.0, ZipfConfig(), is_sorted=True, rng=rng
                ),
                "prepend": sample_insert_keys(
                    gaussian_keys, UPDATE_COUNT, 0.0, ZipfConfig(), is_sorted=True, rng=rng
                ),
            }

        first_run = run_schedule()
        second_run = run_schedule()

        for operation, first_result in first_run.items():
            np.testing.assert_array_equal(first_result, second_run[operation])


if __name__ == "__main__":
    unittest.main()
