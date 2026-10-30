import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fixtures
import numpy as np

from models import Phase, QueryOp, Workload, WriteOp
from dsl.parser import Lark_StandAlone
from dsl.transformer import WorkloadTransformer
from trace_generator import TraceGenerator, TraceBatch
from trace_output import save_trace


class TraceGeneratorTestCase(unittest.TestCase):
    def setUp(self):
        self.parser = Lark_StandAlone()
        self.transformer = WorkloadTransformer()
        self.base_keys = fixtures.trace_base_keys()

    def parse_text(self, text: str) -> Workload:
        return self.transformer.transform(self.parser.parse(text))

    def generate(self, text: str):
        workload = self.parse_text(text)
        generator = TraceGenerator(keys=self.base_keys, seed=42)
        return generator, list(generator.generate_trace(workload))

    def key_span(self) -> float:
        return float(self.base_keys.max() - self.base_keys.min()) or 1.0

    @staticmethod
    def keys_of(trace, op_type):
        return [
            int(k)
            for group in trace
            for b in group
            if b.op_type == op_type
            for k in b.keys.tolist()
        ]

    @staticmethod
    def row_count(trace) -> int:
        return sum(len(b.keys) for group in trace for b in group)


class TestInsertTrace(TraceGeneratorTestCase):
    def test_batches_center_around_the_target_percentile(self):
        for percentile in (10, 50, 90):
            with self.subTest(percentile=percentile):
                _, trace = self.generate(f"P1: I:50,{percentile},g,s")
                inserted_keys = self.keys_of(trace, "insert")

                expected_pivot = float(np.percentile(self.base_keys, percentile))
                rel_error = abs(float(np.median(inserted_keys)) - expected_pivot) / self.key_span()
                self.assertLessEqual(rel_error, 0.10)

    def test_gaussian_fidelity_against_uniform(self):
        _, trace_g = self.generate("P1: I:500,50,g,u")
        _, trace_u = self.generate("P1: I:500,50,r,u")
        keys_g = np.array(self.keys_of(trace_g, "insert"), dtype=np.float64)
        keys_u = np.array(self.keys_of(trace_u, "insert"), dtype=np.float64)

        pivot = float(np.percentile(self.base_keys, 50))
        span = self.key_span()

        self.assertLess(abs(float(np.mean(keys_g)) - pivot) / span, 0.015)
        self.assertLess(abs(float(np.median(keys_g)) - pivot) / span, 0.015)

        self.assertLess(float(np.std(keys_g)), float(np.std(keys_u)))
        self.assertLess(float(np.std(keys_g)) / span, 0.03)

    def test_zipfian_hotspot_centers_its_median_on_the_pivot(self):
        _, trace = self.generate("P1: I:200,50,z,u")
        keys_z = np.array(self.keys_of(trace, "insert"), dtype=np.float64)

        pivot = float(np.percentile(self.base_keys, 50))
        rel_error = abs(float(np.median(keys_z)) - pivot) / self.key_span()
        self.assertLessEqual(rel_error, 0.02)

    def test_sorted_flag_controls_emission_order(self):
        _, trace_s = self.generate("P1: I:60,50,g,s")
        _, trace_u = self.generate("P1: I:60,50,g,u")

        self.assertEqual(self.keys_of(trace_s, "insert"), sorted(self.keys_of(trace_s, "insert")))
        self.assertNotEqual(self.keys_of(trace_u, "insert"), sorted(self.keys_of(trace_u, "insert")))

    def test_keys_are_unique_and_never_collide_with_the_active_set(self):
        generator, trace = self.generate("P1: I:100,50,z,s")
        initial_keys_set = set(int(k) for k in generator.initial_keys)
        inserted_keys = self.keys_of(trace, "insert")

        self.assertEqual(len(inserted_keys), 100)
        self.assertEqual(len(set(inserted_keys)), 100)
        self.assertTrue(set(inserted_keys).isdisjoint(initial_keys_set))
        self.assertTrue(fixtures.strictly_sorted(generator.active_keys))

    def test_multi_phase_inserts_maintain_global_uniqueness(self):
        generator, trace = self.generate("P1: I:50,50,z,s | P2: I:50,50,z,u | P3: I:50,50,g,s")
        all_inserted = self.keys_of(trace, "insert")

        self.assertEqual(len(all_inserted), 150)
        self.assertEqual(len(set(all_inserted)), 150)
        self.assertEqual(len(generator.active_keys), 1000 + 150)
        self.assertEqual(len(set(int(k) for k in generator.active_keys)), 1000 + 150)
        self.assertTrue(fixtures.strictly_sorted(generator.active_keys))

    def test_local_window_slices_maintain_state_monotonicity(self):
        generator, _ = self.generate(
            "P1: I:30,10,g,s | P2: I:40,80,z,u | P3: I:25,20,r,s | P4: I:35,50,g,u"
        )

        self.assertTrue(fixtures.strictly_sorted(generator.active_keys))
        self.assertEqual(len(generator.active_keys), 1000 + 30 + 40 + 25 + 35)

    def test_inserts_work_in_the_upper_uint64_key_space(self):
        base_offset = 1_000_000_000_000
        large_keys = np.sort(
            np.uint64(base_offset)
            + np.random.default_rng(7).choice(
                np.arange(0, 50_000_000, dtype=np.uint64), size=1000, replace=False
            )
        )

        workload = Workload(phases=[Phase(phase_id=1, operations=[
            WriteOp("insert", batch_size=40, percentile=10, distribution="g", sorted=True),
            WriteOp("insert", batch_size=40, percentile=50, distribution="z", sorted=False),
            WriteOp("insert", batch_size=40, percentile=90, distribution="r", sorted=True),
        ])])
        generator = TraceGenerator(keys=large_keys, seed=123)
        trace = list(generator.generate_trace(workload))
        inserted_keys = self.keys_of(trace, "insert")

        self.assertEqual(len(inserted_keys), 120)
        self.assertEqual(len(set(inserted_keys)), 120)
        self.assertTrue(set(inserted_keys).isdisjoint(set(int(k) for k in large_keys)))
        for key in inserted_keys:
            self.assertGreaterEqual(key, int(large_keys[0]))
            self.assertLessEqual(key, int(large_keys[-1]))
        self.assertEqual(generator.active_keys.dtype, np.uint64)
        self.assertTrue(fixtures.strictly_sorted(generator.active_keys))


class TestDeleteTrace(TraceGeneratorTestCase):
    def test_batches_target_the_percentile_region(self):
        for percentile in (20, 80):
            with self.subTest(percentile=percentile):
                _, trace = self.generate(f"P1: D:30,{percentile},g,s")
                deleted_keys = self.keys_of(trace, "delete")

                expected_pivot = float(np.percentile(self.base_keys, percentile))
                rel_error = abs(float(np.median(deleted_keys)) - expected_pivot) / self.key_span()
                self.assertLessEqual(rel_error, 0.20)

    def test_sorted_flag_controls_emission_order(self):
        _, trace_s = self.generate("P1: D:40,50,g,s")
        _, trace_u = self.generate("P1: D:40,50,g,u")

        self.assertEqual(self.keys_of(trace_s, "delete"), sorted(self.keys_of(trace_s, "delete")))
        self.assertNotEqual(self.keys_of(trace_u, "delete"), sorted(self.keys_of(trace_u, "delete")))

    def test_deleted_keys_leave_the_active_state(self):
        generator, trace = self.generate("P1: D:25,50,g,s")
        deleted_keys = self.keys_of(trace, "delete")
        remaining_keys = set(int(k) for k in generator.active_keys)

        self.assertEqual(len(generator.active_keys), 1000 - 25)
        for key in deleted_keys:
            self.assertNotIn(key, remaining_keys)

    def test_state_stays_monotonic_across_phases(self):
        generator, _ = self.generate("P1: D:30,25,g,s | P2: D:30,75,g,u")

        self.assertTrue(fixtures.strictly_sorted(generator.active_keys))
        self.assertEqual(len(generator.active_keys), 1000 - 60)


class TestUpdateTrace(TraceGeneratorTestCase):
    def test_emits_the_drain_batch_then_the_refill_batch(self):
        _, trace = self.generate("P1: U:35,50,g,u")

        self.assertEqual(len(trace), 1)
        group = trace[0]
        self.assertEqual([b.op_type for b in group], ["delete", "insert"])
        self.assertEqual(len(group[0].keys), 35)
        self.assertEqual(len(group[1].keys), 35)
        self.assertEqual(self.row_count(trace), 70)

    def test_drain_targets_the_percentile_region(self):
        for percentile in (25, 75):
            with self.subTest(percentile=percentile):
                _, trace = self.generate(f"P1: U:35,{percentile},g,s")
                drained_keys = self.keys_of(trace, "delete")

                expected_pivot = float(np.percentile(self.base_keys, percentile))
                rel_error = abs(float(np.median(drained_keys)) - expected_pivot) / self.key_span()
                self.assertLessEqual(rel_error, 0.20)

    def test_sorted_flag_controls_both_stream_orders_independently(self):
        _, trace_s = self.generate("P1: U:40,50,g,s")
        drained = self.keys_of(trace_s, "delete")
        refilled = self.keys_of(trace_s, "insert")
        self.assertEqual(drained, sorted(drained))
        self.assertEqual(refilled, sorted(refilled))

        _, trace_u = self.generate("P1: U:40,50,g,u")
        drained_u = self.keys_of(trace_u, "delete")
        refilled_u = self.keys_of(trace_u, "insert")
        self.assertNotEqual(drained_u, sorted(drained_u))
        self.assertNotEqual(refilled_u, sorted(refilled_u))

    def test_updates_preserve_cardinality_and_state_monotonicity(self):
        generator, _ = self.generate("P1: U:30,50,g,s | P2: U:30,80,z,u")

        self.assertEqual(len(generator.active_keys), 1000)
        self.assertEqual(len(set(int(k) for k in generator.active_keys)), 1000)
        self.assertTrue(fixtures.strictly_sorted(generator.active_keys))

    def test_refills_never_collide_with_the_survivors(self):
        generator, trace = self.generate("P1: U:30,50,g,s")
        drained = self.keys_of(trace, "delete")
        refilled = self.keys_of(trace, "insert")
        base_set = set(int(k) for k in self.base_keys)
        surviving = base_set - set(drained)

        self.assertTrue(set(drained).issubset(base_set))
        self.assertTrue(set(refilled).isdisjoint(surviving))
        self.assertEqual(len(set(refilled)), len(refilled))
        self.assertNotEqual(set(int(k) for k in generator.active_keys), base_set)


class TestQueryTrace(TraceGeneratorTestCase):
    def test_point_batch_is_existing_keys_with_empty_param(self):
        trace = self.generate("P1: Q:p,15,50,z")[1]
        self.assertEqual(len(trace), 1)
        self.assertEqual(len(trace[0]), 1)
        batch = trace[0][0]
        self.assertEqual(batch.op_type, "point")
        self.assertEqual(batch.param, "")
        self.assertEqual(batch.phase_id, 1)
        self.assertEqual(len(batch.keys), 15)
        self.assertTrue(np.all(np.isin(batch.keys, self.base_keys)))

    def test_range_batch_carries_the_interval_offset(self):
        trace = self.generate("P1: Q:r,10,50,50,r")[1]
        self.assertEqual(len(trace), 1)
        batch = trace[0][0]
        self.assertEqual(batch.op_type, "range")
        self.assertEqual(batch.param, 50)
        self.assertTrue(np.all(np.isin(batch.keys, self.base_keys)))

    def test_knn_batch_is_domain_points_with_k_param(self):
        trace = self.generate("P1: Q:k,50,10,50,z")[1]
        k_min, k_max = int(self.base_keys[0]), int(self.base_keys[-1])
        self.assertEqual(len(trace), 1)
        batch = trace[0][0]
        self.assertEqual(batch.op_type, "knn")
        self.assertEqual(batch.param, 10)
        self.assertGreaterEqual(int(batch.keys.min()), k_min)
        self.assertLessEqual(int(batch.keys.max()), k_max)

    def test_query_keys_target_the_requested_percentile(self):
        for text, query_type in (
            ("P1: Q:p,200,50,g", "point"),
            ("P1: Q:r,200,50,50,r", "range"),
            ("P1: Q:k,200,50,50,g", "knn"),
        ):
            with self.subTest(query_type=query_type):
                _, trace = self.generate(text)
                sampled = np.array(self.keys_of(trace, query_type), dtype=np.float64)

                expected_pivot = float(np.percentile(self.base_keys, 50))
                rel_error = abs(float(np.median(sampled)) - expected_pivot) / self.key_span()
                self.assertLessEqual(rel_error, 0.05)

    def test_short_form_queries_the_full_active_range(self):
        _, trace = self.generate("P1: Q:p,1000")
        sampled = np.array(self.keys_of(trace, "point"), dtype=np.float64)
        span = float(self.base_keys.max() - self.base_keys.min())

        self.assertGreater(float(np.max(sampled)) - float(np.min(sampled)), 0.9 * span)


class TestWorkloadIntegration(TraceGeneratorTestCase):
    def test_row_count_matches_the_workload_operations(self):
        _, trace = self.generate(
            "P1: Q:r,5,50,10,g; I:50,10,g,s; Q:p,10 | "
            "P2: Q:p,10; U:20,50,z,s; Q:k,15,10,50,z | "
            "P3: Q:p,5,50,z; D:10,25,r,s"
        )

        self.assertEqual(self.row_count(trace), 145)

    def test_identical_seeds_reproduce_the_trace_exactly(self):
        workload = self.parse_text(
            "P1: Q:r,5,50,10,g; I:50,10,g,s; Q:p,10 | P2: U:20,50,z,u; D:15,30,g,s"
        )

        g1 = TraceGenerator(keys=self.base_keys, seed=999)
        g2 = TraceGenerator(keys=self.base_keys, seed=999)
        trace1 = list(g1.generate_trace(workload))
        trace2 = list(g2.generate_trace(workload))

        self.assertEqual(len(trace1), len(trace2))
        for group1, group2 in zip(trace1, trace2):
            self.assertEqual([b.op_type for b in group1], [b.op_type for b in group2])
            for b1, b2 in zip(group1, group2):
                self.assertEqual(b1.phase_id, b2.phase_id)
                self.assertEqual(b1.param, b2.param)
                np.testing.assert_array_equal(b1.keys, b2.keys)

    def test_csv_export_writes_every_trace_row(self):
        generator = TraceGenerator(keys=self.base_keys, seed=42)
        workload = self.parse_text("P1: Q:r,2,10,50,g; I:10,1,g,s; U:5,50,g,s; D:3,10,g,s")

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            total_ops = save_trace(generator.generate_trace(workload), tmp_path)

            self.assertEqual(total_ops, 25)
            with open(tmp_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                self.assertEqual(reader.fieldnames, ["phase_id", "op_type", "key", "param"])
                rows = list(reader)
            self.assertEqual(len(rows), 25)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_spec_example_workload_execution(self):
        workload = self.parse_text(
            "P1: Q:p,1000; Q:r,100,50,50,r; I:100,1,g,s | "
            "P2-5: Q:k,50,10,50,z; U:100,50,z,s"
        )
        generator = TraceGenerator(keys=fixtures.uniform_sparse(), seed=42)
        groups = list(generator.generate_trace(workload))
        ops = [op for phase in workload.phases for op in phase.operations]
        self.assertEqual(len(groups), len(ops))
        expected_operations = 1000 + 100 + 100 + 4 * (50 + 200)
        self.assertEqual(self.row_count(groups), expected_operations)


class TestEmptyKeySetFailFast(TraceGeneratorTestCase):
    def test_empty_initial_dataset_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            TraceGenerator(keys=np.array([], dtype=np.uint64), seed=42)
        self.assertIn("Empty dataset", str(ctx.exception))

    def test_drained_active_set_fails_the_next_operation(self):
        # Draining all keys must cause the subsequent op to fail fast.
        workload = self.parse_text("P1: D:5000,50,r,s | P2: Q:p,10,50,z")
        generator = TraceGenerator(keys=self.base_keys, seed=42)

        batches = []
        with self.assertRaises(RuntimeError) as ctx:
            for group in generator.generate_trace(workload):
                batches.append(group)
        self.assertIn("Active key set is empty", str(ctx.exception))
        self.assertEqual(self.row_count(batches), len(self.base_keys))


class TestOpListenerEvents(TraceGeneratorTestCase):
    def test_events_fire_per_op_before_execution(self):
        workload = self.parse_text("P1: Q:p,5; I:10,50,g,s | P2: Q:p,5,50,z")
        generator = TraceGenerator(keys=self.base_keys, seed=42)

        contexts = []
        generator.add_op_listener(contexts.append)
        list(generator.generate_trace(workload))

        self.assertEqual([c.phase_id for c in contexts], [1, 1, 2])
        self.assertEqual([c.op_index for c in contexts], [0, 1, 0])
        self.assertEqual([c.phase_position for c in contexts], [1, 1, 2])
        self.assertEqual([c.total_phases for c in contexts], [2, 2, 2])
        self.assertEqual([c.phase_op_count for c in contexts], [2, 2, 1])

    def test_events_fire_before_the_op_runs(self):
        # Listeners must be notified before op runs for crash site attribution.
        generator = TraceGenerator(keys=self.base_keys, seed=42)
        workload = self.parse_text("P1: Q:p,5")

        seen = []
        listener = lambda ctx: seen.append(ctx.op_index)
        generator.add_op_listener(listener)
        batches = list(generator.generate_trace(workload))
        self.assertEqual(seen, [0])
        self.assertEqual(self.row_count(batches), 5)

    def test_listeners_survive_reset(self):
        # Listeners are configuration, not execution state.
        generator = TraceGenerator(keys=self.base_keys, seed=42)
        workload = self.parse_text("P1: Q:p,5")

        contexts = []
        generator.add_op_listener(contexts.append)
        list(generator.generate_trace(workload))
        list(generator.generate_trace(workload))
        self.assertEqual(len(contexts), 2)

    def test_emitted_batches_match_the_declared_schema(self):
        # TraceBatch is the contract consumed by all output writers.
        _, trace = self.generate(
            "P1: Q:p,5; Q:r,5,50,50,r; Q:k,5,10,50,z; "
            "I:10,50,g,s; U:10,50,z,u; D:5,50,r,s"
        )
        self.assertEqual(len(trace), 6)
        self.assertEqual([len(g) for g in trace], [1, 1, 1, 1, 2, 1])
        for group in trace:
            for b in group:
                self.assertIsInstance(b, TraceBatch)
                self.assertIsInstance(b.keys, np.ndarray)
                self.assertEqual(b.keys.dtype, np.uint64)


if __name__ == "__main__":
    unittest.main()
