import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fixtures
import numpy as np

from crash_dump import TraceCrashError, TraceCrashGuard, default_crash_dump_root
from models import Phase, QueryOp, Workload, WriteOp, describe_op
from trace_generator import TraceGenerator


class CrashDumpTestCase(unittest.TestCase):
    def setUp(self):
        # Query followed by a saturating insert to trigger fail-fast error.
        self.workload = Workload(
            phases=[
                Phase(
                    phase_id=1,
                    operations=[
                        QueryOp(query_type="point", count=5, percentile=None, distribution="uniform"),
                        WriteOp(operation="insert", batch_size=1000, percentile=50.0, distribution="uniform", sorted=True),
                    ],
                )
            ]
        )
        self.generator = TraceGenerator(keys=fixtures.uniform_dense(), seed=42)

    def generate_until_crash(self, guard, trace_csv: Path) -> TraceCrashError:
        try:
            guard.run_to_csv(trace_csv)
        except TraceCrashError as e:
            return e
        self.fail("expected the saturating insert to raise")

    def test_dump_contains_all_artifacts_and_crash_site(self):
        with tempfile.TemporaryDirectory() as tmp:
            dump_root = Path(tmp) / "dumps"
            trace_csv = Path(tmp) / "trace.csv"
            guard = TraceCrashGuard(self.generator, self.workload, dump_root=dump_root)

            error = self.generate_until_crash(guard, trace_csv)
            crash_dir = error.crash_dir

            self.assertIsInstance(error.error, RuntimeError)
            self.assertTrue((crash_dir / "stacktrace.txt").is_file())
            self.assertTrue("RuntimeError" in (crash_dir / "stacktrace.txt").read_text())

            active_keys = np.load(crash_dir / "active_keys.npy")
            self.assertEqual(len(active_keys), len(fixtures.uniform_dense()))

            report = json.loads((crash_dir / "crash.json").read_text())
            self.assertEqual(report["error"]["type"], "RuntimeError")
            self.assertEqual(report["crash_site"]["phase_id"], 1)
            self.assertEqual(report["crash_site"]["op_index"], 1)
            self.assertEqual(report["crash_site"]["operation"], describe_op(self.workload.phases[0].operations[1]))
            self.assertEqual(report["state"]["rows_emitted_before_crash"], 5)
            self.assertEqual(report["state"]["seed"], 42)

            with open(crash_dir / "partial_trace.csv", encoding="utf-8") as f:
                rows = f.read().strip().splitlines()
            self.assertEqual(len(rows), 6)  # header + the 5 point queries

    def test_crash_location_is_unknown_before_the_first_op(self):
        guard = TraceCrashGuard(self.generator, self.workload, dump_root=Path("/unused"))
        self.assertIn("unknown", guard.crash_location())

    def test_crash_error_location_names_phase_and_operation(self):
        with tempfile.TemporaryDirectory() as tmp:
            guard = TraceCrashGuard(self.generator, self.workload, dump_root=Path(tmp))
            error = self.generate_until_crash(guard, Path(tmp) / "trace.csv")
        self.assertIn("phase 1", error.location)
        self.assertIn("insert", error.location)

    def test_unsupported_output_extension_is_a_usage_error_not_a_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            dump_root = Path(tmp) / "dumps"
            guard = TraceCrashGuard(self.generator, self.workload, dump_root=dump_root)

            with self.assertRaisesRegex(ValueError, "unsupported trace output format"):
                guard.run_to_csv(Path(tmp) / "trace.parquet")

            self.assertFalse(dump_root.exists())
            self.assertEqual(guard.crash_location(), "unknown location (before the first operation)")

    def test_default_dump_root_is_os_local_data_dir(self):
        root = default_crash_dump_root()
        self.assertTrue(root.is_absolute())
        self.assertIn("crash-dumps", root.parts)


if __name__ == "__main__":
    unittest.main()
