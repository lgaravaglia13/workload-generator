import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from trace_generator import TraceBatch
from trace_output import save_trace


class TraceOutputTestCase(unittest.TestCase):
    BATCHES = [
        [TraceBatch(phase_id=1, op_type="point", keys=np.array([10], dtype=np.uint64), param="")],
        [TraceBatch(phase_id=1, op_type="insert", keys=np.array([99], dtype=np.uint64), param="")],
    ]

    def test_csv_writer_streams_batches_and_returns_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.csv"
            count = save_trace(iter(self.BATCHES), path)

            self.assertEqual(count, 2)
            lines = path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(lines[0], "phase_id,op_type,key,param")
            self.assertEqual(lines[1], "1,point,10,")
            self.assertEqual(lines[2], "1,insert,99,")

    def test_empty_trace_writes_only_the_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.csv"
            count = save_trace(iter([]), path)
            self.assertEqual(count, 0)
            lines = path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)

    def test_unsupported_extension_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.parquet"
            with self.assertRaisesRegex(ValueError, "unsupported trace output format"):
                save_trace(iter(self.BATCHES), path)


if __name__ == "__main__":
    unittest.main()
