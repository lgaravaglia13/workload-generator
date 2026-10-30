import io
import sys
import unittest
import logging

from logger import get_logger, normalize_level, setup


class LoggerTestCase(unittest.TestCase):
    def setUp(self):
        self._saved_out, self._saved_err = sys.stdout, sys.stderr
        self.out, self.err = io.StringIO(), io.StringIO()
        sys.stdout, sys.stderr = self.out, self.err
        self.addCleanup(self._restore_streams)

    def _restore_streams(self):
        sys.stdout, sys.stderr = self._saved_out, self._saved_err
        root = logging.getLogger()
        for handler in list(root.handlers):
            root.removeHandler(handler)
        root.setLevel(logging.WARNING)

    def test_progress_below_warning_goes_to_stdout(self):
        setup("info")
        log = get_logger("test")
        log.debug("dbg")
        log.info("inf")
        log.warning("warn")
        log.error("err")

        out, err = self.out.getvalue(), self.err.getvalue()
        self.assertIn("inf", out)
        self.assertNotIn("dbg", out)
        self.assertIn("warn", err)
        self.assertIn("err", err)
        self.assertNotIn("warn", out)
        self.assertNotIn("inf", err)

    def test_debug_level_shows_per_op_detail(self):
        setup("debug")
        get_logger("test").debug("dbg")
        self.assertIn("dbg", self.out.getvalue())

    def test_info_level_hides_debug(self):
        setup("info")
        get_logger("test").debug("dbg")
        self.assertNotIn("dbg", self.out.getvalue())

    def test_error_level_hides_info(self):
        setup("error")
        get_logger("test").info("inf")
        get_logger("test").error("err")
        out, err = self.out.getvalue(), self.err.getvalue()
        self.assertNotIn("inf", out)
        self.assertIn("err", err)

    def test_setup_is_idempotent(self):
        setup("info")
        setup("info")
        get_logger("test").info("once")
        self.assertEqual(self.out.getvalue().count("once"), 1)

    def test_unknown_level_raises(self):
        with self.assertRaisesRegex(ValueError, "unknown log level"):
            setup("verbose")

    def test_normalize_level_is_case_insensitive(self):
        self.assertEqual(normalize_level("info"), logging.INFO)
        self.assertEqual(normalize_level(" Debug "), logging.DEBUG)

    def test_get_logger_namespaces_children(self):
        self.assertEqual(get_logger("trace").name, "workload.trace")
        self.assertEqual(get_logger("workload.trace").name, "workload.trace")

    def test_existing_sampler_drift_logger_flows_through_setup(self):
        setup("debug")
        logging.getLogger("sampler.drift").debug("drift-diagnostic")
        self.assertIn("drift-diagnostic", self.out.getvalue())

    def test_attach_trace_progress_narrates_phases_and_ops(self):
        from logger import attach_trace_progress
        from models import QueryOp, WriteOp
        from trace_generator import OpContext

        class StubGenerator:
            def __init__(self):
                self.listeners = []

            def add_op_listener(self, callback):
                self.listeners.append(callback)

        stub = StubGenerator()
        attach_trace_progress(stub)
        self.assertEqual(len(stub.listeners), 1)
        narrate = stub.listeners[0]

        setup("info")
        narrate(OpContext(3, 1, 4, 0, 2, QueryOp("point", 10, 50.0, "uniform")))
        narrate(OpContext(3, 1, 4, 1, 2, QueryOp("point", 10, 50.0, "uniform")))
        out = self.out.getvalue()
        self.assertEqual(out.count("Phase 1/4: 2 operations"), 1)
        self.assertNotIn("point count=", out)

        setup("debug")
        narrate(OpContext(3, 1, 4, 0, 2, WriteOp("insert", 100, 1.0, "gaussian", True)))
        self.assertIn("Phase 1/4: 2 operations", self.out.getvalue())
        self.assertIn("insert batch_size=100", self.out.getvalue())


if __name__ == "__main__":
    unittest.main()
