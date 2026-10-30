import unittest
import json
from dsl.parser import Lark_StandAlone
from models import QueryOp, WriteOp, Workload
from dsl.transformer import WorkloadTransformer


class TestWorkloadParser(unittest.TestCase):
    def setUp(self):
        self.parser = Lark_StandAlone()
        self.transformer = WorkloadTransformer()

    def parse_text(self, text: str) -> Workload:
        tree = self.parser.parse(text)
        return self.transformer.transform(tree)

    def test_single_phase_query_and_write(self):
        text = "P1: Q:p,1000; I:100,1,g,s; Q:r,100,50,50,r"
        workload = self.parse_text(text)
        self.assertEqual(len(workload.phases), 1)
        phase = workload.phases[0]
        self.assertEqual(phase.phase_id, 1)
        self.assertEqual(len(phase.operations), 3)

        op1 = phase.operations[0]
        self.assertIsInstance(op1, QueryOp)
        self.assertEqual(op1.query_type, "point")
        self.assertEqual(op1.count, 1000)
        self.assertIsNone(op1.percentile)
        self.assertEqual(op1.distribution, "uniform")

        op2 = phase.operations[1]
        self.assertIsInstance(op2, WriteOp)
        self.assertEqual(op2.operation, "insert")
        self.assertEqual(op2.batch_size, 100)
        self.assertEqual(op2.percentile, 1.0)
        self.assertEqual(op2.distribution, "gaussian")
        self.assertTrue(op2.sorted)

        op3 = phase.operations[2]
        self.assertIsInstance(op3, QueryOp)
        self.assertEqual(op3.query_type, "range")
        self.assertEqual(op3.count, 100)
        self.assertEqual(op3.offset, 50)
        self.assertEqual(op3.percentile, 50.0)
        self.assertEqual(op3.distribution, "uniform")

    def test_knn_query_type(self):
        op = self.parse_text("P1: Q:k,50,10,50,z").phases[0].operations[0]
        self.assertIsInstance(op, QueryOp)
        self.assertEqual(op.query_type, "knn")
        self.assertEqual(op.count, 50)
        self.assertEqual(op.k, 10)
        self.assertEqual(op.percentile, 50.0)
        self.assertEqual(op.distribution, "zipf")
        self.assertIsNone(op.offset)

    def test_multi_phase_with_ranges(self):
        text = (
            "P1: Q:p,5; I:100,1,g,s; Q:r,10,20 | "
            "P2-5: Q:r,10,20,50,r; U:100,50,z,s; Q:k,15,10,50,z | "
            "P6-10: Q:p,5; I:100,1,g,s; Q:r,5,10,20,g"
        )
        workload = self.parse_text(text)
        self.assertEqual(len(workload.phases), 10)
        self.assertEqual([p.phase_id for p in workload.phases], list(range(1, 11)))

        p2 = workload.phases[1]
        self.assertEqual(p2.phase_id, 2)
        self.assertEqual(len(p2.operations), 3)
        self.assertEqual(p2.operations[1].operation, "update")
        self.assertEqual(p2.operations[1].distribution, "zipf")

    def test_example_workload_from_the_spec(self):
        workload = self.parse_text(
            "P1: Q:p,1000; Q:r,100,50,50,r; I:100,1,g,s | "
            "P2-5: Q:k,50,10,50,z; U:100,50,z,s"
        )
        self.assertEqual([p.phase_id for p in workload.phases], [1, 2, 3, 4, 5])
        p1_types = [op.query_type for op in workload.phases[0].operations[:2]]
        self.assertEqual(p1_types, ["point", "range"])
        self.assertEqual(workload.phases[4].operations[0].query_type, "knn")

    def test_missing_or_extra_fields_fail_fast(self):
        for text in (
            "P1: Q:r,10",                # range without its offset
            "P1: Q:r,100,50,50",         # truncated long form
            "P1: Q:k,50,10,50",          # truncated long form
            "P1: Q:p,1000,50,50,r",      # point must not take an offset
            "P1: Q:k,50,10,50,z,r",      # extra field
        ):
            with self.subTest(text=text):
                with self.assertRaises(Exception):
                    self.parse_text(text)

    def test_out_of_range_values_fail_fast(self):
        for text in (
            "P1: Q:p,10,150,r",      # percentile > 100
            "P1: Q:p,10,-5,r",       # percentile < 0
            "P1: Q:p,0,50,r",         # count must be >= 1
            "P1: Q:r,10,0,50,r",     # offset must be >= 1
            "P1: Q:k,10,0,50,z",     # k must be >= 1
            "P1: Q:r,10,2.5,50,r",   # offset must be an integer
            "P1: I:100,150,g,s",     # write percentile still validated
        ):
            with self.subTest(text=text):
                with self.assertRaises(Exception):
                    self.parse_text(text)

    def test_invalid_syntax_missing_comma(self):
        text = "P1: Q:r,100 50,50,r; I:100,1,g,s"
        with self.assertRaises(Exception):
            self.parse_text(text)

    def test_invalid_phase_range(self):
        text = "P5-2: Q:p,5"
        with self.assertRaises(Exception):
            self.parse_text(text)

    def test_short_form_selects_the_global_uniform_query(self):
        cases = (
            ("P1: Q:p,100", QueryOp(query_type="point", count=100, percentile=None, distribution="uniform")),
            ("P1: Q:r,10,50", QueryOp(query_type="range", count=10, percentile=None, distribution="uniform", offset=50)),
            ("P1: Q:k,5,10", QueryOp(query_type="knn", count=5, percentile=None, distribution="uniform", k=10)),
        )
        for text, expected in cases:
            with self.subTest(text=text):
                op = self.parse_text(text).phases[0].operations[0]
                self.assertEqual(op, expected)

    def test_unified_json_export(self):
        text = "P1: Q:p,1000; I:100,1,g,s; Q:r,100,50,50,r"
        workload = self.parse_text(text)
        data = workload.to_dict(target_feature="key")

        self.assertIn("phases", data)
        self.assertEqual(len(data["phases"]), 1)
        ops = data["phases"][0]["operations"]
        self.assertEqual(len(ops), 3)
        self.assertEqual(ops[0]["type"], "query")
        self.assertIsNone(ops[0]["percentile"])
        self.assertNotIn("offset", ops[0])
        self.assertNotIn("k", ops[0])
        self.assertEqual(ops[1]["type"], "write")
        self.assertNotIn("details", ops[1])
        self.assertEqual(ops[2]["offset"], 50)
        self.assertNotIn("k", ops[2])

        json_str = workload.to_json(target_feature="key")
        self.assertEqual(json.loads(json_str), data)


if __name__ == "__main__":
    unittest.main()
