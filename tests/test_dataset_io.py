import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from dataset_io import detect_format, load_dataset


def write_sosd_binary(path: Path, keys: np.ndarray) -> None:
    """Writes the SOSD layout: one uint64 size header, then the keys."""
    with open(path, "wb") as f:
        np.array([len(keys)], dtype=np.uint64).tofile(f)
        np.asarray(keys, dtype=np.uint64).tofile(f)


class TestDatasetIO(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_sosd_roundtrip_autodetected(self):
        keys = np.array([3, 10, 1000, 2**63], dtype=np.uint64)
        path = self.tmp_dir / "books"
        write_sosd_binary(path, keys)

        loaded = load_dataset(path)
        self.assertEqual(detect_format(path), "sosd")
        np.testing.assert_array_equal(loaded, keys)
        self.assertEqual(loaded.dtype, np.uint64)

    def test_sosd_explicit_format_ignores_key_col(self):
        keys = np.arange(5, dtype=np.uint64)
        path = self.tmp_dir / "osm"
        write_sosd_binary(path, keys)

        loaded = load_dataset(path, key_col="anything", dataset_format="sosd")
        np.testing.assert_array_equal(loaded, keys)

    def test_csv_autodetected(self):
        keys = np.array([1, 5, 9], dtype=np.uint64)
        path = self.tmp_dir / "keys.csv"
        pd.DataFrame({"key": keys}).to_csv(path, index=False)

        loaded = load_dataset(path)
        self.assertEqual(detect_format(path), "csv")
        np.testing.assert_array_equal(loaded, keys)

    def test_sosd_truncated_file_raises(self):
        path = self.tmp_dir / "truncated"
        with open(path, "wb") as f:
            np.array([10], dtype=np.uint64).tofile(f)
            np.arange(4, dtype=np.uint64).tofile(f)

        with self.assertRaisesRegex(ValueError, "truncated"):
            load_dataset(path)

    def test_sosd_zero_key_header_raises(self):
        path = self.tmp_dir / "zero"
        write_sosd_binary(path, np.array([], dtype=np.uint64))
        with self.assertRaisesRegex(ValueError, "no keys"):
            load_dataset(path, dataset_format="sosd")

    def test_csv_missing_key_col_raises(self):
        path = self.tmp_dir / "keys.csv"
        pd.DataFrame({"id": [1, 2]}).to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "key column"):
            load_dataset(path)

    def test_csv_empty_dataset_raises(self):
        path = self.tmp_dir / "keys.csv"
        pd.DataFrame({"key": []}).to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "no rows"):
            load_dataset(path)

    def test_shuffled_sosd_is_sorted_on_load(self):
        keys = np.array([100, 3, 7, 42], dtype=np.uint64)
        path = self.tmp_dir / "shuffled"
        write_sosd_binary(path, keys)

        loaded = load_dataset(path)
        np.testing.assert_array_equal(loaded, np.sort(keys))

    def test_shuffled_csv_is_sorted_on_load(self):
        keys = np.array([9, 1, 5], dtype=np.uint64)
        path = self.tmp_dir / "keys.csv"
        pd.DataFrame({"key": keys}).to_csv(path, index=False)

        loaded = load_dataset(path)
        np.testing.assert_array_equal(loaded, np.sort(keys))

    def test_duplicate_keys_raise(self):
        keys = np.array([5, 3, 5, 9], dtype=np.uint64)
        path = self.tmp_dir / "dupes"
        write_sosd_binary(path, keys)

        with self.assertRaisesRegex(ValueError, "duplicate"):
            load_dataset(path)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_dataset(self.tmp_dir / "nope")

    def test_unsupported_format_raises(self):
        path = self.tmp_dir / "keys.csv"
        pd.DataFrame({"key": [1]}).to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "unsupported dataset format"):
            load_dataset(path, dataset_format="parquet")


if __name__ == "__main__":
    unittest.main()
