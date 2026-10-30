import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from blocked_array import SortedBlockArray

BLOCK_SIZE = 16


def unique_keys(n: int, seed: int, space: int = 2 ** 24) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.unique(rng.integers(0, space, size=n, dtype=np.uint64))


def strictly_sorted(arr: np.ndarray) -> bool:
    return len(arr) < 2 or bool(np.all(arr[:-1] < arr[1:]))


class SortedBlockArrayTestCase(unittest.TestCase):
    def setUp(self):
        self.keys = unique_keys(500, seed=7)
        self.arr = SortedBlockArray(self.keys, block_size=BLOCK_SIZE)

    def assert_matches_reference(self, arr):
        flat = np.asarray(arr)
        self.assertEqual(len(flat), len(self.keys))
        self.assertTrue(np.array_equal(flat, self.keys), "container diverged from reference")

    def test_init_rejects_empty_dataset(self):
        with self.assertRaises(ValueError):
            SortedBlockArray(np.empty(0, dtype=np.uint64))

    def test_init_rejects_unsorted_keys(self):
        unsorted_keys = np.array([10, 5, 20], dtype=np.uint64)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            SortedBlockArray(unsorted_keys)

    def test_init_rejects_duplicate_keys(self):
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            SortedBlockArray(np.array([5, 5, 10], dtype=np.uint64))

    def test_init_partitions_into_blocks(self):
        self.assertEqual(len(self.arr), len(self.keys))
        self.assertTrue(all(len(b) <= BLOCK_SIZE for b in self.arr.blocks))
        self.assert_matches_reference(self.arr)

    def test_scalar_getitem_matches_reference(self):
        ranks = list(range(len(self.keys))) + [-1, -len(self.keys)]
        for r in ranks:
            self.assertEqual(int(self.arr[r]), int(self.keys[r]))

    def test_slice_getitem_matches_reference(self):
        for lo, hi in [(0, BLOCK_SIZE), (BLOCK_SIZE - 1, BLOCK_SIZE + 1),
                      (5, len(self.keys)), (7, 7), (len(self.keys) - 3, len(self.keys))]:
            self.assertTrue(np.array_equal(self.arr[lo:hi], self.keys[lo:hi]))

    def test_non_unit_step_slice_raises(self):
        with self.assertRaises(ValueError):
            self.arr[::2]

    def test_fancy_getitem_matches_reference(self):
        rng = np.random.default_rng(3)
        ranks = rng.integers(0, len(self.keys), size=64, dtype=np.int64)
        self.assertTrue(np.array_equal(self.arr[ranks], self.keys[ranks]))
        self.assertTrue(np.array_equal(self.arr[np.empty(0, dtype=np.int64)], np.empty(0, dtype=np.uint64)))

    def test_insert_below_min_and_above_max(self):
        new = np.array([1, 2, self.keys.max() + 5, self.keys.max() + 9], dtype=np.uint64)
        self.arr.insert_keys(new)
        self.keys = np.sort(np.append(self.keys, new))
        self.assert_matches_reference(self.arr)

    def test_hotspot_inserts_respect_split_invariant(self):
        lo = int(self.keys[0])
        new = np.arange(lo + 1, lo + 4 * BLOCK_SIZE, dtype=np.uint64) * 3
        self.arr.insert_keys(new)
        self.keys = np.sort(np.append(self.keys, new))
        self.assert_matches_reference(self.arr)
        self.assertTrue(all(len(b) <= 2 * BLOCK_SIZE for b in self.arr.blocks))

    def test_delete_ranks_matches_reference(self):
        rng = np.random.default_rng(11)
        ranks = rng.choice(len(self.keys), size=len(self.keys) - BLOCK_SIZE, replace=False)
        self.arr.delete_ranks(ranks)
        self.keys = np.delete(self.keys, ranks)
        self.assert_matches_reference(self.arr)

    def test_delete_keys_matches_reference(self):
        rng = np.random.default_rng(13)
        victims = self.keys[rng.choice(len(self.keys), size=100, replace=False)]
        self.arr.delete_keys(victims)
        self.keys = np.setdiff1d(self.keys, victims)
        self.assert_matches_reference(self.arr)

    def test_repeated_deletes_respect_merge_invariant(self):
        while len(self.keys) > BLOCK_SIZE:
            ranks = np.arange(len(self.keys) - BLOCK_SIZE, len(self.keys))
            self.arr.delete_ranks(ranks)
            self.keys = self.keys[:-BLOCK_SIZE]
            if len(self.arr.blocks) > 1:
                self.assertTrue(all(len(b) >= BLOCK_SIZE // 2 for b in self.arr.blocks))
        self.assert_matches_reference(self.arr)

    def test_full_drain_yields_empty_container(self):
        self.arr.delete_ranks(np.arange(len(self.keys), dtype=np.int64))
        self.assertEqual(len(self.arr), 0)
        self.assertEqual(len(np.asarray(self.arr)), 0)

    def test_mixed_ops_fuzz_against_reference(self):
        rng = np.random.default_rng(17)
        for _ in range(50):
            action = rng.integers(0, 3)
            if action == 0 or len(self.keys) < 2 * BLOCK_SIZE:
                new = np.unique(rng.integers(int(self.keys.max()) + 1,
                                              int(self.keys.max()) + 2 ** 12, size=8, dtype=np.uint64))
                self.arr.insert_keys(new)
                self.keys = np.sort(np.append(self.keys, new))
            elif action == 1:
                ranks = rng.choice(len(self.keys), size=4, replace=False)
                self.arr.delete_ranks(ranks)
                self.keys = np.delete(self.keys, ranks)
            else:
                victims = self.keys[rng.choice(len(self.keys), size=4, replace=False)]
                self.arr.delete_keys(victims)
                self.keys = np.setdiff1d(self.keys, victims)
            self.assert_matches_reference(self.arr)
            self.assertTrue(strictly_sorted(np.asarray(self.arr)))

    def test_delete_batch_merging_last_block_into_pending_block(self):
        base = np.arange(24, dtype=np.uint64) * 10
        ref = base
        arr = SortedBlockArray(base, block_size=8)
        for chunk in (np.arange(81, 152, 10, dtype=np.uint64),
                      np.arange(161, 232, 10, dtype=np.uint64)):
            arr.insert_keys(chunk)
            ref = np.sort(np.append(ref, chunk))
        batch = np.concatenate([np.arange(22, 24), np.arange(24, 37)])
        arr.delete_ranks(batch)
        ref = np.delete(ref, batch)
        self.assertEqual(len(arr), len(ref))
        self.assertTrue(np.array_equal(np.asarray(arr), ref))

    def test_out_of_bounds_scalar_getitem_raises(self):
        with self.assertRaises(IndexError):
            self.arr[len(self.keys)]
        with self.assertRaises(IndexError):
            self.arr[-len(self.keys) - 1]

    def test_array_protocol_and_iteration(self):
        flat = np.asarray(self.arr)
        self.assertEqual(flat.dtype, np.uint64)
        self.assertEqual(self.arr.dtype, np.uint64)
        self.assertEqual([int(k) for k in self.arr], [int(k) for k in flat])


if __name__ == "__main__":
    unittest.main()
