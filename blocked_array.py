"""Two-level sorted-block array: the active-key container for TraceGenerator."""
import numpy as np


class SortedBlockArray:
    """Sorted unique uint64 keys with block-localized batch edits."""

    def __init__(self, keys: np.ndarray, block_size: int = 32768):
        arr = np.asarray(keys, dtype=np.uint64)
        if len(arr) == 0:
            raise ValueError("Empty dataset: requires at least one key.")
        unsorted_at = np.flatnonzero(arr[:-1] >= arr[1:])
        if unsorted_at.size > 0:
            i = int(unsorted_at[0])
            raise ValueError(
                "Input keys must be strictly increasing (sorted, unique): "
                f"keys[{i}]={arr[i]} >= keys[{i + 1}]={arr[i + 1]}. "
                "Sort and deduplicate the dataset before use."
            )
        self.block_size = block_size
        n_blocks = max(1, (len(arr) + block_size - 1) // block_size)
        self.blocks: list = [b.copy() for b in np.array_split(arr, n_blocks) if len(b) > 0]
        self.lengths = np.fromiter((len(b) for b in self.blocks), dtype=np.int64,
                                   count=len(self.blocks))
        self.mins = np.fromiter((int(b[0]) for b in self.blocks), dtype=np.uint64,
                                count=len(self.blocks))
        self._refresh_offsets()

    def _refresh_offsets(self) -> None:
        self.offsets = np.empty(len(self.blocks) + 1, dtype=np.int64)
        self.offsets[0] = 0
        np.cumsum(self.lengths, out=self.offsets[1:])

    def _set_blocks(self, lo: int, hi: int, pieces) -> None:
        """Mutates blocks [lo:hi] and keeps lengths and mins metadata synchronized."""
        self.blocks[lo:hi] = pieces
        new_lengths = np.array([len(p) for p in pieces], dtype=np.int64)
        new_mins = np.array([int(p[0]) for p in pieces], dtype=np.uint64)
        if len(pieces) == hi - lo:
            self.lengths[lo:hi] = new_lengths
            self.mins[lo:hi] = new_mins
        else:
            self.lengths = np.concatenate([self.lengths[:lo], new_lengths, self.lengths[hi:]])
            self.mins = np.concatenate([self.mins[:lo], new_mins, self.mins[hi:]])

    def __len__(self) -> int:
        return int(self.offsets[-1])

    @property
    def dtype(self):
        return np.uint64

    def __iter__(self):
        return iter(np.asarray(self))

    def __array__(self, dtype=None):
        arr = np.concatenate(self.blocks) if self.blocks else np.empty(0, dtype=np.uint64)
        return arr if dtype is None else arr.astype(dtype)

    def __getitem__(self, item):
        n = len(self)
        if isinstance(item, (int, np.integer)):
            idx = int(item)
            if idx < 0:
                idx += n
            # A negative idx that survives normalization would resolve to
            # the last block below - a silent wrap-around.
            if idx < 0 or idx >= n:
                raise IndexError(f"index {item} is out of bounds for size {n}")
            if idx == 0:
                return self.blocks[0][0]
            if idx == n - 1:
                return self.blocks[-1][-1]
            b = int(np.searchsorted(self.offsets, idx, side="right")) - 1
            return self.blocks[b][idx - int(self.offsets[b])]

        if isinstance(item, slice):
            start, stop, step = item.indices(n)
            if step != 1:
                raise ValueError("Only contiguous unit-step slices are supported.")
            if stop <= start:
                return np.empty(0, dtype=np.uint64)
            b_lo = int(np.searchsorted(self.offsets, start, side="right")) - 1
            b_hi = int(np.searchsorted(self.offsets, stop - 1, side="right")) - 1
            if b_lo == b_hi:
                base = int(self.offsets[b_lo])
                return self.blocks[b_lo][start - base : stop - base]
            parts = [self.blocks[b_lo][start - int(self.offsets[b_lo]) :]]
            parts.extend(self.blocks[b_lo + 1 : b_hi])
            parts.append(self.blocks[b_hi][: stop - int(self.offsets[b_hi])])
            return np.concatenate(parts)

        # Rank array: resolve every index through one offsets search.
        ranks = np.asarray(item, dtype=np.int64)
        if len(ranks) == 0:
            return np.empty(0, dtype=np.uint64)
        b_idx = np.searchsorted(self.offsets, ranks, side="right") - 1
        out = np.empty(len(ranks), dtype=np.uint64)
        for b in np.unique(b_idx):
            mask = b_idx == b
            out[mask] = self.blocks[b][ranks[mask] - self.offsets[b]]
        return out

    def insert_keys(self, new_keys: np.ndarray) -> None:
        """Inserts unique, collision-free keys into their owning blocks."""
        if len(new_keys) == 0:
            return
        s_new = np.sort(np.asarray(new_keys, dtype=np.uint64))
        b_idx = np.clip(np.searchsorted(self.mins, s_new, side="right") - 1,
                        0, len(self.blocks) - 1)
        unique_b, starts, counts = np.unique(b_idx, return_index=True,
                                            return_counts=True)
        # Reverse block order: splitting a block never shifts the blocks
        # still pending at lower indices.
        for b, start, count in zip(unique_b[::-1], starts[::-1], counts[::-1]):
            chunk = s_new[start : start + count]
            blk = self.blocks[b]
            ins_pos = np.searchsorted(blk, chunk)
            merged = np.insert(blk, ins_pos, chunk)
            if len(merged) > 2 * self.block_size:
                n_split = (len(merged) + self.block_size - 1) // self.block_size
                self._set_blocks(b, b + 1,
                                 [sub.copy() for sub in np.array_split(merged, n_split)])
            else:
                self._set_blocks(b, b + 1, [merged])
        self._refresh_offsets()

    def delete_ranks(self, ranks: np.ndarray) -> None:
        """Removes keys at the given global ranks."""
        if len(ranks) == 0:
            return
        s_ranks = np.sort(np.asarray(ranks, dtype=np.int64))
        b_idx = np.searchsorted(self.offsets, s_ranks, side="right") - 1
        unique_b, starts, counts = np.unique(b_idx, return_index=True,
                                            return_counts=True)
        for b, start, count in zip(unique_b[::-1], starts[::-1], counts[::-1]):
            local_ranks = s_ranks[start : start + count] - self.offsets[b]
            blk = self.blocks[b]
            if len(local_ranks) == len(blk):
                self._set_blocks(b, b + 1, [])
            else:
                mask = np.ones(len(blk), dtype=bool)
                mask[local_ranks] = False
                self._set_blocks(b, b + 1, [blk[mask]])
        self._merge_undersized_blocks()
        self._refresh_offsets()

    def delete_keys(self, old_keys: np.ndarray) -> None:
        """Removes existing keys by key value."""
        if len(old_keys) == 0:
            return
        s_old = np.sort(np.asarray(old_keys, dtype=np.uint64))
        b_idx = np.clip(np.searchsorted(self.mins, s_old, side="right") - 1,
                        0, len(self.blocks) - 1)
        unique_b, starts, counts = np.unique(b_idx, return_index=True,
                                            return_counts=True)
        for b, start, count in zip(unique_b[::-1], starts[::-1], counts[::-1]):
            chunk = s_old[start : start + count]
            blk = self.blocks[b]
            local_ranks = np.searchsorted(blk, chunk)
            if len(local_ranks) == len(blk):
                self._set_blocks(b, b + 1, [])
            else:
                mask = np.ones(len(blk), dtype=bool)
                mask[local_ranks] = False
                self._set_blocks(b, b + 1, [blk[mask]])
        self._merge_undersized_blocks()
        self._refresh_offsets()

    def _merge_undersized_blocks(self) -> None:
        """Sweeps blocks in reverse to merge any blocks shrunk below B/2."""
        for b in range(len(self.blocks) - 1, -1, -1):
            self._maybe_merge(b)

    def _maybe_merge(self, b: int) -> None:
        """Merges block `b` with a neighbor if it dropped below B/2."""
        half = self.block_size // 2
        if len(self.blocks) <= 1 or len(self.blocks[b]) >= half:
            return
        neighbor = b + 1 if b + 1 < len(self.blocks) else b - 1
        lo, hi = min(b, neighbor), max(b, neighbor)
        combined = np.concatenate([self.blocks[lo], self.blocks[hi]])
        if len(combined) > 2 * self.block_size:
            mid = len(combined) // 2
            self._set_blocks(lo, hi + 1,
                             [combined[:mid].copy(), combined[mid:].copy()])
        else:
            self._set_blocks(lo, hi + 1, [combined])
