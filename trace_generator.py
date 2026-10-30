from typing import Iterator, List, Callable, NamedTuple, Union

import numpy as np

from blocked_array import SortedBlockArray
from models import Workload, WriteOp, QueryOp
from distributions import from_token
from sampler import (
    RANK_SPREAD_RATIO,
    sample_insert_keys,
    sample_existing_indices,
    sample_point_queries,
    sample_knn_query_points,
    sample_update_pairs,
)


class OpContext(NamedTuple):
    """Context passed to op listeners immediately prior to op execution."""

    phase_id: int
    phase_position: int
    total_phases: int
    op_index: int
    phase_op_count: int
    op: Union[QueryOp, WriteOp]


class TraceBatch(NamedTuple):
    """Emitted batch of trace operations sharing phase_id, op_type, and optional param."""

    phase_id: int
    op_type: str
    keys: np.ndarray
    param: Union[int, str] = ""


class TraceGenerator:
    """Generates benchmark trace operations from an initial set of keys."""

    def __init__(
        self,
        keys: np.ndarray,
        seed: int = 42,
        rank_spread: float = RANK_SPREAD_RATIO,
    ):
        self.seed = seed
        self.rank_spread = rank_spread
        self.base_keys = np.asarray(keys, dtype=np.uint64)
        if len(self.base_keys) == 0:
            raise ValueError("Empty dataset: the workload generator requires at least one key.")
        self._op_listeners: List[Callable[[OpContext], None]] = []
        self.rng = np.random.default_rng(seed)
        self.reset()

    def add_op_listener(self, listener: Callable[[OpContext], None]) -> None:
        """Subscribes a callback fired before every op execution."""
        self._op_listeners.append(listener)

    @property
    def initial_keys(self) -> np.ndarray:
        return self.base_keys

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.active_keys = SortedBlockArray(self.base_keys)

    def _process_delete(self, op: WriteOp, phase_id: int) -> List[TraceBatch]:
        n = len(self.active_keys)
        if op.batch_size <= 0:
            return []

        count = min(op.batch_size, n)
        selected_indices = sample_existing_indices(
            n=n,
            count=count,
            percentile=op.percentile,
            distribution=from_token(op.distribution),
            is_sorted=op.sorted,
            rng=self.rng,
            rank_spread=self.rank_spread,
        )
        deleted_keys = self.active_keys[selected_indices]
        self.active_keys.delete_ranks(selected_indices)

        return [TraceBatch(phase_id, "delete", deleted_keys)]

    def _process_insert(self, op: WriteOp, phase_id: int) -> List[TraceBatch]:
        if op.batch_size <= 0:
            return []

        new_keys = sample_insert_keys(
            active_keys=self.active_keys,
            count=op.batch_size,
            percentile=op.percentile,
            distribution=from_token(op.distribution),
            is_sorted=op.sorted,
            rng=self.rng,
            rank_spread=self.rank_spread,
        )
        self.active_keys.insert_keys(new_keys)

        return [TraceBatch(phase_id, "insert", new_keys)]

    def _process_update(self, op: WriteOp, phase_id: int) -> List[TraceBatch]:
        """Decomposes an update into a drain batch followed by a refill batch."""
        n = len(self.active_keys)
        if op.batch_size <= 0:
            return []

        count = min(op.batch_size, n)
        old_keys, new_keys = sample_update_pairs(
            active_keys=self.active_keys,
            count=count,
            percentile=op.percentile,
            distribution=from_token(op.distribution),
            is_sorted=op.sorted,
            rng=self.rng,
            rank_spread=self.rank_spread,
        )

        self.active_keys.delete_keys(old_keys)
        self.active_keys.insert_keys(new_keys)

        return [
            TraceBatch(phase_id, "delete", old_keys),
            TraceBatch(phase_id, "insert", new_keys),
        ]

    def _process_query(self, op: QueryOp, phase_id: int) -> List[TraceBatch]:
        """Samples query keys for point, range, or knn operations."""
        if op.count <= 0:
            return []

        distribution = from_token(op.distribution)
        if op.query_type == "knn":
            sampled_keys = sample_knn_query_points(
                self.active_keys, op.count, op.percentile, distribution, rng=self.rng,
                rank_spread=self.rank_spread,
            )
            param = op.k
        else:
            sampled_keys = sample_point_queries(
                self.active_keys, op.count, op.percentile, distribution, rng=self.rng,
                rank_spread=self.rank_spread,
            )
            param = "" if op.query_type == "point" else op.offset

        return [TraceBatch(phase_id, op.query_type, sampled_keys, param)]

    def run_op(self, op: Union[WriteOp, QueryOp], phase_id: int) -> List[TraceBatch]:
        """Applies a single operation to the live active keys without resetting."""
        if len(self.active_keys) == 0:
            raise RuntimeError(
                "Active key set is empty: no keys left to apply the operation to."
            )
        if isinstance(op, WriteOp):
            if op.operation == "insert":
                return self._process_insert(op, phase_id)
            if op.operation == "update":
                return self._process_update(op, phase_id)
            if op.operation == "delete":
                return self._process_delete(op, phase_id)
        elif isinstance(op, QueryOp):
            return self._process_query(op, phase_id)
        raise TypeError(f"unsupported operation: {op!r}")

    def generate_trace(self, workload: Workload) -> Iterator[List[TraceBatch]]:
        """Yields one batch group per workload operation, firing registered listeners before each op."""
        self.reset()
        total_phases = len(workload.phases)
        for phase_position, phase in enumerate(workload.phases, start=1):
            phase_op_count = len(phase.operations)
            for op_index, op in enumerate(phase.operations):
                context = OpContext(
                    phase_id=phase.phase_id,
                    phase_position=phase_position,
                    total_phases=total_phases,
                    op_index=op_index,
                    phase_op_count=phase_op_count,
                    op=op,
                )
                for listener in self._op_listeners:
                    listener(context)
                yield self.run_op(op, phase.phase_id)

