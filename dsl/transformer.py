from copy import deepcopy
from typing import List, Union

from dsl.parser import Transformer, Token
from models import QueryOp, WriteOp, Phase, Workload, DistributionType

DISTRIBUTION_MAP = {
    "g": "gaussian",
    "z": "zipf",
    "r": "uniform",
}


def _positive_int(value: Union[int, float], name: str) -> int:
    """Strict validation for integers >= 1."""
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{name} must be an integer >= 1, got {value}")
    iv = int(value)
    if iv < 1:
        raise ValueError(f"{name} must be an integer >= 1, got {value}")
    return iv


def _percentile(value: float) -> float:
    if not (0.0 <= value <= 100.0):
        raise ValueError(f"Percentile must be between 0 and 100, got {value}")
    return value


class WorkloadTransformer(Transformer):
    def NUMBER(self, token: Token) -> Union[int, float]:
        val = str(token)
        return float(val) if "." in val else int(val)

    def DISTRIBUTION(self, token: Token) -> str:
        return str(token)

    def SORTEDNESS(self, token: Token) -> str:
        return str(token)

    def total_query(self, items) -> int:
        return _positive_int(items[0], "count")

    def range_offset(self, items) -> int:
        return _positive_int(items[0], "offset")

    def neighbor_k(self, items) -> int:
        return _positive_int(items[0], "k")

    # The short form (omitted percentile/distribution) implies untargeted uniform query.
    def point_params(self, items) -> QueryOp:
        if len(items) == 1:
            return QueryOp(query_type="point", count=items[0], percentile=None, distribution="uniform")
        return QueryOp(
            query_type="point",
            count=items[0],
            percentile=items[1],
            distribution=items[2],
        )

    def range_params(self, items) -> QueryOp:
        if len(items) == 2:
            return QueryOp(
                query_type="range", count=items[0], percentile=None, distribution="uniform", offset=items[1]
            )
        return QueryOp(
            query_type="range",
            count=items[0],
            offset=items[1],
            percentile=items[2],
            distribution=items[3],
        )

    def knn_params(self, items) -> QueryOp:
        if len(items) == 2:
            return QueryOp(
                query_type="knn", count=items[0], percentile=None, distribution="uniform", k=items[1]
            )
        return QueryOp(
            query_type="knn",
            count=items[0],
            k=items[1],
            percentile=items[2],
            distribution=items[3],
        )

    def query_params(self, items) -> QueryOp:
        return items[0]

    def query(self, items) -> QueryOp:
        return items[0]

    def batch_size(self, items) -> int:
        return int(items[0])

    def percentile_loc(self, items) -> float:
        return _percentile(float(items[0]))

    def distribution(self, items) -> DistributionType:
        return DISTRIBUTION_MAP[str(items[0])]

    def sortedness(self, items) -> bool:
        return str(items[0]) == "s"

    def uid_params(self, items) -> dict:
        return {
            "batch_size": items[0],
            "percentile": items[1],
            "distribution": items[2],
            "sorted": items[3],
        }

    def insert(self, items) -> WriteOp:
        return WriteOp(operation="insert", **items[0])

    def update(self, items) -> WriteOp:
        return WriteOp(operation="update", **items[0])

    def delete(self, items) -> WriteOp:
        return WriteOp(operation="delete", **items[0])

    def operation(self, items) -> Union[QueryOp, WriteOp]:
        return items[0]

    def phase(self, items) -> List[Phase]:
        if isinstance(items[1], (int, float)):
            start_p = int(items[0])
            end_p = int(items[1])
            if start_p > end_p:
                raise ValueError(f"Phase range start ({start_p}) cannot exceed end ({end_p})")
            ops = items[2:]
            # Deepcopy operations to avoid sharing mutable operation objects across phases.
            return [Phase(phase_id=p_id, operations=deepcopy(ops)) for p_id in range(start_p, end_p + 1)]
        else:
            p_id = int(items[0])
            ops = items[1:]
            return [Phase(phase_id=p_id, operations=ops)]

    def start(self, items) -> Workload:
        flattened_phases: List[Phase] = []
        for phase_group in items:
            flattened_phases.extend(phase_group)
        return Workload(phases=flattened_phases)
