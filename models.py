from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union
import json

DistributionType = Literal["gaussian", "zipf", "uniform"]
QueryType = Literal["point", "range", "knn"]
WriteOpType = Literal["insert", "update", "delete"]


@dataclass
class QueryOp:
    """Point, range, or knn query operation specification."""

    query_type: QueryType
    count: int
    percentile: Optional[float]
    distribution: DistributionType
    offset: Optional[int] = None
    k: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "type": "query",
            "query_type": self.query_type,
            "count": self.count,
            "percentile": self.percentile,
            "distribution": self.distribution,
        }
        if self.query_type == "range":
            d["offset"] = self.offset
        elif self.query_type == "knn":
            d["k"] = self.k
        return d


@dataclass
class WriteOp:
    operation: WriteOpType
    batch_size: int
    percentile: float
    distribution: DistributionType
    sorted: bool

    @property
    def op_type(self) -> str:
        return self.operation

    @property
    def percentile_loc(self) -> float:
        return self.percentile

    @property
    def sortedness(self) -> str:
        return "sorted" if self.sorted else "unsorted"

    def to_dict(self, target_feature: str = "key") -> Dict[str, Any]:
        return {
            "type": "write",
            "operation": self.operation,
            "target_feature": target_feature,
            "batch_size": self.batch_size,
            "percentile": self.percentile,
            "distribution": self.distribution,
            "sorted": self.sorted,
        }


Operation = Union[QueryOp, WriteOp]


def describe_op(op: Operation) -> str:
    """Single-line op description in DSL parameter order."""
    if isinstance(op, WriteOp):
        return (
            f"{op.operation} batch_size={op.batch_size} percentile={op.percentile} "
            f"distribution={op.distribution} sortedness={op.sortedness}"
        )
    desc = (
        f"{op.query_type} count={op.count} percentile={op.percentile} "
        f"distribution={op.distribution}"
    )
    if op.query_type == "range":
        desc += f" offset={op.offset}"
    elif op.query_type == "knn":
        desc += f" k={op.k}"
    return desc


@dataclass
class Phase:
    phase_id: int
    operations: List[Operation] = field(default_factory=list)

    def to_dict(self, target_feature: str = "key") -> Dict[str, Any]:
        return {
            "phase_id": self.phase_id,
            "operations": [
                op.to_dict(target_feature=target_feature) if isinstance(op, WriteOp) else op.to_dict()
                for op in self.operations
            ],
        }


@dataclass
class Workload:
    phases: List[Phase] = field(default_factory=list)

    def to_dict(self, target_feature: str = "key") -> Dict[str, Any]:
        return {
            "phases": [phase.to_dict(target_feature=target_feature) for phase in self.phases]
        }

    def to_json(self, indent: int = 2, target_feature: str = "key") -> str:
        return json.dumps(self.to_dict(target_feature=target_feature), indent=indent)
