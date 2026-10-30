"""Multi-format trace output dispatch and streaming writers."""
import logging
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Union

from trace_generator import TraceBatch

_LOG = logging.getLogger("workload.trace")
_PROGRESS_EVERY_ROWS = 1_000_000
# Bounds the transient join string so large batches do not balloon memory.
_WRITE_CHUNK_ROWS = 50_000
TRACE_COLUMNS = ("phase_id", "op_type", "key", "param")


def _write_csv(trace_iter: Iterable[List[TraceBatch]], output_path: Union[str, Path]) -> int:
    """Streams trace batches into a CSV file."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    milestone = _PROGRESS_EVERY_ROWS
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        f.write(",".join(TRACE_COLUMNS) + "\r\n")
        for op_batches in trace_iter:
            for batch in op_batches:
                keys = batch.keys
                if len(keys) == 0:
                    continue
                prefix = f"{batch.phase_id},{batch.op_type},"
                suffix = f",{batch.param}\r\n"
                for lo in range(0, len(keys), _WRITE_CHUNK_ROWS):
                    part = keys[lo:lo + _WRITE_CHUNK_ROWS].tolist()
                    f.write(prefix + (suffix + prefix).join(map(str, part)) + suffix)
                count += len(keys)
                if count >= milestone:
                    _LOG.info("Wrote %s operations", f"{count:,}")
                    milestone += _PROGRESS_EVERY_ROWS
    return count


WRITERS: Dict[str, Callable[[Iterable[List[TraceBatch]], Union[str, Path]], int]] = {
    "csv": _write_csv,
}


def writer_for(output_path: Union[str, Path]) -> Callable[[Iterable[List[TraceBatch]], Union[str, Path]], int]:
    """Resolves the writer function for the target file's extension."""
    path = Path(output_path)
    fmt = path.suffix.lower().lstrip(".")
    if fmt not in WRITERS:
        supported = ", ".join(f".{ext}" for ext in sorted(WRITERS))
        raise ValueError(f"unsupported trace output format '{path.suffix}' (supported: {supported})")
    return WRITERS[fmt]


def save_trace(trace_iter: Iterable[List[TraceBatch]], output_path: Union[str, Path]) -> int:
    """Streams a trace iterator to a file; returns the number of rows written."""
    return writer_for(output_path)(trace_iter, output_path)
