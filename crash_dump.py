"""Crash guard and diagnostic dump handlers for trace generation."""

import json
import os
import platform
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import numpy as np

from models import QueryOp, Workload, WriteOp, describe_op
from trace_generator import TraceGenerator, OpContext
from trace_output import writer_for

APP_DIR_NAME = "workload-generator"
CRASH_DIR_NAME = "crash-dumps"


def default_crash_dump_root() -> Path:
    """Returns OS-specific local data directory for crash dumps."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local"
    else:
        base = os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share"
    return Path(base) / APP_DIR_NAME / CRASH_DIR_NAME


def _count_data_rows(path: Path) -> int:
    """Counts CSV data rows in 1MB chunks."""
    with open(path, "rb") as f:
        lines = sum(buf.count(b"\n") for buf in iter(lambda: f.read(1 << 20), b""))
    return lines - 1


class TraceCrashError(Exception):
    """Raised when trace generation fails, carrying error context and dump path."""

    def __init__(
        self,
        error: BaseException,
        location: str,
        crash_dir: Optional[Path] = None,
        dump_error: Optional[BaseException] = None,
    ):
        super().__init__(f"{location}: {error}")
        self.error = error
        self.location = location
        self.crash_dir = crash_dir
        self.dump_error = dump_error


class TraceCrashGuard:
    """Guards trace generation and dumps diagnostics on failure."""

    def __init__(
        self,
        generator: TraceGenerator,
        workload: Workload,
        dump_root: Optional[Union[str, Path]] = None,
    ):
        self.generator = generator
        self.workload = workload
        self.dump_root = Path(dump_root) if dump_root is not None else default_crash_dump_root()
        self.phase_id: Optional[int] = None
        self.op_index: Optional[int] = None
        self.current_op: Optional[Union[QueryOp, WriteOp]] = None
        generator.add_op_listener(self._track_op)

    def _track_op(self, context: OpContext) -> None:
        self.phase_id = context.phase_id
        self.op_index = context.op_index
        self.current_op = context.op

    def run_to_csv(self, output_path: Union[str, Path]) -> int:
        """Streams the guarded trace to file; dumps crash report and raises TraceCrashError on failure."""
        writer = writer_for(output_path)
        try:
            return writer(self.generator.generate_trace(self.workload), output_path)
        except Exception as e:
            location = self.crash_location()
            try:
                crash_dir = self.dump_crash(e, partial_trace_path=output_path)
            except Exception as dump_error:
                raise TraceCrashError(e, location, dump_error=dump_error) from dump_error
            raise TraceCrashError(e, location, crash_dir=crash_dir) from e

    def crash_location(self) -> str:
        """Returns human-readable description of current crash phase and op."""
        if self.current_op is None:
            return "unknown location (before the first operation)"
        return (
            f"phase {self.phase_id}, operation {self.op_index} "
            f"({describe_op(self.current_op)})"
        )

    def dump_crash(
        self,
        error: BaseException,
        partial_trace_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Writes the crash report and returns its directory. The partial
        trace is copied from the CSV the caller was streaming into (it is
        already on disk and closed by the failed run) instead of being
        duplicated in memory."""
        crash_dir = self.dump_root / datetime.now().strftime("crash-%Y%m%d-%H%M%S")
        crash_dir.mkdir(parents=True, exist_ok=True)

        (crash_dir / "stacktrace.txt").write_text(
            "".join(traceback.format_exception(type(error), error, error.__traceback__)),
            encoding="utf-8",
        )
        np.save(crash_dir / "active_keys.npy", self.generator.active_keys)

        artifacts = {
            "stacktrace": "stacktrace.txt",
            "active_keys": "active_keys.npy",
        }
        rows_emitted: Optional[int] = None
        if partial_trace_path is not None and Path(partial_trace_path).is_file():
            shutil.copy2(partial_trace_path, crash_dir / "partial_trace.csv")
            artifacts["partial_trace"] = "partial_trace.csv"
            rows_emitted = _count_data_rows(crash_dir / "partial_trace.csv")

        (crash_dir / "crash.json").write_text(
            json.dumps(
                {
                    "timestamp": datetime.now().isoformat(),
                    "error": {"type": type(error).__name__, "message": str(error)},
                    "crash_site": {
                        "phase_id": self.phase_id,
                        "op_index": self.op_index,
                        "operation": describe_op(self.current_op)
                        if self.current_op is not None
                        else None,
                    },
                    "state": {
                        "seed": self.generator.seed,
                        "rows_emitted_before_crash": rows_emitted,
                        "active_keys_count": int(len(self.generator.active_keys)),
                    },
                    "workload": self.workload.to_dict(),
                    "environment": {
                        "python": platform.python_version(),
                        "platform": platform.platform(),
                        "numpy": np.__version__,
                    },
                    "artifacts": artifacts,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return crash_dir
