"""Dataset loaders converting supported on-disk formats to 1-d uint64 key arrays."""
from pathlib import Path
from typing import Callable, Dict, Union

import logging
import numpy as np
import pandas as pd

AUTO_FORMAT = "auto"

_LOG = logging.getLogger("workload.dataset")

DatasetPath = Union[str, Path]


def _load_csv(path: Path, key_col: str) -> np.ndarray:
    df = pd.read_csv(path)
    if key_col not in df.columns:
        raise ValueError(f"key column '{key_col}' not found in {path}")
    if df.empty:
        raise ValueError(f"dataset '{path}' has no rows")
    return df[key_col].to_numpy()


def _load_sosd_binary(path: Path, key_col: str) -> np.ndarray:
    """Loads an SOSD binary file: one uint64 size header followed by uint64 keys."""
    del key_col
    with open(path, "rb") as f:
        header = np.fromfile(f, dtype=np.uint64, count=1)
        if header.size == 0:
            raise ValueError(f"dataset '{path}' is empty")
        total_keys = int(header[0])
        if total_keys == 0:
            raise ValueError(f"dataset '{path}' has no keys")
        keys = np.fromfile(f, dtype=np.uint64, count=total_keys)
    if len(keys) != total_keys:
        raise ValueError(
            f"dataset '{path}' is truncated: header declares {total_keys} keys, "
            f"file holds {len(keys)}"
        )
    return keys


LOADERS: Dict[str, Callable[[Path, str], np.ndarray]] = {
    "csv": _load_csv,
    "sosd": _load_sosd_binary,
}


def _normalize_keys(keys: np.ndarray, path: Path) -> np.ndarray:
    """Sorts keys and enforces uniqueness (rejecting duplicates to preserve rank geometry)."""
    keys = np.asarray(keys, dtype=np.uint64)
    if len(keys) < 2 or np.all(keys[:-1] < keys[1:]):
        return keys
    s_keys = np.sort(keys)
    duplicates = np.flatnonzero(s_keys[:-1] == s_keys[1:])
    if duplicates.size > 0:
        raise ValueError(
            f"dataset '{path}' contains {duplicates.size} duplicate keys "
            f"(e.g. {s_keys[duplicates[0]]}): keys must be unique"
        )
    _LOG.warning(
        "dataset '%s' is not sorted: sorting %s keys in memory", path, f"{len(s_keys):,}"
    )
    return s_keys


def detect_format(path: Path) -> str:
    """Detects input format by file extension first, then SOSD byte signature."""
    if path.suffix.lower() == ".csv":
        return "csv"
    size = path.stat().st_size
    if size >= 8:
        with open(path, "rb") as f:
            header = np.fromfile(f, dtype=np.uint64, count=1)
        if header.size == 1 and int(header[0]) > 0 and size == 8 * (int(header[0]) + 1):
            return "sosd"
    return "csv"


def load_dataset(
    dataset_path: DatasetPath,
    key_col: str = "key",
    dataset_format: str = AUTO_FORMAT,
) -> np.ndarray:
    """Loads a dataset as a sorted 1-d unique key array, auto-detecting the
    format unless an explicit one is given."""
    path = Path(dataset_path)
    if not path.is_file():
        raise FileNotFoundError(f"dataset not found: {path}")
    fmt = dataset_format if dataset_format != AUTO_FORMAT else detect_format(path)
    if fmt not in LOADERS:
        supported = ", ".join(sorted(LOADERS))
        raise ValueError(f"unsupported dataset format '{fmt}' (supported: {supported})")
    detection_note = fmt if dataset_format != AUTO_FORMAT else f"{fmt}, auto-detected"
    _LOG.info("Loading dataset '%s' (format: %s)", path, detection_note)
    keys = _normalize_keys(LOADERS[fmt](path, key_col), path)
    _LOG.info(
        "Dataset ready: %s unique sorted keys in [%s, %s]",
        f"{len(keys):,}", keys[0], keys[-1],
    )
    return keys
