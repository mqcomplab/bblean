"""Representation validation and optional-dependency boundaries."""

from __future__ import annotations

import hashlib
from numbers import Integral
from pathlib import Path
import types
import typing as tp

import numpy as np
from numpy.typing import NDArray

POPCOUNT = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)


def positive_int(value: tp.Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def integer_ids(values: tp.Any, name: str) -> NDArray[np.integer]:
    """Validate IDs without float promotion of mixed small/large Python ints."""
    if isinstance(values, np.ndarray):
        a = values
        if (
            a.ndim != 1
            or a.dtype.kind not in "iu"
            or a.dtype.itemsize > 8
            or np.any(a < 0)
        ):
            raise ValueError(f"{name} must be a 1D array of nonnegative uint64 IDs")
        return a
    try:
        items = list(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be an iterable of integer IDs") from exc
    if any(
        isinstance(v, bool)
        or not isinstance(v, Integral)
        or not 0 <= int(v) <= 2**64 - 1
        for v in items
    ):
        raise ValueError(f"{name} must contain nonnegative uint64 IDs")
    return np.array(items, dtype=np.uint64)


def packed(array: tp.Any, n_features: int, *, query: bool = False) -> NDArray[np.uint8]:
    a = np.asarray(array)
    ndim = 1 if query else 2
    if a.ndim != ndim or a.dtype != np.uint8:
        raise ValueError(f"Packed fingerprints must be {ndim}D uint8 arrays")
    width = (n_features + 7) // 8
    if a.shape[-1] != width:
        raise ValueError(f"Expected {width} packed bytes for {n_features} features")
    if n_features % 8 and np.any(a[..., -1] & ((1 << (8 - n_features % 8)) - 1)):
        raise ValueError("Unused low padding bits must be zero (big bit order)")
    return np.ascontiguousarray(a)


def prepare_query(
    query: tp.Any, n_features: int, input_is_packed: bool
) -> NDArray[np.uint8]:
    if input_is_packed:
        return packed(query, n_features, query=True)
    a = np.asarray(query)
    if a.ndim != 1 or len(a) != n_features or not np.all((a == 0) | (a == 1)):
        raise ValueError(f"Expected a binary query with {n_features} features")
    return np.packbits(a.astype(np.uint8), bitorder="big")


def zstandard() -> types.ModuleType:
    try:
        import zstandard as zstd
    except ImportError as exc:
        raise ImportError(
            "Compressed search requires pip install 'bblean[search]'"
        ) from exc
    return zstd


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tanimoto(
    fps: NDArray[np.uint8],
    query: NDArray[np.uint8],
    *,
    empty_similarity: float = 0.0,
) -> NDArray[np.float64]:
    inter = POPCOUNT[np.bitwise_and(fps, query)].sum(axis=1, dtype=np.uint64)
    union = POPCOUNT[np.bitwise_or(fps, query)].sum(axis=1, dtype=np.uint64)
    return np.divide(
        inter,
        union,
        out=np.full(len(fps), empty_similarity, dtype=np.float64),
        where=union != 0,
    )


def native() -> tp.Any:
    # Compiled by the ordinary package build, never at import/query time.
    from bblean.utils import _env_var_is_on

    if _env_var_is_on("BITBIRCH_NO_EXTENSIONS"):
        return None
    try:
        # Compiled extension, has no stubs (as with bblean._cpp_similarity)
        from bblean import _cpp_search  # type: ignore[attr-defined]
    except ImportError:
        return None
    return _cpp_search
