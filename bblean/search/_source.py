"""Access packed arrays or .npy shards without concatenating them."""

from __future__ import annotations

from pathlib import Path
import typing as tp

import numpy as np
from numpy.typing import NDArray

from ._utils import packed, positive_int


class FingerprintSource:
    def __init__(
        self,
        source: tp.Any,
        *,
        input_is_packed: bool = True,
        n_features: int | None = None,
    ) -> None:
        self._owned: list[tp.Any] = []
        if isinstance(source, (str, Path)):
            source = [Path(source)]
        if (
            isinstance(source, (list, tuple))
            and source
            and all(isinstance(p, (str, Path)) for p in source)
        ):
            self.arrays = [
                np.load(p, mmap_mode="r", allow_pickle=False) for p in source
            ]
            self._owned = self.arrays.copy()
        else:
            self.arrays = [np.asarray(source)]
        self.input_is_packed = input_is_packed
        if any(a.ndim != 2 for a in self.arrays):
            raise ValueError("Fingerprints must be 2D arrays or .npy shards")
        widths = {a.shape[1] for a in self.arrays}
        if len(widths) != 1 or next(iter(widths)) == 0:
            raise ValueError("Fingerprint shards must have the same positive width")
        inferred = next(iter(widths)) * (8 if input_is_packed else 1)
        self.n_features = positive_int(
            inferred if n_features is None else n_features, "n_features"
        )
        expected_width = (
            (self.n_features + 7) // 8 if input_is_packed else self.n_features
        )
        if expected_width != next(iter(widths)):
            raise ValueError("n_features does not match the fingerprint width")
        if self.n_features > np.iinfo(np.uint32).max:
            raise ValueError("n_features must fit uint32 popcounts")
        if input_is_packed and any(a.dtype != np.uint8 for a in self.arrays):
            raise ValueError("Packed fingerprints must have dtype uint8")
        self.ends = np.cumsum([len(a) for a in self.arrays], dtype=np.int64)
        self.starts = np.r_[0, self.ends[:-1]]
        self.n = positive_int(int(self.ends[-1]), "number of fingerprints")
        self.width = (self.n_features + 7) // 8

    def _convert(self, a: NDArray[tp.Any]) -> NDArray[np.uint8]:
        if self.input_is_packed:
            return packed(a, self.n_features)
        if not np.all((a == 0) | (a == 1)):
            raise ValueError("Unpacked fingerprints must be binary")
        return np.packbits(a.astype(np.uint8), axis=1, bitorder="big")

    def take(self, rows: tp.Any) -> NDArray[np.uint8]:
        rows = np.asarray(rows, dtype=np.int64)
        if rows.ndim != 1 or np.any(rows < 0) or np.any(rows >= self.n):
            raise ValueError("Fingerprint row out of bounds")
        out = np.empty((len(rows), self.width), np.uint8)
        shards = np.searchsorted(self.ends, rows, side="right")
        for s in np.unique(shards):
            at = np.flatnonzero(shards == s)
            out[at] = self._convert(self.arrays[s][rows[at] - self.starts[s]])
        return out

    def batches(self, size: int) -> tp.Iterator[NDArray[np.uint8]]:
        for start in range(0, self.n, size):
            yield self.take(np.arange(start, min(start + size, self.n)))

    def close(self) -> None:
        for a in self._owned:
            a._mmap.close()
        self._owned.clear()
