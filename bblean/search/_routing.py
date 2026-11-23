"""Sparse quantized summary routing from the November 2026 search experiments.

The score is a heuristic, not an admissible bound on unvisited fingerprints.
"""

from __future__ import annotations

import typing as tp

import numpy as np
from numpy.typing import NDArray

from ._utils import POPCOUNT, native, tanimoto


def split_summaries(
    fps: NDArray[np.uint8], target_rows: int = 32
) -> NDArray[np.uint32]:
    """Deterministic farthest-first initialization and two majority-centroid passes.

    The target controls the number of centers, not a hard population bound.
    Identical fingerprints can remain in one summary, as in the measured code.
    """
    lib = native()
    if lib is not None:
        return lib.split_summaries(fps, target_rows)
    n = len(fps)
    if n <= target_rows:
        return np.zeros(n, np.uint32)
    k = (n + target_rows - 1) // target_rows
    sample_n = min(n, k * 64)
    sample = fps[np.arange(sample_n) * max(1, n // sample_n)]
    centers = np.empty((k, fps.shape[1]), np.uint8)
    centers[0] = fps[0]
    nearest = np.full(sample_n, -1.0)
    for c in range(1, k):
        nearest = np.maximum(
            nearest, tanimoto(sample, centers[c - 1], empty_similarity=1.0)
        )
        centers[c] = sample[np.argmin(nearest)]
    labels = np.zeros(n, np.uint32)
    for _ in range(2):
        best = np.full(n, -1.0)
        for c in range(k):
            scores = tanimoto(fps, centers[c], empty_similarity=1.0)
            mask = scores > best
            labels[mask] = c
            best[mask] = scores[mask]
        for c in range(k):
            chosen = fps[labels == c]
            if len(chosen):
                counts = np.unpackbits(chosen, axis=1, bitorder="big").sum(axis=0)
                centers[c] = np.packbits(2 * counts >= len(chosen), bitorder="big")
    return labels


def quantization() -> tuple[NDArray[np.float64], NDArray[np.float32]]:
    edges = np.linspace(np.sqrt(0.05), 1.0, 8) ** 2
    centers = ((edges[:-1] + edges[1:]) / 2).astype(np.float32)
    centers[-1] = 1.0
    return edges, centers


def encode_gaps(units: tp.Any, previous: int = 0) -> NDArray[np.uint16]:
    # NumPy's stubs resolve diff(int64, prepend=int) to a timedelta overload
    gaps = tp.cast(
        NDArray[np.int64], np.diff(np.asarray(units, np.int64), prepend=previous)
    )
    if np.any(gaps < 0) or np.any(gaps > np.iinfo(np.uint32).max):
        raise ValueError("Posting IDs must be increasing uint32 values")
    big = gaps >= 65535
    width = np.where(big, 3, 1)
    at = np.r_[0, np.cumsum(width)[:-1]]
    out = np.zeros(int(width.sum()), np.uint16)
    out[at[~big]] = gaps[~big]
    out[at[big]] = 65535
    out[at[big] + 1] = gaps[big] & 65535
    out[at[big] + 2] = gaps[big] >> 16
    return out


def decode_gaps(words: NDArray[np.uint16]) -> NDArray[np.uint64]:
    out: list[int] = []
    at = unit = 0
    while at < len(words):
        gap = int(words[at])
        at += 1
        if gap == 65535:
            if at + 2 > len(words):
                raise ValueError("Truncated sparse posting escape")
            gap = int(words[at]) | (int(words[at + 1]) << 16)
            at += 2
        unit += gap
        if out and unit <= out[-1]:
            raise ValueError("Sparse posting IDs must be strictly increasing")
        out.append(unit)
    return np.asarray(out, dtype=np.uint64)


def route(
    query: NDArray[np.uint8], arrays: tp.Mapping[str, tp.Any], n_lists: int
) -> NDArray[np.float64]:
    bit_offsets, code_offsets = arrays["bit_offsets"], arrays["code_offsets"]
    deltas, codes = arrays["deltas"], arrays["codes"]
    means, rows, lists = (
        arrays["mean_pop"],
        arrays["summary_rows"],
        arrays["summary_lists"],
    )
    lut = quantization()[1]
    lib = native()
    if lib is not None:
        return lib.route(
            query,
            bit_offsets,
            code_offsets,
            deltas,
            codes,
            lut,
            means,
            rows,
            lists,
            n_lists,
        )
    mu, var = np.zeros(len(means), np.float32), np.zeros(len(means), np.float32)
    for bit in np.flatnonzero(np.unpackbits(query, bitorder="big")):
        ids = decode_gaps(deltas[int(bit_offsets[bit]) : int(bit_offsets[bit + 1])])
        code = codes[int(code_offsets[bit]) : int(code_offsets[bit + 1])]
        if (
            len(ids) != len(code)
            or np.any(ids >= len(means))
            or np.any(code >= len(lut))
        ):
            raise ValueError("Invalid sparse postings")
        freq = lut[code]
        mu[ids] += freq
        var[ids] += freq * (1 - freq)
    qp = int(POPCOUNT[query].sum())
    log_term = np.sqrt(2 * np.log(np.maximum(rows.astype(np.float64), 1))).astype(
        np.float32
    )
    intersection = np.minimum(
        mu.astype(np.float64)
        + (2 + 0.75 * log_term.astype(np.float64)) * np.sqrt(var.astype(np.float64)),
        qp,
    )
    union = qp + means.astype(np.float64) - intersection
    scores = np.divide(intersection, union, out=np.zeros(len(means)), where=union > 0)
    best = np.full(n_lists, -np.inf)
    np.maximum.at(best, lists, scores)
    return best
