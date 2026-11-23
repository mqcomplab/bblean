"""Transactional index construction with bounded fingerprint working batches.

Membership/permutation metadata are O(N); raw fingerprints are never concatenated
across input shards. Sparse posting construction uses bounded summary batches.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile
from contextlib import ExitStack
import typing as tp

import numpy as np
from numpy.typing import NDArray

from ._routing import encode_gaps, quantization, split_summaries
from ._source import FingerprintSource
from ._utils import (
    POPCOUNT,
    integer_ids,
    packed,
    positive_int,
    sha256,
    tanimoto,
    zstandard,
)

FORMAT = "bblean-search"
VERSION = 1
BLOCK = np.dtype(
    [
        ("list", "<u4"),
        ("rows", "<u8"),
        ("offset", "<u8"),
        ("length", "<u8"),
        ("fp_bytes", "<u8"),
        ("id_bytes", "<u8"),
        ("pop_bytes", "<u8"),
    ]
)
ASSIGNMENT = np.dtype([("start", "<u8"), ("rows", "<u8"), ("list", "<u4")])
SUMMARY = np.dtype([("rows", "<u8"), ("mean_pop", "<f4"), ("list", "<u4")])


def centroid(
    source: FingerprintSource, rows: NDArray[np.integer], batch_size: int
) -> NDArray[np.uint8]:
    counts = np.zeros(source.width * 8, np.uint64)
    for at in range(0, len(rows), batch_size):
        counts += np.unpackbits(source.take(rows[at : at + batch_size]), axis=1).sum(
            axis=0, dtype=np.uint64
        )
    return np.packbits(2 * counts >= len(rows))


def split_capacity(
    source: FingerprintSource,
    rows: NDArray[np.integer],
    capacity: int,
    batch_size: int,
) -> tp.Iterator[NDArray[np.integer]]:
    """Experimental Tanimoto-difference bisection with deterministic row ties."""
    pending = [rows]
    while pending:
        part = pending.pop()
        pieces = math.ceil(len(part) / capacity)
        if pieces == 1:
            yield part
            continue
        sample = source.take(
            part[np.linspace(0, len(part) - 1, min(128, len(part)), dtype=int)]
        )
        center = np.packbits(
            2 * np.unpackbits(sample, axis=1).sum(axis=0) >= len(sample)
        )
        a = sample[np.argmin(tanimoto(sample, center))]
        b = sample[np.argmin(tanimoto(sample, a))]
        scores = np.empty(len(part), np.float64)
        for at in range(0, len(part), batch_size):
            fps = source.take(part[at : at + batch_size])
            scores[at : at + len(fps)] = tanimoto(fps, a) - tanimoto(fps, b)
        order = np.lexsort((part, -scores))
        cut = len(part) * (pieces // 2) // pieces
        pending.append(np.sort(part[order[cut:]]))
        pending.append(np.sort(part[order[:cut]]))


class SummaryWriter:
    def __init__(self, root: Path, work: Path, n_features: int) -> None:
        self.root, self.work = root, work
        self.n_features = n_features
        self.buffer: list[NDArray[np.uint8]] = []
        self.written = 0
        self.meta = (work / "summary.bin").open("wb")
        self.edges, self.lut = quantization()

    def add(self, fps: NDArray[np.uint8], list_id: int) -> None:
        frequencies = np.unpackbits(fps, axis=1)[:, : self.n_features].mean(axis=0)
        code = (
            np.clip(
                np.searchsorted(self.edges, frequencies, side="right") - 1,
                0,
                len(self.lut) - 1,
            ).astype(np.uint8)
            + 1
        )
        code[(frequencies < 0.05) | (frequencies <= 0)] = 0
        self.buffer.append(code)
        np.array(
            (len(fps), POPCOUNT[fps].sum(dtype=np.uint64) / len(fps), list_id),
            dtype=SUMMARY,
        ).tofile(self.meta)
        if len(self.buffer) >= max(1, (8 * 1024**2) // self.n_features):
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        if self.written + len(self.buffer) > np.iinfo(np.uint32).max:
            raise ValueError("The sparse format supports at most 2**32-1 summaries")
        codes = np.stack(self.buffer)
        dtype = np.dtype([("unit", "<u4"), ("code", "u1")])
        for bit in range(self.n_features):
            hit = np.flatnonzero(codes[:, bit])
            if not len(hit):
                continue
            entries = np.empty(len(hit), dtype)
            entries["unit"] = hit + self.written
            entries["code"] = codes[hit, bit] - 1
            with (self.work / f"bit-{bit}.bin").open("ab") as f:
                entries.tofile(f)
        self.written += len(self.buffer)
        self.buffer.clear()

    def finish(self) -> None:
        self.flush()
        self.meta.close()
        # NumPy stubs omit the private _mmap attribute that is closed below
        meta: tp.Any = np.memmap(self.work / "summary.bin", dtype=SUMMARY, mode="r")
        for name, field in [
            ("summary_rows", "rows"),
            ("mean_pop", "mean_pop"),
            ("summary_lists", "list"),
        ]:
            out = np.lib.format.open_memmap(
                self.root / f"{name}.npy",
                mode="w+",
                dtype=meta.dtype[field],
                shape=(len(meta),),
            )
            for at in range(0, len(meta), 65536):
                out[at : at + 65536] = meta[field][at : at + 65536]
            out.flush()
            del out
        meta._mmap.close()
        offsets, code_offsets = [0], [0]
        dtype = np.dtype([("unit", "<u4"), ("code", "u1")])
        with (
            (self.root / "deltas.bin").open("wb") as df,
            (self.root / "codes.bin").open("wb") as cf,
        ):
            for bit in range(self.n_features):
                path = self.work / f"bit-{bit}.bin"
                previous = 0
                if path.exists():
                    with path.open("rb") as f:
                        while True:
                            entries = np.fromfile(f, dtype, count=65536)
                            if not len(entries):
                                break
                            words = encode_gaps(entries["unit"], previous=previous)
                            previous = int(entries["unit"][-1])
                            words.astype("<u2", copy=False).tofile(df)
                            entries["code"].tofile(cf)
                offsets.append(df.tell() // 2)
                code_offsets.append(cf.tell())
        np.save(self.root / "bit_offsets.npy", np.array(offsets, dtype="<u8"))
        np.save(self.root / "code_offsets.npy", np.array(code_offsets, dtype="<u8"))


def build(
    source: FingerprintSource,
    output: str | os.PathLike[str],
    clusters: tp.Iterable[tp.Any],
    *,
    n_lists: int | None = None,
    centrals: tp.Any = None,
    ids: tp.Any = None,
    batch_size: int = 65536,
    summary_size: int = 32,
    block_size: int = 256 * 1024,
) -> Path:
    batch_size = positive_int(batch_size, "batch_size")
    summary_size = positive_int(summary_size, "summary_size")
    block_size = positive_int(block_size, "block_size")
    requested_lists = min(
        source.n,
        positive_int(
            math.ceil(math.sqrt(source.n)) if n_lists is None else n_lists, "n_lists"
        ),
    )
    output = Path(output).absolute()
    if output.exists():
        raise FileExistsError(f"Index destination already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    compressor = zstandard().ZstdCompressor(level=1, write_checksum=True)
    if ids is not None:
        ids = integer_ids(ids, "ids")
        if len(ids) != source.n:
            raise ValueError("ids must contain one nonnegative integer per fingerprint")
    if centrals is not None:
        centrals = packed(centrals, source.n_features)
    unit_dtype = np.dtype(
        [("start", "<u8"), ("rows", "<u8"), ("rep", "u1", (source.width,))]
    )
    with (
        tempfile.TemporaryDirectory(
            prefix=f".{output.name}-", dir=output.parent
        ) as temp,
        ExitStack() as stack,
    ):

        def track(array: tp.Any) -> tp.Any:
            stack.callback(array._mmap.close)
            return array

        root = Path(temp) / "index"
        work = Path(temp) / "work"
        root.mkdir()
        work.mkdir()
        if ids is not None:
            ordered = track(
                np.lib.format.open_memmap(
                    work / "ids.npy", mode="w+", dtype="<u8", shape=(source.n,)
                )
            )
            ordered[:] = ids
            ordered.sort()
            for at in range(1, source.n, batch_size):
                if np.any(
                    ordered[at : at + batch_size]
                    == ordered[at - 1 : min(at + batch_size - 1, source.n - 1)]
                ):
                    raise ValueError("Record IDs must be unique")
            del ordered
        seen = track(
            np.memmap(work / "seen.bin", mode="w+", dtype=np.uint8, shape=(source.n,))
        )
        seen[:] = 0
        total = cluster_count = split_count = 0
        capacity = math.ceil(source.n / requested_lists)
        with (
            (work / "members.bin").open("wb") as mf,
            (work / "units.bin").open("wb") as uf,
        ):
            for ci, raw in enumerate(clusters):
                rows = np.asarray(raw)
                if rows.ndim != 1 or not len(rows) or rows.dtype.kind not in "iu":
                    raise ValueError("Clusters must be nonempty 1D integer row arrays")
                if np.any(rows < 0) or np.any(rows >= source.n):
                    raise ValueError("Cluster member row out of bounds")
                rows = np.sort(rows.astype(np.int64))
                for at in range(0, len(rows), batch_size):
                    chunk = rows[at : at + batch_size]
                    if np.any(seen[chunk]) or len(np.unique(chunk)) != len(chunk):
                        raise ValueError(
                            "Every fingerprint must occur in exactly one cluster"
                        )
                    seen[chunk] = 1
                if centrals is not None and ci >= len(centrals):
                    raise ValueError("One central fingerprint is required per cluster")
                split_count += int(len(rows) > capacity)
                for part in split_capacity(source, rows, capacity, batch_size):
                    rep = (
                        centrals[ci]
                        if centrals is not None and len(rows) <= capacity
                        else centroid(source, part, batch_size)
                    )
                    np.array((total, len(part), rep), dtype=unit_dtype).tofile(uf)
                    part.astype("<u8").tofile(mf)
                    total += len(part)
                cluster_count += 1
        if total != source.n or not cluster_count:
            raise ValueError("Clusters must cover every fingerprint exactly once")
        if centrals is not None and cluster_count != len(centrals):
            raise ValueError("One central fingerprint is required per cluster")
        seen._mmap.close()
        units = track(np.memmap(work / "units.bin", dtype=unit_dtype, mode="r"))
        members = track(np.memmap(work / "members.bin", dtype="<u8", mode="r"))
        k = min(requested_lists, len(units))
        sizes = units["rows"]
        order = np.lexsort((np.arange(len(units)), -sizes.astype(np.int64)))
        pool = order[: min(len(order), 4 * k)]
        reps = np.ascontiguousarray(units["rep"][pool])
        weights = np.sqrt(sizes[pool].astype(float))
        distance, available = np.ones(len(pool)), np.ones(len(pool), bool)
        seeds = []
        for _ in range(k):
            chosen = int(np.argmax(np.where(available, distance * weights, -1)))
            seeds.append(int(pool[chosen]))
            available[chosen] = False
            distance = np.minimum(distance, 1 - tanimoto(reps, reps[chosen]))
        centers = np.ascontiguousarray(units["rep"][seeds])
        loads = np.zeros(k, np.int64)
        lower, upper = math.floor(0.5 * source.n / k), math.ceil(1.5 * source.n / k)
        remaining = source.n
        seeded = np.zeros(len(units), bool)
        seeded[seeds] = True
        with (work / "assignments.bin").open("wb") as af:

            def assign(unit: tp.Any, start: int, count: int, label: int) -> None:
                nonlocal remaining
                np.array(
                    (int(unit["start"]) + start, count, label), dtype=ASSIGNMENT
                ).tofile(af)
                loads[label] += count
                remaining -= count

            for label, u in enumerate(seeds):
                assign(units[u], 0, int(sizes[u]), label)
            for u in order[~seeded[order]]:
                unit = units[u]
                count = int(unit["rows"])
                similarity = tanimoto(centers, unit["rep"])
                needs = np.maximum(lower - loads, 0)
                fits = (upper - loads >= count) & (
                    remaining - count >= needs.sum() - np.minimum(count, needs)
                )
                if fits.any():
                    assign(
                        unit, 0, count, int(np.argmax(np.where(fits, similarity, -1)))
                    )
                else:
                    used = 0
                    for label in np.lexsort((np.arange(k), -similarity)):
                        needs = np.maximum(lower - loads, 0)
                        excess = remaining - int(needs.sum())
                        take = min(
                            count - used,
                            int(upper - loads[label]),
                            int(excess + needs[label]),
                        )
                        if take > 0:
                            assign(unit, used, take, int(label))
                            used += take
                        if used == count:
                            break
                    if used != count:
                        raise RuntimeError("Coarse-list capacity assignment failed")
        if remaining or loads.min() < lower or loads.max() > upper:
            raise RuntimeError("Invalid coarse-list coverage or capacities")
        assignments = track(
            np.memmap(work / "assignments.bin", dtype=ASSIGNMENT, mode="r")
        )
        plan = np.argsort(assignments["list"], kind="stable")
        summaries = SummaryWriter(root, work, source.n_features)
        block_rows = max(1, min(batch_size, block_size // (source.width + 12)))
        buffer_fp = np.empty((block_rows, source.width), np.uint8)
        buffer_ids = np.empty(block_rows, "<u8")
        filled = 0
        blocks = []
        try:
            with (root / "data.bin").open("wb") as payload:

                def flush(label: int) -> None:
                    nonlocal filled
                    if not filled:
                        return
                    fps = buffer_fp[:filled]
                    pops = POPCOUNT[fps].sum(axis=1, dtype=np.uint32).astype("<u4")
                    frames = [
                        compressor.compress(x.tobytes())
                        for x in (fps, buffer_ids[:filled], pops)
                    ]
                    lengths = [len(f) for f in frames]
                    blocks.append(
                        (label, filled, payload.tell(), sum(lengths), *lengths)
                    )
                    for f in frames:
                        payload.write(f)
                    filled = 0

                last_label = 0
                for ai in plan:
                    assignment = assignments[ai]
                    label = int(assignment["list"])
                    if label != last_label:
                        flush(last_label)
                        last_label = label
                    rows = members[
                        int(assignment["start"]) : int(
                            assignment["start"] + assignment["rows"]
                        )
                    ]
                    # Units are bounded by coarse capacity; no database-wide FP matrix.
                    fps = source.take(rows)
                    labels = split_summaries(fps, summary_size)
                    for piece in np.unique(labels):
                        selected = np.flatnonzero(labels == piece)
                        piece_fps = fps[selected]
                        summaries.add(piece_fps, label)
                        at = 0
                        while at < len(selected):
                            count = min(len(selected) - at, block_rows - filled)
                            buffer_fp[filled : filled + count] = piece_fps[
                                at : at + count
                            ]
                            row_ids = rows[selected[at : at + count]]
                            buffer_ids[filled : filled + count] = (
                                row_ids if ids is None else ids[row_ids]
                            )
                            filled += count
                            at += count
                            if filled == block_rows:
                                flush(label)
                flush(last_label)
                payload.flush()
                os.fsync(payload.fileno())
            summaries.finish()
        finally:
            if not summaries.meta.closed:
                summaries.meta.close()
            units._mmap.close()
            members._mmap.close()
            assignments._mmap.close()
        np.save(root / "blocks.npy", np.array(blocks, dtype=BLOCK))
        np.save(root / "list_rows.npy", loads.astype("<u8"))
        manifest = dict(
            format=FORMAT,
            version=VERSION,
            n_records=source.n,
            n_features=source.n_features,
            n_lists=k,
            compression="zstd",
            summary_size=summary_size,
            block_size=block_size,
            original_clusters=cluster_count,
            split_clusters=split_count,
            routing="sparse-3bit-sqrt-theta0.05-max",
            exclusion="exclude_identical=True by default",
            files={
                p.name: dict(bytes=p.stat().st_size, sha256=sha256(p))
                for p in sorted(root.iterdir())
            },
        )
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if output.exists():
            raise FileExistsError(f"Index destination already exists: {output}")
        os.rename(root, output)
    return output
