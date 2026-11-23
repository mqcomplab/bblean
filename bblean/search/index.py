"""Persistent cluster-routed binary fingerprint search on local storage or S3."""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
import typing as tp

import numpy as np
from numpy.typing import NDArray
import typing_extensions as tpx

from ._build import BLOCK, FORMAT, VERSION, build
from ._routing import route
from ._source import FingerprintSource
from ._storage import LocalStore, S3Store, plan_ranges, s3_client, s3_location
from ._utils import (
    integer_ids,
    native,
    positive_int,
    prepare_query,
    sha256,
    tanimoto,
    zstandard,
)
from .types import SearchResult, SearchStats

_ARRAYS = {
    "blocks": BLOCK,
    "list_rows": np.dtype("<u8"),
    "summary_rows": np.dtype("<u8"),
    "mean_pop": np.dtype("<f4"),
    "summary_lists": np.dtype("<u4"),
    "bit_offsets": np.dtype("<u8"),
    "code_offsets": np.dtype("<u8"),
}
_FILES = {"data.bin", "deltas.bin", "codes.bin", *(f"{name}.npy" for name in _ARRAYS)}


class SearchIndex:
    """An immutable index with approximate routing and exact candidate scores.

    Build from packed fingerprints, ``.npy`` shards, an existing BitBirch tree,
    or explicit zero-based row memberships. ``n_probe=None`` scans all lists
    and supplies an exhaustive control. Perfect (Tanimoto=1) matches are excluded
    by default (nearest-other-neighbor protocol); set ``exclude_identical=False``
    for identity lookup. Empty/empty similarity is zero, as in core BBLean and
    RDKit.
    """

    location: str
    path: Path
    metadata: dict[str, tp.Any]
    n_records: int
    n_features: int
    n_lists: int
    _arrays: dict[str, tp.Any]
    _cache: tempfile.TemporaryDirectory[str] | None
    _closed: bool
    _store: LocalStore | S3Store
    _list_block_offsets: NDArray[np.intp]

    @classmethod
    def build(
        cls,
        fingerprints: tp.Any,
        directory: str | os.PathLike[str],
        *,
        clusters: tp.Any = None,
        centrals: tp.Any = None,
        n_lists: int | None = None,
        ids: tp.Any = None,
        input_is_packed: bool = True,
        n_features: int | None = None,
        bitbirch_kwargs: dict[str, tp.Any] | None = None,
        batch_size: int = 65536,
        summary_size: int = 32,
        block_size: int = 256 * 1024,
    ) -> tpx.Self:
        """Build a new index without overwriting an existing destination.

        ``clusters`` is a fitted BitBirch object or an iterable of nonempty
        integer row arrays forming an exact partition. With ``clusters=None``,
        BitBirch is fitted in batches. Input shard order defines row IDs;
        optional ``ids`` replaces returned row positions with unique uint64 IDs.
        """
        batch_size = positive_int(batch_size, "batch_size")
        if str(directory).startswith("s3://"):
            raise ValueError(
                "Build locally, then explicitly upload the completed index"
            )
        if Path(directory).exists():
            raise FileExistsError(f"Index destination already exists: {directory}")
        source = FingerprintSource(
            fingerprints, input_is_packed=input_is_packed, n_features=n_features
        )
        try:
            if clusters is None:
                from bblean import BitBirch

                tree = BitBirch(**(bitbirch_kwargs or {}))
                for batch in source.batches(batch_size):
                    tree.fit(batch, input_is_packed=True, n_features=source.n_features)
                clusters = tree
            if hasattr(clusters, "get_cluster_mol_ids"):
                if clusters.num_fitted_fps != source.n:
                    raise ValueError(
                        "Tree and fingerprint source have different row counts"
                    )
                if centrals is None:
                    centrals = np.asarray(
                        clusters.get_centroids(sort=False, packed=True)
                    )
                clusters = clusters.get_cluster_mol_ids(sort=False)
            path = build(
                source,
                directory,
                clusters,
                n_lists=n_lists,
                centrals=centrals,
                ids=ids,
                batch_size=batch_size,
                summary_size=summary_size,
                block_size=block_size,
            )
        finally:
            source.close()
        return cls.open(path)

    @classmethod
    def from_bitbirch(
        cls,
        tree: tp.Any,
        fingerprints: tp.Any,
        directory: str | os.PathLike[str],
        **kwargs: tp.Any,
    ) -> tpx.Self:
        """Reuse a fitted tree without fitting or modifying it."""
        return cls.build(fingerprints, directory, clusters=tree, **kwargs)

    @classmethod
    def from_clusters(
        cls,
        fingerprints: tp.Any,
        members: tp.Iterable[tp.Any],
        directory: str | os.PathLike[str],
        **kwargs: tp.Any,
    ) -> tpx.Self:
        """Reuse saved zero-based memberships without loading pickle files."""
        return cls.build(fingerprints, directory, clusters=members, **kwargs)

    @classmethod
    def open(
        cls,
        directory: str | os.PathLike[str],
        *,
        client: tp.Any = None,
        cache_dir: str | os.PathLike[str] | None = None,
        verify: bool = True,
    ) -> tpx.Self:
        """Open local storage or an S3 prefix without downloading its FP payload.

        S3 routing metadata is cached locally under a manifest-content hash;
        payloads are read with range GETs during queries and are not cached.
        Loading/verification of routing metadata is initialization, not query time.
        ``verify`` checks metadata hashes; :meth:`verify` also hashes the local
        fingerprint payload when an end-to-end integrity check is desired.
        """
        self = cls.__new__(cls)
        self._arrays = {}
        self._cache = None
        self._closed = False
        self.location = str(directory)
        try:
            if str(directory).startswith("s3://"):
                self._store = S3Store(directory, client)
                raw = self._store.read_file("manifest.json")
                if len(raw) > 1024 * 1024:
                    raise ValueError("Search manifest is too large")
                if cache_dir is None:
                    self._cache = tempfile.TemporaryDirectory(prefix="bblean-search-")
                    cache_dir = self._cache.name
                self.path = Path(cache_dir) / hashlib.sha256(raw).hexdigest()
                self.path.mkdir(parents=True, exist_ok=True)
                self.metadata = json.loads(raw)
                self._check_manifest()
                for name, info in self.metadata["files"].items():
                    if name == "data.bin":
                        continue
                    dest = self.path / name
                    if (
                        dest.is_file()
                        and dest.stat().st_size == info["bytes"]
                        and sha256(dest) == info["sha256"]
                    ):
                        continue
                    # Independent temporary files make concurrent cache fills safe.
                    with tempfile.NamedTemporaryFile(dir=self.path, delete=False) as f:
                        tmp = Path(f.name)
                    try:
                        self._store.download_file(name, tmp, info["bytes"])
                        if sha256(tmp) != info["sha256"]:
                            raise ValueError(f"Corrupt S3 search metadata: {name}")
                        tmp.replace(dest)
                    finally:
                        tmp.unlink(missing_ok=True)
                (self.path / "manifest.json").write_bytes(raw)
            else:
                self.path = Path(directory).absolute()
                self._store = LocalStore(self.path)
                self.metadata = json.loads((self.path / "manifest.json").read_text())
                self._check_manifest()
            for name, info in self.metadata["files"].items():
                if name == "data.bin" and isinstance(self._store, S3Store):
                    continue
                p = self.path / name
                if not p.is_file() or p.stat().st_size != info["bytes"]:
                    raise ValueError(f"Missing or truncated index file: {name}")
                if verify and name != "data.bin" and sha256(p) != info["sha256"]:
                    raise ValueError(f"Index metadata checksum mismatch: {name}")
            for name in _ARRAYS:
                self._arrays[name] = np.load(
                    self.path / f"{name}.npy", mmap_mode="r", allow_pickle=False
                )
            for name, dtype in [("deltas", "<u2"), ("codes", "u1")]:
                p = self.path / f"{name}.bin"
                self._arrays[name] = (
                    np.memmap(p, dtype=dtype, mode="r")
                    if p.stat().st_size
                    else np.empty(0, dtype)
                )
            self.n_records = self.metadata["n_records"]
            self.n_features = self.metadata["n_features"]
            self.n_lists = self.metadata["n_lists"]
            self._validate_arrays()
            return self
        except Exception:
            self.close()
            raise

    def _check_manifest(self) -> None:
        m = self.metadata
        if (
            m.get("format") != FORMAT
            or m.get("version") != VERSION
            or m.get("compression") != "zstd"
        ):
            raise ValueError("Unsupported search index format/version/codec")
        for name in ("n_records", "n_features", "n_lists"):
            positive_int(m.get(name), name)
        if m["n_features"] > np.iinfo(np.uint32).max:
            raise ValueError("n_features must fit uint32 popcounts")
        if m["n_lists"] > min(m["n_records"], np.iinfo(np.uint32).max):
            raise ValueError("Invalid number of search lists")
        if set(m.get("files", {})) != _FILES:
            raise ValueError("Invalid search index file inventory")
        for info in m["files"].values():
            if (
                not isinstance(info.get("bytes"), int)
                or info["bytes"] < 0
                or not isinstance(info.get("sha256"), str)
                or len(info["sha256"]) != 64
            ):
                raise ValueError("Invalid file size/checksum in search manifest")

    def _validate_arrays(self) -> None:
        a = self._arrays
        for name, dtype in _ARRAYS.items():
            if a[name].ndim != 1 or a[name].dtype != dtype:
                raise ValueError(f"Invalid search array: {name}")
        b = a["blocks"]
        if not len(b) or np.any(b["rows"] == 0) or np.any(b["length"] == 0):
            raise ValueError("Empty or invalid block directory")
        if (
            np.any(b["list"] >= self.n_lists)
            or np.any(b["list"][1:] < b["list"][:-1])
            or int(b["offset"][0]) != 0
            or np.any(b["offset"][1:] != b["offset"][:-1] + b["length"][:-1])
            or int(b["offset"][-1]) + int(b["length"][-1])
            != self.metadata["files"]["data.bin"]["bytes"]
            or np.any(b["length"] != b["fp_bytes"] + b["id_bytes"] + b["pop_bytes"])
        ):
            raise ValueError("Invalid compressed block ranges")
        list_rows = np.zeros(self.n_lists, np.uint64)
        np.add.at(list_rows, b["list"], b["rows"])
        if (
            not np.array_equal(list_rows, a["list_rows"])
            or int(list_rows.sum()) != self.n_records
            or np.any(list_rows == 0)
        ):
            raise ValueError("Invalid search list populations")
        count = len(a["summary_rows"])
        if (
            not count
            or len(a["mean_pop"]) != count
            or len(a["summary_lists"]) != count
            or np.any(a["summary_rows"] == 0)
            or np.any(a["summary_lists"] >= self.n_lists)
            or np.any(~np.isfinite(a["mean_pop"]))
            or np.any(a["mean_pop"] < 0)
            or np.any(a["mean_pop"] > self.n_features)
        ):
            raise ValueError("Invalid routing summaries")
        summary_counts = np.zeros(self.n_lists, np.uint64)
        np.add.at(summary_counts, a["summary_lists"], a["summary_rows"])
        if not np.array_equal(summary_counts, list_rows):
            raise ValueError("Summary/list populations disagree")
        for name, final in [
            ("bit_offsets", len(a["deltas"])),
            ("code_offsets", len(a["codes"])),
        ]:
            arr = a[name]
            if (
                len(arr) != self.n_features + 1
                or arr[0] != 0
                or arr[-1] != final
                or np.any(arr[1:] < arr[:-1])
            ):
                raise ValueError(f"Invalid sparse offsets: {name}")
        self._list_block_offsets = np.searchsorted(
            b["list"], np.arange(self.n_lists + 1)
        )

    def search(
        self,
        query: tp.Any,
        k: int = 10,
        n_probe: int | None = 1,
        *,
        threshold: float = 0.0,
        exclude_identical: bool = True,
        exclude_ids: tp.Iterable[int] = (),
        input_is_packed: bool = True,
        io_workers: int = 8,
    ) -> list[SearchResult]:
        """Return exact scores sorted by decreasing score, then increasing ID.

        Only selected lists are searched; ``threshold`` filters their results,
        not the whole library. Probe all lists with ``n_probe=None`` for exhaustive
        retrieval. Results can contain fewer than ``k`` eligible records.
        """
        return self.search_with_stats(
            query,
            k,
            n_probe,
            threshold=threshold,
            exclude_identical=exclude_identical,
            exclude_ids=exclude_ids,
            input_is_packed=input_is_packed,
            io_workers=io_workers,
        )[0]

    def search_with_stats(
        self,
        query: tp.Any,
        k: int = 10,
        n_probe: int | None = 1,
        *,
        threshold: float = 0.0,
        exclude_identical: bool = True,
        exclude_ids: tp.Iterable[int] = (),
        input_is_packed: bool = True,
        io_workers: int = 8,
    ) -> tuple[list[SearchResult], SearchStats]:
        """Return ``(results, stats)`` without shared per-query mutable state."""
        if self._closed:
            raise RuntimeError("Search index is closed")
        k = positive_int(k, "k")
        io_workers = positive_int(io_workers, "io_workers")
        n_probe = (
            self.n_lists
            if n_probe is None
            else min(positive_int(n_probe, "n_probe"), self.n_lists)
        )
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("threshold must be finite and between 0 and 1")
        excluded = integer_ids(exclude_ids, "exclude_ids").astype(np.uint64, copy=False)
        query = prepare_query(query, self.n_features, input_is_packed)
        started = time.perf_counter()
        stats = SearchStats(lists_probed=n_probe)
        routing_started = time.perf_counter()
        if n_probe == self.n_lists:
            lists = np.arange(self.n_lists)
        else:
            scores = route(query, self._arrays, self.n_lists)
            lists = np.lexsort((np.arange(self.n_lists), -scores))[:n_probe]
        stats.routing_seconds = time.perf_counter() - routing_started
        # Directory slices avoid a database-wide boolean block mask per query.
        blocks = self._arrays["blocks"]
        chosen = np.concatenate(
            [
                blocks[self._list_block_offsets[i] : self._list_block_offsets[i + 1]]
                for i in np.sort(lists)
            ]
        )
        ranges = iter(plan_ranges(chosen, row_bytes=(self.n_features + 7) // 8 + 12))
        best_ids, best_scores = np.empty(0, np.uint64), np.empty(0, np.float64)
        kernel = native()

        def consume(item: tp.Any) -> None:
            nonlocal best_ids, best_scores
            decoded, read_stats, decode_seconds = item
            for name in (
                "read_requests",
                "s3_get_attempts",
                "retries",
                "fetched_bytes",
                "read_seconds",
            ):
                setattr(stats, name, getattr(stats, name) + getattr(read_stats, name))
            stats.fetched_bytes_lower_bound |= read_stats.fetched_bytes_lower_bound
            stats.decode_seconds += decode_seconds
            stats.blocks_read += len(decoded)
            for fps, ids, pops in decoded:
                t = time.perf_counter()
                if len(excluded):
                    keep = ~np.isin(ids, excluded)
                    fps, ids, pops = fps[keep], ids[keep], pops[keep]
                stats.candidates += len(ids)
                if kernel is not None:
                    hit_ids, hit_scores = kernel.topk(
                        fps,
                        ids,
                        pops,
                        query,
                        min(k, len(ids)),
                        threshold,
                        exclude_identical,
                    )
                else:
                    scores = tanimoto(fps, query)
                    keep = (scores >= threshold) & (
                        (scores < 1) if exclude_identical else True
                    )
                    hit_ids, hit_scores = ids[keep], scores[keep]
                    ix = np.lexsort((hit_ids, -hit_scores))[:k]
                    hit_ids, hit_scores = hit_ids[ix], hit_scores[ix]
                merged_ids = np.concatenate([best_ids, hit_ids])
                merged_scores = np.concatenate([best_scores, hit_scores])
                ix = np.lexsort((merged_ids, -merged_scores))[:k]
                best_ids, best_scores = merged_ids[ix], merged_scores[ix]
                stats.score_seconds += time.perf_counter() - t

        task: tuple[int, int, tp.Any] | None
        if io_workers == 1:
            for task in ranges:
                consume(self._read_decode(task))
        else:
            with ThreadPoolExecutor(max_workers=io_workers) as pool:
                pending: deque[Future[tp.Any]] = deque()
                for _ in range(io_workers):
                    task = next(ranges, None)
                    if task is None:
                        break
                    pending.append(pool.submit(self._read_decode, task))
                while pending:
                    consume(pending.popleft().result())
                    task = next(ranges, None)
                    if task is not None:
                        pending.append(pool.submit(self._read_decode, task))
        stats.total_seconds = time.perf_counter() - started
        return [
            SearchResult(int(i), float(s)) for i, s in zip(best_ids, best_scores)
        ], stats

    def _read_decode(
        self, task: tuple[int, int, tp.Any]
    ) -> tuple[list[tp.Any], SearchStats, float]:
        start, length, blocks = task
        t = time.perf_counter()
        data, stats = self._store.read(start, length)
        stats.read_seconds = time.perf_counter() - t
        t = time.perf_counter()
        decoder = zstandard().ZstdDecompressor()
        width = (self.n_features + 7) // 8
        decoded = []
        for b in blocks:
            pos, rows = int(b["offset"]) - start, int(b["rows"])
            columns = []
            for name, dtype, size in [
                ("fp_bytes", "u1", width),
                ("id_bytes", "<u8", 1),
                ("pop_bytes", "<u4", 1),
            ]:
                end = pos + int(b[name])
                raw = decoder.decompress(
                    data[pos:end],
                    max_output_size=rows * np.dtype(dtype).itemsize * size,
                )
                if len(raw) != rows * np.dtype(dtype).itemsize * size:
                    raise ValueError("Decoded search block has invalid length")
                columns.append(np.frombuffer(raw, dtype=dtype))
                pos = end
            decoded.append((columns[0].reshape(rows, width), columns[1], columns[2]))
        return decoded, stats, time.perf_counter() - t

    def verify(self) -> None:
        """Verify every local file including the full compressed payload."""
        if self._closed:
            raise RuntimeError("Search index is closed")
        if isinstance(self._store, S3Store):
            raise ValueError("Full payload verification requires a local index")
        for name, info in self.metadata["files"].items():
            if sha256(self.path / name) != info["sha256"]:
                raise ValueError(f"Index checksum mismatch: {name}")

    def upload(self, uri: str, *, client: tp.Any = None) -> None:
        """Explicitly upload a local index to a new S3 prefix, manifest last.

        Uses boto3's managed multipart upload for payloads larger than 5 GB.
        Existing published manifests are refused; use a new prefix per version.
        """
        self.verify()
        bucket, prefix = s3_location(uri)
        client = s3_client(client)
        key = lambda name: f"{prefix}/{name}" if prefix else name
        try:
            client.head_object(Bucket=bucket, Key=key("manifest.json"))
        except Exception as exc:
            code = getattr(exc, "response", {}).get("Error", {}).get("Code")
            if not isinstance(exc, FileNotFoundError) and code not in (
                "404",
                "NoSuchKey",
                "NotFound",
            ):
                raise
        else:
            raise FileExistsError("S3 index prefix already has a published manifest")
        for name in [*self.metadata["files"], "manifest.json"]:
            client.upload_file(str(self.path / name), bucket, key(name))

    def close(self) -> None:
        """Release metadata mappings and any temporary S3 metadata cache."""
        if getattr(self, "_closed", True):
            return
        self._closed = True
        for a in self._arrays.values():
            mapping = getattr(a, "_mmap", None)
            if mapping is not None:
                mapping.close()
        self._arrays.clear()
        if self._cache is not None:
            self._cache.cleanup()

    def __enter__(self) -> tpx.Self:
        if self._closed:
            raise RuntimeError("Search index is closed")
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
