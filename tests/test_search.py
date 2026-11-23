"""Persistent search correctness against independent exhaustive references.

Small deterministic fixtures exercise the algorithm and storage paths; these
tests do not reproduce the original experiments' large-scale timings or recall.
"""

import io
import json
import os
from pathlib import Path
import subprocess
import sys
import typing as tp

import numpy as np
from numpy.typing import NDArray
import pytest
from rdkit import DataStructs

pytest.importorskip("zstandard")

from bblean import BitBirch  # noqa: E402
from bblean.search import SearchIndex  # noqa: E402
from bblean.search import _routing  # noqa: E402
from bblean.search._storage import S3Store  # noqa: E402
from bblean.search._utils import POPCOUNT, native  # noqa: E402


def oracle(
    fps: NDArray[np.uint8],
    query: NDArray[np.uint8],
    ids: tp.Any = None,
    k: int = 10,
    exclude: bool = True,
    threshold: float = 0,
) -> list[tuple[int, float]]:
    """RDKit's binary Tanimoto with deterministic record-ID ties."""
    q = DataStructs.CreateFromBitString("".join(map(str, np.unpackbits(query))))
    sims = []
    for row in fps:
        fp = DataStructs.CreateFromBitString("".join(map(str, np.unpackbits(row))))
        sims.append(DataStructs.TanimotoSimilarity(q, fp))
    scores = np.array(sims)
    ids = np.arange(len(fps), dtype=np.uint64) if ids is None else ids
    keep = (scores >= threshold) & ((scores < 1) if exclude else True)
    ids, scores = ids[keep], scores[keep]
    order = np.lexsort((ids, -scores))[:k]
    return [(int(i), float(s)) for i, s in zip(ids[order], scores[order])]


def pairs(results: tp.Iterable[tp.Any]) -> list[tuple[int, float]]:
    return [(r.index, r.similarity) for r in results]


@pytest.fixture
def fps() -> NDArray[np.uint8]:
    a = np.random.default_rng(31).integers(0, 256, (127, 32), dtype=np.uint8)
    a[3] = a[0]
    a[4:6] = 0
    return a


@pytest.fixture
def index(tmp_path: Path, fps: NDArray[np.uint8]) -> tp.Iterator[SearchIndex]:
    with SearchIndex.from_clusters(
        fps,
        [np.arange(i, min(i + 30, len(fps))) for i in range(0, len(fps), 30)],
        tmp_path / "index",
        n_lists=8,
        block_size=512,
    ) as idx:
        yield idx


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("include_identical", [False, True])
def test_full_probe_matches_rdkit(
    index: SearchIndex,
    fps: NDArray[np.uint8],
    monkeypatch: pytest.MonkeyPatch,
    fallback: bool,
    include_identical: bool,
) -> None:
    if fallback:
        monkeypatch.setenv("BITBIRCH_NO_EXTENSIONS", "1")
    for query in (fps[0], fps[4], np.full(fps.shape[1], 1, np.uint8)):
        expected = oracle(fps, query, k=200, exclude=not include_identical)
        results, stats = index.search_with_stats(
            query,
            k=200,
            n_probe=None,
            exclude_identical=not include_identical,
            io_workers=2,
        )
        assert pairs(results) == expected
        assert stats.candidates == len(fps)
        assert stats.lists_probed == index.n_lists
        assert stats.fetched_bytes == (index.path / "data.bin").stat().st_size
        assert stats.s3_get_attempts == 0
        assert stats.blocks_read == len(index._arrays["blocks"])


def test_probe_recall_is_monotone(index: SearchIndex, fps: NDArray[np.uint8]) -> None:
    for query in fps[:4]:
        truth = {i for i, _ in oracle(fps, query)}
        recalls, work = [], []
        for n in range(1, index.n_lists + 1):
            results, stats = index.search_with_stats(query, n_probe=n, io_workers=1)
            recalls.append(len(truth.intersection(r.index for r in results)))
            work.append(stats.candidates)
            assert all(
                r.similarity == dict(oracle(fps, query, k=len(fps)))[r.index]
                for r in results
            )
        assert recalls == sorted(recalls)
        assert work == sorted(work)
        assert recalls[-1] == len(truth)


def test_strict_lists_even_when_ranges_include_unselected_bytes(
    index: SearchIndex, fps: NDArray[np.uint8], monkeypatch: pytest.MonkeyPatch
) -> None:
    scores = np.zeros(index.n_lists)
    scores[0], scores[-1] = 3, 2
    monkeypatch.setattr("bblean.search.index.route", lambda *args: scores)
    results, stats = index.search_with_stats(
        fps[10], k=len(fps), n_probe=2, io_workers=1, exclude_identical=False
    )
    expected: set[int] = set()
    for i in (0, index.n_lists - 1):
        b = index._arrays["blocks"]
        selected = b[b["list"] == i]
        for block in selected:
            decoded, _, _ = index._read_decode(
                (int(block["offset"]), int(block["length"]), np.array([block]))
            )
            expected.update(map(int, decoded[0][1]))
    assert {r.index for r in results} == expected
    assert stats.candidates == len(expected)


def test_reopen_and_verify(index: SearchIndex, fps: NDArray[np.uint8]) -> None:
    directory = index.path
    expected = pairs(index.search(fps[9], n_probe=4))
    index.close()
    with SearchIndex.open(directory) as reopened:
        reopened.verify()
        assert pairs(reopened.search(fps[9], n_probe=4)) == expected
    with pytest.raises(RuntimeError, match="closed"):
        reopened.search(fps[0])


def test_concurrent_queries_and_bounded_range_workers(
    index: SearchIndex, fps: NDArray[np.uint8], monkeypatch: pytest.MonkeyPatch
) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from bblean.search._storage import plan_ranges

    # Force separate range tasks even on this small fixture.
    monkeypatch.setattr(
        "bblean.search.index.plan_ranges",
        lambda blocks, **kwargs: plan_ranges(blocks, max_span=1, **kwargs),
    )

    def run(query: NDArray[np.uint8]) -> list[tuple[int, float]]:
        results, stats = index.search_with_stats(query, n_probe=None, io_workers=3)
        assert stats.read_requests == stats.blocks_read
        return pairs(results)

    with ThreadPoolExecutor(max_workers=3) as pool:
        actual = list(pool.map(run, fps[:6]))
    assert actual == [oracle(fps, query) for query in fps[:6]]


def test_custom_uint64_ids_and_exclusions(
    tmp_path: Path, fps: NDArray[np.uint8]
) -> None:
    ids = np.arange(len(fps), dtype=np.uint64)[::-1] + np.uint64(2**63 + 7)
    with SearchIndex.from_clusters(
        fps, [np.arange(len(fps))], tmp_path / "ids", ids=ids, n_lists=1, summary_size=9
    ) as idx:
        assert pairs(
            idx.search(fps[0], n_probe=None, exclude_identical=False)
        ) == oracle(fps, fps[0], ids, exclude=False)
        dropped = [int(ids[0]), int(ids[3])]
        out = idx.search(
            fps[0], n_probe=None, exclude_identical=False, exclude_ids=dropped
        )
        mask = ~np.isin(ids, dropped)
        assert pairs(out) == oracle(fps[mask], fps[0], ids[mask], exclude=False)
        assert all(r.id == r.index for r in out)


def test_mixed_small_and_large_python_integer_ids(tmp_path: Path) -> None:
    fps = np.array([[128], [192], [64]], dtype=np.uint8)
    ids = [2**64 - 1, 0, 2**63]
    with SearchIndex.from_clusters(
        fps, [[0, 1, 2]], tmp_path / "mixed-ids", n_lists=1, ids=ids
    ) as idx:
        hits = idx.search(
            fps[0], n_probe=None, exclude_identical=False, exclude_ids=[2**64 - 1, 0]
        )
        assert pairs(hits) == [(2**63, 0.0)]


def test_shards_unpacked_and_padding(tmp_path: Path) -> None:
    bits = np.random.default_rng(4).integers(0, 2, (33, 13), dtype=np.uint8)
    p = np.packbits(bits, axis=1)
    for name, array in [("a", bits[:10]), ("b", bits[10:])]:
        np.save(tmp_path / f"{name}.npy", array)
    with SearchIndex.from_clusters(
        [tmp_path / "a.npy", tmp_path / "b.npy"],
        [np.arange(33)],
        tmp_path / "shards",
        input_is_packed=False,
        n_features=13,
        block_size=64,
        batch_size=7,
    ) as idx:
        assert pairs(
            idx.search(bits[0], n_probe=None, input_is_packed=False)
        ) == oracle(p, p[0])
        with pytest.raises(ValueError, match="padding"):
            idx.search(np.array([0, 1], np.uint8))


def test_tree_reuse_does_not_refit_or_change_memberships(
    tmp_path: Path, fps: NDArray[np.uint8], monkeypatch: pytest.MonkeyPatch
) -> None:
    tree = BitBirch(threshold=0.5).fit(fps)
    before = [list(x) for x in tree.get_cluster_mol_ids(sort=False)]
    monkeypatch.setattr(
        BitBirch, "fit", lambda *args, **kwargs: pytest.fail("Must reuse tree")
    )
    with SearchIndex.from_bitbirch(tree, fps, tmp_path / "reuse") as idx:
        assert pairs(idx.search(fps[7], n_probe=None)) == oracle(fps, fps[7])
    assert before == tree.get_cluster_mol_ids(sort=False)


def test_build_clusters_new_fingerprints(
    tmp_path: Path, fps: NDArray[np.uint8]
) -> None:
    with SearchIndex.build(
        fps, tmp_path / "automatic", batch_size=19, bitbirch_kwargs={"threshold": 0.4}
    ) as idx:
        assert pairs(idx.search(fps[11], n_probe=None)) == oracle(fps, fps[11])
        assert idx.metadata["original_clusters"] > 0


@pytest.mark.parametrize("n,lists", [(1, 1), (6, 4), (9, 12)])
def test_zero_fingerprints_and_small_list_limits(
    tmp_path: Path, n: int, lists: int
) -> None:
    fps = np.zeros((n, 2), dtype=np.uint8)
    with SearchIndex.from_clusters(
        fps, [np.arange(n)], tmp_path / "small", n_lists=lists, block_size=1
    ) as idx:
        assert 1 <= idx.n_lists <= min(lists, n)
        assert not len(idx._arrays["codes"])
        assert pairs(idx.search(fps[0], k=n, n_probe=None)) == oracle(fps, fps[0], k=n)
        assert all(r.similarity == 0 for r in idx.search(fps[0]))


@pytest.mark.parametrize(
    "clusters",
    [
        [[0, 1, 1], [2, 3]],
        [[0, 1], [2]],
        [[-1], [0, 1, 2, 3]],
        [[0, 1], [2, 4]],
        [[], [0, 1, 2, 3]],
        [[0.0, 1.0], [2, 3]],
        [[0, 1, 2, 3, 0]],
    ],
)
def test_invalid_partition_is_transactional(
    tmp_path: Path, fps: NDArray[np.uint8], clusters: list[list[tp.Any]]
) -> None:
    output = tmp_path / "bad"
    with pytest.raises(ValueError):
        SearchIndex.from_clusters(fps[:4], clusters, output, batch_size=2)
    assert not output.exists()
    assert not list(tmp_path.glob(".bad-*"))


def test_reject_overwrite_and_duplicate_ids(
    index: SearchIndex, fps: NDArray[np.uint8], tmp_path: Path
) -> None:
    with pytest.raises(FileExistsError):
        SearchIndex.from_clusters(fps, [np.arange(len(fps))], index.path)
    with pytest.raises(ValueError, match="unique"):
        SearchIndex.from_clusters(
            fps,
            [np.arange(len(fps))],
            tmp_path / "duplicate-ids",
            ids=np.zeros(len(fps), dtype=np.uint64),
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(k=0),
        dict(k=1.5),
        dict(n_probe=0),
        dict(n_probe=True),
        dict(threshold=-1),
        dict(threshold=1.1),
        dict(threshold=float("nan")),
        dict(io_workers=0),
        dict(exclude_ids=[-1]),
    ],
)
def test_invalid_query_options(
    index: SearchIndex, fps: NDArray[np.uint8], kwargs: dict[str, tp.Any]
) -> None:
    with pytest.raises(ValueError):
        index.search(fps[0], **kwargs)


def test_threshold_and_ties(index: SearchIndex, fps: NDArray[np.uint8]) -> None:
    for threshold in (0, 0.3, 0.7, 1):
        assert pairs(
            index.search(
                fps[0], n_probe=None, threshold=threshold, exclude_identical=False
            )
        ) == oracle(fps, fps[0], exclude=False, threshold=threshold)


def test_corrupt_metadata_and_payload(index: SearchIndex) -> None:
    path = index.path
    index.close()
    meta = path / "mean_pop.npy"
    raw = meta.read_bytes()
    meta.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    with pytest.raises(ValueError, match="checksum"):
        SearchIndex.open(path)
    meta.write_bytes(raw)
    with SearchIndex.open(path) as idx:
        payload = path / "data.bin"
        original = payload.read_bytes()
        payload.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
        with pytest.raises(ValueError, match="checksum"):
            idx.verify()


def test_manifest_path_and_version_validation(index: SearchIndex) -> None:
    path = index.path
    index.close()
    manifest = path / "manifest.json"
    m = json.loads(manifest.read_text())
    m["version"] = 900
    manifest.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="Unsupported"):
        SearchIndex.open(path)
    m["version"] = 1
    m["files"]["../outside"] = dict(bytes=1, sha256="a" * 64)
    manifest.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="inventory"):
        SearchIndex.open(path)


class FakeS3:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.calls: list[tuple[str, str | None]] = []
        self.uploaded: list[str] = []
        self.bodies: list[io.BytesIO] = []
        self.interrupt = False

    def head_object(self, Bucket: str, Key: str) -> dict[str, int]:
        if (Bucket, Key) not in self.objects:
            raise FileNotFoundError(Key)
        return {"ContentLength": len(self.objects[Bucket, Key])}

    def upload_file(self, filename: str, bucket: str, key: str) -> None:
        self.objects[bucket, key] = Path(filename).read_bytes()
        self.uploaded.append(key)

    def get_object(self, Bucket: str, Key: str, **kwargs: tp.Any) -> dict[str, tp.Any]:
        self.calls.append((Key, kwargs.get("Range")))
        raw = self.objects[Bucket, Key]
        response: dict[str, tp.Any] = {}
        if "Range" in kwargs:
            start, end = map(int, kwargs["Range"].removeprefix("bytes=").split("-"))
            response["ContentRange"] = f"bytes {start}-{end}/{len(raw)}"
            raw = raw[start : end + 1]
        body = io.BytesIO(raw)
        if self.interrupt and "Range" in kwargs:
            self.interrupt = False

            class Interrupted(io.BytesIO):
                def read(self, n: int | None = -1) -> bytes:
                    if self.tell():
                        raise OSError("Interrupted body")
                    return super().read(min(-1 if n is None else n, 7))

            body = Interrupted(raw)
        self.bodies.append(body)
        response["Body"] = body
        return response


def test_s3_open_range_reads_retries_and_cache(
    index: SearchIndex, fps: NDArray[np.uint8], tmp_path: Path
) -> None:
    client = FakeS3()
    index.upload("s3://testing/index", client=client)
    assert client.uploaded[-1] == "index/manifest.json"
    with pytest.raises(FileExistsError):
        index.upload("s3://testing/index", client=client)
    with SearchIndex.open(
        "s3://testing/index", client=client, cache_dir=tmp_path / "cache"
    ) as cloud:
        assert not any(key.endswith("data.bin") for key, _ in client.calls)
        assert not (cloud.path / "data.bin").exists()
        client.interrupt = True
        results, stats = cloud.search_with_stats(fps[0], n_probe=None, io_workers=1)
        assert pairs(results) == oracle(fps, fps[0])
        assert stats.retries == 1
        assert stats.fetched_bytes_lower_bound
        assert stats.fetched_bytes == (index.path / "data.bin").stat().st_size
        payload_calls = [r for key, r in client.calls if key.endswith("data.bin")]
        assert str(payload_calls[1]).startswith("bytes=7-")
        assert stats.s3_get_attempts == len(payload_calls)
        cloud.search(fps[0], n_probe=None)
        assert len([r for key, r in client.calls if key.endswith("data.bin")]) > len(
            payload_calls
        )
    before = len(client.calls)
    with SearchIndex.open(
        "s3://testing/index", client=client, cache_dir=tmp_path / "cache"
    ):
        assert client.calls[before:] == [("index/manifest.json", None)]
    assert all(b.closed for b in client.bodies)


def test_s3_range_failure_never_returns_partial_results() -> None:
    client = FakeS3()
    client.objects["testing", "empty/data.bin"] = b""
    store = S3Store("s3://testing/empty", client=client, max_retries=1)
    with pytest.raises(EOFError):
        store.read(0, 10)
    assert len(client.calls) == 2 and all(b.closed for b in client.bodies)


def test_s3_sdk_retry_attempt_accounting(
    index: SearchIndex, fps: NDArray[np.uint8]
) -> None:
    class RetryingClient(FakeS3):
        def get_object(
            self, Bucket: str, Key: str, **kwargs: tp.Any
        ) -> dict[str, tp.Any]:
            response = super().get_object(Bucket=Bucket, Key=Key, **kwargs)
            response["ResponseMetadata"] = {"RetryAttempts": 2}
            return response

    client = RetryingClient()
    index.upload("s3://testing/sdk-retries", client=client)
    with SearchIndex.open("s3://testing/sdk-retries", client=client) as cloud:
        hits, stats = cloud.search_with_stats(fps[0], n_probe=None, io_workers=1)
        calls = sum(r is not None for _, r in client.calls)
        assert pairs(hits) == oracle(fps, fps[0])
        assert stats.s3_get_attempts == 3 * calls
        assert stats.retries == 2 * calls


def test_s3_service_retry_and_permission_error() -> None:
    from botocore.exceptions import ClientError

    class Client(FakeS3):
        error = "SlowDown"
        status = 503
        attempts = 0

        def get_object(
            self, Bucket: str, Key: str, **kwargs: tp.Any
        ) -> dict[str, tp.Any]:
            self.attempts += 1
            if self.attempts == 1:
                raise ClientError(
                    {
                        "Error": {"Code": self.error},
                        "ResponseMetadata": {"HTTPStatusCode": self.status},
                    },
                    "GetObject",
                )
            return super().get_object(Bucket=Bucket, Key=Key, **kwargs)

    client = Client()
    client.objects["testing", "service/data.bin"] = b"12345"
    store = S3Store("s3://testing/service", client=client)
    data, stats = store.read(0, 5)
    assert data == b"12345" and stats.retries == 1
    assert stats.s3_get_attempts == 2
    client.attempts = 0
    client.error, client.status = "AccessDenied", 403
    with pytest.raises(ClientError):
        store.read(0, 5)
    assert client.attempts == 1


def test_missing_extra_has_actionable_error(
    tmp_path: Path, fps: NDArray[np.uint8], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "zstandard", None)
    with pytest.raises(ImportError, match=r"bblean\[search\]"):
        SearchIndex.from_clusters(
            fps, [np.arange(len(fps))], tmp_path / "missing-extra"
        )


@pytest.mark.parametrize("width", [1, 3, 8, 21, 129, 256])
def test_native_and_fallback_agree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, width: int
) -> None:
    lib = native()
    if lib is None:
        if os.environ.get("BITBIRCH_CANT_SKIP_CPP_TESTS") == "1":
            pytest.fail("Search native extension was required")
        pytest.skip("Native extension is not installed")
    fps = np.random.default_rng(2).integers(0, 256, (95, width), dtype=np.uint8)
    labels = _routing.split_summaries(fps, 10)
    monkeypatch.setenv("BITBIRCH_NO_EXTENSIONS", "1")
    np.testing.assert_array_equal(labels, _routing.split_summaries(fps, 10))
    with SearchIndex.from_clusters(
        fps, [np.arange(len(fps))], tmp_path / "compare", n_lists=5, summary_size=7
    ) as idx:
        python_route = _routing.route(fps[0], idx._arrays, idx.n_lists)
        python_results = pairs(idx.search(fps[0], n_probe=2))
        monkeypatch.delenv("BITBIRCH_NO_EXTENSIONS")
        np.testing.assert_array_equal(
            python_route, _routing.route(fps[0], idx._arrays, idx.n_lists)
        )
        assert pairs(idx.search(fps[0], n_probe=2)) == python_results
        pops = POPCOUNT[fps].sum(axis=1, dtype=np.uint32)
        ids = np.arange(len(fps), dtype=np.uint64)[::-1]
        got_ids, got_scores = lib.topk(fps, ids, pops, fps[0], 13, 0.2, True)
        assert list(zip(map(int, got_ids), map(float, got_scores))) == oracle(
            fps, fps[0], ids, 13, threshold=0.2
        )


def test_sparse_escape_and_invalid_postings() -> None:
    units = np.array([0, 1, 65535, 65536, 2**32 - 1], np.uint64)
    np.testing.assert_array_equal(
        _routing.decode_gaps(_routing.encode_gaps(units)), units
    )
    np.testing.assert_array_equal(
        _routing.encode_gaps(np.array([70000]), previous=69999), [1]
    )
    with pytest.raises(ValueError, match="escape"):
        _routing.decode_gaps(np.array([65535, 0], np.uint16))


def test_optional_dependencies_not_imported_with_core() -> None:
    code = """
import sys
class Block:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in ('zstandard', 'boto3'):
            raise AssertionError('Optional dependency imported eagerly: ' + fullname)
sys.meta_path.insert(0, Block())
import bblean
import bblean.search
"""
    subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )


def test_cli_build_reusing_assignments_and_query(
    tmp_path: Path, fps: NDArray[np.uint8]
) -> None:
    np.save(tmp_path / "fps.npy", fps)
    np.save(tmp_path / "q.npy", fps[0])
    np.save(tmp_path / "labels.npy", np.arange(len(fps)) // 15)
    command = [sys.executable, "-m", "bblean.search.cli"]
    build = subprocess.run(
        [
            *command,
            "build",
            str(tmp_path / "fps.npy"),
            "-o",
            str(tmp_path / "cli"),
            "--assignments",
            str(tmp_path / "labels.npy"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(build.stdout)["records"] == len(fps)
    query = subprocess.run(
        [
            *command,
            "query",
            str(tmp_path / "cli"),
            str(tmp_path / "q.npy"),
            "--exhaustive",
            "--stats",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    out = json.loads(query.stdout)
    assert [(x["index"], x["similarity"]) for x in out["results"]] == oracle(
        fps, fps[0]
    )
    assert out["stats"]["candidates"] == len(fps)
