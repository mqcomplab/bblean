"""Local and S3 byte-range access. No credentials or cloud resources are created."""

from __future__ import annotations

from pathlib import Path
import typing as tp
from urllib.parse import urlsplit

from .types import SearchStats


def s3_location(uri: tp.Any) -> tuple[str, str]:
    parsed = urlsplit(str(uri))
    if parsed.scheme != "s3" or not parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError("Expected s3://bucket/prefix")
    return parsed.netloc, parsed.path.strip("/")


def s3_client(client: tp.Any = None) -> tp.Any:
    if client is not None:
        return client
    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:
        raise ImportError("S3 search requires pip install 'bblean[search]'") from exc
    # Own retries are visible to query accounting, including streamed body retries.
    return boto3.client(
        "s3", config=Config(retries={"max_attempts": 0}, max_pool_connections=32)
    )


class LocalStore:
    def __init__(self, root: str | Path) -> None:
        self.path = Path(root) / "data.bin"

    def read(self, start: int, length: int) -> tuple[bytes, SearchStats]:
        stats = SearchStats(read_requests=1)
        with self.path.open("rb") as f:
            f.seek(start)
            data = f.read(length)
        stats.fetched_bytes = len(data)
        if len(data) != length:
            raise EOFError("Truncated search payload")
        return data, stats


class S3Store:
    def __init__(
        self, uri: tp.Any, client: tp.Any = None, max_retries: int = 3
    ) -> None:
        self.bucket, self.prefix = s3_location(uri)
        self.client = s3_client(client)
        self.max_retries = max_retries
        self.retryable: tuple[type[BaseException], ...] = (
            OSError,
            EOFError,
            TimeoutError,
        )
        try:
            from botocore.exceptions import (
                ConnectionError,
                HTTPClientError,
                IncompleteReadError,
                ReadTimeoutError,
                ResponseStreamingError,
            )
        except ImportError:
            pass  # An injected client can operate without boto3 installed.
        else:
            self.retryable += (
                ConnectionError,
                HTTPClientError,
                IncompleteReadError,
                ReadTimeoutError,
                ResponseStreamingError,
            )

    def key(self, name: str) -> str:
        return f"{self.prefix}/{name}" if self.prefix else name

    def read_file(self, name: str) -> bytes:
        body = self.client.get_object(Bucket=self.bucket, Key=self.key(name))["Body"]
        try:
            data = body.read(1024 * 1024 + 1)
            if len(data) > 1024 * 1024:
                raise ValueError("Search manifest is too large")
            return data
        finally:
            body.close()

    def download_file(
        self, name: str, destination: str | Path, expected_bytes: int
    ) -> None:
        body = self.client.get_object(Bucket=self.bucket, Key=self.key(name))["Body"]
        total = 0
        try:
            with Path(destination).open("wb") as f:
                while True:
                    chunk = body.read(min(1024 * 1024, expected_bytes - total + 1))
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > expected_bytes:
                        raise ValueError(f"Oversized S3 metadata: {name}")
                    f.write(chunk)
            if total != expected_bytes:
                raise EOFError(f"Truncated S3 metadata: {name}")
        finally:
            body.close()

    def read(self, start: int, length: int) -> tuple[bytes, SearchStats]:
        stats = SearchStats(read_requests=1)
        data = bytearray()
        for attempt in range(self.max_retries + 1):
            body = None
            try:
                stats.s3_get_attempts += 1
                response = self.client.get_object(
                    Bucket=self.bucket,
                    Key=self.key("data.bin"),
                    Range=f"bytes={start + len(data)}-{start + length - 1}",
                )
                body = response["Body"]
                sdk_retries = int(
                    response.get("ResponseMetadata", {}).get("RetryAttempts", 0)
                )
                stats.s3_get_attempts += sdk_retries
                stats.retries += sdk_retries
                content_range = response.get("ContentRange")
                expected = f"bytes {start + len(data)}-{start + length - 1}/"
                if content_range is not None and not content_range.startswith(expected):
                    raise ValueError("S3 returned an unexpected byte range")
                while len(data) < length:
                    chunk = body.read(min(1024 * 1024, length - len(data)))
                    if not chunk:
                        raise EOFError("Incomplete S3 range response")
                    if len(chunk) > length - len(data):
                        raise ValueError("S3 returned more bytes than requested")
                    data.extend(chunk)
                    stats.fetched_bytes += len(chunk)
                return bytes(data), stats
            except Exception as exc:
                response = getattr(exc, "response", {})
                code = response.get("Error", {}).get("Code")
                status = response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0)
                if (
                    not isinstance(exc, self.retryable)
                    and code not in ("SlowDown", "RequestTimeout")
                    and status not in (429, 500, 502, 503, 504)
                ):
                    raise
                sdk_retries = int(
                    response.get("ResponseMetadata", {}).get("RetryAttempts", 0)
                )
                stats.s3_get_attempts += sdk_retries
                stats.retries += sdk_retries
                if body is not None and not isinstance(exc, EOFError):
                    stats.fetched_bytes_lower_bound = True
                if attempt == self.max_retries:
                    raise
                stats.retries += 1
            finally:
                if body is not None:
                    body.close()
        raise RuntimeError("Unreachable range retry state")


def plan_ranges(
    blocks: tp.Any,
    amplification: float = 2.0,
    max_span: int = 8 * 1024**2,
    row_bytes: int = 268,
    max_decoded: int = 16 * 1024**2,
) -> tp.Iterator[tuple[int, int, tp.Any]]:
    """Merge selected compressed blocks without scoring any intervening blocks."""
    at = 0
    while at < len(blocks):
        stop = at + 1
        start = int(blocks[at]["offset"])
        selected = int(blocks[at]["length"])
        decoded = int(blocks[at]["rows"]) * row_bytes
        end = start + selected
        while stop < len(blocks):
            b = blocks[stop]
            next_end = int(b["offset"]) + int(b["length"])
            if (
                next_end - start > max_span
                or decoded + int(b["rows"]) * row_bytes > max_decoded
                or next_end - start > amplification * (selected + int(b["length"]))
            ):
                break
            selected += int(b["length"])
            decoded += int(b["rows"]) * row_bytes
            end = next_end
            stop += 1
        yield start, end - start, blocks[at:stop]
        at = stop
