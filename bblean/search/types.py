"""Public result types for the search extension."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SearchResult:
    """A record ID and its exact binary Tanimoto score.

    ``index`` is the original row position, or the user-supplied uint64 ID.
    ``smi`` is reserved for compatibility with the original IVF interface;
    persistent search does not store or hydrate molecular structures.
    """

    index: int
    similarity: float
    smi: str | None = None

    @property
    def id(self) -> int:
        return self.index


@dataclass
class SearchStats:
    """Per-query work; S3 requests are attempts, including resumed body reads."""

    lists_probed: int = 0
    candidates: int = 0
    blocks_read: int = 0
    read_requests: int = 0
    s3_get_attempts: int = 0
    retries: int = 0
    fetched_bytes: int = 0
    fetched_bytes_lower_bound: bool = False
    routing_seconds: float = 0.0
    read_seconds: float = 0.0
    decode_seconds: float = 0.0
    score_seconds: float = 0.0
    total_seconds: float = 0.0
