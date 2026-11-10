.. _bblean-search:

Similarity search
=================

The optional search extension turns BitBIRCH clusters into persistent lists of
packed binary fingerprints. Sparse summaries rank the lists; selected lists are
read from compressed local storage or directly from S3 and scored with exact
Tanimoto similarity. Routing is approximate. A full scan is available as an
exhaustive reference.

Installation
------------

This feature is on the development branch. Install from this checkout until it
is released::

    pip install -e '.[search]'
    bb-search --help

The extra adds ``zstandard`` for compressed storage and ``boto3`` for S3. After
release, the equivalent installation is ``pip install 'bblean[search]'``. Normal
package builds include the native search kernels. ``BITBIRCH_PY_ONLY=1`` builds
a Python-only package, and ``BITBIRCH_NO_EXTENSIONS=1`` selects the NumPy search
fallback at runtime. Core clustering does not import either optional dependency.

Building a new index
--------------------

Use an array of packed uint8 fingerprints or one or more ``.npy`` shards. Shard
order defines original row positions; shards are not concatenated into one
fingerprint array. The default representation is NumPy's big bit order. Pass
``n_features`` for fingerprints whose width is not divisible by eight; unused
low bits in the last byte must be zero.

.. code-block:: python

    from bblean.search import SearchIndex

    with SearchIndex.build(
        ["fingerprints-000.npy", "fingerprints-001.npy"],
        "my-search-index",
        bitbirch_kwargs={"threshold": 0.30, "branching_factor": 3500},
    ) as index:
        print(index.n_records, index.n_lists)

``build`` fits BitBirch in batches when no clusters are supplied. It does not
overwrite an existing directory. Construction writes a temporary sibling
directory, validates memberships and IDs, then publishes the completed directory.
An incomplete build does not leave a valid index at the requested destination.
The original fingerprint files and tree are not modified.

Reusing existing clustering
---------------------------

Reusing a fitted tree avoids clustering again:

.. code-block:: python

    from bblean import BitBirch
    from bblean.search import SearchIndex

    tree = BitBirch(threshold=0.30).fit(packed_fingerprints)
    with SearchIndex.from_bitbirch(
        tree, packed_fingerprints, "index-from-tree"
    ) as index:
        results = index.search(query, k=10, n_probe=16)

For a previous multiround run or saved memberships, pass the original zero-based
row positions directly:

.. code-block:: python

    with SearchIndex.from_clusters(
        ["part-000.npy", "part-001.npy"],
        cluster_members,                  # iterable of integer row arrays
        "index-from-memberships",
        centrals=packed_cluster_centroids,  # optional; same order as memberships
        n_lists=1000,
        ids=record_ids,                   # optional unique uint64 record IDs
    ) as index:
        results = index.search(query, n_probe=32)

Memberships must cover every source row exactly once. Gaps, duplicates, empty
clusters, out-of-range rows and duplicate record IDs are rejected. When explicit
IDs are omitted, results use original row positions, including across shards.
``input_is_packed=False`` accepts binary arrays and packs working batches.
Centroids, when supplied, are always packed.

The CLI accepts integer assignments saved as a ``.npy`` array, including the
one-based labels returned by BitBirch::

    bb-search build fingerprints.npy -o my-index --assignments assignments.npy
    bb-search build part-000.npy part-001.npy -o another-index --n-lists 1000

Coarse lists and summary construction
-------------------------------------

The default target is ``ceil(sqrt(N))`` lists. Oversized fine clusters are split
by deterministic Tanimoto-difference bisection. Population-weighted farthest-first
seeds and capacity-constrained assignment group fine units into lists, targeting
populations between half and one-and-a-half times the average. If fewer units
exist than the requested number of lists, the actual count is reduced. Fine
units are preserved unless a split is required to satisfy capacity constraints.

Within units, farthest-first seeds and two majority-centroid passes construct
roughly 32-row routing summaries. ``summary_size`` is a population target, not
a hard maximum; identical fingerprints can remain together. Frequencies below
0.05 are dropped; retained frequencies use three-bit square-root-spaced
quantization. Sparse postings store delta-coded summary IDs and frequency codes.
Each list is ranked by its best constituent summary score, using the experimental
mean/variance heuristic with coefficients 2 and 0.75.

Payload blocks contain separate, checksummed zstd level-one columns for
fingerprints, uint64 IDs and uint32 popcounts. ``block_size`` defaults to 256 KiB
before compression.
Blocks end at list boundaries. Coalesced reads can include intervening bytes,
but those bytes never add candidates from unselected lists. Queries bound both
compressed and decoded range sizes and the number of in-flight tasks.

Construction uses O(N) membership/permutation metadata, including scratch
memory-mapped arrays and sorting workspace. It needs temporary disk space for
memberships and sparse postings. Fingerprint working memory depends on the
largest coarse unit and ``batch_size``; a very small ``n_lists`` can create large
units. This implementation has not been benchmarked at a billion records.

Searching and interpreting results
-----------------------------------

.. code-block:: python

    import numpy as np
    from bblean.search import SearchIndex

    query = np.load("query.npy")
    with SearchIndex.open("my-search-index") as index:
        hits, stats = index.search_with_stats(query, k=10, n_probe=32)
        for hit in hits:
            print(hit.index, hit.similarity)
        print(stats.candidates, stats.fetched_bytes, stats.total_seconds)

        exact = index.search(query, k=10, n_probe=None)
        identity_included = index.search(
            query, k=10, n_probe=None, exclude_identical=False
        )

Returned scores are exact binary Tanimoto values. Results are ordered by
decreasing similarity, then increasing original record ID. Perfect
``Tanimoto=1`` matches are excluded by default, matching the nonempty-query
protocol in the study. Set ``exclude_identical=False`` to include them, or use
``exclude_ids=[id1, id2]`` to exclude particular records. Empty/empty similarity
is zero, consistent with core BBLean and RDKit.

``n_probe=None`` (or a budget at least as large as ``n_lists``) scans every list
without routing and returns the exhaustive result for the same exclusions.
``threshold`` filters only the selected candidates; approximate routing cannot
guarantee all database hits above a threshold. Fewer than ``k`` eligible results
are returned when necessary. Choose a probe budget on representative validation
queries; the small overlapping development panels in the paper are not a
general recall guarantee.

``SearchStats`` reports list and candidate counts, selected blocks, logical
range reads, GET attempts and retries, observed payload bytes, and timings.
Routing initialization is outside query time. Worker read/decode times are sums
of task durations and can overlap; ``total_seconds`` is elapsed query time.
Payload bytes exclude HTTP/TLS overhead. Interrupted bodies can leave unknown
partial bytes; ``fetched_bytes_lower_bound`` flags this qualification.

The CLI outputs JSON::

    bb-search query my-index query.npy -k 10 --n-probe 32 --stats
    bb-search query my-index query.npy --exhaustive --include-identical

S3 storage
-----------

Upload is explicit and separate from building:

.. code-block:: python

    with SearchIndex.open("my-index") as index:
        index.upload("s3://my-bucket/indexes/library-v1")

    with SearchIndex.open(
        "s3://my-bucket/indexes/library-v1",
        cache_dir="routing-cache",
    ) as index:
        hits, stats = index.search_with_stats(query, n_probe=32, io_workers=32)

The default client uses boto3's normal credentials and configuration chain.
Pass ``client=your_boto3_client`` to ``open`` or ``upload`` for custom endpoints,
credentials or client configuration. Upload uses managed multipart transfers
and writes the manifest last; published manifests are not overwritten. Use a
new prefix per version and serialize publication to a given prefix.

Opening S3 downloads and verifies routing metadata into a local cache keyed by
manifest content. It does not download ``data.bin``. Query payloads use range
GETs with bounded read amplification, resume interrupted responses and are not
cached between queries. The default client disables hidden SDK retries; the
reader tracks its own attempts and includes any SDK retry counts reported by
an injected client. No bucket, instance, IAM policy or other infrastructure is
created by the extension.

The routing summaries must be available locally for each worker. Worker startup
and routing-index download/loading are not included in query latency. Sharing
one open index supports concurrent queries; close it after those queries finish.

Persistence and compatibility
------------------------------

``manifest.json`` contains the format version, representation, populations and
file SHA-256 hashes. Arrays use explicit little-endian numeric dtypes and
``allow_pickle=False``. ``open`` checks metadata integrity and payload size;
``index.verify()`` additionally hashes the full local payload. Keep immutable
versions for reproducibility; the current format does not support insert/delete
updates in place.

``bblean.ivf.IVFIndex`` and the branch's older hidden CLI commands remain
available for existing in-memory medoid indexes. ``bblean.search.SearchIndex``
is the preferred API for sparse-summary routing and compressed storage.
The new format does not read the research worker's directory layout directly;
reuse its original memberships/fingerprint shards to build a package index.

Algorithm provenance
---------------------

The sparse routing heuristic, split-summary scheme, balanced reuse of fine
clusters, strict candidate accounting, zstd layouts and S3 range strategy derive
from the November 2026 clustered-search experiments. The curated scientific
record is `mqcomplab/bb-clustered-search
<https://github.com/mqcomplab/bb-clustered-search>`_, snapshot ``1990812``.
The package uses current main's portable native kernels instead of the worker's
runtime-built AVX-512 libraries, and a versioned portable storage format.
It does not claim the worker's measured latency, cost or billion-record recall.

API
----

.. autoclass:: bblean.search.SearchIndex
   :members:

.. autoclass:: bblean.search.SearchResult
   :members:

.. autoclass:: bblean.search.SearchStats
   :members:
