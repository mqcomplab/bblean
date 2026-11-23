"""``bb-search``: build and query optional persistent search indexes."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import typing as tp

import numpy as np
from numpy.typing import NDArray

from .index import SearchIndex


def members_from_assignments(
    labels: NDArray[np.integer],
) -> tp.Iterator[NDArray[np.intp]]:
    """Translate any integer label vocabulary into zero-based row memberships."""
    if labels.ndim != 1 or not len(labels) or labels.dtype.kind not in "iu":
        raise ValueError("Assignments must be a nonempty 1D integer array")
    order = np.argsort(labels, kind="stable")
    ordered = labels[order]
    cuts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1, len(order)]
    return (order[a:b] for a, b in zip(cuts[:-1], cuts[1:]))


def main(argv: tp.Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="Create a new compressed index")
    build.add_argument(
        "fingerprints", nargs="+", help="Packed .npy shards in row order"
    )
    build.add_argument("-o", "--output", required=True)
    build.add_argument(
        "--assignments", help="Reuse existing integer labels in a .npy file"
    )
    build.add_argument("--ids", help="Optional unique uint64 IDs in a .npy file")
    build.add_argument("--n-lists", type=int)
    build.add_argument("--n-features", type=int)
    build.add_argument("--unpacked", action="store_true")
    build.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="BitBirch threshold when clustering new fingerprints",
    )
    query = commands.add_parser("query", help="Search a local index or s3:// prefix")
    query.add_argument("index")
    query.add_argument("query", help="One packed query fingerprint in a .npy file")
    query.add_argument("-k", type=int, default=10)
    query.add_argument("--n-probe", type=int, default=1)
    query.add_argument("--exhaustive", action="store_true")
    query.add_argument("--threshold", type=float, default=0)
    query.add_argument("--include-identical", action="store_true")
    query.add_argument("--unpacked", action="store_true")
    query.add_argument("--stats", action="store_true")
    query.add_argument("--cache-dir")
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            members = None
            if args.assignments:
                members = members_from_assignments(
                    np.load(args.assignments, mmap_mode="r", allow_pickle=False)
                )
            ids = (
                np.load(args.ids, mmap_mode="r", allow_pickle=False)
                if args.ids
                else None
            )
            with SearchIndex.build(
                args.fingerprints,
                args.output,
                clusters=members,
                n_lists=args.n_lists,
                n_features=args.n_features,
                ids=ids,
                input_is_packed=not args.unpacked,
                bitbirch_kwargs={"threshold": args.threshold},
            ) as index:
                print(
                    json.dumps(
                        {
                            "index": str(index.path),
                            "records": index.n_records,
                            "lists": index.n_lists,
                        }
                    )
                )
        else:
            q = np.load(args.query, allow_pickle=False)
            if q.ndim == 2 and len(q) == 1:
                q = q[0]
            with SearchIndex.open(args.index, cache_dir=args.cache_dir) as index:
                results, stats = index.search_with_stats(
                    q,
                    args.k,
                    None if args.exhaustive else args.n_probe,
                    threshold=args.threshold,
                    exclude_identical=not args.include_identical,
                    input_is_packed=not args.unpacked,
                )
                out: dict[str, tp.Any] = {"results": [asdict(r) for r in results]}
                if args.stats:
                    out["stats"] = asdict(stats)
                print(json.dumps(out))
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(2, f"bb-search: {exc}\n")


if __name__ == "__main__":
    main()
