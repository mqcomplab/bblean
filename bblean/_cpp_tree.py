r"""C++ BitBirch tree"""

from __future__ import annotations

import itertools
import typing as tp

import numpy as np
from numpy.typing import NDArray

from bblean._memory import _ArrayMemPagesManager, _Input
from bblean.fingerprints import pack_fingerprints, unpack_fingerprints
from bblean.merges import _cpp_merge_params
from bblean.utils import _env_var_is_on

if tp.TYPE_CHECKING:
    from bblean.bitbirch import _BFSubcluster, _FitParams

_cpp_bitbirch: tp.Any
if _env_var_is_on("BITBIRCH_NO_EXTENSIONS") or _env_var_is_on("BITBIRCH_NO_CPP_TREE"):
    _cpp_bitbirch = None
else:
    try:
        from bblean import _cpp_bitbirch  # type: ignore
    except ImportError:
        _cpp_bitbirch = None

# Rows per call to the C++ tree when fitting arrays that are not mmapped files, which
# bounds the size of the temporary packed copies of unpacked fingerprints
_CPP_FIT_CHUNK = 65536
_NO_CPP_TREE_HINT = "set BITBIRCH_NO_CPP_TREE=1 to use the python tree"


class _CppTree:
    r"""BitBirch tree of the C++ extension, with the same interface as `_PyTree`"""

    def __init__(self, n_features: int) -> None:
        self.n_features = n_features
        self.num_fitted_fps = 0
        self.has_discarded = False  # The builtin criteria never discard
        self._tree = _cpp_bitbirch.Tree(n_features)

    @property
    def _only_has_leaves(self) -> bool:
        return self._tree.only_has_leaves

    def _checked_merge_params(
        self, n_features: int, params: _FitParams
    ) -> tuple[str, float, float, float, float]:
        if self.n_features != n_features:
            raise ValueError(
                f"n_features ({n_features}) is different from the number of features"
                f" of the fitted tree ({self.n_features})"
            )
        merge = _cpp_merge_params(params.merge_accept_fn, params.threshold)
        if merge is None:
            raise ValueError(
                f"Merge criterion {params.merge_accept_fn!r} is not supported by"
                " the C++ BitBirch tree, which was used for the previous fits,"
                f" {_NO_CPP_TREE_HINT}"
            )
        return merge

    def fit(
        self,
        X: _Input,
        n_features: int,
        input_is_packed: bool,
        reinsert_indices: tp.Iterable[int] | None,
        weights: tp.Iterable[int] | None,
        mmanager: _ArrayMemPagesManager,
        params: _FitParams,
    ) -> None:
        merge = self._checked_merge_params(n_features, params)
        if not _cpp_tree_supports_input(X, input_is_packed):
            raise ValueError(
                "The C++ BitBirch tree requires packed uint8 or unpacked integer"
                f" fingerprints, {_NO_CPP_TREE_HINT}"
            )
        num = len(X)
        ids = None
        if reinsert_indices is not None:
            ids = np.fromiter(
                itertools.islice(reinsert_indices, num), dtype=np.int64, count=-1
            )
            num = len(ids)
        w = None
        if weights is not None:
            w = np.fromiter(itertools.islice(weights, num), dtype=np.uint64, count=-1)
            if len(w) < num:
                raise ValueError("There are fewer weights than fingerprints")
        # Fit in chunks of one 'page' of the memory manager, to release memory
        # of mmapped files in the same way as the python tree
        chunk = mmanager.iters_per_page if mmanager.can_release else _CPP_FIT_CHUNK
        for begin in range(0, num, chunk):
            end = min(begin + chunk, num)
            self._tree.fit(
                _cpp_packed_rows(X, begin, end, input_is_packed, n_features),
                None if ids is None else ids[begin:end],
                self.num_fitted_fps,
                None if w is None else w[begin:end],
                merge,
                params.branching_factor,
            )
            self.num_fitted_fps += end - begin
            if mmanager.can_release and mmanager.should_release_curr_page(end):
                mmanager.release_curr_page_and_update_addr()

    def fit_buffers(
        self,
        X: _Input,
        n_features: int,
        reinsert_index_seqs: tp.Iterable[tp.Sequence[int]] | None,
        check_indices: bool,
        mmanager: _ArrayMemPagesManager,
        params: _FitParams,
    ) -> None:
        merge = self._checked_merge_params(n_features, params)
        num = len(X)
        # Molecule indices of buffer i: mol_idxs[offsets[i] : offsets[i + 1]]
        mol_idxs: NDArray[np.int64] = np.empty(0, dtype=np.int64)
        offsets: NDArray[np.int64] = np.empty(0, dtype=np.int64)
        if reinsert_index_seqs is not None:
            seqs = list(itertools.islice(reinsert_index_seqs, num))
            num = len(seqs)
            offsets = np.zeros(num + 1, dtype=np.int64)
            np.cumsum([len(q) for q in seqs], out=offsets[1:])
            mol_idxs = np.fromiter(
                itertools.chain.from_iterable(seqs),
                dtype=np.int64,
                count=int(offsets[-1]),
            )
        chunk = mmanager.iters_per_page if mmanager.can_release else _CPP_FIT_CHUNK
        for begin in range(0, num, chunk):
            end = min(num, begin + chunk)
            if isinstance(X, list):
                bufs = np.stack(X[begin:end])
            else:
                bufs = np.ascontiguousarray(X[begin:end])
            if reinsert_index_seqs is None:
                self._tree.fit_buffers(bufs, None, None, merge, params.branching_factor)
            else:
                chunk_offsets = offsets[begin : end + 1]
                if check_indices:
                    _check_buffer_n_samples(bufs, chunk_offsets)
                self._tree.fit_buffers(
                    bufs,
                    mol_idxs[chunk_offsets[0] : chunk_offsets[-1]],
                    chunk_offsets - chunk_offsets[0],
                    merge,
                    params.branching_factor,
                )
                self.num_fitted_fps += int(chunk_offsets[-1] - chunk_offsets[0])
            if mmanager.can_release and mmanager.should_release_curr_page(end):
                mmanager.release_curr_page_and_update_addr()

    def delete_internal_nodes(self) -> None:
        self._tree.delete_internal_nodes()

    def _get_leaf_bfs(self, sort: bool = True) -> list[_BFSubcluster]:
        # Copies of the C++ leaf subclusters, as python objects. Imported here since
        # bblean.bitbirch imports this module
        from bblean.bitbirch import _BFSubcluster

        return [
            _BFSubcluster(buf, ids, centroid, check_indices=False)
            for buf, ids, centroid in zip(
                self._tree.get_leaf_buffers(sort),
                self._tree.get_cluster_mol_ids(sort),
                self._tree.get_centroids(sort),
            )
        ]

    def get_cluster_mol_ids(self, sort: bool = True) -> list[list[int]]:
        return self._tree.get_cluster_mol_ids(sort)

    def get_centroids(
        self, sort: bool = True, packed: bool = True
    ) -> list[NDArray[np.uint8]]:
        centroids = self._tree.get_centroids(sort)
        if not packed:
            centroids = unpack_fingerprints(centroids, self.n_features)
        return list(centroids)

    def get_assignments(self, sort: bool = True) -> NDArray[np.uint64]:
        return self._tree.get_assignments(self.num_fitted_fps, sort)


def _cpp_tree_supports(params: _FitParams, n_features: int) -> bool:
    return (
        _cpp_bitbirch is not None
        and n_features % 8 == 0
        and _cpp_merge_params(params.merge_accept_fn, params.threshold) is not None
    )


def _cpp_tree_supports_input(X: tp.Any, input_is_packed: bool) -> bool:
    if isinstance(X, list):
        arrays = X
        if not all(isinstance(a, np.ndarray) and a.ndim == 1 for a in X):
            return False
        if len({a.shape for a in X}) > 1:
            return False
    elif isinstance(X, np.ndarray) and X.ndim == 2:
        arrays = [X]
    else:
        return False
    if input_is_packed:
        return all(a.dtype == np.uint8 for a in arrays)
    return all(a.dtype.kind in "bui" for a in arrays)


def _cpp_packed_rows(
    X: tp.Any, begin: int, end: int, input_is_packed: bool, n_features: int
) -> NDArray[np.uint8]:
    rows = np.stack(X[begin:end]) if isinstance(X, list) else X[begin:end]
    if input_is_packed:
        return np.ascontiguousarray(rows[:, : n_features // 8])
    if not _is_binary(rows):
        raise ValueError("Unpacked fingerprints must only contain 0 and 1")
    return pack_fingerprints(rows.astype(np.uint8, copy=False))


def _is_binary(a: NDArray[tp.Any]) -> bool:
    # Avoids temporaries the size of a
    if a.dtype.kind == "b":
        return True
    if a.dtype.kind not in "ui":
        return False
    return a.size == 0 or (a.max() <= 1 and (a.dtype.kind == "u" or a.min() >= 0))


def _check_buffer_n_samples(bufs: NDArray[tp.Any], offsets: NDArray[np.int64]) -> None:
    lens = np.diff(offsets)
    nonempty = lens > 0
    if (bufs[nonempty, -1].astype(np.uint64) != lens[nonempty]).any():
        raise ValueError("len mol_indices must be equal to buffer[-1] if specified")
