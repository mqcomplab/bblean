import itertools
import os
from pathlib import Path
import pickle
import typing as tp

import numpy as np
import pytest

import bblean.bitbirch
from bblean.bitbirch import BitBirch
from bblean.fingerprints import make_fake_fingerprints, unpack_fingerprints
from bblean.merges import DiameterMerge

from tree_cases import CASES, PYTHON_ONLY, SEED, flat_cluster_ids  # type: ignore

if bblean.bitbirch._cpp_bitbirch is None and os.getenv("BITBIRCH_CANT_SKIP_CPP_TESTS"):
    raise ImportError("The C++ BitBirch tree is not available")

_REFERENCE = np.load(Path(__file__).parent / "tree-reference.npz")
_TREES = ["python"]
if bblean.bitbirch._cpp_bitbirch is not None:
    _TREES.append("cpp")


def _uses_cpp(tree: BitBirch) -> bool:
    return isinstance(tree._tree, bblean.bitbirch._CppTree)


@pytest.fixture(params=_TREES)
def tree_kind(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    # The python tree is used if the C++ extension is not available
    if request.param == "python":
        monkeypatch.setattr(bblean.bitbirch, "_cpp_bitbirch", None)
    return request.param


@pytest.mark.parametrize("name", list(CASES))
def test_tree_reference(name: str, tree_kind: str) -> None:
    # Same clusters, cluster order and molecule order as the reference results
    tree = CASES[name]()
    uses_cpp = tree_kind == "cpp" and name not in PYTHON_ONLY
    assert _uses_cpp(tree) == uses_cpp
    ids, offsets = flat_cluster_ids(tree)
    assert ids.tolist() == _REFERENCE[f"{name}/ids"].tolist()
    assert offsets.tolist() == _REFERENCE[f"{name}/offsets"].tolist()
    assert tree.num_fitted_fps == _REFERENCE[f"{name}/num_fitted"]


def _state(tree: BitBirch) -> tuple[tp.Any, ...]:
    return (
        tree.get_cluster_mol_ids(),
        tree.get_cluster_mol_ids(sort=False),
        tree.get_assignments().tolist(),
        tree.get_assignments(sort=False).tolist(),
        [c.tolist() for c in tree.get_centroids()],
        [c.tolist() for c in tree.get_centroids(packed=False)],
        [(bf._buffer.dtype.name, bf._buffer.tolist()) for bf in tree._get_leaf_bfs()],
        tree.num_fitted_fps,
    )


def _fit_cpp_and_python(
    fit: tp.Callable[[], BitBirch], monkeypatch: pytest.MonkeyPatch
) -> tuple[BitBirch, BitBirch]:
    cpp = fit()
    with monkeypatch.context() as m:
        m.setattr(bblean.bitbirch, "_cpp_bitbirch", None)
        py = fit()
    assert _uses_cpp(cpp) and not _uses_cpp(py)
    return cpp, py


@pytest.mark.skipif("cpp" not in _TREES, reason="C++ BitBirch tree not available")
def test_tree_state_python_and_cpp(monkeypatch: pytest.MonkeyPatch) -> None:
    # Centroids, linear sums and assignments are the same for both trees
    fps = make_fake_fingerprints(800, n_features=2048, seed=SEED, pack=True)
    cpp, py = _fit_cpp_and_python(
        lambda: BitBirch(threshold=0.3, branching_factor=7).fit(fps), monkeypatch
    )
    assert _state(cpp) == _state(py)


@pytest.mark.skipif("cpp" not in _TREES, reason="C++ BitBirch tree not available")
@pytest.mark.parametrize("weight", [70_000, 2**32 + 1])
def test_tree_large_weights(weight: int, monkeypatch: pytest.MonkeyPatch) -> None:
    # Linear sums that need uint32 and uint64 values
    fps = make_fake_fingerprints(300, n_features=2048, seed=SEED, pack=True)
    cpp, py = _fit_cpp_and_python(
        lambda: BitBirch(threshold=0.4, branching_factor=10).fit(
            fps, weights=itertools.repeat(weight)
        ),
        monkeypatch,
    )
    assert _state(cpp) == _state(py)


@pytest.mark.skipif("cpp" not in _TREES, reason="C++ BitBirch tree not available")
def test_tree_fit_in_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrays are passed to the C++ tree in chunks (packed one at a time)
    fps = make_fake_fingerprints(500, n_features=2048, seed=SEED, pack=True)
    unpacked = unpack_fingerprints(fps)

    def fit(X: tp.Any, input_is_packed: bool) -> BitBirch:
        tree = BitBirch(threshold=0.4, branching_factor=10).fit(
            X,
            input_is_packed=input_is_packed,
            reinsert_indices=range(499, -1, -1),
            weights=(i % 5 + 1 for i in range(500)),
        )
        assert _uses_cpp(tree)
        return tree

    expect = _state(fit(fps, True))
    monkeypatch.setattr(bblean.bitbirch, "_CPP_FIT_CHUNK", 7)
    assert _state(fit(fps, True)) == expect
    assert _state(fit(unpacked, False)) == expect
    assert _state(fit(list(unpacked.astype(bool)), False)) == expect


def test_tree_load_old_pickle(monkeypatch: pytest.MonkeyPatch) -> None:
    # Trees pickled before the C++ tree was added keep the python tree in _root and
    # _dummy_leaf
    monkeypatch.setattr(bblean.bitbirch, "_cpp_bitbirch", None)
    fps = make_fake_fingerprints(200, n_features=2048, seed=SEED, pack=True)
    tree = BitBirch(threshold=0.4, branching_factor=10).fit(fps)
    expect = tree.get_cluster_mol_ids()
    py_tree = tree.__dict__.pop("_tree")
    tree.__dict__.update(_root=py_tree._root, _dummy_leaf=py_tree._dummy_leaf)
    loaded = pickle.loads(pickle.dumps(tree))
    assert loaded.get_cluster_mol_ids() == expect
    assert loaded.fit(fps).num_fitted_fps == 400

    empty = BitBirch()
    del empty.__dict__["_tree"]
    empty.__dict__.update(
        _root=None,
        _dummy_leaf=bblean.bitbirch._BFNode(branching_factor=2, n_features=0),
    )
    assert not pickle.loads(pickle.dumps(empty)).is_init


@pytest.mark.skipif("cpp" not in _TREES, reason="C++ BitBirch tree not available")
def test_tree_selection() -> None:
    fps = make_fake_fingerprints(50, n_features=2048, seed=SEED, pack=True)
    assert _uses_cpp(BitBirch().fit(fps))
    assert _uses_cpp(BitBirch(merge_criterion=DiameterMerge()).fit(fps))
    assert _uses_cpp(BitBirch().fit(list(fps)))
    unpacked = unpack_fingerprints(fps)
    assert _uses_cpp(BitBirch().fit(unpacked, input_is_packed=False))

    # The python tree is used for user defined merge criteria, non-binary or non
    # uint8 inputs, and n_features that are not a multiple of 8
    class Custom(DiameterMerge):
        pass

    assert not _uses_cpp(BitBirch(merge_criterion=Custom()).fit(fps))
    assert not _uses_cpp(BitBirch().fit(unpacked * 2, input_is_packed=False))
    assert not _uses_cpp(BitBirch().fit(fps.astype(np.int64)))
    assert bblean.bitbirch._cpp_packed_rows(fps, True, 2044) is None

    # A C++ tree can't continue with a user defined criterion or non-binary input
    tree = BitBirch().fit(fps)
    with pytest.raises(ValueError):
        tree.fit(unpacked * 2, input_is_packed=False)
    tree.set_merge(Custom())
    with pytest.raises(ValueError):
        tree.fit(fps)


def test_tree_pickle_and_continue(tree_kind: str) -> None:
    fps = make_fake_fingerprints(600, n_features=2048, seed=SEED, pack=True)
    expect = BitBirch(threshold=0.4, branching_factor=10).fit(fps).get_cluster_mol_ids()
    tree = BitBirch(threshold=0.4, branching_factor=10).fit(fps[:300])
    tree = pickle.loads(pickle.dumps(tree))
    assert tree.fit(fps[300:]).get_cluster_mol_ids() == expect


def test_tree_delete_internal_nodes(tree_kind: str) -> None:
    fps = make_fake_fingerprints(300, n_features=2048, seed=SEED, pack=True)
    tree = BitBirch(threshold=0.4, branching_factor=5).fit(fps)
    ids = tree.get_cluster_mol_ids()
    tree.delete_internal_nodes()
    assert tree.get_cluster_mol_ids() == ids
    with pytest.raises(ValueError):
        tree.fit(fps)
    tree.reset()
    assert not tree.is_init
    assert tree.fit(fps).get_cluster_mol_ids() == ids
