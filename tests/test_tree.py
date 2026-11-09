import json
from pathlib import Path
import pickle

import pytest

import bblean.bitbirch
from bblean.bitbirch import BitBirch
from bblean.fingerprints import make_fake_fingerprints

from tree_cases import CASES, SEED, summary  # type: ignore

_REFERENCE = json.loads((Path(__file__).parent / "tree-reference.json").read_text())


@pytest.mark.parametrize("name", list(CASES))
def test_tree_reference(name: str) -> None:
    # Same clusters, cluster order and molecule order as the reference results
    tree = CASES[name]()
    assert summary(tree) == _REFERENCE[name]


def test_tree_load_old_pickle() -> None:
    # Trees pickled before the tree was moved to _PyTree keep it in _root and
    # _dummy_leaf
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


def test_tree_pickle_and_continue() -> None:
    fps = make_fake_fingerprints(600, n_features=2048, seed=SEED, pack=True)
    expect = BitBirch(threshold=0.4, branching_factor=10).fit(fps).get_cluster_mol_ids()
    tree = BitBirch(threshold=0.4, branching_factor=10).fit(fps[:300])
    tree = pickle.loads(pickle.dumps(tree))
    assert tree.fit(fps[300:]).get_cluster_mol_ids() == expect


def test_tree_delete_internal_nodes() -> None:
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
