import itertools
import pytest
import numpy as np

from bblean.bitbirch import BitBirch
from bblean.fingerprints import pack_fingerprints, make_fake_fingerprints

from inline_snapshot import snapshot

# NOTE: Results on this file don't depend on branching factor / threshold


def test_bb_lean_defaults() -> None:
    tree = BitBirch()
    assert tree.branching_factor == 50
    assert tree.threshold == 0.65
    assert tree.merge_criterion == "diameter"


def test_bb_cluster_empty_input() -> None:
    fp = pack_fingerprints(np.zeros((0, 2048), dtype=np.uint8))
    # At least 1 fp should be present
    with pytest.raises(ValueError):
        _ = BitBirch().fit(fp, n_features=2048).get_cluster_mol_ids()


def test_bb_cluster_simple_repeated_fps() -> None:
    for repeats in (1, 2, 10):
        zeros_fp = pack_fingerprints(np.zeros((repeats, 2048), dtype=np.uint8))
        ids = BitBirch().fit(zeros_fp, n_features=2048).get_cluster_mol_ids()
        assert ids == [list(range(repeats))]

        ones_fp = pack_fingerprints(np.ones((repeats, 2048), dtype=np.uint8))
        ids = BitBirch().fit(ones_fp, n_features=2048).get_cluster_mol_ids()
        assert ids == [list(range(repeats))]

        rng = np.random.default_rng(12620509540149709235)
        mixed_fp = pack_fingerprints(
            np.tile(rng.integers(0, 2, (1, 2048), dtype=np.uint8), (repeats, 1))
        )
        ids = BitBirch().fit(mixed_fp, n_features=2048).get_cluster_mol_ids()
        assert ids == [list(range(repeats))]


def test_bb_cluster_3_fps() -> None:
    fps = make_fake_fingerprints(3, n_features=8, seed=12620509540149709235, pack=True)

    data = BitBirch().fit(fps).get_cluster_mol_ids()
    assert data == snapshot([[0], [1], [2]])
    data = BitBirch().fit(fps, weights=itertools.repeat(5)).get_cluster_mol_ids()
    assert data == snapshot([[1, 2], [0]])
    data = BitBirch().fit(fps, weights=itertools.repeat(10000)).get_cluster_mol_ids()
    assert data == snapshot([[1, 2], [0]])
    data = BitBirch().fit(fps, weights=itertools.repeat(1000000)).get_cluster_mol_ids()
    assert data == snapshot([[1, 2], [0]])
