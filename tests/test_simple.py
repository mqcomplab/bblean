import numpy as np

from bblean.bitbirch import BitBirch  # type: ignore
from bblean.packing import pack_fingerprints


def test_bb_lean_defaults() -> None:
    tree = BitBirch()
    assert tree.branching_factor == 50
    assert tree.threshold == 0.65
    assert tree.merge_criterion == "diameter"


def test_bb_cluster_empty_input() -> None:
    fp = pack_fingerprints(np.zeros((0, 2048), dtype=np.uint8))
    ids = BitBirch().fit(fp, n_features=2048).get_cluster_mol_ids()
    assert ids == []


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
