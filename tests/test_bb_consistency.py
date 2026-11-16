from pathlib import Path
import typing as tp

import numpy as np
import pytest

from bblean._legacy.bb_int64 import (  # type: ignore
    BitBirch as BitBirchInt64,
    set_merge as set_merge_int64,
)
from bblean._legacy.bb_uint8 import (  # type: ignore
    BitBirch as BitBirchUint8,
    set_merge as set_merge_uint8,
)
from bblean.bitbirch import BitBirch  # type: ignore
from bblean.fingerprints import make_fake_fingerprints, unpack_fingerprints
from bblean.merges import DiameterMerge, DiscardSubcluster


def test_random_fps_consistency() -> None:
    fps = make_fake_fingerprints(
        3000, n_features=2048, seed=12620509540149709235, pack=True
    )
    ids_expect = [
        [2195, 2196, 2378, 2440, 2443, 2454, 2463, 2464, 2465, 2467, 2527, 2544],
        [199, 228, 255, 270, 273, 438, 457, 458, 461, 470, 477, 496],
        [700, 728, 773, 798, 825, 891, 919, 962, 963, 968, 998],
        [1448, 1567, 1590, 1606, 1612, 1637, 1640, 1648, 1686, 1694],
        [1059, 1065, 1072, 1077, 1154, 1194, 1301],
        [1779, 1802, 1807, 1828, 1856, 1864],
        [2826, 2896, 2970, 2973, 2975],
        [1986, 2107, 2139, 2141],
        [1933, 1949],
        [2233, 2294],
        [1551, 1552],
        [1219, 1226],
        [614, 637],
    ]
    tree = BitBirch(branching_factor=50, threshold=0.65, merge_criterion="diameter")
    tree.fit(fps, n_features=2048)
    output = tree.get_cluster_mol_ids()
    assert output[:13] == ids_expect

    # Check consistency with "upacked" bitbirch
    fps = unpack_fingerprints(fps, n_features=2048)
    tree = BitBirch(branching_factor=50, threshold=0.65, merge_criterion="diameter")
    tree.fit(fps, n_features=2048, input_is_packed=False)
    output = tree.get_cluster_mol_ids()
    assert output[:13] == ids_expect

    # Check consistency with "legacy" bitbirchs (uint8 and int64)
    set_merge_uint8("diameter")
    tree = BitBirchUint8(threshold=0.65, branching_factor=50)
    tree.fit(fps, n_features=2048, input_is_packed=False)
    output = tree.get_cluster_mol_ids()
    assert output[:13] == ids_expect

    fps = fps.astype(np.int64)
    set_merge_int64("diameter")
    tree = BitBirchInt64(threshold=0.65, branching_factor=50)
    tree.fit(fps, n_features=2048, input_is_packed=False)
    output = tree.get_cluster_mol_ids()
    assert output[:13] == ids_expect


def test_random_fps_radius_consistency() -> None:
    fps = make_fake_fingerprints(
        1000, n_features=2048, seed=12620509540149709235, pack=True
    )
    tree = BitBirch(merge_criterion="radius", threshold=0.65, branching_factor=50)
    tree.fit(fps, n_features=2048)
    ids_expect = [
        [749, 751, 766, 798, 825, 827, 840, 845],
        [463, 464, 470, 477, 496, 498],
        [0, 3, 32, 53],
        [607, 609, 614, 615],
        [542, 557, 560, 561],
        [30, 36, 45],
        [647, 650, 652],
        [689, 694, 745],
        [762, 764, 773],
        [771, 775, 789],
        [520, 522, 551],
        [60, 66, 99],
        [248, 389, 390],
        [336, 391, 405],
        [199, 230, 231],
        [71, 125],
        [82, 155],
    ]
    output = tree.get_cluster_mol_ids()
    assert output[2:19] == ids_expect

    # Check consistency with "upacked" bitbirch
    fps = unpack_fingerprints(fps, n_features=2048)
    tree = BitBirch(merge_criterion="radius", threshold=0.65, branching_factor=50)
    tree.fit(fps, n_features=2048, input_is_packed=False)
    output = tree.get_cluster_mol_ids()
    assert output[2:19] == ids_expect

    # Check consistency with "legacy" bitbirchs (uint8 and int64)
    set_merge_uint8("radius")
    tree = BitBirchUint8(threshold=0.65, branching_factor=50)
    tree.fit(fps, n_features=2048, input_is_packed=False)
    output = tree.get_cluster_mol_ids()
    assert output[2:19] == ids_expect

    fps = fps.astype(np.int64)
    set_merge_int64("radius")
    tree = BitBirchInt64(threshold=0.65, branching_factor=50)
    tree.fit(fps, n_features=2048, input_is_packed=False)
    output = tree.get_cluster_mol_ids()
    assert output[2:19] == ids_expect


def test_random_fps_tolerance_consistency() -> None:
    fps = make_fake_fingerprints(
        500, n_features=2048, seed=12620509540149709235, pack=True
    )
    tree = BitBirch(
        merge_criterion="tolerance-legacy",
        branching_factor=50,
        threshold=0.65,
        tolerance=0.05,
    )
    tree.fit(fps, n_features=2048)
    output = tree.get_cluster_mol_ids()
    ids_expect = [[182, 255, 311, 389, 405, 438, 457, 461, 470], [107, 228], [13], [0]]
    assert output[:4] == ids_expect

    # Check consistency with "upacked" bitbirch
    fps = unpack_fingerprints(fps, n_features=2048)
    tree = BitBirch(
        merge_criterion="tolerance-legacy",
        branching_factor=50,
        threshold=0.65,
        tolerance=0.05,
    )
    tree.fit(fps, n_features=2048, input_is_packed=False)
    output = tree.get_cluster_mol_ids()
    assert output[:4] == ids_expect

    # Check consistency with "legacy" bitbirchs (uint8 and int64)
    set_merge_uint8("tolerance-legacy", 0.05)
    tree = BitBirchUint8(threshold=0.65, branching_factor=50)
    tree.fit(fps, n_features=2048, input_is_packed=False)
    output = tree.get_cluster_mol_ids()
    assert output[:4] == ids_expect

    fps = fps.astype(np.int64)
    set_merge_int64("tolerance-legacy", 0.05)
    tree = BitBirchInt64(threshold=0.65, branching_factor=50)
    tree.fit(fps, n_features=2048, input_is_packed=False)
    output = tree.get_cluster_mol_ids()
    assert output[:4] == ids_expect


def test_incremental_fit_consistency() -> None:
    # Twelve disjoint fingerprints, each in its own cluster, split the root twice
    fps = np.packbits(np.eye(12, 16, dtype=np.uint8), axis=1)
    complete = BitBirch(threshold=1.0, branching_factor=2).fit(fps)
    incremental = BitBirch(threshold=1.0, branching_factor=2)
    for batch in (fps[:1], fps[1:5], fps[5:9], fps[9:]):
        incremental.fit(batch)
    for tree in (complete, incremental):
        assert tree.num_fitted_fps == 12
        ids = tree.get_cluster_mol_ids()
        assert sorted(ids) == [[i] for i in range(12)]
        np.testing.assert_array_equal(tree.get_centroids(), fps[[i[0] for i in ids]])
        labels = tree.get_assignments()
        assert [labels[i[0]] for i in ids] == list(range(1, 13))
    assert incremental.get_cluster_mol_ids() == complete.get_cluster_mol_ids()


class _DiscardOddIds(DiameterMerge):
    def on_check_merge_end(
        self, accepted: bool, old_idxs: tp.Sequence[int], nominee_idxs: tp.Sequence[int]
    ) -> None:
        if nominee_idxs[0] % 2:
            raise DiscardSubcluster


def test_discard_consistency() -> None:
    tree = BitBirch(threshold=1.0, merge_criterion=_DiscardOddIds())
    fps = np.full((6, 1), 240, dtype=np.uint8)
    tree.fit(fps[:3])
    tree.fit(fps[3:])
    assert tree.num_fitted_fps == 6
    assert tree.get_cluster_mol_ids() == [[0, 2, 4]]
    np.testing.assert_array_equal(tree.get_assignments(), [1, 0, 1, 0, 1, 0])

    tree = BitBirch(threshold=1.0, merge_criterion=_DiscardOddIds())
    buffers = np.tile(np.array([2, 2, 2, 2, 0, 0, 0, 0, 2], dtype=np.uint8), (3, 1))
    tree._fit_buffers(buffers, reinsert_index_seqs=[[0, 4], [1, 3], [2, 5]])
    assert tree.num_fitted_fps == 6
    assert tree.get_cluster_mol_ids() == [[0, 4, 2, 5]]
    np.testing.assert_array_equal(tree.get_assignments(), [1, 0, 1, 0, 1, 1])
    np.testing.assert_array_equal(tree.get_centroids(), [[0b11110000]])


@pytest.mark.filterwarnings("ignore:(Saving|Loading) large BitBIRCH")
def test_save_load_and_continue_consistency(tmp_path: Path) -> None:
    fps = np.array([[240], [15], [170], [240]], dtype=np.uint8)
    BitBirch(threshold=1.0, branching_factor=2).fit(fps).save(tmp_path / "tree.pkl")
    tree = BitBirch.load(tmp_path / "tree.pkl")
    assert tree.get_cluster_mol_ids() == [[0, 3], [1], [2]]
    np.testing.assert_array_equal(tree.get_centroids(), [[240], [15], [170]])
    tree.fit(fps[:2])
    assert tree.num_fitted_fps == 6
    assert tree.get_cluster_mol_ids() == [[0, 3, 4], [1, 5], [2]]
    np.testing.assert_array_equal(tree.get_assignments(), [1, 2, 3, 1, 1, 2])


def test_delete_internal_nodes_and_reset_consistency() -> None:
    fps = np.array([[240], [15], [170], [240]], dtype=np.uint8)
    tree = BitBirch(threshold=1.0, branching_factor=2).fit(fps)
    tree.delete_internal_nodes()
    assert tree.get_cluster_mol_ids() == [[0, 3], [1], [2]]
    np.testing.assert_array_equal(tree.get_assignments(), [1, 2, 3, 1])
    with pytest.raises(ValueError, match="Internal nodes were released"):
        tree.fit(fps)
    tree.reset()
    assert not tree.is_init and tree.num_fitted_fps == 0
    with pytest.raises(ValueError, match="not been fitted"):
        tree.get_cluster_mol_ids()
    assert tree.fit(fps).get_cluster_mol_ids() == [[0, 3], [1], [2]]
