import numpy as np

from bbtools.bb_lean import (  # type: ignore
    BitBirch,
    unpack_fingerprints,
)

from bbtools.legacy.bb_int64_dense import (  # type: ignore
    BitBirch as BitBirchInt64,
    set_merge as set_merge_int64,
)
from bbtools.legacy.bb_uint8 import (  # type: ignore
    BitBirch as BitBirchUint8,
    set_merge as set_merge_uint8,
)
from bbtools.fake_fingerprints import make_fake_fingerprints


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
    tree = BitBirch()
    tree.fit(fps, n_features=2048)
    output = tree.get_cluster_mol_ids()
    assert output[:13] == ids_expect

    # Check consistency with "upacked" bitbirch
    fps = unpack_fingerprints(fps, n_features=2048)
    tree = BitBirch()
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
    tree = BitBirch(merge_criterion="radius")
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
    tree = BitBirch(merge_criterion="radius")
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
    tree = BitBirch(merge_criterion="tolerance")
    tree.fit(fps, n_features=2048)
    output = tree.get_cluster_mol_ids()
    ids_expect = [[182, 255, 311, 389, 405, 438, 457, 461, 470], [107, 228], [13], [0]]
    assert output[:4] == ids_expect

    # Check consistency with "upacked" bitbirch
    fps = unpack_fingerprints(fps, n_features=2048)
    tree = BitBirch(merge_criterion="tolerance")
    tree.fit(fps, n_features=2048, input_is_packed=False)
    output = tree.get_cluster_mol_ids()
    assert output[:4] == ids_expect

    # Check consistency with "legacy" bitbirchs (uint8 and int64)
    set_merge_uint8("tolerance")
    tree = BitBirchUint8(threshold=0.65, branching_factor=50)
    tree.fit(fps, n_features=2048, input_is_packed=False)
    output = tree.get_cluster_mol_ids()
    assert output[:4] == ids_expect

    fps = fps.astype(np.int64)
    set_merge_int64("tolerance")
    tree = BitBirchInt64(threshold=0.65, branching_factor=50)
    tree.fit(fps, n_features=2048, input_is_packed=False)
    output = tree.get_cluster_mol_ids()
    assert output[:4] == ids_expect


def test_random_fps_tolerance_tough_consistency() -> None:
    fps = make_fake_fingerprints(
        500, n_features=2048, seed=12620509540149709235, pack=True
    )
    threshold = 0.4
    tree = BitBirch(merge_criterion="tolerance_tough", threshold=threshold)
    tree.fit(fps, n_features=2048)
    output = tree.get_cluster_mol_ids()
    ids_expect = [
        [3, 21, 26, 30, 32, 36, 37, 38, 45, 46, 53],
        [305, 306, 311, 315, 336],
        [248, 252, 255, 270, 273],
        [197, 199, 206, 209, 215],
    ]
    assert output[1:5] == ids_expect

    # Check consistency with "upacked" bitbirch
    fps = unpack_fingerprints(fps, n_features=2048)
    tree = BitBirch(merge_criterion="tolerance_tough", threshold=threshold)
    tree.fit(fps, n_features=2048, input_is_packed=False)
    output = tree.get_cluster_mol_ids()
    assert output[1:5] == ids_expect

    # Check consistency with "legacy" bitbirchs (uint8 and int64)
    set_merge_uint8("tolerance_tough")
    tree = BitBirchUint8(threshold=threshold, branching_factor=50)
    tree.fit(fps, n_features=2048, input_is_packed=False)
    output = tree.get_cluster_mol_ids()
    assert output[1:5] == ids_expect

    fps = fps.astype(np.int64)
    set_merge_int64("tolerance_tough")
    tree = BitBirchInt64(threshold=threshold, branching_factor=50)
    tree.fit(fps, n_features=2048, input_is_packed=False)
    output = tree.get_cluster_mol_ids()
    assert output[1:5] == ids_expect
