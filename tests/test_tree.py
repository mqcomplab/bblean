import pickle
import typing as tp
from pathlib import Path

import numpy as np
import pytest

import bblean.bitbirch
from bblean.bitbirch import BitBirch
from bblean.merges import (
    DiameterMerge,
    DiscardSubcluster,
    FlexibleToleranceDiameterMerge,
    ToleranceDiameterMerge,
    ToleranceRadiusMerge,
)


@pytest.mark.parametrize(
    "merge",
    [
        "diameter", "radius", "tolerance-diameter", "flexible-tolerance-diameter",
        "tolerance-radius", "tolerance-legacy",
    ],
)
@pytest.mark.parametrize("n_features", [7, 8, 64, 1032, 2048])
def test_tree_merge_and_split(merge: str, n_features: int) -> None:
    fingerprints = np.zeros((6, n_features), dtype=np.uint8)
    fingerprints[np.arange(6), [0, 1, 0, 2, 1, 0]] = 1
    tree = BitBirch(threshold=1.0, branching_factor=2, merge_criterion=merge)
    tree.fit(np.packbits(fingerprints, axis=1), n_features=n_features)

    assert tree.num_fitted_fps == 6
    assert tree.get_cluster_mol_ids() == [[0, 2, 5], [1, 4], [3]]
    np.testing.assert_array_equal(tree.get_assignments(), [1, 2, 1, 3, 2, 1])
    np.testing.assert_array_equal(
        tree.get_centroids(packed=False), fingerprints[[0, 1, 3]]
    )


@pytest.mark.parametrize("branching_factor", [2, 3, 5])
def test_tree_multiple_levels_and_incremental_fit(branching_factor: int) -> None:
    # Twelve disjoint fingerprints cannot share a cluster at threshold 1.
    fingerprints = np.packbits(np.eye(12, 16, dtype=np.uint8), axis=1)
    expected = [[index] for index in range(12)]
    complete = BitBirch(threshold=1.0, branching_factor=branching_factor)
    complete.fit(fingerprints)
    incremental = BitBirch(threshold=1.0, branching_factor=branching_factor)
    for batch in (fingerprints[:1], fingerprints[1:5], fingerprints[5:9], fingerprints[9:]):
        incremental.fit(batch)

    for tree in (complete, incremental):
        assert tree.num_fitted_fps == 12
        assert sorted(tree.get_cluster_mol_ids(sort=False)) == expected
        assert tree.get_cluster_mol_ids() == tree.get_cluster_mol_ids(sort=False)
        for sort in (False, True):
            members = tree.get_cluster_mol_ids(sort=sort)
            np.testing.assert_array_equal(
                tree.get_centroids(sort=sort),
                fingerprints[[group[0] for group in members]],
            )
            labels = tree.get_assignments(sort=sort)
            for label, group in enumerate(members, 1):
                np.testing.assert_array_equal(labels[group], [label])
    assert incremental.get_cluster_mol_ids(False) == complete.get_cluster_mol_ids(False)


@pytest.mark.parametrize("input_kind", ["packed", "unpacked", "list", "file"])
def test_tree_input_formats(input_kind: str, tmp_path: Path) -> None:
    packed = np.array([[240], [15], [15], [240], [15]], dtype=np.uint8)
    unpacked = np.unpackbits(packed, axis=1)
    inputs = {
        "packed": packed,
        "unpacked": unpacked,
        "list": list(packed),
        "file": tmp_path / "fingerprints.npy",
    }
    if input_kind == "file":
        np.save(inputs["file"], packed)
    tree = BitBirch(threshold=1.0).fit(
        inputs[input_kind], input_is_packed=input_kind != "unpacked"
    )

    assert tree.get_cluster_mol_ids(False) == [[0, 3], [1, 2, 4]]
    assert tree.get_cluster_mol_ids() == [[1, 2, 4], [0, 3]]
    for sort, expected_labels, expected_centroids in (
        (False, [1, 2, 2, 1, 2], [[240], [15]]),
        (True, [2, 1, 1, 2, 1], [[15], [240]]),
    ):
        np.testing.assert_array_equal(tree.get_assignments(sort=sort), expected_labels)
        result = tree.get_centroids_mol_ids(sort=sort)
        assert result["mol_ids"] == tree.get_cluster_mol_ids(sort)
        np.testing.assert_array_equal(result["centroids"], expected_centroids)
        np.testing.assert_array_equal(
            tree.get_centroids(sort, packed=False),
            np.unpackbits(np.array(expected_centroids, dtype=np.uint8), axis=1),
        )


@pytest.mark.parametrize("weight", [1, 300, 70_000])
def test_tree_weights_and_centroids(weight: int) -> None:
    fingerprints = np.array([[0b11110000], [0b11000000], [0b11000000]], dtype=np.uint8)
    tree = BitBirch(threshold=0.0).fit(fingerprints, weights=[weight, 1, 1])

    assert tree.num_fitted_fps == 3
    assert tree.get_cluster_mol_ids() == [[0, 1, 2]]
    np.testing.assert_array_equal(tree.get_assignments(), [1, 1, 1])
    expected_centroid = 0b11000000 if weight == 1 else 0b11110000
    np.testing.assert_array_equal(tree.get_centroids(), [[expected_centroid]])
    subcluster = tree._get_leaf_bfs()[0]
    assert subcluster.n_samples == weight + 2
    expected_sum = np.array(
        [weight + 2, weight + 2, weight, weight, 0, 0, 0, 0], dtype=np.uint64
    )
    np.testing.assert_array_equal(subcluster.linear_sum, expected_sum)


def test_tree_uint64_sample_counts() -> None:
    tree = BitBirch(threshold=1.0).fit(
        np.zeros((2, 1), dtype=np.uint8), weights=[2**32 + 1, 1]
    )
    assert tree.num_fitted_fps == 2
    assert tree.get_cluster_mol_ids() == [[0, 1]]
    assert tree._get_leaf_bfs()[0].n_samples == 2**32 + 2
    np.testing.assert_array_equal(tree.get_centroids(), [[0]])
    np.testing.assert_array_equal(tree.get_assignments(), [1, 1])


@pytest.mark.parametrize("merge,boundary", [("diameter", 0.5), ("radius", 0.75)])
@pytest.mark.parametrize("position", ["below", "equal", "above"])
def test_tree_threshold_boundary(merge: str, boundary: float, position: str) -> None:
    thresholds = {
        "below": np.nextafter(boundary, 0.0),
        "equal": boundary,
        "above": np.nextafter(boundary, 1.0),
    }
    tree = BitBirch(threshold=thresholds[position], merge_criterion=merge)
    tree.fit(np.array([[0b11110000], [0b11000000]], dtype=np.uint8))
    expected = [[0], [1]] if position == "above" else [[0, 1]]
    assert tree.get_cluster_mol_ids() == expected
    if position != "above":
        # Bits set in exactly half the members are included in the centroid.
        np.testing.assert_array_equal(tree.get_centroids(), [[0b11110000]])


@pytest.mark.parametrize("merge", ["diameter", "radius", "tolerance-diameter", "never"])
def test_tree_change_merge_between_fits(merge: str) -> None:
    fingerprints = np.array([[240], [240], [15]], dtype=np.uint8)
    tree = BitBirch(threshold=1.0, merge_criterion="never").fit(fingerprints[:1])
    tree.set_merge(merge)
    tree.fit(fingerprints[1:])
    expected = [[0], [1], [2]] if merge == "never" else [[0, 1], [2]]
    assert tree.get_cluster_mol_ids() == expected
    assert tree.num_fitted_fps == 3


@pytest.mark.parametrize(
    "merge_type",
    [ToleranceDiameterMerge, FlexibleToleranceDiameterMerge, ToleranceRadiusMerge],
)
@pytest.mark.parametrize("adaptive,n_max", [(False, 2), (True, 2), (True, 1000)])
def test_tree_adaptive_tolerance(
    merge_type: type[ToleranceDiameterMerge] | type[FlexibleToleranceDiameterMerge],
    adaptive: bool,
    n_max: int,
) -> None:
    # The third fingerprint lowers diameter similarity to 2/3 and radius to 5/6.
    merge = merge_type(tolerance=0.6, adaptive=adaptive, n_max=n_max, decay=0.001)
    tree = BitBirch(threshold=0.5, merge_criterion=merge)
    tree.fit(np.array([[0b11110000], [0b11110000], [0b11000000]], dtype=np.uint8))
    rejects = adaptive and n_max == 2 and merge_type is not FlexibleToleranceDiameterMerge
    expected = [[0, 1], [2]] if rejects else [[0, 1, 2]]
    assert tree.get_cluster_mol_ids() == expected


def test_tree_custom_threshold_hook() -> None:
    class RelaxedDiameterMerge(DiameterMerge):
        def on_check_merge_start(self, threshold: float, *args: tp.Any) -> float:
            return threshold / 2

    tree = BitBirch(threshold=1.0, merge_criterion=RelaxedDiameterMerge())
    tree.fit(np.array([[0b11110000], [0b11000000]], dtype=np.uint8))
    assert tree.get_cluster_mol_ids() == [[0, 1]]
    assert tree.threshold == 1.0
    np.testing.assert_array_equal(tree.get_centroids(), [[0b11110000]])


class _DiscardOddIds(DiameterMerge):
    def on_check_merge_end(
        self, accepted: bool, old_idxs: tp.Sequence[int], nominee_idxs: tp.Sequence[int]
    ) -> None:
        if nominee_idxs[0] % 2:
            raise DiscardSubcluster


def test_tree_discard_and_continue() -> None:
    fingerprints = np.full((6, 1), 240, dtype=np.uint8)
    tree = BitBirch(threshold=1.0, merge_criterion=_DiscardOddIds())
    tree.fit(fingerprints[:3])
    tree.fit(fingerprints[3:])

    assert tree.num_fitted_fps == 6
    assert tree.get_cluster_mol_ids() == [[0, 2, 4]]
    np.testing.assert_array_equal(tree.get_assignments(), [1, 0, 1, 0, 1, 0])
    assert tree._get_leaf_bfs()[0].n_samples == 3
    np.testing.assert_array_equal(tree.get_centroids(), [[240]])


def test_tree_discard_buffers() -> None:
    linear_sum = np.array([2, 2, 2, 2, 0, 0, 0, 0, 2], dtype=np.uint8)
    tree = BitBirch(threshold=1.0, merge_criterion=_DiscardOddIds())
    tree._fit_buffers(
        np.tile(linear_sum, (3, 1)), reinsert_index_seqs=[[0, 4], [1, 3], [2, 5]]
    )
    assert tree.num_fitted_fps == 6
    assert tree.get_cluster_mol_ids() == [[0, 4, 2, 5]]
    assert tree._get_leaf_bfs()[0].n_samples == 4
    np.testing.assert_array_equal(tree.get_assignments(), [1, 0, 1, 0, 1, 1])
    np.testing.assert_array_equal(tree.get_centroids(), [[0b11110000]])


@pytest.mark.parametrize("operation", ["refine", "recluster"])
def test_tree_refinement_preserves_identical_groups(operation: str) -> None:
    fingerprints = np.array([[240], [15], [240], [15], [15]], dtype=np.uint8)
    tree = BitBirch(threshold=1.0).fit(fingerprints)
    if operation == "refine":
        tree.refine_inplace(fingerprints, n_largest=1)
    else:
        tree.recluster_inplace(iterations=2)
    assert tree.num_fitted_fps == 5
    assert [sorted(group) for group in tree.get_cluster_mol_ids()] == [[1, 3, 4], [0, 2]]
    np.testing.assert_array_equal(tree.get_centroids(), [[15], [240]])
    np.testing.assert_array_equal(tree.get_assignments(), [2, 1, 2, 1, 1])


@pytest.mark.parametrize("buffers", [False, True])
def test_tree_reinsert_indices(buffers: bool) -> None:
    tree = BitBirch(threshold=1.0)
    if buffers:
        unpacked = np.unpackbits(np.array([[240], [15]], dtype=np.uint8), axis=1)
        linear_sums = unpacked * np.array([[2], [3]], dtype=np.uint8)
        tree._fit_buffers(
            np.column_stack([linear_sums, [2, 3]]).astype(np.uint8),
            reinsert_index_seqs=[[4, 0], [3, 1, 2]],
        )
    else:
        tree.fit(
            np.array([[240], [15], [15], [15], [240]], dtype=np.uint8),
            reinsert_indices=[4, 3, 1, 2, 0],
        )
    assert tree.num_fitted_fps == 5
    assert tree.get_cluster_mol_ids(False) == [[4, 0], [3, 1, 2]]
    np.testing.assert_array_equal(tree.get_assignments(sort=False), [1, 2, 2, 2, 1])
    np.testing.assert_array_equal(tree.get_centroids(False), [[240], [15]])


def test_tree_reject_invalid_inputs() -> None:
    tree = BitBirch()
    with pytest.raises(ValueError, match="at least 1 fingerprint"):
        tree.fit(np.empty((0, 1), dtype=np.uint8))
    assert not tree.is_init
    with pytest.raises(ValueError, match="n_features is larger"):
        tree.fit(np.array([[240]], dtype=np.uint8), n_features=9)
    assert not tree.is_init
    with pytest.raises(ValueError, match="len mol_indices"):
        tree._fit_buffers(
            np.array([[1, 0, 0, 0, 0, 0, 0, 0, 2]], dtype=np.uint8), [[0]]
        )
    tree.reset()
    tree.fit(np.array([[240], [15]], dtype=np.uint8), reinsert_indices=[0, 0])
    with pytest.raises(ValueError, match="unasigned molecules"):
        tree.get_assignments()
    np.testing.assert_array_equal(tree.get_assignments(check_valid=False), [2, 0])


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("state", ["unfitted", "fitted", "leaves"])
def test_tree_pickle_and_continue(legacy: bool, state: str) -> None:
    fingerprints = np.array([[240], [15], [170], [240]], dtype=np.uint8)
    tree = BitBirch(threshold=1.0, branching_factor=2)
    if state != "unfitted":
        tree.fit(fingerprints)
    if state == "leaves":
        tree.delete_internal_nodes()
    if legacy:
        # Old pickle states store nodes directly on BitBirch.
        py_tree = tree.__dict__.pop("_tree")
        tree.__dict__.update(
            _root=None if py_tree is None else py_tree._root,
            _dummy_leaf=(
                bblean.bitbirch._BFNode(2, 0)
                if py_tree is None else py_tree._dummy_leaf
            ),
        )
    loaded = pickle.loads(pickle.dumps(tree))
    assert loaded.is_init == (state != "unfitted")
    if state == "unfitted":
        loaded.fit(fingerprints)
    else:
        assert loaded.get_cluster_mol_ids() == [[0, 3], [1], [2]]
        np.testing.assert_array_equal(loaded.get_centroids(), [[240], [15], [170]])
        np.testing.assert_array_equal(loaded.get_assignments(), [1, 2, 3, 1])
        if state == "leaves":
            with pytest.raises(ValueError, match="Internal nodes were released"):
                loaded.fit(fingerprints)
            loaded.reset()
            loaded.fit(fingerprints)
        else:
            loaded.fit(fingerprints[:2])
            assert loaded.num_fitted_fps == 6
            assert loaded.get_cluster_mol_ids() == [[0, 3, 4], [1, 5], [2]]
            np.testing.assert_array_equal(
                loaded.get_assignments(), [1, 2, 3, 1, 1, 2]
            )
            return
    assert loaded.num_fitted_fps == 4
    assert loaded.get_cluster_mol_ids() == [[0, 3], [1], [2]]


def test_tree_delete_internal_nodes_and_reset() -> None:
    fingerprints = np.array([[240], [15], [170], [240]], dtype=np.uint8)
    tree = BitBirch(threshold=1.0, branching_factor=2).fit(fingerprints)
    tree.delete_internal_nodes()
    tree.delete_internal_nodes()
    assert tree.get_cluster_mol_ids() == [[0, 3], [1], [2]]
    np.testing.assert_array_equal(tree.get_assignments(), [1, 2, 3, 1])
    np.testing.assert_array_equal(tree.get_centroids(), [[240], [15], [170]])
    with pytest.raises(ValueError, match="Internal nodes were released"):
        tree.fit(fingerprints)
    tree.reset()
    assert not tree.is_init
    assert tree.num_fitted_fps == 0
    assert tree.threshold == 1.0
    assert tree.branching_factor == 2
    assert tree.merge_criterion == "diameter"
    with pytest.raises(ValueError, match="not been fitted"):
        tree.get_cluster_mol_ids()
    tree.fit(fingerprints)
    assert tree.get_cluster_mol_ids() == [[0, 3], [1], [2]]
    assert tree.num_fitted_fps == 4
