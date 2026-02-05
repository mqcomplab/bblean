from numpy.typing import NDArray
import numpy as np

from inline_snapshot import snapshot

from legacy_merges import (  # type: ignore
    merge_radius,
    merge_diameter,
    merge_tolerance,
)
from bblean import BitBirch
from bblean.merges import (
    NeverMerge,
    RadiusMerge,
    DiameterMerge,
    ToleranceLegacyMerge,
    ToleranceDiameterMerge,
    ToleranceRadiusMerge,
)
from bblean.fingerprints import make_fake_fingerprints
from bblean.similarity import centroid_from_sum

# Cases to test for all merges:
# low|high tolerance


def get_old_and_nom(
    fps: NDArray[np.integer], j: int, case: str = "1, 1"
) -> tuple[NDArray[np.integer], NDArray[np.integer]]:
    if case == "1, 1":
        old = fps[j : j + 1]
        nom = fps[j + 1 : j + 2]
        return old, nom
    if case == "1, >1":
        old = fps[j : j + 1]
        nom = fps[j + 1 : j + 100]
        return old, nom
    if case == ">1, 1":
        old = fps[j : j + 100]
        nom = fps[j + 101 : j + 102]
        return old, nom
    if case == ">1, >1":
        old = fps[j : j + 100]
        nom = fps[j + 101 : j + 200]
        return old, nom
    raise ValueError("Unknown case")


# low|high threshold
# old = 1, nom = 1
# old = 1, nom > 1
# old > 1, nom = 1
# old > 1, nom > 1
def test_non_tolerance() -> None:
    fps = make_fake_fingerprints(
        500, n_features=2048, seed=12620509540149709235, pack=False
    )
    legacy_fns = (
        merge_radius,
        merge_diameter,
    )
    oop_fns = (
        RadiusMerge,
        DiameterMerge,
    )
    thresholds = (0.65, 0.65, 0.3, 0.3)

    for fn_expect, Fn, thresh in zip(legacy_fns, oop_fns, thresholds):
        fn = Fn()
        for case in ("1, 1", "1, >1", ">1, 1", ">1, >1"):
            for j in range(200):
                old, nom = get_old_and_nom(fps, j, case)
                old_ls = old.sum(0)
                nom_ls = nom.sum(0)
                new_ls = old_ls + nom_ls
                old_n = len(old)
                nom_n = len(nom)
                new_n = old_n + nom_n
                cent = centroid_from_sum(new_ls, new_n, pack=False)
                val_expect = fn_expect(
                    thresh, new_ls, cent, new_n, old_ls, nom_ls, old_n, nom_n
                )
                val = fn(thresh, new_ls, new_n, old_ls, nom_ls, old_n, nom_n, [], [])
                assert val == val_expect


# These are designed to trip all cases of tolerance
def test_tolerance_radius() -> None:
    fps = make_fake_fingerprints(
        500, n_features=2048, seed=12620509540149709235, pack=False
    )
    tolerances = (0.00, 1e-8, 0.05, 0.05, 0.9, 0.5)

    expect = [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        True,
    ]
    idx = 0
    for thresh in (0.23, 1e-3):
        for j, tol in enumerate(tolerances):
            old, nom = get_old_and_nom(fps, j, ">1, >1")
            old_ls = old.sum(0)
            nom_ls = nom.sum(0)
            new_ls = old_ls + nom_ls
            old_n = len(old)
            nom_n = len(nom)
            new_n = old_n + nom_n
            fn = ToleranceRadiusMerge(tolerance=tol)
            val = fn(thresh, new_ls, new_n, old_ls, nom_ls, old_n, nom_n, [], [])
            assert val == expect[idx]
            idx += 1


def test_never_merge() -> None:
    fps = make_fake_fingerprints(
        500, n_features=2048, seed=12620509540149709235, pack=False
    )
    tolerances = range(1, 10)
    for thresh in (0.23, 0.2):
        for j, tol in enumerate(tolerances):
            old, nom = get_old_and_nom(fps, j, ">1, >1")
            old_ls = old.sum(0)
            nom_ls = nom.sum(0)
            new_ls = old_ls + nom_ls
            old_n = len(old)
            nom_n = len(nom)
            new_n = old_n + nom_n
            fn = NeverMerge(tolerance=tol)
            val = fn(thresh, new_ls, new_n, old_ls, nom_ls, old_n, nom_n, [], [])
            assert not val


# These are designed to trip all cases of tolerance
def test_tolerance_diameter() -> None:
    fps = make_fake_fingerprints(
        500, n_features=2048, seed=12620509540149709235, pack=False
    )
    tolerances = (0.00, 1e-8, 0.05, 0.05, 0.9, 0.5)

    expect = [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        True,
        True,
    ]
    idx = 0
    for thresh in (0.23, 0.2):
        for j, tol in enumerate(tolerances):
            old, nom = get_old_and_nom(fps, j, ">1, >1")
            old_ls = old.sum(0)
            nom_ls = nom.sum(0)
            new_ls = old_ls + nom_ls
            old_n = len(old)
            nom_n = len(nom)
            new_n = old_n + nom_n
            fn = ToleranceDiameterMerge(tolerance=tol)
            val = fn(thresh, new_ls, new_n, old_ls, nom_ls, old_n, nom_n, [], [])
            assert val == expect[idx]
            idx += 1


# These are designed to trip all cases of tolerance
def test_tolerance() -> None:
    fps = make_fake_fingerprints(
        500, n_features=2048, seed=12620509540149709235, pack=False
    )
    legacy_fns = (
        merge_tolerance,
        merge_tolerance,
    )
    oop_fns = (
        ToleranceLegacyMerge,
        ToleranceLegacyMerge,
    )
    thresholds = (0.2, 0.2, 0.2, 0.2)
    tolerances = (0.05, 0.05, 0.90, 0.90)
    for fn_expect, Fn, thresh, tol in zip(legacy_fns, oop_fns, thresholds, tolerances):
        fn = Fn(tol)
        fn._backwards_compat = True  # type: ignore
        for case in ("1, 1", "1, >1", ">1, 1", ">1, >1"):
            for j in range(200):
                old, nom = get_old_and_nom(fps, j, case)
                old_ls = old.sum(0)
                nom_ls = nom.sum(0)
                new_ls = old_ls + nom_ls
                old_n = len(old)
                nom_n = len(nom)
                new_n = old_n + nom_n
                cent = centroid_from_sum(new_ls, new_n, pack=False)
                val_expect = fn_expect(
                    thresh, new_ls, cent, new_n, old_ls, nom_ls, old_n, nom_n, tol
                )
                val = fn(thresh, new_ls, new_n, old_ls, nom_ls, old_n, nom_n, [], [])
                assert val == val_expect


def test_custom_merge():
    class TrackingDiameterMerge(DiameterMerge):
        def __init__(self, max_cluster_size: int) -> None:
            self.max_cluster_size = max_cluster_size
            self.redundant_mol_idxs = []

        def on_after_check_merge(self, accepted, old_idxs, nominee_idxs):
            if accepted and len(old_idxs) >= self.max_cluster_size:
                self.redundant_mol_idxs.extend(nominee_idxs)
                self.discard()

    merge_fn = TrackingDiameterMerge(max_cluster_size=32)

    fps = make_fake_fingerprints(
        1000, n_features=2048, seed=12620509540149709235, pack=False
    )
    tree = BitBirch(threshold=0.3, merge_criterion=merge_fn)
    tree.fit(fps)
    assert merge_fn.redundant_mol_idxs == snapshot(
        [
            610,
            611,
            612,
            613,
            614,
            615,
            616,
            617,
            618,
            620,
            621,
            622,
            623,
            625,
            627,
            628,
            629,
            646,
            658,
            679,
            681,
            682,
            685,
            688,
            695,
            696,
            708,
            713,
            714,
            716,
            719,
            720,
            722,
            733,
            734,
            735,
            737,
            740,
            741,
            742,
            743,
            744,
            745,
            746,
            747,
            748,
            749,
            750,
            751,
            752,
            753,
            756,
            757,
            758,
            759,
            761,
            765,
            766,
            767,
            774,
            775,
            777,
            778,
            781,
            786,
            790,
            794,
            795,
            797,
        ]
    )
    assert sum(tree.get_assignments() == 0) == snapshot(69)
