from numpy.typing import NDArray
import numpy as np

from legacy_merges import (  # type: ignore
    merge_radius,
    merge_diameter,
    merge_tolerance,
)
from bblean._merges import (
    RadiusMerge,
    DiameterMerge,
    ToleranceMerge,
)
from bblean.fingerprints import make_fake_fingerprints, centroid_from_sum

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
                val = fn(thresh, new_ls, new_n, old_ls, nom_ls, old_n, nom_n)
                assert val == val_expect


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
        ToleranceMerge,
        ToleranceMerge,
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
                val = fn(thresh, new_ls, new_n, old_ls, nom_ls, old_n, nom_n)
                assert val == val_expect
