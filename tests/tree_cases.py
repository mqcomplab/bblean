r"""Clustering cases with stored reference results (tests/tree-reference.npz)

The reference results were produced by the python BitBirch tree of bblean (commit
e83709a, before the C++ tree was added) by running this module as a script:
``python tests/tree_cases.py <out.npz>``, which always uses the python tree. Each case
builds a tree using only the public API, and the stored result is the full cluster
membership in the tree's leaf order (``get_cluster_mol_ids(sort=False)``), which
determines the clusters, their order, and the order of the molecules inside them.
"""

import itertools
import sys
import typing as tp

import numpy as np

import bblean.bitbirch
from bblean.bitbirch import BitBirch
from bblean.fingerprints import make_fake_fingerprints, unpack_fingerprints
from bblean.merges import (
    DiameterMerge,
    DiscardSubcluster,
    FlexibleToleranceDiameterMerge,
    ToleranceDiameterMerge,
    ToleranceRadiusMerge,
)

SEED = 12620509540149709235
MERGES = [
    "diameter",
    "radius",
    "tolerance-diameter",
    "flexible-tolerance-diameter",
    "tolerance-radius",
    "tolerance-legacy",
    "never",
]


def _fps(num: int, n_features: int = 2048, pack: bool = True) -> tp.Any:
    return make_fake_fingerprints(num, n_features=n_features, seed=SEED, pack=pack)


class _ThresholdHookMerge(DiameterMerge):
    # Overrides a hook, so the tree must call back into python
    def on_check_merge_start(self, threshold: float, *args: tp.Any) -> float:
        new_n = args[1]
        return threshold + 0.1 if new_n > 5 else threshold


class _DiscardMerge(DiameterMerge):
    def on_check_merge_start(self, threshold: float, *args: tp.Any) -> float:
        if args[-1][0] % 7 == 0:
            raise DiscardSubcluster
        return threshold


def _cases() -> dict[str, tp.Callable[[], BitBirch]]:
    cases: dict[str, tp.Callable[[], BitBirch]] = {}

    for merge, (bf, thresh) in itertools.product(
        MERGES, [(50, 0.65), (7, 0.3), (3, 0.5)]
    ):

        def fit(merge: str = merge, bf: int = bf, thresh: float = thresh) -> BitBirch:
            tree = BitBirch(
                threshold=thresh, branching_factor=bf, merge_criterion=merge
            )
            return tree.fit(_fps(800))

        cases[f"{merge}-bf{bf}-t{thresh}"] = fit

    for n_features in (8, 64, 1024, 1032):

        def fit_features(n_features: int = n_features) -> BitBirch:
            fps = _fps(500, n_features)
            return BitBirch(branching_factor=5).fit(fps, n_features=n_features)

        cases[f"features-{n_features}"] = fit_features

    def fit_partial() -> BitBirch:
        fps = _fps(600)
        tree = BitBirch(threshold=0.4, branching_factor=10)
        tree.fit(fps[:200], weights=itertools.repeat(300))
        tree.fit(fps[200:400], reinsert_indices=range(1000, 1200))
        tree.set_merge("tolerance-diameter", tolerance=0.1, threshold=0.5)
        tree.fit(unpack_fingerprints(fps[400:]), input_is_packed=False)
        return tree

    cases["partial-fits-weights-indices"] = fit_partial

    def fit_list() -> BitBirch:
        return BitBirch(threshold=0.4, branching_factor=8).fit(list(_fps(400)))

    cases["list-input"] = fit_list

    def fit_refine_recluster() -> BitBirch:
        fps = _fps(1000)
        tree = BitBirch(threshold=0.3, branching_factor=20).fit(fps)
        tree.set_merge("tolerance-diameter", tolerance=0.05, threshold=0.35)
        tree.refine_inplace(fps, n_largest=2)
        tree.recluster_inplace(iterations=2, extra_threshold=0.01)
        return tree

    cases["refine-recluster"] = fit_refine_recluster

    for name, merge_fn in [
        ("tol-diam", ToleranceDiameterMerge),
        ("flex-tol-diam", FlexibleToleranceDiameterMerge),
        ("tol-radius", ToleranceRadiusMerge),
    ]:

        def fit_adaptive(merge_fn: tp.Any = merge_fn) -> BitBirch:
            # Clusters grow past n_max, where the adaptive tolerance becomes 0
            merge = merge_fn(tolerance=0.2, n_max=10, decay=0.1)
            tree = BitBirch(threshold=0.3, branching_factor=20, merge_criterion=merge)
            return tree.fit(_fps(1500))

        cases[f"adaptive-{name}"] = fit_adaptive

    def fit_constant_tolerance() -> BitBirch:
        merge = ToleranceDiameterMerge(tolerance=0.1, adaptive=False)
        tree = BitBirch(threshold=0.3, branching_factor=20, merge_criterion=merge)
        return tree.fit(_fps(1000))

    cases["constant-tolerance"] = fit_constant_tolerance

    def fit_hook() -> BitBirch:
        tree = BitBirch(
            threshold=0.3, branching_factor=10, merge_criterion=_ThresholdHookMerge()
        )
        return tree.fit(_fps(600))

    cases["python-hook-merge"] = fit_hook

    def fit_discard() -> BitBirch:
        tree = BitBirch(
            threshold=0.3, branching_factor=10, merge_criterion=_DiscardMerge()
        )
        return tree.fit(_fps(600))

    cases["python-discard-merge"] = fit_discard

    def fit_duplicates() -> BitBirch:
        fps = _fps(100)
        fps = np.concatenate([fps, fps[:50], np.zeros((30, 256), np.uint8), fps[:20]])
        return BitBirch(threshold=0.5, branching_factor=4).fit(fps)

    cases["duplicates-and-zeros"] = fit_duplicates
    return cases


CASES = _cases()

# Cases with user defined merge criteria, which always use the python tree
PYTHON_ONLY = {"python-hook-merge", "python-discard-merge"}


def flat_cluster_ids(tree: BitBirch) -> tuple[tp.Any, tp.Any]:
    r"""Molecule ids of all clusters in leaf order, concatenated, and the offsets"""
    clusters = tree.get_cluster_mol_ids(sort=False)
    ids = np.array([i for c in clusters for i in c], dtype=np.int64)
    offsets = np.cumsum([0] + [len(c) for c in clusters], dtype=np.int64)
    return ids, offsets


if __name__ == "__main__":
    # The reference results are those of the python tree
    bblean.bitbirch._cpp_bitbirch = None
    out: dict[str, tp.Any] = {}
    for name, make in CASES.items():
        tree = make()
        ids, offsets = flat_cluster_ids(tree)
        out[f"{name}/ids"] = ids
        out[f"{name}/offsets"] = offsets
        out[f"{name}/num_fitted"] = np.array(tree.num_fitted_fps)
        print(name, len(offsets) - 1, "clusters")
    np.savez_compressed(sys.argv[1], **out)
