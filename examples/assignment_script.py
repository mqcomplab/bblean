import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pickle
from bblean.fingerprints import unpack_fingerprints

num_fps_to_assign = 1000
directory = "test-assignment-rdkit5"
use_medoids = False
# very high branching factor is needed for good recall (~3500-5000)
# Probably a branching factor of ~sqrt(N_fps) * 3.5-5 is good
k = 4  # More than 4 is too slow, number of searches increases as factorial(k)
# k = 1 seems to always give ~30% recall, mostly independent of branching factor
# For example, k=4 starts with 4 searches, then each search splits into 3, then
# each splits into 2, for a total of 24 searches. k=1 or k=2 is ideal
# The actual growth is slower since some nodes have les branches, but it is bounded by
# the factorial
#
# Mergeable recall is similar to the normal recall (a bit higher in general)
# k = 4 is around 75% recal with branching factor of 3500-5000
# k = 2 seems to be ~ 10x faster than flat search (but it is non-parallelizable)
# k = 4 is only ~2x / ~1.5x faster
# k = 10 is ~ 4x SLOWER and gets around 96% recall (spawns around 90 searches)

fps = np.load("../10M/packed-fps-rdkit-uint8-7f343532.006.npy")[:num_fps_to_assign]
with open(f"./{directory}/bitbirch.pkl", mode="rb") as f:
    tree = pickle.load(f)
    # NOTE: Using tolerance-diamenter doesn't change the number of mergeable fps very
    # much
    # tree.set_merge("tolerance-diameter", tolerance=0.05)
    #
    # Similar time for medoids or centroids
    if use_medoids:
        unpacked_fitted_fps = unpack_fingerprints(
            np.load(list(Path(f"./{directory}/input-fps/").glob("*.npy"))[0])
        )
        flat_assignments = tree.assign(
            fps, kind="flat", use_medoids=True, unpacked_fitted_fps=unpacked_fitted_fps
        )
        tree_assignments = tree.assign(
            fps,
            kind="tree",
            use_medoids=True,
            unpacked_fitted_fps=unpacked_fitted_fps,
            k_search=k,
        )
    else:
        _start = time.perf_counter()
        tree_assignments = tree.assign(fps, kind="tree", k_search=k)
        print(f"Time elapsed tree: {time.perf_counter() - _start} s", flush=True)
        _start = time.perf_counter()
        flat_assignments = tree.assign(fps, kind="flat")
        print(f"Time elapsed flat: {time.perf_counter() - _start} s", flush=True)

    correct_assignments = flat_assignments["is_mergeable"]

    # Clearly what is happening in the rdkit fps is that the first centroid has a *ton*
    # of 1s and the rest have much less 1s
    idxs = flat_assignments["cluster_label"]
    values, counts = np.unique(idxs, return_counts=True)
    fig, ax = plt.subplots()
    ax.bar(values, counts, width=200, alpha=0.25)
    ax.set_ylabel(r"Counts")
    ax.set_xlabel(r"Label")
    ax.set_title(f"Flat ({'medoid' if use_medoids else 'centroid'}) assignment")
    plt.show(block=False)

    idxs = tree_assignments["cluster_label"]
    values, counts = np.unique(idxs, return_counts=True)
    fig, ax = plt.subplots()
    ax.bar(values, counts, width=200, alpha=0.25)
    ax.set_ylabel(r"Counts")
    ax.set_xlabel(r"Label")
    ax.set_title(f"Tree ({'medoid' if use_medoids else 'centroid'}) assignment")
    plt.show()

    num_matches = (
        flat_assignments["cluster_label"] == tree_assignments["cluster_label"]
    ).sum()
    num_mergeable_matches = (
        flat_assignments["cluster_label"][correct_assignments]
        == tree_assignments["cluster_label"][correct_assignments]
    ).sum()

    print(f"Total: {len(flat_assignments)}")
    print(f"Total mergeable: {correct_assignments.sum()}")
    print(f"Recall: {num_matches * 100 / len(flat_assignments)}")
    print(
        f"Mergeable recall: {num_mergeable_matches * 100 / len(flat_assignments[correct_assignments])}"  # noqa
    )
