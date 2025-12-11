# type: ignore
import bblean
import numpy as np
import pandas as pd
import bblean.similarity as iSIM


def recluster_iteration_metrics(
    fps: np.ndarray,
    threshold: float = None,
    sigmas: float = 3.5,
    extra: float = 0,
    branching_factor: int = 1024,
    iterations: int = 11,
) -> pd.DataFrame:

    if threshold is None:
        # Obtain the iSIM and iSIM-sigma
        isim = iSIM.jt_isim_packed(fps)
        isim_sigma = iSIM.estimate_jt_std(fps, n_samples=50)
        threshold = isim + sigmas * isim_sigma
    else:
        isim_sigma = iSIM.estimate_jt_std(fps, n_samples=50)

    # Do the initial clustering
    bbc = bblean.BitBirch(
        threshold=threshold,
        branching_factor=branching_factor,
        merge_criterion="diameter",
    )
    bbc.fit(fps)
    clusters = bbc.get_cluster_mol_ids()

    # Collect the initial statistics
    n_clusters = [len(clusters)]
    n_singletons = [len([c for c in clusters if len(c) == 1])]
    medoids = bbc.get_medoids(fps=fps)
    dispersion = [iSIM.jt_isim(medoids)]
    dispersion_no_singletons = [
        iSIM.jt_isim(medoids[: n_clusters[0] - n_singletons[0]])
    ]
    top_sizes = [[len(c) for c in clusters[:10]]]

    # Do refinement steps
    for _ in range(iterations):
        bbc.recluster_inplace(iterations=1, extra_threshold=extra * isim_sigma)
        clusters = bbc.get_cluster_mol_ids()

        # Collect statistics
        n_clusters.append(len(clusters))
        n_singletons.append(len([c for c in clusters if len(c) == 1]))
        medoids = bbc.get_medoids(fps=fps)
        dispersion.append(iSIM.jt_isim(medoids))
        end_idx = n_clusters[-1] - n_singletons[-1]
        dispersion_no_singletons.append(iSIM.jt_isim(medoids[:end_idx]))
        top_sizes.append([len(c) for c in clusters[:10]])

        if bbc.threshold >= 1.0:
            break

    # Save the results
    results = {
        "iteration": list(range(len(n_clusters))),
        "n_clusters": n_clusters,
        "n_singletons": n_singletons,
        "dispersions": dispersion,
        "dispersions_no_singletons": dispersion_no_singletons,
        "top_sizes": top_sizes,
    }

    df = pd.DataFrame(results)

    return df


def threshold_scan(
    fps: np.ndarray,
    max_sigmas: float = 10,
    branching_factor: int = 1024,
):

    # Obtain the iSIM and iSIM-sigma
    isim = iSIM.jt_isim_packed(fps)
    isim_sigma = iSIM.estimate_jt_std(fps, n_samples=50)

    # Do the clustering for different thresholds
    thresholds = []
    n_clusters = []
    n_singletons = []
    dispersions = []
    dispersions_non_singleton = []
    top_sizes = []
    top_range = min(isim + max_sigmas * isim_sigma, 1.0)
    for threshold in np.arange(isim, top_range, isim_sigma):
        bbc = bblean.BitBirch(
            threshold=threshold,
            branching_factor=branching_factor,
            merge_criterion="diameter",
        )
        bbc.fit(fps)
        clusters = bbc.get_cluster_mol_ids()

        medoids = bbc.get_medoids(fps=fps)

        # Collect statistics
        thresholds.append(threshold)
        n_clusters.append(len(clusters))
        n_singletons.append(len([c for c in clusters if len(c) == 1]))
        dispersions.append(iSIM.jt_isim(medoids))
        end_idx = n_clusters[-1] - n_singletons[-1]
        dispersions_non_singleton.append(iSIM.jt_isim(medoids[:end_idx]))
        top_sizes.append([len(c) for c in clusters[:10]])

        # Save the results
        results = {
            "thresholds": thresholds,
            "n_clusters": n_clusters,
            "n_singletons": n_singletons,
            "dispersions": dispersions,
            "dispersions_non_singleton": dispersions_non_singleton,
            "top_sizes": top_sizes,
        }

        df = pd.DataFrame(results)

    return df


def branching_factor_scan(
    fps: np.ndarray,
    sigmas_list: list[float],
    branching_factor_list: list[int] = [64, 128, 256, 512, 1024],
    save_path: str = None,
) -> pd.DataFrame:
    # Obtain the iSIM and iSIM-sigma
    isim = iSIM.jt_isim_packed(fps)
    isim_sigma = iSIM.estimate_jt_std(fps, n_samples=50)

    # Do the clustering for different thresholds
    thresholds_list = [isim + k * isim_sigma for k in sigmas_list]
    thresholds = []
    n_clusters = []
    n_singletons = []
    dispersions = []
    dispersions_non_singleton = []
    top_sizes = []
    branching_factors = []
    times = []
    memory = []
    for threshold in thresholds_list:
        for bf in branching_factor_list:
            bbc = bblean.BitBirch(
                threshold=threshold, branching_factor=bf, merge_criterion="diameter"
            )
            bbc.fit(fps)

            clusters = bbc.get_cluster_mol_ids()

            medoids = bbc.get_medoids(fps=fps)

            # Collect statistics
            thresholds.append(threshold)
            n_clusters.append(len(clusters))
            n_singletons.append(len([c for c in clusters if len(c) == 1]))
            branching_factors.append(bf)
            dispersions.append(iSIM.jt_isim(medoids))
            end_idx = n_clusters[-1] - n_singletons[-1]
            dispersions_non_singleton.append(iSIM.jt_isim(medoids[:end_idx]))
            top_sizes.append([len(c) for c in clusters[:10]])

            # Save the results
            results = {
                "thresholds": thresholds,
                "n_clusters": n_clusters,
                "n_singletons": n_singletons,
                "dispersions": dispersions,
                "dispersions_non_singleton": dispersions_non_singleton,
                "top_sizes": top_sizes,
                "branching_factor": branching_factors,
                "time": times,
                "memory_gb": memory,
            }

    df = pd.DataFrame(results)
    if save_path is not None:
        df.to_csv(save_path, index=False)
    else:
        return df
