from numpy.typing import NDArray
import numpy as np
from bblean.similarity import jt_isim_from_sum, jt_isim_medoid, centroid_from_sum


# For Dunn
def _intra_sim(clust1: NDArray[np.uint8], clust2: NDArray[np.uint8]) -> float:
    """Similarity between clusters from the iSIM of their union"""
    n1 = len(clust1)
    n2 = len(clust2)
    combined = np.sum(clust1, axis=0) + np.sum(clust2, axis=0)
    return jt_isim_from_sum(combined, n1 + n2)


# For Dunn
def _inter_sim(clust1, clust2) -> float:
    """Similarity between clusters from the average distance between their elements"""
    n1 = len(clust1)
    n2 = len(clust2)
    n = n1 + n2
    c_total1 = np.sum(clust1, axis=0)
    c_total2 = np.sum(clust2, axis=0)
    combined = c_total1 + c_total2
    # This function has a fatal bug TODO: Check how to implement correctly
    raise ValueError()
    # return (
    # jt_isim_from_sum(combined, n1 + n2) * n * (n - 1)
    # - (
    # jt_isim_from_sum(c_total1) * n1 * (n1 - 1)
    # + jt_isim_from_sum(c_total2) * n2 * (n2 - 1)
    # )
    # ) / (2 * n1 * n2)

    # if central_kind == "centroid":
    # all_fps_central = centroid_from_fps(fps)
    # elif central_kind == "medoid":
    # _, all_fps_central = jt_isim_medoid(fps)


def chi(
    cluster_fps: list[NDArray[np.uint8]],
    centrals: list[NDArray[np.uint8]],
    all_fps: NDArray[np.uint8],
    all_fps_central: NDArray[np.uint8],
) -> float:
    """Calinski-Harabasz index

    Note
    ----
    Higher values are better
    """
    fps_num = len(all_fps)
    clusters_num = len(cluster_fps)

    if clusters_num <= 1:
        return 0

    wcss = 0  # within-cluster sum of squares
    bcss = 0  # between-cluster sum of squares
    for central, clust_fps in zip(centrals, cluster_fps):
        bcss += len(clust_fps) * jt_sim_packed(all_fps_central, central) ** 2
        d = 1 - jt_sim_packed(central, clust_fps)
        wcss += np.dot(d, d)

    return bcss * (fps_num - clusters_num) / (wcss * (fps_num - 1))


# NOTE: Input must be packed for now
def dbi(
    cluster_fps: list[NDArray[np.uint8]], centrals: list[NDArray[np.uint8]]
) -> float:
    # Centrals can be 'medoids' or 'centroids'
    fps_num = 0
    S = []
    for central, clust_fps in zip(centrals, cluster_fps):
        size = len(clust_fps)
        S.append(np.sum(1 - jt_sim_packed(central, clust_fps)) / size)
        fps_num += size

    if fps_num == 0:
        return 0

    # Quadratic scaling on num. clusters I believe
    numerator = 0
    for i, central in enumerate(centrals):
        maxd = 0
        for j, other_central in enumerate(centrals):
            if i == j:
                continue
            Mij = 1 - jt_sim_packed(central, other_central)
            maxd = max(maxd, (S[i] + S[j]) / Mij)
        numerator += maxd

    return numerator / fps_num


def dunn(
    clusters,
    within_sim="isim",
    cluster_sim="intra",
    reps=False,
    rep_type="centroid",
    min_size=1,
):
    """Dunn index

    within_sim : {'isim', 'mean'} type of intra cluster similarity
        isim : isim of the cluster
        mean : mean distances to the cluster representative

    cluster_sim : {'intra', 'inter'} type of similarity between clusters
        intra : intra_sim
        inter : inter_sim

    clusters : list of np.arrays containing the clusters

    reps : {bool, list} indicates if cluster representatives are given

    rep_type : type of representative, medoid or centroid

    curated : bool indicates if singletons have been removed

    min_size : int size below which clusters will be ignored

    Note
    ----
    Higher values are better

    """
    clusters = select_large(clusters, min_size)

    D = []

    if within_sim == "isim":
        for clust in clusters:
            n_samples = len(clust)
            linear_sum = np.sum(clust, axis=0)
            D.append(jt_isim_from_sum(linear_sum, n_samples))
    elif within_sim == "mean":
        if not reps:
            for clust in clusters:
                n_samples = len(clust)
                if rep_type == "centroid":
                    linear_sum = np.sum(clust, axis=0)
                    rep = centroid_from_sum(linear_sum, n_samples)
                elif rep_type == "medoid":
                    medoid = jt_isim_medoid(clust)
                    rep = clust[medoid]
                d = (
                    1
                    - (
                        jt_isim_from_sum(linear_sum + rep, n_samples + 1)
                        * (n_samples + 1)
                        - jt_isim_from_sum(linear_sum, n_samples) * (n_samples - 1)
                    )
                    / 2
                )
                D.append(d)
        else:
            for i, clust in enumerate(clusters):
                n_samples = len(clust)
                d = (
                    1
                    - (
                        jt_isim_from_sum(linear_sum + reps[i], n_samples + 1)
                        * (n_samples + 1)
                        - jt_isim_from_sum(linear_sum, n_samples) * (n_samples - 1)
                    )
                    / 2
                )
                D.append(d)
    if not D:
        return 0


def dunn_isim(
    clusters: list[NDArray[np.uint8]],
    cluster_sim="intra",
) -> float:
    D = [jt_isim_packed(clust_fps) for clust_fps in clusters]
    min_d = 1.00
    # Quadratic scaling on num. clusters I believe
    for i, clust1 in enumerate(clusters[:-1]):
        for j, clust2 in enumerate(clusters[i + 1 :]):
            if cluster_sim == "intra":
                dij = 1 - _intra_sim(clust1, clust2)
            elif cluster_sim == "inter":
                dij = 1 - _inter_sim(clust1, clust2)
            min_d = min(dij, min_d)
    return min_d / max(D)


def clust_dispersion(clusters, reps=False, rep_type="centroid", min_size=1):
    """Cluster dispersion

    cd = isim(representatives)/<isim(clusters)>

    clusters : list of np.arrays containing the clusters

    reps : {bool, list} indicates if cluster representatives are given

    rep_type : type of representative, medoid or centroid

    curated : bool indicates if singletons have been removed

    min_size : int size below which clusters will be ignored

    Note
    ----
    Lower values are better TODO: DOUBLE CHECK THIS!!!!
    """
    clusters = select_large(clusters, min_size)

    n_clusters = len(clusters)

    if n_clusters == 1:
        return -1

    isim_clusts = 0

    if not reps:
        representatives = []
    else:
        representatives = reps

    for clust in clusters:
        n_samples = len(clust)
        linear_sum = np.sum(clust, axis=0)
        isim_clusts += jt_isim_from_sum(linear_sum, n_samples)
        if not reps:
            if rep_type == "centroid":
                representatives.append(centroid_from_sum(linear_sum, n_samples))
            elif rep_type == "medoid":
                medoid = jt_isim_medoid(clust)
                representatives.append(clust[medoid])

    av_isim_clusts = isim_clusts / n_clusters

    representatives = np.array(representatives)
    rep_linear_sum = np.sum(representatives, axis=0)
    rep_isim = jt_isim_from_sum(rep_linear_sum, n_clusters)

    # ??
    if av_isim_clusts == 0:
        return 0
    return rep_isim / av_isim_clusts
