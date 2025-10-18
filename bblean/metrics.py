import numpy as np
from bblean.similarity import jt_isim_from_sum as jt_isim


def intra_sim(clust1, clust2):
    """Similarity between clusters from the iSIM of their union"""
    n1 = len(clust1)
    n2 = len(clust2)
    combined = np.sum(clust1, axis=0) + np.sum(clust2, axis=0)
    return jt_isim(combined, n1 + n2)


def inter_sim(clust1, clust2):
    """Similarity between clusters from the average distance between their elements"""
    n1 = len(clust1)
    n2 = len(clust2)
    n = n1 + n2
    c_total1 = np.sum(clust1, axis=0)
    c_total2 = np.sum(clust2, axis=0)
    combined = c_total1 + c_total2
    return (
        jt_isim(combined, n1 + n2) * n * (n - 1)
        - (jt_isim(c_total1) * n1 * (n1 - 1) + jt_isim(c_total2) * n2 * (n2 - 1))
    ) / (2 * n1 * n2)


def dbi(clusters, reps=False, rep_type="centroid", min_size=1):
    """Davies-Bouldin index

    clusters : list of np.arrays containing the clusters

    reps : {bool, list} indicates if cluster representatives are given

    rep_type : type of representative, medoid or centroid

    min_size : int size below which clusters will be ignored

    Note
    ----
    Lower values are better
    """
    clusters = select_large(clusters, min_size)

    n = 0

    S = []

    if not reps:
        reps = []
        for clust in clusters:
            n_samples = len(clust)
            n += n_samples
            if rep_type == "centroid":
                linear_sum = np.sum(clust, axis=0)
                rep = calculate_centroid(linear_sum, n_samples)
            elif rep_type == "medoid":
                medoid = calculate_medoid(clust)
                rep = clust[medoid]
            reps.append(rep)
            S.append(np.sum(1 - jt_one_to_many(rep, clust)) / n_samples)
    else:
        for i, clust in enumerate(clusters):
            n_samples = len(clust)
            n += n_samples
            S.append(np.sum(1 - jt_one_to_many(reps[i], clust)) / n_samples)

    db = 0

    for i, clust in enumerate(clusters):
        d = []
        for j, other_clust in enumerate(clusters):
            if i == j:
                d.append(-1)
            else:
                Mij = 1 - jt_pair(reps[i], reps[j])
                Rij = (S[i] + S[j]) / Mij
                d.append(Rij)
        db += max(d)

    try:
        value = db / n
    except:
        value = 0

    return value


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
            D.append(jt_isim(linear_sum, n_samples))
    elif intra_sim == "mean":
        if not reps:
            for clust in clusters:
                n_samples = len(clust)
                if rep_type == "centroid":
                    linear_sum = np.sum(clust, axis=0)
                    rep = calculate_centroid(linear_sum, n_samples)
                elif rep_type == "medoid":
                    medoid = calculate_medoid(clust)
                    rep = clust[medoid]
                d = (
                    1
                    - (
                        jt_isim(linear_sum + rep, n_samples + 1) * (n_samples + 1)
                        - jt_isim(linear_sum, n_samples) * (n_samples - 1)
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
                        jt_isim(linear_sum + reps[i], n_samples + 1) * (n_samples + 1)
                        - jt_isim(linear_sum, n_samples) * (n_samples - 1)
                    )
                    / 2
                )
                D.append(d)

    if len(D) == 0:
        Dm = 0
    else:
        Dm = max(D)

    # initial min_d value could be any number > 1
    min_d = 3.08

    for i, clust1 in enumerate(clusters):
        for j, clust2 in enumerate(clusters):
            if i == j:
                pass
            else:
                if cluster_sim == "intra":
                    dij = 1 - intra_sim(clust1, clust2)
                elif cluster_sim == "inter":
                    dij = 1 - inter_sim(clust1, clust2)
                if dij < min_d:
                    min_d = dij

    try:
        value = min_d / Dm
    except:
        value = 0

    return value


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
        isim_clusts += jt_isim(linear_sum, n_samples)
        if not reps:
            if rep_type == "centroid":
                representatives.append(calculate_centroid(linear_sum, n_samples))
            elif rep_type == "medoid":
                medoid = calculate_medoid(clust)
                representatives.append(clust[medoid])

    av_isim_clusts = isim_clusts / n_clusters

    representatives = np.array(representatives)
    rep_linear_sum = np.sum(representatives, axis=0)
    rep_isim = jt_isim(rep_linear_sum, n_clusters)

    try:
        value = rep_isim / av_isim_clusts
    except:
        value = 0

    return value
