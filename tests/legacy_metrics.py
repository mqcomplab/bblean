# type: ignore
import numpy as np


def intra_sim(clust1, clust2):
    """Similarity between clusters from the iSIM of their union"""
    n1 = len(clust1)
    n2 = len(clust2)
    combined = np.sum(clust1, axis=0) + np.sum(clust2, axis=0)
    return jt_isim(combined, n1 + n2)


def calculate_comp_sim(data):
    """Returns vector of complementary similarities"""
    n_objects = len(data) - 1
    c_total = np.sum(data, axis=0)
    comp_matrix = c_total - data
    a = comp_matrix * (comp_matrix - 1) / 2
    comp_sims = np.sum(a, axis=1) / np.sum(
        (a + comp_matrix * (n_objects - comp_matrix)), axis=1
    )
    return comp_sims


def calculate_medoid(data):
    """Returns index of medoid"""
    return np.argmin(calculate_comp_sim(data))


def jt_isim(c_total, n_objects):
    """iSIM Tanimoto calculation"""
    sum_kq = np.sum(c_total)
    sum_kqsq = np.dot(c_total, c_total)
    a = (sum_kqsq - sum_kq) / 2
    return a / (a + n_objects * sum_kq - sum_kqsq)


def jt_pair(mol1, mol2):
    """Tanimoto between two molecules"""
    a = np.dot(mol1, mol2)
    return a / (np.sum(mol1) + np.sum(mol2) - a)


def jt_one_to_many(mol, mol_set):
    """Tanimoto similarities between a molecule and a set of molecules

    mol: np.array of a single molecule
    mol_set: np.array containing the fingerprints of a set of molecules
    """
    pop_counts = np.sum(mol_set, axis=1)
    a = np.dot(mol_set, mol)
    sims = a / (np.sum(mol) + pop_counts - a)
    return sims


def calculate_centroid(linear_sum, n_samples):
    """Calculates centroid"""
    return np.floor(linear_sum / n_samples + 0.5)


def legacy_chi(clusters, reps=False, rep_type="centroid"):
    """Calinski-Harabasz index

    clusters : list of np.arrays containing the clusters

    reps : {bool, list} indicates if cluster representatives are given

    rep_type : type of representative, medoid or centroid

    curated : bool indicates if singletons have been removed

    min_size : int size below which clusters will be ignored

    Note
    ----
    Higher values are better

    """
    # n : total number of points
    n = 0

    # k : number of clusters
    k = len(clusters)

    # wcss : within-cluster sum of squares
    wcss = 0

    # bcss : between-cluster sum of squares
    bcss = 0
    total_data = []
    for clust in clusters:
        for mol in clust:
            total_data.append(mol)
    total_data = np.array(total_data)
    if rep_type == "centroid":
        linear_sum = np.sum(total_data, axis=0)
        n_samples = len(total_data)
        c = calculate_centroid(linear_sum, n_samples)
    elif rep_type == "medoid":
        medoid = calculate_medoid(total_data)
        c = total_data[medoid]
    if not reps:
        for clust in clusters:
            n_samples = len(clust)
            n += n_samples
            if rep_type == "centroid":
                linear_sum = np.sum(clust, axis=0)
                rep = calculate_centroid(linear_sum, n_samples)
            elif rep_type == "medoid":
                medoid = calculate_medoid(clust)
                rep = clust[medoid]
            distances = 1 - jt_one_to_many(rep, clust)
            wcss += np.dot(distances, distances)
            # NOTE: Fix fatal bug here, original impl lacks 1 - ...
            bcss += n_samples * (1 - jt_pair(c, rep)) ** 2
    else:
        for i, clust in enumerate(clusters):
            n_samples = len(clust)
            n += n_samples
            # NOTE: Fix fatal bug here, original impl lacks 1 - ...
            bcss += n_samples * (1 - jt_pair(c, reps[i])) ** 2
            distances = 1 - jt_one_to_many(reps[i], clust)
            wcss += np.dot(distances, distances)

    try:
        value = bcss * (n - k) / (wcss * (k - 1))
    except Exception:
        value = 0
    return value


def legacy_dbi(clusters, reps=False, rep_type="centroid"):
    """Davies-Bouldin index

    clusters : list of np.arrays containing the clusters

    reps : {bool, list} indicates if cluster representatives are given

    rep_type : type of representative, medoid or centroid

    curated : bool indicates if singletons have been removed

    min_size : int size below which clusters will be ignored

    Note
    ----
    Lower values are better

    """
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
    except Exception:
        value = 0

    return value


def legacy_dunn(
    clusters,
    cluster_sim="intra",
    reps=False,
    rep_type="centroid",
):
    """Dunn index

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
    D = []

    for clust in clusters:
        n_samples = len(clust)
        linear_sum = np.sum(clust, axis=0)
        D.append(jt_isim(linear_sum, n_samples))

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
                dij = 1 - intra_sim(clust1, clust2)
                if dij < min_d:
                    min_d = dij

    try:
        value = min_d / Dm
    except Exception:
        value = 0

    return value
