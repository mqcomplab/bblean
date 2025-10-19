r"""Clustering metrics using Tanimoto similarity"""

from numpy.typing import NDArray
import numpy as np

from bblean.similarity import (
    jt_isim_from_sum,
    jt_sim_packed,
    jt_isim_packed,
    jt_isim_unpacked,
    centroid as centroid_from_fps,
    centroid_from_sum,
    jt_isim_medoid,
)
from bblean.fingerprints import unpack_fingerprints, pack_fingerprints

__all__ = ["jt_isim_chi", "jt_isim_dunn", "jt_dbi"]


def _calc_centrals(
    cluster_fps: list[NDArray[np.uint8]],
    kind: str,
    input_is_packed: bool = True,
    n_features: int | None = None,
    pack: bool = True,
) -> list[NDArray[np.uint8]]:
    if kind == "medoid":
        return [
            jt_isim_medoid(
                c, input_is_packed=input_is_packed, n_features=n_features, pack=pack
            )[1]
            for c in cluster_fps
        ]
    elif kind == "centroid":
        return [
            centroid_from_fps(
                c, input_is_packed=input_is_packed, n_features=n_features, pack=pack
            )
            for c in cluster_fps
        ]
    raise ValueError(f"Unknown arg {kind} use 'medoids|centroids'")


def jt_isim_chi(
    cluster_fps: list[NDArray[np.uint8]],
    all_fps_central: NDArray[np.uint8] | str = "centroid",
    centrals: list[NDArray[np.uint8]] | str = "centroid",
    input_is_packed: bool = True,
    n_features: int | None = None,
) -> float:
    """Calinski-Harabasz clustering index

    An approximation to the CHI index using the Tanimoto iSIM. *Higher* is better.
    """
    all_fps_num = sum(len(c) for c in cluster_fps)
    if isinstance(all_fps_central, str):
        if not all_fps_central == "centroid":
            raise NotImplementedError("Currently only 'centroid' implemented for CHI")
        if input_is_packed:
            unpacked_clusts = [unpack_fingerprints(c, n_features) for c in cluster_fps]
        else:
            unpacked_clusts = cluster_fps
        total_linear_sum = sum(np.sum(c, axis=0) for c in unpacked_clusts)
        all_fps_central = centroid_from_sum(total_linear_sum, all_fps_num)

    # TODO: Remove reshapes once jt_sim_packed is generic over shapes
    all_fps_central = all_fps_central.reshape(1, -1)
    if isinstance(centrals, str):
        if not centrals == "centroid":
            raise NotImplementedError("Currently only 'centroid' implemented for CHI")
        centrals = _calc_centrals(cluster_fps, centrals, input_is_packed, n_features)
    else:
        if not input_is_packed:
            centrals = [pack_fingerprints(c) for c in centrals]

    clusters_num = len(cluster_fps)
    # Packed cluster_fps required for CHI
    if not input_is_packed:
        cluster_fps = [pack_fingerprints(c) for c in cluster_fps]

    if clusters_num <= 1:
        return 0

    wcss = 0.0  # within-cluster sum of squares
    bcss = 0.0  # between-cluster sum of squares
    for central, clust in zip(centrals, cluster_fps):
        # TODO: In the original implementation there isn't a (1 - jt...) here (!)
        # Is it correct like this?
        bcss += len(clust) * (1 - jt_sim_packed(all_fps_central, central).item()) ** 2
        d = 1 - jt_sim_packed(clust, central)
        wcss += np.dot(d, d)
    # TODO: When can the denom be 0?
    return bcss * (all_fps_num - clusters_num) / (wcss * (clusters_num - 1))


def jt_dbi(
    cluster_fps: list[NDArray[np.uint8]],
    centrals: list[NDArray[np.uint8]] | str = "centroid",
    input_is_packed: bool = True,
    n_features: int | None = None,
) -> float:
    """Davies-Bouldin clustering index

    DBI index using the Tanimoto distance. *Lower* is better.
    """
    if isinstance(centrals, str):
        centrals = _calc_centrals(cluster_fps, centrals, input_is_packed, n_features)
    else:
        if not input_is_packed:
            centrals = [pack_fingerprints(c) for c in centrals]

    # Centrals can be 'medoids' or 'centroids'
    if not input_is_packed:
        cluster_fps = [pack_fingerprints(c) for c in cluster_fps]
    # Packed cluster_fps required for DBI

    fps_num = 0
    S: list[float] = []
    for central, clust_fps in zip(centrals, cluster_fps):
        size = len(clust_fps)
        S.append(np.sum(1 - jt_sim_packed(clust_fps, central)) / size)
        fps_num += size

    if fps_num == 0:
        return 0

    # Quadratic scaling on num. clusters
    numerator = 0.0
    for i, central in enumerate(centrals):
        # TODO: Remove reshape once jt_sim_packed is generic over shapes
        central = central.reshape(1, -1)
        max_d = 0.0
        for j, other_central in enumerate(centrals):
            if i == j:
                continue
            Mij = 1 - jt_sim_packed(central, other_central).item()
            max_d = max(max_d, (S[i] + S[j]) / Mij)
        numerator += max_d
    return numerator / fps_num


# This is the Dunn varaint used in the original BitBirch article
def jt_isim_dunn(
    cluster_fps: list[NDArray[np.uint8]],
    input_is_packed: bool = True,
    n_features: int | None = None,
) -> float:
    """Dunn clustering index

    An approximation to the Dunn index using the Tanimoto iSIM. *Higher* is better.
    """
    # Unpacked cluster_fps required for Dunn
    if input_is_packed:
        D = [jt_isim_packed(clust) for clust in cluster_fps]
        cluster_fps = [unpack_fingerprints(clust, n_features) for clust in cluster_fps]
    else:
        D = [jt_isim_unpacked(clust) for clust in cluster_fps]
    max_d = max(D)
    if max_d == 0:
        # TODO: Unclear what to return in this case, probably 1.0 is safer?
        return 1
    min_d = 1.00
    # Quadratic scaling on num. clusters
    for i, clust1 in enumerate(cluster_fps[:-1]):
        for j, clust2 in enumerate(cluster_fps[i + 1 :]):
            combined = np.sum(clust1, axis=0) + np.sum(clust2, axis=0)
            dij = 1 - jt_isim_from_sum(combined, len(clust1) + len(clust2))
            min_d = min(dij, min_d)
    return min_d / max(D)
