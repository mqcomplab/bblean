r"""IVF (Inverted File) search index implementation using BitBIRCH clustering.

IVF is efficient search index for chemical fingerprints that uses BitBIRCH to partition
the cluster space. Searches in this space use approximate nearest neighbors (ANN).
"""

import pickle
from pathlib import Path
import typing_extensions as tpx
import dataclasses
import math
import typing as tp
import numpy as np
from numpy.typing import NDArray

from bblean.bitbirch import BitBirch
from bblean.smiles import load_smiles
from bblean.similarity import jt_sim_packed
from bblean.fingerprints import pack_fingerprints, unpack_fingerprints


@dataclasses.dataclass
class SearchResult:
    index: int
    similarity: float
    smi: str | None

    def __repr__(self) -> str:
        sim = self.similarity
        if sim > 1e-4:
            sim_str = f"{sim:.4f}"
        elif sim > 0:
            sim_str = f"{sim:.4e}"
        elif sim == 0:
            sim_str = "0"
        else:
            raise RuntimeError("Negative similarity found")
        out = f"SearchResult(index={self.index}, similarity={sim_str}"
        if self.smi is not None:
            return f"{out}, smi='{self.smi}')"
        return f"{out})"


class IVFIndex:
    r"""
    Inverted File (IVF) index for efficient similarity search of chemical fingerprints.

    The index uses BitBIRCH clustering to partition fingerprints into clusters,
    then at query time, only the most relevant clusters are searched, providing
    a significant speedup over exhaustive search.
    """

    def __init__(
        self,
        medoids_packed: NDArray[np.uint8],
        members: tp.Sequence[list[int]],
        fps: NDArray[np.uint8],
        smiles: tp.Sequence[str] | NDArray[np.str_] = (),
        input_is_packed: bool = True,
        n_features: int | None = None,
    ):
        # Build directly from global clusters
        self._medoids_packed = medoids_packed
        self._members = list(members)
        fps = fps.astype(np.uint8, copy=False)
        if not input_is_packed:
            fps = pack_fingerprints(fps)
        self._packed_fps = fps
        self._smiles = np.asarray(smiles, dtype=np.str_)

    @classmethod
    def from_dir(cls, idx_path: Path) -> tpx.Self:
        global_cluster_medoids_path = idx_path / "global-cluster-medoids-packed.npy"
        global_clusters_path = idx_path / "global-clusters.pkl"
        fps_path = idx_path / "fps.npy"
        smiles_path = idx_path / "smiles.smi"
        with open(global_clusters_path, "rb") as f:
            members = pickle.load(f)
        medoids_packed = np.load(global_cluster_medoids_path)
        fps = np.load(fps_path)
        smiles = load_smiles(smiles_path)
        return cls(medoids_packed, members, fps, smiles)

    @classmethod
    def from_bitbirch_clusters(
        cls,
        members: tp.Sequence[list[int]],
        centrals: NDArray[np.uint8],
        fps: NDArray[np.uint8],
        smiles: tp.Sequence[str] | NDArray[np.str_] = (),
        method: str = "kmeans",
        n_clusters: int | None = None,
        input_is_packed: bool = True,
        n_features: int | None = None,
        sort: bool = True,
        **method_kwargs: tp.Any,
    ) -> tpx.Self:
        """Build the IVF index from bitbirch clusters"""
        n_samples = fps.shape[0]
        if n_clusters is None:
            n_clusters = max(int(math.sqrt(n_samples)), 1)
        if n_clusters is not None and n_clusters <= 0:
            raise ValueError("n_clusters must be a positive integer or None")

        fps = fps.astype(np.uint8, copy=False)
        if input_is_packed:
            fps = unpack_fingerprints(fps, n_features)
            centrals = unpack_fingerprints(centrals, n_features)
        labels = BitBirch._centrals_global_clustering(
            centrals, n_clusters, method=method, **method_kwargs
        )

        num_centrals = len(centrals)
        n_clusters = n_clusters if num_centrals > n_clusters else num_centrals
        mol_ids = BitBirch._new_ids_from_labels(members, labels - 1, n_clusters)
        if sort:
            mol_ids.sort(key=lambda x: len(x), reverse=True)
        medoids = BitBirch._unpacked_medoids_from_members(fps, mol_ids)
        fps = pack_fingerprints(fps)
        return cls(pack_fingerprints(medoids), mol_ids, fps, smiles)

    def _find_candidate_idxs(
        self, query_fp_packed: NDArray[np.uint8], n_probe: int
    ) -> NDArray[np.int64]:
        """
        Find the n_probe nearest clusters to the query fingerprint.

        Args:
            query_fp: Query fingerprint (numpy array or RDKit ExplicitBitVect)
            n_probe: Number of clusters to return

        Returns:
            list of cluster IDs, sorted by similarity to query
        """
        similarities = jt_sim_packed(self._medoids_packed, query_fp_packed)
        # Get indices of top n_probe most similar medoids, limiting to avail clusters
        n_probe = min(n_probe, len(self._medoids_packed))
        top_indices = np.argsort(similarities)[-n_probe:]
        candidates = []
        for idx in top_indices:
            candidates.extend(self._members[idx])
        return np.array(candidates)

    def search(
        self,
        query_fp: NDArray[np.uint8],
        k: int = 10,
        n_probe: int = 1,
        threshold: float = 0.0,
        input_is_packed: bool = True,
        n_features: int | None = None,
    ) -> list[SearchResult]:
        """
        Search for the k most similar fingerprints to the query.

        Args:
            query_fp: Query fingerprint (numpy array or RDKit ExplicitBitVect)
            k: Number of results to return
            n_probe: Number of clusters to search
            threshold: Minimum similarity threshold (0.0 means no threshold)
            n_features: provided for API consistency only, does nothing.

        Returns:
            list of SearchResult, in sorted order, each with:
                - index: Index of the fingerprint
                - similarity: Tanimoto similarity to query
                - smi: SMILES string (if available, else None)
        """
        if k <= 0:
            raise ValueError("k must be > 0")
        if n_probe <= 0:
            raise ValueError("n_probe must be > 0")

        if not input_is_packed:
            query_fp = pack_fingerprints(query_fp)

        candidates = self._find_candidate_idxs(query_fp, n_probe)
        similarities = jt_sim_packed(self._packed_fps[candidates], query_fp)

        # Apply threshold filter
        if threshold > 0.0:
            is_selected = similarities >= threshold
            similarities = similarities[is_selected]
            candidates = candidates[is_selected]

        # Sort by similarity (descending) and prepare results
        sorted_indices = np.argsort(similarities)[::-1][:k]
        results = []
        for idx in sorted_indices:
            fp_idx = candidates[idx].item()
            smi = self._smiles[fp_idx].strip() if self._smiles.size > 0 else None
            results.append(SearchResult(fp_idx, similarities[idx].item(), smi))
        return results
