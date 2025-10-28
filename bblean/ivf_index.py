r"""IVF (Inverted File) search index implementation using BitBIRCH clustering.

IVF is efficient search index for chemical fingerprints that uses BitBIRCH to partition
the cluster space. Searches in this space use approximate nearest neighbors (ANN).
"""

from collections import defaultdict
from numpy.typing import NDArray
import math
import time
import typing as tp
import numpy as np

from bblean.bitbirch import BitBirch
from bblean.similarity import jt_sim_packed
from bblean.fingerprints import pack_fingerprints


class IVFIndex:
    """
    Inverted File (IVF) index for efficient similarity search of chemical fingerprints.

    The index uses BitBIRCH clustering to partition fingerprints into clusters,
    then at query time, only the most relevant clusters are searched, providing
    a significant speedup over exhaustive search.

    Attributes:
        n_clusters (int): Number of clusters to use. If None, uses sqrt(n_samples)
        threshold (float): Similarity threshold for BitBIRCH clustering
        branching_factor (int): Branching factor for BitBIRCH clustering
        cluster_centroids (np.ndarray): Centroids of each cluster
        cluster_members (Dict[int, list[int]]): Mapping of cluster IDs to member
        fingerprint indices
        fingerprints (np.ndarray): Stored fingerprints for similarity search
        smiles (list[str]): Optional SMILES strings corresponding to fingerprints
        built (bool): Whether the index has been built
    """

    def __init__(
        self,
        n_clusters: int | None = None,
        threshold: float = 0.7,
        branching_factor: int = 50,
    ):
        """
        Initialize the IVF index.

        Args:
            n_clusters: Number of clusters to use (required)
            threshold: Similarity threshold for BitBIRCH clustering (used only for tree
            building)
            branching_factor: Branching factor for BitBIRCH clustering
        """
        if n_clusters is not None and n_clusters <= 0:
            raise ValueError("n_clusters must be a positive integer or None")

        self.n_clusters = n_clusters
        self.threshold = threshold
        self.branching_factor = branching_factor

        # Will be populated during build_index
        self.cluster_centroids_packed = np.array([], dtype=np.uint8)  # packed
        self.cluster_members: dict[int, list[int]] = {}
        self.fingerprints = np.array([], dtype=np.uint8)
        self.smiles: list[str] = []
        self.is_built = False

    def build_index(
        self,
        fingerprints: NDArray[np.uint8],
        smiles: tp.Sequence[str] = (),
        method: str = "kmeans-normalized",
        input_is_packed: bool = False,
        n_features: int | None = None,
        verbose: bool = False,
        **method_kwargs: tp.Any,
    ) -> None:
        """
        Build the IVF index by clustering fingerprints using BitBIRCH.

        Args:
            fingerprints: Binary fingerprints of shape (n_samples, n_features)
            smiles: Optional list of SMILES strings corresponding to fingerprints
        """
        n_samples = fingerprints.shape[0]
        if self.n_clusters is None:
            n_clusters = max(int(math.sqrt(n_samples)), 1)
        else:
            n_clusters = self.n_clusters

        fingerprints = fingerprints.astype(np.uint8, copy=False)
        if not input_is_packed:
            fingerprints = pack_fingerprints(fingerprints)

        # Store (packed) fingerprints and smiles for later use
        self.fingerprints = fingerprints
        self.smiles = list(smiles)

        # Always use k-clusters functionality since n_clusters is required
        if verbose:
            print(f"Clustering {n_samples} fps into exactly {n_clusters} clusters...")

        # Initialize BitBIRCH for clustering
        birch = BitBirch(
            threshold=self.threshold, branching_factor=self.branching_factor
        )
        birch.fit(fingerprints)
        birch.global_clustering(method=method, n_clusters=n_clusters, **method_kwargs)

        # Fetch new cluster centroids and members
        bf_labels = birch._global_clustering_centroid_labels
        unique_clusters = np.unique(birch._global_clustering_centroid_labels)
        if verbose:
            print(f"Found {len(unique_clusters)} unique clusters")
        root = birch._root
        assert root is not None  # mypy
        cluster_ls = np.zeros((len(unique_clusters), root.n_features), dtype=np.uint64)
        cluster_samples = np.zeros(len(unique_clusters), dtype=np.uint64)
        cluster_members: dict[int, list[int]] = defaultdict(list)
        for i, bf in enumerate(birch._get_leaf_bfs()):
            cluster_ls[bf_labels[i]] += bf.linear_sum
            cluster_samples[bf_labels[i]] += bf.n_samples
            cluster_members[bf_labels[i]].extend(bf.mol_indices)

        centroids = (cluster_ls >= cluster_samples * 0.5).view(np.uint8)
        self.cluster_centroids_packed = np.packbits(centroids, axis=-1)
        self.cluster_members = cluster_members

        self.is_built = True
        if verbose:
            print(f"IVF index built with {len(self.cluster_centroids_packed)} clusters")

    def _find_candidates(
        self, query_fp_packed: NDArray[np.uint8], n_probe: int, verbose: bool = False
    ) -> NDArray[np.int64]:
        """
        Find the n_probe nearest clusters to the query fingerprint.

        Args:
            query_fp: Query fingerprint (numpy array or RDKit ExplicitBitVect)
            n_probe: Number of clusters to return

        Returns:
            list of cluster IDs, sorted by similarity to query
        """

        # Limit n_probe to available clusters
        n_probe = min(n_probe, len(self.cluster_centroids_packed))

        t1 = time.time()
        similarities = jt_sim_packed(self.cluster_centroids_packed, query_fp_packed)
        centroid_sim_time = time.time() - t1

        t2 = time.time()
        # Get indices of top n_probe most similar centroids
        # TODO: Probably inefficient
        top_indices = np.argsort(similarities)[-n_probe:][::-1]  # Sort descending
        # Map index to cluster ID - fix the mapping!
        members = self.cluster_members
        candidates = []
        for idx in top_indices:
            candidates.extend(members[idx.item()])
        sort_time = time.time() - t2

        if verbose:
            print("  Cluster search details:")
            print(
                f"    Centroid sims:"
                f" {centroid_sim_time*1000:.2f}ms"
                f" (vs {len(self.cluster_centroids_packed)} centroids)"
            )
            print(f"    Sorting/mapping/gathering: {sort_time*1000:.2f}ms")
        return np.array(candidates)

    def search(
        self,
        query_fp: NDArray[np.uint8],
        k: int = 10,
        n_probe: int = 1,
        threshold: float = 0.0,
        input_is_packed: bool = False,
        verbose: bool = False,
    ) -> list[dict[str, tp.Union[int, float, str]]]:
        """
        Search for the k most similar fingerprints to the query.

        Args:
            query_fp: Query fingerprint (numpy array or RDKit ExplicitBitVect)
            k: Number of results to return
            n_probe: Number of clusters to search
            threshold: Minimum similarity threshold (0.0 means no threshold)

        Returns:
            list of dictionaries containing search results, each with:
                - 'index': Index of the fingerprint
                - 'similarity': Tanimoto similarity to query
                - 'smiles': SMILES string (if available)
        """

        if not self.is_built:
            raise RuntimeError("Index has not been built. Call build_index first.")
        if not input_is_packed:
            query_fp = pack_fingerprints(query_fp)
        # Find nearest clusters
        t1 = time.time()
        candidate_indices = self._find_candidates(query_fp, n_probe, verbose=verbose)
        cluster_time = time.time() - t1

        # Calculate similarities based on method and available formats
        t3 = time.time()
        # Get candidate fingerprints and convert if needed
        similarities = jt_sim_packed(self.fingerprints[candidate_indices], query_fp)
        sim_time = time.time() - t3

        # Apply threshold filter
        t4 = time.time()
        if threshold > 0.0:
            valid_idxs = (similarities > threshold).nonzero()[0].reshape(-1)
            similarities = similarities[valid_idxs]
            candidate_indices = candidate_indices[valid_idxs]

        # Sort by similarity (descending)
        sorted_indices = np.argsort(similarities)[::-1][:k]

        # Prepare results
        # TODO: Inefficient
        results = []
        for idx in sorted_indices:
            result = {
                "index": candidate_indices[idx],
                "similarity": similarities[idx],
            }

            # Add SMILES if available
            if self.smiles:
                result["smiles"] = self.smiles[candidate_indices[idx]]
            results.append(result)
        post_time = time.time() - t4

        # Print timing breakdown
        total_time = cluster_time + sim_time + post_time
        if verbose:
            print("IVF Search timing breakdown:")
            print(
                f"  Find clusters and gather candidates: {cluster_time*1000:.2f}ms ({cluster_time/total_time*100:.1f}%)"
            )
            print(
                f"  Similarity calc: {sim_time*1000:.2f}ms ({sim_time/total_time*100:.1f}%)"
            )
            print(
                f"  Post-processing: {post_time*1000:.2f}ms ({post_time/total_time*100:.1f}%)"
            )
            print(
                f"  Total: {total_time*1000:.2f}ms, candidates: {len(candidate_indices)}"
            )
        return results
