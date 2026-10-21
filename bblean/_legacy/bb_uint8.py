# type: ignore
# BitBIRCH is an open-source clustering module based on iSIM
#
# Please, cite the BitBIRCH paper: https://doi.org/10.1039/D5DD00030K
#
# BitBIRCH is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# BitBIRCH is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# BitBIRCH License: GPL-3.0 https://www.gnu.org/licenses/gpl-3.0.en.html
#
# Memory-efficient BitBIRCH authors: Ramon Alain Miranda Quintana <ramirandaq@gmail.com>, <quintana@chem.ufl.edu>
#                                    Krizstina Zsigmond <kzsigmond@ufl.edu>
#
### Part of the tree-management code was derived from https://scikit-learn.org/stable/modules/generated/sklearn.cluster.Birch.html
### Authors: Manoj Kumar <manojkumarsivaraj334@gmail.com>
###          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
###          Joel Nothman <joel.nothman@gmail.com>
### License: BSD 3 clause
# Parts of the BitBIRCH algorithm were previously released under the LGPL-3.0 license by:
# Ramon Alain Miranda Quintana <ramirandaq@gmail.com>, <quintana@chem.ufl.edu>
# Vicky (Vic) Jung <jungvicky@ufl.edu>
# Kenneth Lopez Perez <klopezperez@chem.ufl.edu>
# Kate Huddleston <kdavis2@chem.ufl.edu>
from pathlib import Path

import numpy as np
from scipy import sparse


def safe_sum(nmax, np1, np2):
    if nmax >= 4294967294:
        return np1.astype("uint64") + np2.astype("uint64")
    elif nmax >= 65534:
        return np1.astype("uint32") + np2.astype("uint32")
    elif nmax >= 254:
        return np1.astype("uint16") + np2.astype("uint16")
    else:
        return np1 + np2


def _copy_or_unpack(x, n_features, input_is_packed: bool = True):
    return unpack_fingerprints(x, n_features) if input_is_packed else x.copy()


def unpack_fingerprints(a, n_features: int):
    """Unpacks uint8 arrays into boolean arrays"""
    # n_features is required to discard padded zeros if it is not a multiple of 8
    return np.unpackbits(a, axis=-1, count=n_features)


# Utility function to validate the n_features argument for packed inputs
def _validate_n_features(X, input_is_packed: bool, n_features: int | None) -> int:
    if input_is_packed:
        if n_features is None:
            raise ValueError("n_features is required for packed inputs")
        return n_features

    x_n_features = X.shape[1]
    if n_features is not None:
        if n_features != x_n_features:
            raise ValueError(
                "n_features is redundant for non-packed inputs"
                " if passed, it must be equal to X.shape[1]."
                f" For passed X, X.shape[1] = {X.shape[1]}."
                " If this value is not what you expected,"
                " make sure the passed X is actually unpacked."
            )
    return x_n_features


def set_merge(merge_criterion, tolerance=0.05):
    """
    Sets merge_accept function for merge_subcluster, based on user specified merge_criteria.

    Radius: merge subcluster based on comparison to centroid of the cluster
    Diameter: merge subcluster based on instant Tanimoto similarity of cluster
    Tolerance: applies tolerance threshold to diameter merge criteria, which will merge subcluster with stricter threshold for newly added molecules

    Parameters:
    -----------
    merge_criterion: str();
                        radius, diameter or tolerance
    tolerance: float;
                        sets penalty value for similarity threshold when callng tolerance merge criteria

    Returns:
    --------
    merge_accept(): function
                        if cluster is accepted to merge, merge the cluster based on the criteria specified
    """
    if merge_criterion == "radius":

        def merge_accept(
            threshold, new_ls, new_centroid, new_n, old_ls, nom_ls, old_n, nom_n
        ):
            jt_sim = jt_isim(new_ls + new_centroid, new_n + 1) * (new_n + 1) - jt_isim(
                new_ls, new_n
            ) * (new_n - 1)
            return jt_sim >= threshold * 2

    elif merge_criterion == "diameter":

        def merge_accept(
            threshold, new_ls, new_centroid, new_n, old_ls, nom_ls, old_n, nom_n
        ):
            jt_radius = jt_isim(new_ls, new_n)
            return jt_radius >= threshold

    elif merge_criterion == "tolerance_tough":

        def merge_accept(
            threshold, new_ls, new_centroid, new_n, old_ls, nom_ls, old_n, nom_n
        ):
            jt_radius = jt_isim(new_ls, new_n)
            if jt_radius < threshold:
                return False
            else:
                if old_n == 1 and nom_n == 1:
                    return True
                elif nom_n == 1:
                    return (
                        jt_isim(old_ls + nom_ls, old_n + 1) * (old_n + 1)
                        - jt_isim(old_ls, old_n) * (old_n - 1)
                    ) / 2 >= jt_isim(old_ls, old_n) - tolerance and (
                        jt_radius >= threshold
                    )
                else:
                    return (
                        jt_isim(old_ls + nom_ls, old_n + nom_n)
                        * (old_n + nom_n)
                        * (old_n + nom_n - 1)
                        - jt_isim(old_ls, old_n) * old_n * (old_n - 1)
                        - jt_isim(nom_ls, nom_n) * nom_n * (nom_n - 1)
                    ) / (2 * old_n * nom_n) >= jt_isim(old_ls, old_n) - tolerance and (
                        jt_radius >= threshold
                    )

    elif merge_criterion in ["tolerance", "tolerance-legacy"]:

        def merge_accept(
            threshold, new_ls, new_centroid, new_n, old_ls, nom_ls, old_n, nom_n
        ):
            jt_radius = jt_isim(new_ls, new_n)
            if jt_radius < threshold:
                return False
            else:
                if old_n == 1 and nom_n == 1:
                    return True
                elif nom_n == 1:
                    return (
                        jt_isim(old_ls + nom_ls, old_n + 1) * (old_n + 1)
                        - jt_isim(old_ls, old_n) * (old_n - 1)
                    ) / 2 >= jt_isim(old_ls, old_n) - tolerance and (
                        jt_radius >= threshold
                    )
                else:
                    return True

    globals()["merge_accept"] = merge_accept


def jt_isim(c_total, n_objects):
    """iSIM Tanimoto calculation

    https://pubs.rsc.org/en/content/articlelanding/2024/dd/d4dd00041b

    Parameters
    ----------
    c_total : np.ndarray
              Sum of the elements column-wise

    n_objects : int
                Number of elements

    Returns
    ----------
    isim : float
           iSIM Jaccard-Tanimoto value
    """
    x = c_total.astype("uint64")
    sum_kq = np.sum(x)
    sum_kqsq = np.dot(x, x)
    a = (sum_kqsq - sum_kq) / 2
    return a / (a + n_objects * sum_kq - sum_kqsq)


def max_separation(Y):
    """Finds two objects in X that are very separated
    This is an approximation (not guaranteed to find
    the two absolutely most separated objects), but it is
    a very robust O(N) implementation. Quality of clustering
    does not diminish in the end.

    Algorithm:
    a) Find centroid of X
    b) mol1 is the molecule most distant from the centroid
    c) mol2 is the molecule most distant from mol1

    Returns
    -------
    (mol1, mol2) : (int, int)
                   indices of mol1 and mol2
    1 - sims_mol1 : np.ndarray
                   Distances to mol1
    1 - sims_mol2: np.ndarray
                   Distances to mol2
    These are needed for node1_dist and node2_dist in _split_node
    """
    # Get the centroid of the set
    X = Y.astype("uint64")
    n_samples = len(X)
    linear_sum = np.sum(X, axis=0)
    centroid = calc_centroid(linear_sum, n_samples)

    # Get the similarity of each molecule to the centroid
    pop_counts = np.sum(X, axis=1)
    a_centroid = np.dot(X, centroid)
    sims_med = a_centroid / (pop_counts + np.sum(centroid) - a_centroid)

    # Get the least similar molecule to the centroid
    mol1 = np.argmin(sims_med)

    # Get the similarity of each molecule to mol1
    a_mol1 = np.dot(X, X[mol1])
    sims_mol1 = a_mol1 / (pop_counts + pop_counts[mol1] - a_mol1)

    # Get the least similar molecule to mol1
    mol2 = np.argmin(sims_mol1)

    # Get the similarity of each molecule to mol2
    a_mol2 = np.dot(X, X[mol2])
    sims_mol2 = a_mol2 / (pop_counts + pop_counts[mol2] - a_mol2)

    return (mol1, mol2), sims_mol1, sims_mol2


def calc_centroid(linear_sum, n_samples):
    """Calculates centroid

    Parameters
    ----------

    linear_sum : np.ndarray
                 Sum of the elements column-wise
    n_samples : int
                Number of samples

    Returns
    -------
    centroid : np.ndarray
               Centroid fingerprints of the given set
    """
    cent = np.where(linear_sum >= n_samples * 0.5, 1, 0)
    return cent.astype("bool")


def _iterate_sparse_X(X):
    """This little hack returns a densified row when iterating over a sparse
    matrix, instead of constructing a sparse matrix for every row that is
    expensive.
    """
    n_samples, n_features = X.shape
    X_indices = X.indices
    X_data = X.data
    X_indptr = X.indptr

    for i in range(n_samples):
        row = np.zeros(n_features)
        startptr, endptr = X_indptr[i], X_indptr[i + 1]
        nonzero_indices = X_indices[startptr:endptr]
        row[nonzero_indices] = X_data[startptr:endptr]
        yield row


def _split_node(node, threshold, branching_factor):
    """The node has to be split if there is no place for a new subcluster
    in the node.
    1. Two empty nodes and two empty subclusters are initialized.
    2. The pair of distant subclusters are found.
    3. The properties of the empty subclusters and nodes are updated
       according to the nearest distance between the subclusters to the
       pair of distant subclusters.
    4. The two nodes are set as children to the two subclusters.
    """
    new_subcluster1 = _BFSubcluster()
    new_subcluster2 = _BFSubcluster()
    new_node1 = _BFNode(
        threshold=threshold,
        branching_factor=branching_factor,
        is_leaf=node.is_leaf,
        n_features=node.n_features,
        dtype=node.init_centroids_.dtype,
    )
    new_node2 = _BFNode(
        threshold=threshold,
        branching_factor=branching_factor,
        is_leaf=node.is_leaf,
        n_features=node.n_features,
        dtype=node.init_centroids_.dtype,
    )
    new_subcluster1.child_ = new_node1
    new_subcluster2.child_ = new_node2

    if node.is_leaf:
        if node.prev_leaf_ is not None:
            node.prev_leaf_.next_leaf_ = new_node1
        new_node1.prev_leaf_ = node.prev_leaf_
        new_node1.next_leaf_ = new_node2
        new_node2.prev_leaf_ = new_node1
        new_node2.next_leaf_ = node.next_leaf_
        if node.next_leaf_ is not None:
            node.next_leaf_.prev_leaf_ = new_node2

    # O(N) implementation of max separation
    farthest_idx, node1_dist, node2_dist = max_separation(node.centroids_)
    # Notice that max_separation is returning similarities and not distances
    node1_closer = node1_dist > node2_dist
    # Make sure node1 is closest to itself even if all distances are equal.
    # This can only happen when all node.centroids_ are duplicates leading to all
    # distances between centroids being zero.
    node1_closer[farthest_idx[0]] = True

    for idx, subcluster in enumerate(node.subclusters_):
        if node1_closer[idx]:
            new_node1.append_subcluster(subcluster)
            new_subcluster1.update(subcluster)
            # if not singly:
            #    subcluster.parent_ = new_subcluster1
        else:
            new_node2.append_subcluster(subcluster)
            new_subcluster2.update(subcluster)
            # if not singly:
            #    subcluster.parent_ = new_subcluster2
    return new_subcluster1, new_subcluster2


class _BFNode:
    """Each node in a BFTree is called a BFNode.

    The BFNode can have a maximum of branching_factor
    number of BFSubclusters.

    Parameters
    ----------
    threshold : float
        Threshold needed for a new subcluster to enter a BFSubcluster.

    branching_factor : int
        Maximum number of BF subclusters in each node.

    is_leaf : bool
        We need to know if the BFNode is a leaf or not, in order to
        retrieve the final subclusters.

    n_features : int
        The number of features.

    Attributes
    ----------
    subclusters_ : list
        List of subclusters for a particular BFNode.

    prev_leaf_ : _BFNode
        Useful only if is_leaf is True.

    next_leaf_ : _BFNode
        next_leaf. Useful only if is_leaf is True.
        the final subclusters.

    init_centroids_ : ndarray of shape (branching_factor + 1, n_features)
        Manipulate ``init_centroids_`` throughout rather than centroids_ since
        the centroids are just a view of the ``init_centroids_`` .

    centroids_ : ndarray of shape (branching_factor + 1, n_features)
        View of ``init_centroids_``.

    """

    def __init__(self, *, threshold, branching_factor, is_leaf, n_features, dtype):
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.is_leaf = is_leaf
        self.n_features = n_features

        # The list of subclusters, centroids and squared norms
        # to manipulate throughout.
        self.subclusters_ = []
        self.init_centroids_ = np.zeros((branching_factor + 1, n_features), dtype=dtype)
        self.prev_leaf_ = None
        self.next_leaf_ = None

    def append_subcluster(self, subcluster):
        n_samples = len(self.subclusters_)
        self.subclusters_.append(subcluster)
        self.init_centroids_[n_samples] = subcluster.centroid_

        # Keep centroids as views. In this way
        # if we change init_centroids, it is sufficient
        self.centroids_ = self.init_centroids_[: n_samples + 1, :]

    def update_split_subclusters(self, subcluster, new_subcluster1, new_subcluster2):
        """Remove a subcluster from a node and update it with the
        split subclusters.
        """

        ind = self.subclusters_.index(subcluster)
        self.subclusters_[ind] = new_subcluster1
        self.init_centroids_[ind] = new_subcluster1.centroid_
        self.centroids_[ind] = new_subcluster1.centroid_
        self.append_subcluster(new_subcluster2)

    def insert_bf_subcluster(self, subcluster, set_bits):
        """Insert a new subcluster into the node."""
        if not self.subclusters_:
            self.append_subcluster(subcluster)
            return False

        threshold = self.threshold
        branching_factor = self.branching_factor
        # We need to find the closest subcluster among all the
        # subclusters so that we can insert our new subcluster.
        sub_centroids = self.centroids_.astype("uint16")
        in_centroid = subcluster.centroid_.astype("uint16")
        a = np.dot(sub_centroids, in_centroid)
        sim_matrix = a / (np.sum(sub_centroids, axis=1) + set_bits - a)
        closest_index = np.argmax(sim_matrix)
        closest_subcluster = self.subclusters_[closest_index]

        # If the subcluster has a child, we need a recursive strategy.
        if closest_subcluster.child_ is not None:

            split_child = closest_subcluster.child_.insert_bf_subcluster(
                subcluster, set_bits
            )

            if not split_child:
                # If it is determined that the child need not be split, we
                # can just update the closest_subcluster
                closest_subcluster.update(subcluster)
                self.init_centroids_[closest_index] = self.subclusters_[
                    closest_index
                ].centroid_
                self.centroids_[closest_index] = self.subclusters_[
                    closest_index
                ].centroid_
                return False

            # things not too good. we need to redistribute the subclusters in
            # our child node, and add a new subcluster in the parent
            # subcluster to accommodate the new child.
            else:
                new_subcluster1, new_subcluster2 = _split_node(
                    closest_subcluster.child_, threshold, branching_factor
                )
                self.update_split_subclusters(
                    closest_subcluster, new_subcluster1, new_subcluster2
                )

                if len(self.subclusters_) > self.branching_factor:
                    return True
                return False

        # good to go!
        else:
            merged = closest_subcluster.merge_subcluster(subcluster, self.threshold)
            if merged:
                self.centroids_[closest_index] = closest_subcluster.centroid_
                self.init_centroids_[closest_index] = closest_subcluster.centroid_

                return False

            # not close to any other subclusters, and we still
            # have space, so add.
            elif len(self.subclusters_) < self.branching_factor:
                self.append_subcluster(subcluster)

                return False

            # We do not have enough space nor is it closer to an
            # other subcluster. We need to split.
            else:
                self.append_subcluster(subcluster)
                return True


class _BFSubcluster:
    """Each subcluster in a BFNode is called a BFSubcluster.

    A BFSubcluster can have a BFNode has its child.

    Parameters
    ----------
    linear_sum : ndarray of shape (n_features,), default=None
        Sample. This is kept optional to allow initialization of empty
        subclusters.

    Attributes
    ----------
    n_samples_ : int
        Number of samples that belong to each subcluster.

    linear_sum_ : ndarray
        Linear sum of all the samples in a subcluster. Prevents holding
        all sample data in memory.

    centroid_ : ndarray of shape (branching_factor + 1, n_features)
        Centroid of the subcluster. Prevent recomputing of centroids when
        ``BFNode.centroids_`` is called.

    mol_indices : list, default=[]
        List of indices of molecules included in the given cluster.

    child_ : _BFNode
        Child Node of the subcluster. Once a given _BFNode is set as the child
        of the _BFNode, it is set to ``self.child_``.
    """

    def __init__(self, *, linear_sum=None, mol_indices=[]):
        if linear_sum is None:
            self.n_samples_ = 0
            self.centroid_ = self.linear_sum_ = np.zeros((2048,), dtype="bool")
            self.mol_indices = []
        else:
            self.n_samples_ = 1
            self.centroid_ = self.linear_sum_ = linear_sum
            self.mol_indices = mol_indices

        self.child_ = None
        self.parent_ = None

    def update(self, subcluster):
        self.n_samples_ += np.uint64(subcluster.n_samples_)
        self.linear_sum_ = safe_sum(
            self.n_samples_, self.linear_sum_, subcluster.linear_sum_
        )
        self.mol_indices += subcluster.mol_indices
        self.centroid_ = calc_centroid(self.linear_sum_, self.n_samples_)

    def merge_subcluster(self, nominee_cluster, threshold):
        """Check if a cluster is worthy enough to be merged. If
        yes then merge.
        """
        new_n = np.uint64(self.n_samples_) + np.uint64(nominee_cluster.n_samples_)
        new_ls = safe_sum(new_n, self.linear_sum_, nominee_cluster.linear_sum_)
        new_centroid = calc_centroid(new_ls, new_n)

        if merge_accept(
            threshold,
            new_ls,
            new_centroid,
            new_n,
            self.linear_sum_,
            nominee_cluster.linear_sum_,
            self.n_samples_,
            nominee_cluster.n_samples_,
        ):
            (self.n_samples_, self.linear_sum_, self.centroid_, self.mol_indices) = (
                new_n,
                new_ls,
                new_centroid,
                self.mol_indices + nominee_cluster.mol_indices,
            )
            return True
        return False


class BitBirch:
    """Implements the BitBIRCH clustering algorithm.

    BitBIRCH paper:

    Memory- and time-efficient, online-learning algorithm.
    It constructs a tree data structure with the cluster centroids being read off the leaf.

    Parameters
    ----------
    threshold : float, default=0.5
        The similarity radius of the subcluster obtained by merging a new sample and the
        closest subcluster should be greater than the threshold. Otherwise a new
        subcluster is started. Setting this value to be very low promotes
        splitting and vice-versa.

    branching_factor : int, default=50
        Maximum number of BF subclusters in each node. If a new samples enters
        such that the number of subclusters exceed the branching_factor then
        that node is split into two nodes with the subclusters redistributed
        in each. The parent subcluster of that node is removed and two new
        subclusters are added as parents of the 2 split nodes.

    Attributes
    ----------
    root_ : _BFNode
        Root of the BFTree.

    dummy_leaf_ : _BFNode
        Start pointer to all the leaves.

    subcluster_centers_ : ndarray
        Centroids of all subclusters read directly from the leaves.

    Notes
    -----
    The tree data structure consists of nodes with each node consisting of
    a number of subclusters. The maximum number of subclusters in a node
    is determined by the branching factor. Each subcluster maintains a
    linear sum, mol_indices and the number of samples in that subcluster.
    In addition, each subcluster can also have a node as its child, if the
    subcluster is not a member of a leaf node.

    For a new point entering the root, it is merged with the subcluster closest
    to it and the linear sum, mol_indices and the number of samples of that
    subcluster are updated. This is done recursively till the properties of
    the leaf node are updated.
    """

    def __init__(
        self,
        *,
        threshold=0.5,
        branching_factor=50,
    ):
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.index_tracker = 0
        self.first_call = True

    def fit(
        self,
        X,
        store_centroids: bool = False,
        input_is_packed: bool = True,
        n_features: int | None = None,
        max_fps: int | None = None,
    ):
        """
        Build a BF Tree for the input data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        self
            Fitted estimator.
        """
        if isinstance(X, Path):
            X = np.load(X, mmap_mode="r")[:max_fps]
        else:
            X = X[:max_fps]
        threshold = self.threshold
        branching_factor = self.branching_factor
        n_features = _validate_n_features(X, input_is_packed, n_features)
        d_type = X.dtype

        # If partial_fit is called for the first time or fit is called, we
        # start a new tree.
        if self.first_call:
            # The first root is the leaf. Manipulate this object throughout.
            self.root_ = _BFNode(
                threshold=threshold,
                branching_factor=branching_factor,
                is_leaf=True,
                n_features=n_features,
                dtype=d_type,
            )

            # To enable getting back subclusters.
            self.dummy_leaf_ = _BFNode(
                threshold=threshold,
                branching_factor=branching_factor,
                is_leaf=True,
                n_features=n_features,
                dtype=d_type,
            )
            self.dummy_leaf_.next_leaf_ = self.root_
            self.root_.prev_leaf_ = self.dummy_leaf_

        # Cannot vectorize. Enough to convince to use cython.
        if not sparse.issparse(X):
            iter_func = iter
        else:
            iter_func = _iterate_sparse_X

        for sample in iter_func(X):
            unpack = _copy_or_unpack(sample, n_features, input_is_packed)
            set_bits = np.sum(unpack.astype("uint64"))
            subcluster = _BFSubcluster(
                linear_sum=unpack, mol_indices=[self.index_tracker]
            )
            split = self.root_.insert_bf_subcluster(subcluster, set_bits)

            if split:
                new_subcluster1, new_subcluster2 = _split_node(
                    self.root_, threshold, branching_factor
                )
                del self.root_
                self.root_ = _BFNode(
                    threshold=threshold,
                    branching_factor=branching_factor,
                    is_leaf=False,
                    n_features=n_features,
                    dtype=d_type,
                )
                self.root_.append_subcluster(new_subcluster1)
                self.root_.append_subcluster(new_subcluster2)

            self.index_tracker += 1
        if store_centroids:
            centroids = np.concatenate([leaf.centroids_ for leaf in self._get_leaves()])
            self.subcluster_centers_ = centroids
            self._n_features_out = self.subcluster_centers_.shape[0]

        self.first_call = False
        return self

    def fit_np(self, X):
        threshold = self.threshold
        branching_factor = self.branching_factor

        n_features = X.shape[1] - 1
        d_type = X.dtype

        # If partial_fit is called for the first time or fit is called, we
        # start a new tree.
        if self.first_call:
            # The first root is the leaf. Manipulate this object throughout.
            self.root_ = _BFNode(
                threshold=threshold,
                branching_factor=branching_factor,
                is_leaf=True,
                n_features=n_features,
                dtype=d_type,
            )

            # To enable getting back subclusters.
            self.dummy_leaf_ = _BFNode(
                threshold=threshold,
                branching_factor=branching_factor,
                is_leaf=True,
                n_features=n_features,
                dtype=d_type,
            )
            self.dummy_leaf_.next_leaf_ = self.root_
            self.root_.prev_leaf_ = self.dummy_leaf_

        # Cannot vectorize. Enough to convince to use cython.
        if not sparse.issparse(X):
            iter_func = iter
        else:
            iter_func = _iterate_sparse_X

        for sample in iter_func(X):
            sample_copy = sample.copy()
            subcluster = _BFSubcluster(
                linear_sum=sample_copy[:-1], mol_indices=[self.index_tracker]
            )
            n_samples = sample_copy[-1]
            if n_samples > 1:
                subcluster.n_samples_ = n_samples
                subcluster.centroid_ = calc_centroid(sample_copy[:-1], n_samples)
            set_bits = np.sum(subcluster.centroid_.astype("uint64"))
            split = self.root_.insert_bf_subcluster(subcluster, set_bits)

            if split:
                new_subcluster1, new_subcluster2 = _split_node(
                    self.root_, threshold, branching_factor
                )
                del self.root_
                self.root_ = _BFNode(
                    threshold=threshold,
                    branching_factor=branching_factor,
                    is_leaf=False,
                    n_features=n_features,
                    dtype=d_type,
                )
                self.root_.append_subcluster(new_subcluster1)
                self.root_.append_subcluster(new_subcluster2)

            self.index_tracker += 1

        self.first_call = False
        return self

    def fit_np_reinsert(self, X, reinsert_indices):
        threshold = self.threshold
        branching_factor = self.branching_factor

        n_features = X.shape[1] - 1
        d_type = X.dtype

        # If partial_fit is called for the first time or fit is called, we
        # start a new tree.
        if self.first_call:
            # The first root is the leaf. Manipulate this object throughout.
            self.root_ = _BFNode(
                threshold=threshold,
                branching_factor=branching_factor,
                is_leaf=True,
                n_features=n_features,
                dtype=d_type,
            )

            # To enable getting back subclusters.
            self.dummy_leaf_ = _BFNode(
                threshold=threshold,
                branching_factor=branching_factor,
                is_leaf=True,
                n_features=n_features,
                dtype=d_type,
            )
            self.dummy_leaf_.next_leaf_ = self.root_
            self.root_.prev_leaf_ = self.dummy_leaf_

        # Cannot vectorize. Enough to convince to use cython.
        if not sparse.issparse(X):
            iter_func = iter
        else:
            iter_func = _iterate_sparse_X

        for sample, mol_inds in zip(iter_func(X), reinsert_indices):
            sample_copy = sample.copy()
            subcluster = _BFSubcluster(
                linear_sum=sample_copy[:-1], mol_indices=mol_inds
            )
            n_samples = sample_copy[-1]
            if n_samples > 1:
                subcluster.n_samples_ = n_samples
                subcluster.centroid_ = calc_centroid(sample_copy[:-1], n_samples)
            set_bits = np.sum(subcluster.centroid_.astype("uint64"))
            split = self.root_.insert_bf_subcluster(subcluster, set_bits)

            if split:
                new_subcluster1, new_subcluster2 = _split_node(
                    self.root_, threshold, branching_factor
                )
                del self.root_
                self.root_ = _BFNode(
                    threshold=threshold,
                    branching_factor=branching_factor,
                    is_leaf=False,
                    n_features=n_features,
                    dtype=d_type,
                )
                self.root_.append_subcluster(new_subcluster1)
                self.root_.append_subcluster(new_subcluster2)

        self.first_call = False
        return self

    def fit_reinsert(
        self,
        X,
        reinsert_indices,
        store_centroids: bool = False,
        input_is_packed: bool = True,
        n_features: int | None = None,
    ):
        """X corresponds to only the molecules that will be reinserted into the tree
        reinsert indices are the indices of the molecules that will be reinserted into the tree
        """
        threshold = self.threshold
        branching_factor = self.branching_factor
        n_features = _validate_n_features(X, input_is_packed, n_features)
        d_type = X.dtype

        # If partial_fit is called for the first time or fit is called, we
        # start a new tree.
        if self.first_call:
            # The first root is the leaf. Manipulate this object throughout.
            self.root_ = _BFNode(
                threshold=threshold,
                branching_factor=branching_factor,
                is_leaf=True,
                n_features=n_features,
                dtype=d_type,
            )

            # To enable getting back subclusters.
            self.dummy_leaf_ = _BFNode(
                threshold=threshold,
                branching_factor=branching_factor,
                is_leaf=True,
                n_features=n_features,
                dtype=d_type,
            )
            self.dummy_leaf_.next_leaf_ = self.root_
            self.root_.prev_leaf_ = self.dummy_leaf_

        # Cannot vectorize. Enough to convince to use cython.
        if not sparse.issparse(X):
            iter_func = iter
        else:
            iter_func = _iterate_sparse_X

        for sample, mol_ind in zip(iter_func(X), reinsert_indices):
            unpack = _copy_or_unpack(sample, n_features, input_is_packed)
            set_bits = np.sum(unpack.astype("uint64"))
            subcluster = _BFSubcluster(linear_sum=unpack, mol_indices=[mol_ind])
            split = self.root_.insert_bf_subcluster(subcluster, set_bits)
            if split:
                new_subcluster1, new_subcluster2 = _split_node(
                    self.root_, threshold, branching_factor
                )
                del self.root_
                self.root_ = _BFNode(
                    threshold=threshold,
                    branching_factor=branching_factor,
                    is_leaf=False,
                    n_features=n_features,
                    dtype=d_type,
                )
                self.root_.append_subcluster(new_subcluster1)
                self.root_.append_subcluster(new_subcluster2)

        if store_centroids:
            centroids = np.concatenate([leaf.centroids_ for leaf in self._get_leaves()])
            self.subcluster_centers_ = centroids
            self._n_features_out = self.subcluster_centers_.shape[0]

        self.first_call = False
        return self

    def _get_leaves(self):
        """
        Retrieve the leaves of the BF Node.

        Returns
        -------
        leaves : list of shape (n_leaves,)
            List of the leaf nodes.
        """
        leaf_ptr = self.dummy_leaf_.next_leaf_
        leaves = []
        while leaf_ptr is not None:
            leaves.append(leaf_ptr)
            leaf_ptr = leaf_ptr.next_leaf_
        return leaves

    def get_centroids_mol_ids(self):
        """Method to return a dictionary containing the centroids and mol indices of the leaves"""
        if self.first_call:
            raise ValueError("The model has not been fitted yet.")

        centroids = []
        mol_ids = []
        for leaf in self._get_leaves():
            for subcluster in leaf.subclusters_:
                centroids.append(subcluster.centroid_)
                mol_ids.append(subcluster.mol_indices)

        dict_centroids_mol_ids = {"centroids": centroids, "mol_ids": mol_ids}

        return dict_centroids_mol_ids

    def get_centroids(self):
        """Method to return a list of Numpy arrays containing the centroids' fingerprints"""
        if self.first_call:
            raise ValueError("The model has not been fitted yet.")

        centroids = []
        for leaf in self._get_leaves():
            for subcluster in leaf.subclusters_:
                centroids.append(subcluster.centroid_)

        return centroids

    def get_cluster_mol_ids(self):
        """Method to return the indices of molecules in each cluster"""
        if self.first_call:
            raise ValueError("The model has not been fitted yet.")

        clusters_mol_id = []
        for leaf in self._get_leaves():
            for subcluster in leaf.subclusters_:
                clusters_mol_id.append(subcluster.mol_indices)

        # Sort the clusters by the number of samples in the cluster
        clusters_mol_id = sorted(clusters_mol_id, key=lambda x: len(x), reverse=True)

        return clusters_mol_id

    def _get_BFs(self):
        """Method to return the BitFeatures of the leaves"""
        if self.first_call:
            raise ValueError("The model has not been fitted yet.")

        BFs = []
        for leaf in self._get_leaves():
            for subcluster in leaf.subclusters_:
                BFs.append(subcluster)

        # Sort the BitFeatures by the number of samples in the cluster
        BFs = sorted(BFs, key=lambda x: x.n_samples_, reverse=True)

        return BFs

    def bf_to_np_refine(
        self,
        fps,
        initial_mol=0,
        input_is_packed: bool = True,
        n_features: int | None = None,
    ):
        """Method to prepare the BitFeatures of the largest cluster and the rest of the clusters"""
        if self.first_call:
            raise ValueError("The model has not been fitted yet.")
        n_features = _validate_n_features(fps, input_is_packed, n_features)
        BFs = self._get_BFs()
        big, rest = BFs[0], BFs[1:]

        fp_64 = []
        fp_32 = []
        fp_16 = []
        fp_8 = []

        mols_64 = []
        mols_32 = []
        mols_16 = []
        mols_8 = []

        for BF in rest:
            if BF.n_samples_ >= 4294967294:
                fp_64.append(BF.linear_sum_)
                fp_64.append(BF.n_samples_)
                mols_64.append(BF.mol_indices)
            elif BF.n_samples_ >= 65534:
                fp_32.append(BF.linear_sum_)
                fp_32.append(BF.n_samples_)
                mols_32.append(BF.mol_indices)
            elif BF.n_samples_ >= 254:
                fp_16.append(list(BF.linear_sum_))
                fp_16[-1].append(BF.n_samples_)
                mols_16.append(BF.mol_indices)
            else:
                fp_8.append(list(BF.linear_sum_))
                fp_8[-1].append(BF.n_samples_)
                mols_8.append(BF.mol_indices)

        for mol in big.mol_indices:
            if input_is_packed:
                fp_8.append(
                    list(unpack_fingerprints(fps[mol - initial_mol], n_features))
                )
            else:
                fp_8.append(list(fps[mol - initial_mol]))
            fp_8[-1].append(1)
            mols_8.append(BF.mol_indices)

        fps_bfs = []
        mols_bfs = []
        if len(fp_64) != 0:
            fp_64 = np.array(fp_64, dtype=np.uint64)
            fps_bfs.append(fp_64)
            mols_bfs.append(mols_64)
        if len(fp_32) != 0:
            fp_32 = np.array(fp_32, dtype=np.uint32)
            fps_bfs.append(fp_32)
            mols_bfs.append(mols_32)
        if len(fp_16) != 0:
            fp_16 = np.array(fp_16, dtype=np.uint16)
            fps_bfs.append(fp_16)
            mols_bfs.append(mols_16)
        if len(fp_8) != 0:
            fp_8 = np.array(fp_8, dtype=np.uint8)
            fps_bfs.append(fp_8)
            mols_bfs.append(mols_8)

        return fps_bfs, mols_bfs

    def bf_to_np(self):
        """Method to prepare the BitFeatures of the largest cluster and the rest of the clusters"""
        if self.first_call:
            raise ValueError("The model has not been fitted yet.")

        BFs = self._get_BFs()

        fp_64 = []
        fp_32 = []
        fp_16 = []
        fp_8 = []

        mols_64 = []
        mols_32 = []
        mols_16 = []
        mols_8 = []

        for BF in BFs:
            if BF.n_samples_ >= 4294967294:
                fp_64.append(BF.linear_sum_)
                fp_64.append(BF.n_samples_)
                mols_64.append(BF.mol_indices)
            elif BF.n_samples_ >= 65534:
                fp_32.append(BF.linear_sum_)
                fp_32.append(BF.n_samples_)
                mols_32.append(BF.mol_indices)
            elif BF.n_samples_ >= 254:
                fp_16.append(list(BF.linear_sum_))
                fp_16[-1].append(BF.n_samples_)
                mols_16.append(BF.mol_indices)
            else:
                fp_8.append(list(BF.linear_sum_))
                fp_8[-1].append(BF.n_samples_)
                mols_8.append(BF.mol_indices)

        fps_bfs = []
        mols_bfs = []
        if len(fp_64) != 0:
            fp_64 = np.array(fp_64, dtype=np.uint64)
            fps_bfs.append(fp_64)
            mols_bfs.append(mols_64)
        if len(fp_32) != 0:
            fp_32 = np.array(fp_32, dtype=np.uint32)
            fps_bfs.append(fp_32)
            mols_bfs.append(mols_32)
        if len(fp_16) != 0:
            fp_16 = np.array(fp_16, dtype=np.uint16)
            fps_bfs.append(fp_16)
            mols_bfs.append(mols_16)
        if len(fp_8) != 0:
            fp_8 = np.array(fp_8, dtype=np.uint8)
            fps_bfs.append(fp_8)
            mols_bfs.append(mols_8)

        return fps_bfs, mols_bfs

    def get_assignments(self, n_mols):
        clustered_ids = self.get_cluster_mol_ids()

        assignments = np.full(n_mols, -1, dtype=int)
        for i, cluster in enumerate(clustered_ids):
            assignments[cluster] = i + 1

        # Check that there are no unassigned molecules
        assert np.all(assignments != -1)

        return assignments
