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
# Memory-efficient BitBIRCH authors:
#     Ramon Alain Miranda Quintana <ramirandaq@gmail.com>, <quintana@chem.ufl.edu>
#     Krizstina Zsigmond <kzsigmond@ufl.edu>
#
# Part of the tree-management code was derived from:
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.Birch.html
# Authors: Manoj Kumar <manojkumarsivaraj334@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Joel Nothman <joel.nothman@gmail.com>
# License: BSD 3 clause
# Parts of the BitBIRCH algorithm were previously released under LGPL-3.0 license by:
#     Ramon Alain Miranda Quintana <ramirandaq@gmail.com>, <quintana@chem.ufl.edu>
#     Vicky (Vic) Jung <jungvicky@ufl.edu>
#     Kenneth Lopez Perez <klopezperez@chem.ufl.edu>
#     Kate Huddleston <kdavis2@chem.ufl.edu>

import typing as tp
import itertools
from collections import defaultdict
from weakref import WeakSet

import numpy as np
from scipy import sparse


# Returns the minimum uint dtype that safely holds a (positive) python int
# Input must be a positive python integer
def min_safe_uint(nmax: int) -> np.dtype:
    out = np.min_scalar_type(nmax)
    # Check if the dtype is a pointer to a python bigint
    if out.hasobject:
        raise ValueError(f"n_samples: {nmax} is too large to hold in a uint64 array")
    return out


# For backwards compatibility with the global "set_merge", keep weak references to all
# the BitBirch instances and update them when set_merge is called
_BITBIRCH_INSTANCES: set["BitBirch"] = WeakSet()


# For backwards compatibility: global function used to accept merges
_global_merge_accept = None


# For backwards compatibility: set the global merge_accept function
def set_merge(merge_criterion: str, tolerance=0.05):
    r"""Sets the global criteria for merging subclusters in any BitBirch tree

    ..  warning::
        The use of this function is discouraged, instead please use either `bb_tree =
        BitBirch(...); BitBirch.set_merge(merge_criterion=..., tolerance=...)`
        or directly `bb_tree = BitBirch(..., merge_criterion=..., tolerance=...)`.

    Parameters:
    -----------
    merge_criterion: str
        radius, diameter or tolerance
        radius: merge subcluster based on comparison to centroid of the cluster
        diameter: merge subcluster based on instant Tanimoto similarity of cluster
        tolerance: applies tolerance threshold to diameter merge criteria, which will
            merge subcluster with stricter threshold for newly added molecules

    tolerance: float
        Penalty value for similarity threshold of the 'tolerance' merge criteria
    """
    # Set the global merge_accept function
    global _global_merge_accept
    _global_merge_accept = _get_merge_accept_fn(merge_criterion, tolerance)
    for bbirch in _BITBIRCH_INSTANCES:
        bbirch._merge_accept_fn = _global_merge_accept


# Utility function, either copies or unpacks a fingerprint
# NOTE: Copy of 'sample' is not required if input is packed,
# since unpack_fingerprints allocates a new array
def _copy_or_unpack(x, n_features, input_is_packed: bool = True):
    return unpack_fingerprints(x, n_features) if input_is_packed else x.copy()


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


class MergeAcceptFunction:
    # For the merge functions, although outputs of jt_isim f64, directly using f64 is
    # *not* faster than starting with uint64
    def __call__(self, threshold, new_ls, new_n, old_ls, nom_ls, old_n, nom_n) -> bool:
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class RadiusMerge(MergeAcceptFunction):
    name = "radius"

    def __call__(self, threshold, new_ls, new_n, old_ls, nom_ls, old_n, nom_n):
        # NOTE: Use uint64 sum since jt_isim casts to uint64 internally
        # This prevents multiple casts
        new_unpacked_centroid = calc_centroid(new_ls, new_n, pack=False)
        new_ls_1 = np.add(new_ls, new_unpacked_centroid, dtype=np.uint64)
        new_n_1 = new_n + 1
        new_jt = jt_isim(new_ls, new_n)
        new_jt_1 = jt_isim(new_ls_1, new_n_1)
        return new_jt_1 * new_n_1 - new_jt * (new_n - 1) >= threshold * 2


class DiameterMerge(MergeAcceptFunction):
    name = "diameter"

    def __call__(self, threshold, new_ls, new_n, old_ls, nom_ls, old_n, nom_n):
        return jt_isim(new_ls, new_n) >= threshold


class ToleranceMerge(MergeAcceptFunction):
    name = "tolerance"

    def __init__(self, tolerance: float = 0.05) -> None:
        self._tolerance = tolerance

    def __call__(self, threshold, new_ls, new_n, old_ls, nom_ls, old_n, nom_n):
        # First two branches are equivalent to 'diameter'
        new_jt = jt_isim(new_ls, new_n)
        if new_jt < threshold:
            return False
        if old_n == 1 or nom_n != 1:
            return True
        # 'new_jt >= threshold' and 'new_n == old_n + 1' are guaranteed here
        old_jt = jt_isim(old_ls, old_n)
        return (new_jt * new_n - old_jt * (old_n - 1)) / 2 >= old_jt - self._tolerance

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._tolerance})"


class ToleranceToughMerge(ToleranceMerge):
    name = "tolerance_tough"

    def __call__(self, threshold, new_ls, new_n, old_ls, nom_ls, old_n, nom_n):
        # First two branches are equivalent to 'diameter', third to 'tolerance'
        new_jt = jt_isim(new_ls, new_n)
        if new_jt < threshold:
            return False
        if old_n == 1 and nom_n == 1:
            return True

        old_jt = jt_isim(old_ls, old_n)
        if nom_n == 1:
            # 'new_jt >= threshold' and 'new_n == old_n + 1' are guaranteed here
            return (
                new_jt * new_n - old_jt * (old_n - 1)
            ) / 2 >= old_jt - self._tolerance

        # "tough" branch
        nom_jt = jt_isim(nom_ls, nom_n)
        new_term = new_jt * new_n * (new_n - 1)
        old_term = old_jt * old_n * (old_n - 1)
        nom_term = nom_jt * nom_n * (nom_n - 1)
        denom = 2 * old_n * nom_n
        return (new_term - old_term - nom_term) / denom >= old_jt - self._tolerance


def _get_merge_accept_fn(
    merge_criterion: str, tolerance: float = 0.05
) -> MergeAcceptFunction:
    if merge_criterion == "radius":
        return RadiusMerge()
    elif merge_criterion == "diameter":
        return DiameterMerge()
    elif merge_criterion == "tolerance":
        return ToleranceMerge(tolerance)
    elif merge_criterion == "tolerance_tough":
        return ToleranceToughMerge(tolerance)
    raise ValueError(
        f"Unknown merge criterion {merge_criterion}"
        "Valid criteria are: radius|diameter|tolerance|tolerance_tough"
    )


# Requires numpy >= 2.0
def popcount(a):
    # a is packed uint8 array with last axis = bytes
    # Sum bit-counts across bytes to get per-object totals

    # If the array has columns that are a multiple of 8, doing a bitwise count
    # over the buffer reinterpreted as uint64 is slightly faster.
    # This is zero cost if the exception is not triggered. Not having a be a multiple of
    # 8 is a very unlikely scenario, since fps are typically 1024 or 2048
    try:
        b = a.view(np.uint64)
    except ValueError:
        b = a
    return np.bitwise_count(b).sum(axis=-1, dtype=np.uint32)


def pack_fingerprints(a):
    """Packs boolean or integer arrays into uint8 arrays"""
    # packbits may pad with zeros if n_features is not a multiple of 8
    return np.packbits(a, axis=-1)


def unpack_fingerprints(a, n_features: int):
    """Unpacks uint8 arrays into boolean arrays"""
    # n_features is required to discard padded zeros if it is not a multiple of 8
    return np.unpackbits(a, axis=-1, count=n_features)


def tanimoto_sim_packed(arr, vec, cardinalities=None):
    """Tanimoto similarity between a matrix of packed fingerprints and a single packed
    fingerprint.

    If "cardinalities" is passed, it must be the result of calling popcount(arr).
    """
    intersection = popcount(np.bitwise_and(arr, vec))
    if cardinalities is None:
        cardinalities = popcount(arr)
    # Return value requires an out-of-place operation since it casts uints to f64
    return intersection / (cardinalities + popcount(vec) - intersection)


def jt_isim(c_total, n_objects: int):
    r"""iSIM Tanimoto calculation

    iSIM Tanimoto was first propsed in:
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
    x = c_total.astype("uint64", copy=False)
    sum_kq = np.sum(x)
    sum_kqsq = np.dot(x, x)
    a = (sum_kqsq - sum_kq) / 2
    return a / (a + n_objects * sum_kq - sum_kqsq)


def max_separation(Y, n_features: int):
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
    X_unpacked = unpack_fingerprints(Y, n_features)
    n_samples = len(X_unpacked)
    # np.sum() automatically promotes to uint64 unless forced to a smaller dtype
    linear_sum = np.sum(X_unpacked, axis=0, dtype=min_safe_uint(n_samples))
    centroid_packed = calc_centroid(linear_sum, n_samples, pack=True)

    cardinalities = popcount(Y)
    # Get the similarity of each molecule to the centroid
    sims_med = tanimoto_sim_packed(Y, centroid_packed, cardinalities)

    # Get the least similar molecule to the centroid
    mol1 = np.argmin(sims_med)

    # Get the similarity of each molecule to mol1
    sims_mol1 = tanimoto_sim_packed(Y, Y[mol1], cardinalities)

    # Get the least similar molecule to mol1
    mol2 = np.argmin(sims_mol1)

    # Get the similarity of each molecule to mol2
    sims_mol2 = tanimoto_sim_packed(Y, Y[mol2], cardinalities)

    return (mol1, mol2), sims_mol1, sims_mol2


def calc_centroid(linear_sum, n_samples, *, pack: bool):
    """Calculates centroid

    Parameters
    ----------

    linear_sum : np.ndarray
                 Sum of the elements column-wise
    n_samples : int
                Number of samples
    pack : bool
        Whether to pack the resulting fingerprints

    Returns
    -------
    centroid : np.ndarray[np.uint8]
               Centroid fingerprints of the given set
    """
    # NOTE: I believe np guarantees bools are stored as 0xFF -> True and 0x00 -> False,
    # so this view is fully safe
    if n_samples <= 1:
        cent = linear_sum.astype(np.uint8, copy=False)
    cent = (linear_sum >= n_samples * 0.5).view(np.uint8)
    if pack:
        return pack_fingerprints(cent)
    return cent


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


def _split_node(node):
    """The node has to be split if there is no place for a new subcluster
    in the node.
    1. Two empty nodes and two empty subclusters are initialized.
    2. The pair of distant subclusters are found.
    3. The properties of the empty subclusters and nodes are updated
       according to the nearest distance between the subclusters to the
       pair of distant subclusters.
    4. The two nodes are set as children to the two subclusters.
    """
    n_features = node.n_features
    branching_factor = node.branching_factor
    new_subcluster1 = _BFSubcluster(n_features=n_features)
    new_subcluster2 = _BFSubcluster(n_features=n_features)

    new_node = _BFNode(branching_factor, n_features)
    new_subcluster1.child_ = new_node
    new_subcluster2.child_ = node

    if node.is_leaf:
        new_node.prev_leaf_ = node.prev_leaf_
        node.prev_leaf_.next_leaf_ = new_node
        new_node.next_leaf_ = node
        node.prev_leaf_ = new_node

    # O(N) implementation of max separation
    separated_idxs, node1_dist, node2_dist = max_separation(node.centroids_, n_features)
    # Notice that max_separation is returning similarities and not distances
    node1_closer = node1_dist > node2_dist
    # Make sure node1 is closest to itself even if all distances are equal.
    # This can only happen when all node.centroids_ are duplicates leading to all
    # distances between centroids being zero.
    node1_closer[separated_idxs[0]] = True
    subclusters = node.subclusters_.copy()  # Shallow copy
    node.subclusters_ = []
    node.centroids_ = node.init_centroids_[:0, :]
    for idx, subcluster in enumerate(subclusters):
        if node1_closer[idx]:
            new_node.append_subcluster(subcluster)
            new_subcluster1.update(subcluster)
        else:
            node.append_subcluster(subcluster)
            new_subcluster2.update(subcluster)
    return new_subcluster1, new_subcluster2


class _BFNode:
    """Each node in a BFTree is called a BFNode.

    The BFNode can have a maximum of branching_factor
    number of BFSubclusters.

    Parameters
    ----------
    branching_factor : int
        Maximum number of BF subclusters in each node.

    n_features : int
        The number of features.

    Attributes
    ----------
    subclusters_ : list
        List of subclusters for a particular BFNode.

    prev_leaf_ : _BFNode
        Only useful for leaf nodes, otherwise None

    next_leaf_ : _BFNode
        Only useful for leaf nodes, otherwise None

    init_centroids_ : ndarray of shape (branching_factor + 1, n_features)
        Manipulate ``init_centroids_`` throughout rather than centroids_ since
        the centroids are just a view of the ``init_centroids_`` .

    centroids_ : ndarray of shape (branching_factor, n_features)
        View of ``init_centroids_``.

    is_leaf : bool
        True if next_leaf_ is present
    """

    # NOTE: Slots deactivates __dict__, and thus reduces memory usage of python objects
    __slots__ = (
        "n_features",
        "subclusters_",
        "init_centroids_",
        "centroids_",
        "prev_leaf_",
        "next_leaf_",
    )

    def __init__(self, branching_factor: int, n_features: int):
        self.n_features = n_features

        # The list of subclusters, centroids and squared norms
        # to manipulate throughout.
        self.subclusters_ = []
        # Centroids are stored packed. All centroids up to branching_factor are
        # allocated in a contiguous array
        self.init_centroids_ = np.empty(
            (branching_factor + 1, (n_features + 7) // 8), dtype=np.uint8
        )
        # centroids_ is a view of init_centroids. Modifying init_centroids is sufficient
        # The view starts with size 0 (no subclusters)
        self.centroids_ = self.init_centroids_[:0, :]
        self.prev_leaf_ = None
        self.next_leaf_ = None

    @property
    def is_leaf(self) -> bool:
        return self.prev_leaf_ is not None

    @property
    def branching_factor(self) -> int:
        return self.init_centroids_.shape[0] - 1

    def append_subcluster(self, subcluster):
        n_samples = len(self.subclusters_)
        self.subclusters_.append(subcluster)
        # Update init_centroids with the new centroid
        self.init_centroids_[n_samples] = subcluster.centroid_
        # Move the view since now there is an extra centroid
        # This is only necessary when a subcluster is appended
        self.centroids_ = self.init_centroids_[: n_samples + 1, :]

    def update_split_subclusters(self, subcluster, new_subcluster1, new_subcluster2):
        """Remove a subcluster from a node and update it with the
        split subclusters.
        """
        # Replace subcluster with new_subcluster1
        ind = self.subclusters_.index(subcluster)
        self.subclusters_[ind] = new_subcluster1
        self.init_centroids_[ind] = new_subcluster1.centroid_
        # Append new_subcluster2
        self.append_subcluster(new_subcluster2)

    def insert_bf_subcluster(self, subcluster, merge_accept_fn, threshold):
        """Insert a new subcluster into the node."""
        # Reusing tree with different features is forbidden
        if not self.subclusters_:
            self.append_subcluster(subcluster)
            return False

        # We need to find the closest subcluster among all the
        # subclusters so that we can insert our new subcluster.
        sim_matrix = tanimoto_sim_packed(self.centroids_, subcluster.centroid_)
        closest_index = np.argmax(sim_matrix)
        closest_subcluster = self.subclusters_[closest_index]

        # If the subcluster has a child, we need a recursive strategy.
        if closest_subcluster.child_ is not None:

            split_child = closest_subcluster.child_.insert_bf_subcluster(
                subcluster, merge_accept_fn, threshold
            )

            if not split_child:
                # If it is determined that the child need not be split, we
                # can just update the closest_subcluster
                closest_subcluster.update(subcluster)
                self.init_centroids_[closest_index] = self.subclusters_[
                    closest_index
                ].centroid_
                return False

            # things not too good. we need to redistribute the subclusters in
            # our child node, and add a new subcluster in the parent
            # subcluster to accommodate the new child.
            else:
                new_subcluster1, new_subcluster2 = _split_node(
                    closest_subcluster.child_
                )
                self.update_split_subclusters(
                    closest_subcluster, new_subcluster1, new_subcluster2
                )

                if len(self.subclusters_) > self.branching_factor:
                    return True
                return False

        # good to go!
        else:
            merged = closest_subcluster.merge_subcluster(
                subcluster, threshold, merge_accept_fn
            )
            if merged:
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
    r"""Each subcluster in a BFNode is called a BFSubcluster.

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

    # NOTE: Slots deactivates __dict__, and thus reduces memory usage of python objects
    __slots__ = (
        "_buffer",
        "centroid_",
        "mol_indices",
        "child_",
    )

    def __init__(
        self, *, linear_sum=None, mol_indices=None, n_features=2048, buffer=None
    ):
        # NOTE: Internally, _buffer holds both "linear_sum" and "n_samples"
        # It is guaranteed to always have the minimum required uint dtype
        # It should not be accessed by external classes, only used internally.
        # The individual parts can be accessed in a read-only way using the
        # linear_sum_ and n_samples_ properties.
        #
        # IMPROTANT: To mutate instances of this class, *always* use the public API
        # given by replace|add_to_n_samples_and_linear_sum(...)
        if buffer is not None:
            if linear_sum is not None:
                raise ValueError("'linear_sum' and 'buffer' are mutually exclusive")
            if len(mol_indices) != buffer[-1]:
                raise ValueError("Expected len(mol_indices) == n_samples")
            self._buffer = buffer
            self.centroid_ = calc_centroid(buffer[:-1], buffer[-1], pack=True)
        else:
            if linear_sum is not None:
                if len(mol_indices) != 1:
                    raise ValueError("Expected len(mol_indices) == 1")
                buffer = np.empty((len(linear_sum) + 1,), dtype=np.uint8)
                buffer[:-1] = linear_sum
                buffer[-1] = 1
                self._buffer = buffer
                self.centroid_ = pack_fingerprints(
                    linear_sum.astype(np.uint8, copy=False)
                )
            else:
                # Empty subcluster
                if mol_indices is not None:
                    raise ValueError("Expected mol_indices = None for empty subcluster")
                self._buffer = np.zeros((n_features + 1,), dtype=np.uint8)
                self.centroid_ = np.zeros(((n_features + 7) // 8,), dtype=np.uint8)

        self.mol_indices = mol_indices if mol_indices is not None else []

        self.child_ = None
        # self.parent_ = None

    @property
    def n_features(self) -> int:
        return len(self._buffer) - 1

    @property
    def dtype_name(self):
        return self._buffer.dtype.name

    @property
    def linear_sum_(self):
        read_only_view = self._buffer[:-1]
        read_only_view.flags.writeable = False
        return read_only_view

    @property
    def n_samples_(self):
        # Returns a python int, which is guaranteed to never overflow in sums, so
        # n_samples_ can always be safely added when accessed through this property
        return self._buffer.item(-1)

    # NOTE: Part of the contract is that all elements of linear sum must always be
    # less or equal to n_samples. This function does not check this
    def replace_n_samples_and_linear_sum(self, n_samples, linear_sum):
        # Cast to the minimum uint that can hold the inputs
        self._buffer = self._buffer.astype(min_safe_uint(n_samples), copy=False)
        # NOTE: Assignments are safe and do not recast the buffer
        self._buffer[:-1] = linear_sum
        self._buffer[-1] = n_samples
        self.centroid_ = calc_centroid(linear_sum, n_samples, pack=True)

    # NOTE: Part of the contract is that all elements of linear sum must always be
    # less or equal n_samples. This function does not check this
    def add_to_n_samples_and_linear_sum(self, n_samples, linear_sum):
        # Cast to the minimum uint that can hold the inputs
        new_n_samples = self.n_samples_ + n_samples
        self._buffer = self._buffer.astype(min_safe_uint(new_n_samples), copy=False)
        # NOTE: Assignment and inplace add are safe and do not recast the buffer
        self._buffer[:-1] += linear_sum
        self._buffer[-1] = new_n_samples
        self.centroid_ = calc_centroid(self._buffer[:-1], new_n_samples, pack=True)

    def update(self, subcluster):
        self.add_to_n_samples_and_linear_sum(
            subcluster.n_samples_, subcluster.linear_sum_
        )
        self.mol_indices.extend(subcluster.mol_indices)

    def merge_subcluster(self, nominee_cluster, threshold, merge_accept_fn):
        """Check if a cluster is worthy enough to be merged. If yes, merge."""
        old_n = self.n_samples_
        nom_n = nominee_cluster.n_samples_
        new_n = old_n + nom_n
        old_ls = self.linear_sum_
        nom_ls = nominee_cluster.linear_sum_
        # np.add with explicit dtype is safe from overflows, e.g. :
        # np.add(np.uint8(255), np.uint8(255), dtype=np.uint16) = np.uint16(510)
        new_ls = np.add(old_ls, nom_ls, dtype=min_safe_uint(new_n))
        if merge_accept_fn(threshold, new_ls, new_n, old_ls, nom_ls, old_n, nom_n):
            self.replace_n_samples_and_linear_sum(new_n, new_ls)
            self.mol_indices.extend(nominee_cluster.mol_indices)
            return True
        return False


class BitBirch:
    r"""Implements the BitBIRCH clustering algorithm.

    BitBIRCH paper:

    Memory- and time-efficient, online-learning algorithm.
    It constructs a tree data structure with the cluster centroids being read off the
    leaf.

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
        branching_factor: int = 50,
        merge_criterion: str | None = None,
        tolerance: float | None = None,
    ):
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.index_tracker = 0
        self.first_call = True

        if _global_merge_accept is not None:
            # Backwards compat
            if tolerance is not None:
                raise ValueError(
                    "tolerance can only be passed if "
                    "the *global* set_merge function has *not* been used"
                )
            if merge_criterion is not None:
                raise ValueError(
                    "merge_criterion can only be passed if "
                    "the *global* set_merge function has *not* been used"
                )
            self._merge_accept_fn = _global_merge_accept
        else:
            merge_criterion = "diameter" if merge_criterion is None else merge_criterion
            tolerance = 0.05 if tolerance is None else tolerance
            self._merge_accept_fn = _get_merge_accept_fn(merge_criterion, tolerance)

        # For backwards compatibility, weak-register in global state
        _BITBIRCH_INSTANCES.add(self)

    def set_merge(
        self, merge_criterion: str = "diameter", tolerance: float = 0.05
    ) -> None:
        r"""Sets the criteria for merging subclusters in this BitBirch tree

        Parameters:
        -----------
        merge_criterion: str
            radius, diameter or tolerance
            radius: merge subcluster based on comparison to centroid of the cluster
            diameter: merge subcluster based on instant Tanimoto similarity of cluster
            tolerance: applies tolerance threshold to diameter merge criteria, which
                will merge subcluster with stricter threshold for newly added molecules

        tolerance: float
            Penalty value for similarity threshold of the 'tolerance' merge criteria
        """
        if _global_merge_accept is not None:
            raise ValueError(
                "merge_criterion can only be set if "
                "the global set_merge function has *not* been used"
            )
        self._merge_accept_fn = _get_merge_accept_fn(merge_criterion, tolerance)

    def _initialize_tree(self, n_features: int) -> None:
        # Initialize the root (and a dummy node to get back the subclusters
        self.root_ = _BFNode(self.branching_factor, n_features)
        self.dummy_leaf_ = _BFNode(self.branching_factor, n_features)
        self.dummy_leaf_.next_leaf_ = self.root_
        self.root_.prev_leaf_ = self.dummy_leaf_

    def fit(
        self,
        X,
        reinsert_indices: tp.Iterator[int] | None = None,
        store_centroids: bool = True,
        input_is_packed: bool = True,
        n_features: int | None = None,
    ):
        r"""Build a BF Tree for the input data.

        if `reinsert_indices` is passed, X corresponds only to the molecules that will
        be reinserted into the tree, and `reinsert_indices` are the indices associated
        with these molecules.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        self
            Fitted estimator.
        """
        n_features = _validate_n_features(X, input_is_packed, n_features)
        # Start a new tree the first time this function is called
        if self.first_call:
            self._initialize_tree(n_features)

        if not sparse.issparse(X):
            iter_func = iter
        else:
            iter_func = _iterate_sparse_X

        if reinsert_indices is None:
            idx_provider = itertools.count(self.index_tracker)
        else:
            idx_provider = reinsert_indices

        threshold = self.threshold
        branching_factor = self.branching_factor
        merge_accept_fn = self._merge_accept_fn
        for idx, sample in zip(idx_provider, iter_func(X)):
            subcluster = _BFSubcluster(
                linear_sum=_copy_or_unpack(sample, n_features, input_is_packed),
                mol_indices=[idx],
                n_features=n_features,
            )
            split = self.root_.insert_bf_subcluster(
                subcluster, merge_accept_fn, threshold
            )

            if split:
                new_subcluster1, new_subcluster2 = _split_node(self.root_)
                self.root_ = _BFNode(branching_factor, n_features)
                self.root_.append_subcluster(new_subcluster1)
                self.root_.append_subcluster(new_subcluster2)

            if reinsert_indices is not None:
                self.index_tracker += 1

        if store_centroids:
            centroids = np.concatenate([leaf.centroids_ for leaf in self._get_leaves()])
            self.subcluster_centers_ = centroids
            self._n_features_out = self.subcluster_centers_.shape[0]

        self.first_call = False
        return self

    # Provided for backwards compatibility
    def fit_reinsert(
        self,
        X,
        reinsert_indices: tp.Iterator[int],
        store_centroids: bool = True,
        input_is_packed: bool = True,
        n_features: int | None = None,
    ):
        return self.fit(
            X, reinsert_indices, store_centroids, input_is_packed, n_features
        )

    def fit_np(self, X):
        # NOTE: X can be a numpy array or a list
        if isinstance(X, list):
            n_features = len(X[0]) - 1
        else:
            n_features = X.shape[1] - 1

        # Start a new tree the first time this function is called
        if self.first_call:
            self._initialize_tree(n_features)

        if not sparse.issparse(X):
            iter_func = iter
        else:
            iter_func = _iterate_sparse_X

        merge_accept_fn = self._merge_accept_fn
        threshold = self.threshold
        branching_factor = self.branching_factor
        # A copy is only required when iterating over a numpy array
        copy_if_arr = (lambda x: x) if isinstance(X, list) else (lambda x: x.copy())
        for sample in iter_func(X):
            subcluster = _BFSubcluster(
                buffer=copy_if_arr(sample),
                mol_indices=[self.index_tracker],
                n_features=n_features,
            )
            split = self.root_.insert_bf_subcluster(
                subcluster, merge_accept_fn, threshold
            )

            if split:
                new_subcluster1, new_subcluster2 = _split_node(self.root_)
                self.root_ = _BFNode(branching_factor, n_features)
                self.root_.append_subcluster(new_subcluster1)
                self.root_.append_subcluster(new_subcluster2)

            self.index_tracker += 1

        self.first_call = False
        return self

    def fit_np_reinsert(self, X, reinsert_indices, store_centroids: bool = True):
        # NOTE: X can be a numpy array or a list
        if isinstance(X, list):
            n_features = len(X[0]) - 1
        else:
            n_features = X.shape[1] - 1

        # Start a new tree the first time this function is called
        if self.first_call:
            self._initialize_tree(n_features)

        if not sparse.issparse(X):
            iter_func = iter
        else:
            iter_func = _iterate_sparse_X

        merge_accept_fn = self._merge_accept_fn

        threshold = self.threshold
        branching_factor = self.branching_factor
        # A copy is only required when iterating over a numpy array
        copy_if_arr = (lambda x: x) if isinstance(X, list) else (lambda x: x.copy())
        for sample, mol_indices in zip(iter_func(X), reinsert_indices):
            subcluster = _BFSubcluster(
                buffer=copy_if_arr(sample),
                mol_indices=mol_indices,
                n_features=n_features,
            )
            split = self.root_.insert_bf_subcluster(
                subcluster, merge_accept_fn, threshold
            )

            if split:
                new_subcluster1, new_subcluster2 = _split_node(self.root_)
                self.root_ = _BFNode(branching_factor, n_features)
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
        """Get a dictionary containing the centroids and mol indices of the leaves"""
        if self.first_call:
            raise ValueError("The model has not been fitted yet.")

        centroids = []
        mol_ids = []
        for leaf in self._get_leaves():
            for subcluster in leaf.subclusters_:
                centroids.append(subcluster.centroid_)
                mol_ids.append(subcluster.mol_indices)
        return {"centroids": centroids, "mol_ids": mol_ids}

    def get_centroids(self):
        """Get a list of Numpy arrays containing the centroids' fingerprints"""
        if self.first_call:
            raise ValueError("The model has not been fitted yet.")

        centroids = []
        for leaf in self._get_leaves():
            for subcluster in leaf.subclusters_:
                centroids.append(subcluster.centroid_)
        return centroids

    def get_cluster_mol_ids(self):
        """Get the indices of the molecules in each cluster"""
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
        """Get the BitFeatures of the leaves"""
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
    ):
        """Prepare BitFeatures of the largest cluster and the rest of the clusters

        Input fps must have the same n_features as the ones currently in the tree.
        """
        if self.first_call:
            raise ValueError("The model has not been fitted yet.")

        BFs = self._get_BFs()
        big, rest = BFs[0], BFs[1:]
        n_features = big.n_features
        dtypes_to_fp, dtypes_to_mols = self._prepare_bf_to_buffer_dicts(rest)
        # Add fps and mol indices of the "big" cluster
        for mol_idx in big.mol_indices:
            unpacked_fp = _copy_or_unpack(
                fps[mol_idx - initial_mol], n_features, input_is_packed
            )
            buffer = np.empty(unpacked_fp.shape[0] + 1, dtype=np.uint8)
            buffer[:-1] = unpacked_fp
            buffer[-1] = 1
            dtypes_to_fp["uint8"].append(buffer)
            dtypes_to_mols["uint8"].append([mol_idx])
        return dtypes_to_fp, dtypes_to_mols

    def bf_to_np(self):
        """Prepare BitFeatures of the largest cluster and the rest of the clusters"""
        if self.first_call:
            raise ValueError("The model has not been fitted yet.")

        return self._prepare_bf_to_buffer_dicts(self._get_BFs())

    @staticmethod
    def _prepare_bf_to_buffer_dicts(BFs):
        # Helper function used when returning lists of subclusters
        dtypes_to_fp = defaultdict(list)
        dtypes_to_mols = defaultdict(list)
        for BF in BFs:
            dtypes_to_fp[BF.dtype_name].append(BF._buffer)
            dtypes_to_mols[BF.dtype_name].append(BF.mol_indices)
        return dtypes_to_fp, dtypes_to_mols

    def get_assignments(self, n_mols):
        clustered_ids = self.get_cluster_mol_ids()

        assignments = np.full(n_mols, -1, dtype=int)
        for i, cluster in enumerate(clustered_ids):
            assignments[cluster] = i + 1
        # Check that there are no unassigned molecules
        assert np.all(assignments != -1)
        return assignments

    def __repr__(self) -> str:
        fn = self._merge_accept_fn
        _str = f"{self.__class__.__name__}(threshold={self.threshold}, branching_factor={self.branching_factor}, merge_criterion='{fn.name}'"  # noqa:E501
        if isinstance(fn, ToleranceMerge):
            _str += f", tolerance={fn._tolerance})"
        else:
            _str += ")"
        return _str
