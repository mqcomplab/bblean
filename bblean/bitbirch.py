# BitBIRCH-Lean Python Package: An open-source clustering module based on iSIM.
#
# If you find this software useful please cite the following articles:
# - BitBIRCH: efficient clustering of large molecular libraries:
#   https://doi.org/10.1039/D5DD00030K
# - BitBIRCH Clustering Refinement Strategies:
#   https://doi.org/10.1021/acs.jcim.5c00627
# - BitBIRCH-Lean: TO-BE-ADDED
#
# Copyright (C) 2025  The Miranda-Quintana Lab and other BitBirch developers, including:
# - Ramon Alain Miranda Quintana <ramirandaq@gmail.com>, <quintana@chem.ufl.edu>
# - Krisztina Zsigmond <kzsigmond@ufl.edu>
# - Ignacio Pickering <ipickering@chem.ufl.edu>
# - Kenneth Lopez Perez <klopezperez@chem.ufl.edu>
# - Miroslav Lzicar <miroslav.lzicar@deepmedchem.com>
#
# Authors of ./bblean/multiround.py are:
# - Ramon Alain Miranda Quintana <ramirandaq@gmail.com>, <quintana@chem.ufl.edu>
# - Ignacio Pickering <ipickering@chem.ufl.edu>
#
# This program is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software Foundation,
# version 3 (SPDX-License-Identifier: GPL-3.0-only).
#
# Portions of this file are licensed under the BSD 3-Clause License
# Copyright (c) 2007-2024 The scikit-learn developers. All rights reserved.
# (SPDX-License-Identifier: BSD-3-Clause). Copies or reproductions of code in this
# file must in addition adhere to the BSD-3-Clause license terms. A
# copy of the BSD-3-Clause license can be located at the root of this repository, under
# ./LICENSES/BSD-3-Clause.txt.
#
# Portions of this file were previously licensed under the LGPL 3.0
# license (SPDX-License-Identifier: LGPL-3.0-only), they are relicensed in this program
# as GPL-3.0, with permission of all original copyright holders:
# - Ramon Alain Miranda Quintana <ramirandaq@gmail.com>, <quintana@chem.ufl.edu>
# - Vicky (Vic) Jung <jungvicky@ufl.edu>
# - Kenneth Lopez Perez <klopezperez@chem.ufl.edu>
# - Kate Huddleston <kdavis2@chem.ufl.edu>
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with this
# program. This copy can be located at the root of this repository, under
# ./LICENSES/GPL-3.0-only.txt.  If not, see <http://www.gnu.org/licenses/gpl-3.0.html>.
# type: ignore
import warnings
import typing as tp
from collections import defaultdict
from weakref import WeakSet

from scipy import sparse
import numpy as np
from numpy.typing import NDArray, DTypeLike

from bblean.utils import calc_centroid, min_safe_uint
from bblean.packing import (
    pack_fingerprints,
    unpack_fingerprints,
    popcount,
    jt_sim_packed,
)
from bblean.merges import get_merge_accept_fn, MergeAcceptFunction


# For backwards compatibility with the global "set_merge", keep weak references to all
# the BitBirch instances and update them when set_merge is called
_BITBIRCH_INSTANCES: WeakSet["BitBirch"] = WeakSet()


# For backwards compatibility: global function used to accept merges
_global_merge_accept: MergeAcceptFunction | None = None


# For backwards compatibility: set the global merge_accept function
def set_merge(merge_criterion: str, tolerance: float = 0.05) -> None:
    r"""Sets the global criteria for merging subclusters in any BitBirch tree

    For usage see `BitBirch.set_merge`

    ..  warning::

        Use of this function is highly discouraged, instead use either:
            bb_tree = BitBirch(...)
            bb_tree.set_merge(merge_criterion=..., tolerance=...)
        or directly: `bb_tree = BitBirch(..., merge_criterion=..., tolerance=...)`."
    """
    msg = (
        "Use of the global `set_merge` function is highly discouraged,\n"
        " instead use either: "
        "    bb_tree = BitBirch(...)\n"
        "    bb_tree.set_merge(merge_criterion=..., tolerance=...)\n"
        " or directly: `bb_tree = BitBirch(..., merge_criterion=..., tolerance=...)`."
    )
    warnings.warn(msg, UserWarning)
    # Set the global merge_accept function
    global _global_merge_accept
    _global_merge_accept = get_merge_accept_fn(merge_criterion, tolerance)
    for bbirch in _BITBIRCH_INSTANCES:
        bbirch._merge_accept_fn = _global_merge_accept


# Utility function to validate the n_features argument for packed inputs
def _validate_n_features(
    X, input_is_packed: bool, n_features: int | None = None
) -> int:
    if input_is_packed:
        if n_features is None:
            raise ValueError("n_features is required for packed inputs")
        return n_features

    x_n_features = len(X[0]) if isinstance(X, list) else X.shape[1]
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


def _max_separation(Y: NDArray[np.uint8], n_features: int):
    """Finds two objects in Y that are very separated
    This is not guaranteed to find
    the two absolutely most separated objects, but it is
    a very robust O(N) approximation. Quality of clustering
    does not diminish in the end.

    Algorithm:
    a) Find centroid of Y
    b) mol1 is the molecule most distant from the centroid
    c) mol2 is the molecule most distant from mol1

    Returns
    -------
    (mol1, mol2) : (int, int)
                   indices of mol1 and mol2
    1 - sims_mol1 : np.ndarray
                   Similarities to mol1
    1 - sims_mol2: np.ndarray
                   Similarities to mol2

    These are needed for node1_sim and node2_sim in _split_node
    """
    # Get the centroid of the set
    Y_unpacked = unpack_fingerprints(Y, n_features)
    n_samples = len(Y_unpacked)
    # np.sum() automatically promotes to uint64 unless forced to a smaller dtype
    linear_sum = np.sum(Y_unpacked, axis=0, dtype=min_safe_uint(n_samples))
    centroid_packed = calc_centroid(linear_sum, n_samples, pack=True)

    cardinalities = popcount(Y)

    # Get the similarity of each molecule to the centroid, and the least similar idx
    sims_med = jt_sim_packed(Y, centroid_packed, cardinalities)
    mol1 = np.argmin(sims_med)

    # Get the similarity of each molecule to mol1, and the least similar idx
    sims_mol1 = jt_sim_packed(Y, Y[mol1], cardinalities)
    mol2 = np.argmin(sims_mol1)

    # Get the similarity of each molecule to mol2
    sims_mol2 = jt_sim_packed(Y, Y[mol2], cardinalities)
    return (mol1, mol2), sims_mol1, sims_mol2


def _split_node(node):
    """The node has to be split if there is no place for a new subcluster
    in the node.
    1. An extra empty node and two empty subclusters are initialized.
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
    separated_idxs, node1_sim, node2_sim = _max_separation(node.centroids_, n_features)
    # _max_separation returns similarities, not distances
    node1_closer = node1_sim > node2_sim
    # Make sure node1 and node2 are closest to themselves, even if all sims are equal.
    # This can only happen when all node.centroids_ are duplicates leading to all
    # distances between centroids being zero.
    node1_closer[separated_idxs[0]] = True
    node1_closer[separated_idxs[1]] = False
    subclusters = node.subclusters_.copy()  # Shallow copy
    node.subclusters_ = []  # Reset the node
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
        # Nodes that are leaves have a non-null prev_leaf_
        self.prev_leaf_ = None
        self.next_leaf_ = None

    @property
    def is_leaf(self) -> bool:
        return self.prev_leaf_ is not None

    @property
    def branching_factor(self) -> int:
        return self.init_centroids_.shape[0] - 1

    @property
    def centroids_(self) -> NDArray[np.uint8]:
        # centroids_ returns a view of the (packed) init_centroids. Modifying
        # init_centroids is sufficient.
        return self.init_centroids_[: len(self.subclusters_), :]

    def append_subcluster(self, subcluster):
        n_samples = len(self.subclusters_)
        self.subclusters_.append(subcluster)
        self.init_centroids_[n_samples] = subcluster.centroid_

    def update_split_subclusters(self, subcluster, new_subcluster1, new_subcluster2):
        """Remove a subcluster from a node and update it with the
        split subclusters.
        """
        # Replace subcluster with new_subcluster1
        idx = self.subclusters_.index(subcluster)
        self.subclusters_[idx] = new_subcluster1
        self.init_centroids_[idx] = new_subcluster1.centroid_
        # Append new_subcluster2
        self.append_subcluster(new_subcluster2)

    def insert_bf_subcluster(self, subcluster, merge_accept_fn, threshold) -> bool:
        """Insert a new subcluster into the node."""
        # Reusing tree with different features is forbidden
        if not self.subclusters_:
            self.append_subcluster(subcluster)
            return False

        # We need to find the closest subcluster among all the
        # subclusters so that we can insert our new subcluster.
        sim_matrix = jt_sim_packed(self.centroids_, subcluster.centroid_)
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

    A BFSubcluster can have a BFNode as its child.

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
    __slots__ = ("_buffer", "centroid_", "child_", "mol_indices")

    def __init__(
        self, *, linear_sum=None, mol_indices=(), n_features: int = 2048, buffer=None
    ):
        # NOTE: Internally, _buffer holds both "linear_sum" and "n_samples" It is
        # guaranteed to always have the minimum required uint dtype It should not be
        # accessed by external classes, only used internally. The individual parts can
        # be accessed in a read-only way using the linear_sum_ and n_samples_
        # properties.
        #
        # IMPORTANT: To mutate instances of this class, *always* use the public API
        # given by replace|add_to_n_samples_and_linear_sum(...)
        if buffer is not None:
            if linear_sum is not None:
                raise ValueError("'linear_sum' and 'buffer' are mutually exclusive")
            if len(mol_indices) != buffer[-1]:
                raise ValueError(
                    "Expected len(mol_indices) == buffer[-1],"
                    f" but found {len(mol_indices)} != {buffer[-1]}"
                )
            self._buffer = buffer
            self.centroid_ = calc_centroid(buffer[:-1], buffer[-1], pack=True)
        else:
            if linear_sum is not None:
                if len(mol_indices) != 1:
                    raise ValueError(
                        "Expected len(mol_indices) == 1,"
                        f" but found {len(mol_indices)} != 1"
                    )
                buffer = np.empty((len(linear_sum) + 1,), dtype=np.uint8)
                buffer[:-1] = linear_sum
                buffer[-1] = 1
                self._buffer = buffer
                self.centroid_ = pack_fingerprints(
                    linear_sum.astype(np.uint8, copy=False)
                )
            else:
                # Empty subcluster
                if len(mol_indices) != 0:
                    raise ValueError(
                        "Expected len(mol_indices) == 0 for empty subcluster,"
                        f" but found {len(mol_indices)} != 0"
                    )
                self._buffer = np.zeros((n_features + 1,), dtype=np.uint8)
                self.centroid_ = np.empty(0, dtype=np.uint8)  # Will be overwritten
        self.mol_indices = list(mol_indices)
        self.child_ = None

    @property
    def n_features(self) -> int:
        return len(self._buffer) - 1

    @property
    def dtype_name(self) -> str:
        return self._buffer.dtype.name

    @property
    def linear_sum_(self):
        read_only_view = self._buffer[:-1]
        read_only_view.flags.writeable = False
        return read_only_view

    @property
    def n_samples_(self) -> int:
        # Returns a python int, which is guaranteed to never overflow in sums, so
        # n_samples_ can always be safely added when accessed through this property
        return self._buffer.item(-1)

    # NOTE: Part of the contract is that all elements of linear sum must always be
    # less or equal to n_samples. This function does not check this
    def replace_n_samples_and_linear_sum(self, n_samples, linear_sum) -> None:
        # Cast to the minimum uint that can hold the inputs
        self._buffer = self._buffer.astype(min_safe_uint(n_samples), copy=False)
        # NOTE: Assignments are safe and do not recast the buffer
        self._buffer[:-1] = linear_sum
        self._buffer[-1] = n_samples
        self.centroid_ = calc_centroid(linear_sum, n_samples, pack=True)

    # NOTE: Part of the contract is that all elements of linear sum must always be
    # less or equal to n_samples. This function does not check this
    def add_to_n_samples_and_linear_sum(self, n_samples, linear_sum) -> None:
        # Cast to the minimum uint that can hold the inputs
        new_n_samples = self.n_samples_ + n_samples
        self._buffer = self._buffer.astype(min_safe_uint(new_n_samples), copy=False)
        # NOTE: Assignment and inplace add are safe and do not recast the buffer
        self._buffer[:-1] += linear_sum
        self._buffer[-1] = new_n_samples
        self.centroid_ = calc_centroid(self._buffer[:-1], new_n_samples, pack=True)

    def update(self, subcluster) -> None:
        self.add_to_n_samples_and_linear_sum(
            subcluster.n_samples_, subcluster.linear_sum_
        )
        self.mol_indices.extend(subcluster.mol_indices)

    def merge_subcluster(self, nominee_cluster, threshold, merge_accept_fn) -> bool:
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

    Memory- and time-efficient, online-learning algorithm. It constructs a tree data
    structure with the cluster centroids being read off the leaf.

    Parameters
    ----------
    threshold : float, default=0.65
        The similarity radius of the subcluster obtained by merging a new sample and the
        closest subcluster should be greater than the threshold. Otherwise a new
        subcluster is started. Setting this value to be very low promotes splitting and
        vice-versa.

    branching_factor : int, default=50
        Maximum number of BF subclusters in each node. If a new samples enters such that
        the number of subclusters exceed the branching_factor then that node is split
        into two nodes with the subclusters redistributed in each. The parent subcluster
        of that node is removed and two new subclusters are added as parents of the 2
        split nodes.

    Attributes
    ----------
    root_ : _BFNode
        Root of the BFTree.

    dummy_leaf_ : _BFNode
        Start pointer to all the leaves.

    subcluster_centers_ : ndarray
        Centroids of all subclusters read directly from the leaves.
        Only generated if `store_centroids` is passed to `BitBirch.fit()`

    Notes
    -----
    The tree data structure consists of nodes with each node consisting of a number of
    subclusters. The maximum number of subclusters in a node is determined by the
    branching factor. Each subcluster maintains a linear sum, mol_indices and the number
    of samples in that subcluster. In addition, each subcluster can also have a node as
    its child, if the subcluster is not a member of a leaf node.

    For a new point entering the root, it is merged with the subcluster closest to it
    and the linear sum, mol_indices and the number of samples of that subcluster are
    updated. This is done recursively till the properties of the leaf node are updated.
    """

    def __init__(
        self,
        *,
        threshold: float = 0.65,
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
            self._merge_accept_fn = get_merge_accept_fn(merge_criterion, tolerance)
        # For backwards compatibility, weak-register in global state This is used to
        # update the merge_accept function if the global set_merge() is called
        _BITBIRCH_INSTANCES.add(self)

    @property
    def merge_criterion(self) -> str:
        return self._merge_accept_fn.name

    def set_merge(
        self,
        *,
        merge_criterion: str = "diameter",
        tolerance: float = 0.05,
        threshold: float | None = None,
        branching_factor: int | None = None,
    ) -> None:
        r"""Sets the criteria for merging subclusters in this BitBirch tree

        Parameters:
        -----------
        merge_criterion: str
            radius, diameter or tolerance
            - radius: merge subcluster based on comparison to centroid of the cluster
            - diameter: merge subcluster based on instant Tanimoto similarity of cluster
            - tolerance: applies tolerance threshold to diameter merge criteria, which
                will merge subcluster with stricter threshold for newly added molecules
        tolerance: float
            Penalty value for similarity threshold of the 'tolerance' merge criteria
        """
        if _global_merge_accept is not None:
            raise ValueError(
                "merge_criterion can only be set if "
                "the global set_merge function has *not* been used"
            )
        self._merge_accept_fn = get_merge_accept_fn(merge_criterion, tolerance)
        if threshold is not None:
            self.threshold = threshold
        if branching_factor is not None:
            self.branching_factor = branching_factor

    def fit(
        self,
        X,
        reinsert_indices: tp.Iterator[int] | None = None,
        store_centroids: bool = False,
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

        # The array iterator either copies, un-sparsifies, or does nothing
        # with the array rows, depending on the kind of X passed
        arr_iterator = _get_array_iterator(X, input_is_packed, n_features)
        if reinsert_indices is None:
            iterable = enumerate(arr_iterator, self.index_tracker)
        else:
            iterable = zip(reinsert_indices, arr_iterator)

        threshold = self.threshold
        branching_factor = self.branching_factor
        merge_accept_fn = self._merge_accept_fn
        for idx, fp in iterable:
            subcluster = _BFSubcluster(
                linear_sum=fp, mol_indices=[idx], n_features=n_features
            )
            split = self.root_.insert_bf_subcluster(
                subcluster, merge_accept_fn, threshold
            )

            if split:
                new_subcluster1, new_subcluster2 = _split_node(self.root_)
                self.root_ = _BFNode(branching_factor, n_features)
                self.root_.append_subcluster(new_subcluster1)
                self.root_.append_subcluster(new_subcluster2)

        if reinsert_indices is None and len(X) > 0:
            self.index_tracker = idx + 1
        if store_centroids:
            self._store_centroids_array()
        self.first_call = False
        return self

    def fit_np(
        self,
        X,
        reinsert_index_sequences: tp.Iterator[tp.Sequence[int]] | None = None,
        store_centroids: bool = False,
    ):
        r"""Build a BF Tree starting from buffers

        Buffers are arrays of the form:
            - buffer[0:-1] = linear_sum
            - buffer[-1] = n_samples
        And X is either an array or a list of such buffers

        if `reinsert_index_sequences` is passed, X corresponds only to the buffers to be
        reinserted into the tree, and `reinsert_index_sequences` are the sequences
        of indices associated with such buffers.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples + 1, n_features)
            Input data.

        Returns
        -------
        self
            Fitted estimator.
        """
        n_features = _validate_n_features(X, input_is_packed=False) - 1

        if self.first_call:
            self._initialize_tree(n_features)

        # The array iterator either copies, un-sparsifies, or does nothing with the
        # array rows, depending on the kind of X passed
        arr_iterator = _get_array_iterator(X, input_is_packed=False, dtype=X[0].dtype)
        merge_accept_fn = self._merge_accept_fn
        threshold = self.threshold
        branching_factor = self.branching_factor
        idx_provider: tp.Iterator[tp.Sequence[int]]
        if reinsert_index_sequences is None:
            idx_provider = map(list, range(self.index_tracker))  # type: ignore
        else:
            idx_provider = reinsert_index_sequences
        for idxs, buf in zip(idx_provider, arr_iterator):
            subcluster = _BFSubcluster(
                buffer=buf, mol_indices=idxs, n_features=n_features
            )
            split = self.root_.insert_bf_subcluster(
                subcluster, merge_accept_fn, threshold
            )
            if split:
                new_subcluster1, new_subcluster2 = _split_node(self.root_)
                self.root_ = _BFNode(branching_factor, n_features)
                self.root_.append_subcluster(new_subcluster1)
                self.root_.append_subcluster(new_subcluster2)
        if reinsert_index_sequences is None and len(X) > 0:
            self.index_tracker = idxs[0] + 1
        if store_centroids:
            self._store_centroids_array()
        self.first_call = False
        return self

    # Provided for backwards compatibility
    def fit_reinsert(
        self,
        X,
        reinsert_indices: tp.Iterator[int],
        store_centroids: bool = False,
        input_is_packed: bool = True,
        n_features: int | None = None,
    ):
        return self.fit(
            X, reinsert_indices, store_centroids, input_is_packed, n_features
        )

    # Provided for backwards compatibility
    def fit_np_reinsert(
        self,
        X,
        reinsert_index_sequences: tp.Iterator[tp.Sequence[int]],
        store_centroids: bool = False,
    ):
        return self.fit_np(X, reinsert_index_sequences, store_centroids)

    def _initialize_tree(self, n_features: int) -> None:
        # Initialize the root (and a dummy node to get back the subclusters
        self.root_ = _BFNode(self.branching_factor, n_features)
        self.dummy_leaf_ = _BFNode(self.branching_factor, n_features)
        self.dummy_leaf_.next_leaf_ = self.root_
        self.root_.prev_leaf_ = self.dummy_leaf_

    def _store_centroids_array(self) -> None:
        centroids = np.concatenate([leaf.centroids_ for leaf in self._get_leaves()])
        self.subcluster_centers_ = centroids
        self._n_features_out = self.subcluster_centers_.shape[0]

    def _get_leaves(self) -> tp.Iterator[_BFNode]:
        r"""Iterate over the leaf nodes of the tree

        Yields
        -------
        leaf: _BFNode
            Node of the tree that is a leaf
        """
        leaf = self.dummy_leaf_.next_leaf_
        while leaf is not None:
            yield leaf
            leaf = leaf.next_leaf_

    def get_centroids_mol_ids(self, sort: bool = True):
        """Get a dict with centroids and mol indices of the leaves"""
        # NOTE: This is different from the original bitbirch, here outputs are sorted
        # by default
        if self.first_call:
            raise ValueError("The model has not been fitted yet.")
        centroids = []
        mol_ids = []
        for subcluster in self._get_BFs(sort=sort):
            centroids.append(subcluster.centroid_)
            mol_ids.append(subcluster.mol_indices)
        return {"centroids": centroids, "mol_ids": mol_ids}

    def get_centroids(self, sort: bool = True) -> list[NDArray[np.uint8]]:
        """Get a list of arrays with the centroids' fingerprints"""
        # NOTE: This is different from the original bitbirch, here outputs are sorted
        # by default
        if self.first_call:
            raise ValueError("The model has not been fitted yet.")
        return [s.centroid_ for s in self._get_BFs(sort=sort)]

    def get_cluster_mol_ids(self, sort: bool = True) -> list[list[int]]:
        """Get the indices of the molecules in each cluster"""
        if self.first_call:
            raise ValueError("The model has not been fitted yet.")
        return [s.mol_indices for s in self._get_BFs(sort=sort)]

    def _get_BFs(self, sort: bool = True) -> list[_BFSubcluster]:
        """Get the BitFeatures of the leaves"""
        if self.first_call:
            raise ValueError("The model has not been fitted yet.")
        bfs = [s for leaf in self._get_leaves() for s in leaf.subclusters_]
        if sort:
            # Sort the BitFeatures by the number of samples in the cluster
            bfs.sort(key=lambda x: x.n_samples_, reverse=True)
        return bfs

    def bf_to_np_refine(
        self,
        X,
        initial_mol: int = 0,
        input_is_packed: bool = True,
    ):
        """Prepare numpy buffers ('np') for BitFeatures, splitting the biggest cluster

        The largest cluster is split into singletons. In order to perform this split,
        the original fingerprint array used to fit the tree (X) has to be provided,
        together with the index associated with the first fingerprint.

        The split is only performed for the returned 'np' buffers, the cluster in the
        tree itself is not modified
        """
        if self.first_call:
            raise ValueError("The model has not been fitted yet.")

        BFs = self._get_BFs()
        big, rest = BFs[0], BFs[1:]
        n_features = big.n_features
        dtypes_to_fp, dtypes_to_mols = self._prepare_bf_to_buffer_dicts(rest)
        # Add X and mol indices of the "big" cluster
        for mol_idx in big.mol_indices:
            fp = X[mol_idx - initial_mol]
            fp = unpack_fingerprints(fp, n_features) if input_is_packed else fp.copy()
            buffer = np.empty(fp.shape[0] + 1, dtype=np.uint8)
            buffer[:-1] = fp
            buffer[-1] = 1
            dtypes_to_fp["uint8"].append(buffer)
            dtypes_to_mols["uint8"].append([mol_idx])
        return dtypes_to_fp, dtypes_to_mols

    def bf_to_np(self):
        """Prepare numpy buffers ('np') for BitFeatures of all clusters"""
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

    def get_assignments(self, n_mols: int) -> NDArray[np.uint64]:
        clustered_ids = self.get_cluster_mol_ids()
        assignments = np.full(n_mols, -1, dtype=np.uint64)
        for i, cluster in enumerate(clustered_ids, 1):
            assignments[cluster] = i
        # Check that there are no unassigned molecules
        assert np.all(assignments != -1)
        return assignments

    def refine_inplace(
        self,
        X,
        initial_mol: int = 0,
        input_is_packed: bool = True,
    ):
        # Extract the BitFeatures of the leaves, breaking the largest cluster apart into
        # singleton subclusters
        fps_bfs, mols_bfs = self.bf_to_np_refine(
            X, initial_mol=initial_mol, input_is_packed=input_is_packed
        )

        # Reset the whole tree
        del self.root_
        self.first_call = False
        self.index_tracker = 0

        # Rebuild the tree again from scratch, reinserting all the subclusters
        for bufs, mol_idxs in zip(fps_bfs.values(), mols_bfs.values()):
            self.fit_np(bufs, reinsert_index_sequences=mol_idxs)
        return self

    def __repr__(self) -> str:
        fn = self._merge_accept_fn
        _str = f"{self.__class__.__name__}(threshold={self.threshold}, branching_factor={self.branching_factor}, merge_criterion='{fn.name}'"  # noqa:E501
        if hasattr(fn, "tolerance"):
            _str += f", tolerance={fn.tolerance})"
        else:
            _str += ")"
        return _str


# There are 4 cases here:
# (1) The input is a scipy.sparse array
# (2) The input is a list of dense arrays (nothing required)
# (3) The input is a packed array or list of packed arrays (unpack required)
# (4) The input is a dense array (copy required)
# NOTE: Sparse iteration hack is taken from sklearn
# It returns a densified row when iterating over a sparse matrix, instead
# of constructing a sparse matrix for every row that is expensive.
#
# Output is *always* of dtype uint8, but input (if unpacked) can be of arbitrary dtype
# It is most efficient for input to be uint8 to prevent copies
def _get_array_iterator(
    X,
    input_is_packed: bool = True,
    n_features: int | None = None,
    dtype: DTypeLike = np.uint8,
) -> tp.Iterator[NDArray[np.integer]]:

    if input_is_packed:
        if n_features is None:
            raise ValueError("n_features is required for packed inputs")
        return (np.unpackbits(a, axis=-1, count=n_features) for a in X)  # type: ignore
    if isinstance(X, list):
        return (a.astype(dtype, copy=False) for a in X)
    if sparse.issparse(X):
        if input_is_packed:
            raise ValueError("Packed input not supported for scipy sparse arrays")
        return _iter_sparse(X)
    # A copy is required here to avoid keeping a ref to the full array alive
    return (a.astype(dtype, copy=True) for a in X)


def _iter_sparse(X) -> tp.Iterator[NDArray[np.uint8]]:
    n_samples, n_features = X.shape
    X_indices = X.indices  # type: ignore
    X_data = X.data
    X_indptr = X.indptr  # type: ignore
    for i in range(n_samples):
        a = np.zeros(n_features, dtype=np.uint8)
        startptr, endptr = X_indptr[i], X_indptr[i + 1]
        nonzero_indices = X_indices[startptr:endptr]
        a[nonzero_indices] = X_data[startptr:endptr].astype(np.uint8, copy=False)
        yield a
