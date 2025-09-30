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
r"""BitBirch 'Lean' class for fast, memory-efficient O(N) clustering"""
from __future__ import annotations  # Stringize type annotations for no runtime overhead
import typing_extensions as tpx
from pathlib import Path
import warnings
import typing as tp
from typing import cast
from collections import defaultdict
from weakref import WeakSet

import numpy as np
from numpy.typing import NDArray, DTypeLike

from bblean._memory import _mmap_file_and_madvise_sequential, _ArrayMemPagesManager
from bblean._merges import get_merge_accept_fn, MergeAcceptFunction, BUILTIN_MERGES
from bblean.utils import min_safe_uint
from bblean.fingerprints import (
    pack_fingerprints,
    calc_centroid,
    _get_fingerprints_from_file_seq,
)
from bblean.similarity import jt_sim_packed, jt_most_dissimilar_packed

try:
    # NOTE: There are small gains from using this fn but only ~3%, so don't warn for now
    # if this fails, and don't expose it
    from bblean._cpp_similarity import unpack_fingerprints as _unpack_fingerprints
except ImportError:
    from bblean.fingerprints import unpack_fingerprints as _unpack_fingerprints

__all__ = ["BitBirch"]


# For backwards compatibility with the global "set_merge", keep weak references to all
# the BitBirch instances and update them when set_merge is called
_BITBIRCH_INSTANCES: WeakSet["BitBirch"] = WeakSet()


# For backwards compatibility: global function used to accept merges
_global_merge_accept: MergeAcceptFunction | None = None

_Input = tp.Union[NDArray[np.integer], list[NDArray[np.integer]]]


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
    X: _Input, input_is_packed: bool, n_features: int | None = None
) -> int:
    if len(X) == 0:
        raise ValueError("Input must have at least 1 fingerprint")
    if input_is_packed:
        _padded_n_features = len(X[0]) * 8 if isinstance(X, list) else X.shape[1] * 8
        if n_features is None:
            # Assume multiple of 8
            return _padded_n_features
        if _padded_n_features < n_features:
            raise ValueError(
                "n_features is larger than the padded length, which is inconsistent"
            )
        return n_features

    x_n_features = len(X[0]) if isinstance(X, list) else X.shape[1]
    if n_features is not None:
        if n_features != x_n_features:
            raise ValueError(
                "n_features is redundant for non-packed inputs"
                " if passed, it must be equal to X.shape[1] (or len(X[0]))."
                f" For passed X the inferred n_features was {x_n_features}."
                " If this value is not what you expected,"
                " make sure the passed X is actually unpacked."
            )
    return x_n_features


def _split_node(node: "_BFNode") -> tuple["_BFSubcluster", "_BFSubcluster"]:
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

    node1 = _BFNode(branching_factor, n_features)
    node2 = node  # Rename for clarity
    new_subcluster1.child = node1
    new_subcluster2.child = node2

    if node2.is_leaf:
        # If is_leaf, _prev_leaf is guaranteed to be not None
        # NOTE: cast seems to have a small overhead here for some reason
        node1._prev_leaf = node2._prev_leaf
        node2._prev_leaf._next_leaf = node1  # type: ignore
        node1._next_leaf = node2
        node2._prev_leaf = node1

    # O(N) approximation to obtain "most dissimilar fingerprints" within an array
    node1_idx, _, node1_sim, node2_sim = jt_most_dissimilar_packed(
        node2.packed_centroids, n_features
    )
    node1_closer = node1_sim > node2_sim
    # Make sure node1 and node2 are closest to themselves, even if all sims are equal.
    # This can only happen when all node.packed_centroids are duplicates leading to all
    # distances between centroids being zero.

    # TODO: Currently this behavior is buggy (?), seems like in some cases one of the
    # subclusters may *never* get updated, double check this logic
    node1_closer[node1_idx] = True
    subclusters = node2._subclusters.copy()  # Shallow copy
    node2._subclusters = []  # Reset the node
    for idx, subcluster in enumerate(subclusters):
        if node1_closer[idx]:
            node1.append_subcluster(subcluster)
            new_subcluster1.update(subcluster)
        else:
            node2.append_subcluster(subcluster)
            new_subcluster2.update(subcluster)
    return new_subcluster1, new_subcluster2


class _BFNode:
    """Each node in a BitBirch tree is a _BFNode.

    The _BFNode holds a maximum of branching_factor _BFSubclusters.

    Parameters
    ----------
    branching_factor : int
        Maximum number of _BFSubcluster in the node.

    n_features : int
        The number of features.

    Attributes
    ----------
    _subclusters : list
        List of _BFSubcluster for thre _BFNode.

    _prev_leaf : _BFNode
        Only useful for leaf nodes, otherwise None

    _next_leaf : _BFNode
        Only useful for leaf nodes, otherwise None

    _packed_centroids_buf : NDArray[np.uint8]
        Packed array of shape (branching_factor + 1, (n_features + 7) // 8) The code
        internally manipulates this buf rather than packed_centroids, which is just a
        view of this buf.

    packed_centroids : ndarray of shape (branching_factor, n_features)
        Packed array of shape (len(_subclusters), (n_features + 7) // 8)
        View of the valid section of ``_packed_centroids_buf``.
    """

    # NOTE: Slots deactivates __dict__, and thus reduces memory usage of python objects
    __slots__ = (
        "n_features",
        "_subclusters",
        "_packed_centroids_buf",
        "_prev_leaf",
        "_next_leaf",
    )

    def __init__(self, branching_factor: int, n_features: int):
        self.n_features = n_features
        # The list of subclusters, centroids and squared norms
        # to manipulate throughout.
        self._subclusters: list["_BFSubcluster"] = []
        # Centroids are stored packed. All centroids up to branching_factor are
        # allocated in a contiguous array
        self._packed_centroids_buf = np.empty(
            (branching_factor + 1, (n_features + 7) // 8), dtype=np.uint8
        )
        # Nodes that are leaves have a non-null _prev_leaf
        self._prev_leaf: tp.Optional["_BFNode"] = None
        self._next_leaf: tp.Optional["_BFNode"] = None

    @property
    def is_leaf(self) -> bool:
        return self._prev_leaf is not None

    @property
    def branching_factor(self) -> int:
        return self._packed_centroids_buf.shape[0] - 1

    @property
    def packed_centroids(self) -> NDArray[np.uint8]:
        # packed_centroids returns a view of the valid part of _packed_centroids_buf.
        return self._packed_centroids_buf[: len(self._subclusters), :]

    def append_subcluster(self, subcluster: "_BFSubcluster") -> None:
        n_samples = len(self._subclusters)
        self._subclusters.append(subcluster)
        self._packed_centroids_buf[n_samples] = subcluster.packed_centroid

    def update_split_subclusters(
        self,
        subcluster: "_BFSubcluster",
        new_subcluster1: "_BFSubcluster",
        new_subcluster2: "_BFSubcluster",
    ) -> None:
        """Remove a subcluster from a node and update it with the
        split subclusters.
        """
        # Replace subcluster with new_subcluster1
        idx = self._subclusters.index(subcluster)
        self._subclusters[idx] = new_subcluster1
        self._packed_centroids_buf[idx] = new_subcluster1.packed_centroid
        # Append new_subcluster2
        self.append_subcluster(new_subcluster2)

    def insert_bf_subcluster(
        self,
        subcluster: "_BFSubcluster",
        merge_accept_fn: MergeAcceptFunction,
        threshold: float,
    ) -> bool:
        """Insert a new subcluster into the node."""
        if not self._subclusters:
            self.append_subcluster(subcluster)
            return False

        # Within this node, find the closest subcluster to the one to-be-inserted
        sim_matrix = jt_sim_packed(self.packed_centroids, subcluster.packed_centroid)
        closest_idx = np.argmax(sim_matrix)
        closest_subcluster = self._subclusters[closest_idx]
        closest_node = closest_subcluster.child
        if closest_node is None:
            # The subcluster doesn't have a child node (this is a leaf node)
            # attempt direct merge
            merge_was_successful = closest_subcluster.merge_subcluster(
                subcluster, threshold, merge_accept_fn
            )
            if not merge_was_successful:
                # Could not merge due to criteria
                # Append subcluster, and check if splitting *this node* is needed
                self.append_subcluster(subcluster)
                return len(self._subclusters) > self.branching_factor
            # Merge success, update the centroid
            self._packed_centroids_buf[closest_idx] = closest_subcluster.packed_centroid
            return False

        # Hard case: the closest subcluster has a child (is 'tracking'), use recursion
        child_must_be_split = closest_node.insert_bf_subcluster(
            subcluster, merge_accept_fn, threshold
        )
        if child_must_be_split:
            # Split the child node and redistribute subclusters. Update
            # this node with the 'tracking' subclusters of the two new children.
            # Then, check if *this node* needs splitting too
            new_subcluster1, new_subcluster2 = _split_node(closest_node)
            self.update_split_subclusters(
                closest_subcluster, new_subcluster1, new_subcluster2
            )
            return len(self._subclusters) > self.branching_factor

        # Child need not be split, update the *tracking* closest subcluster
        closest_subcluster.update(subcluster)
        self._packed_centroids_buf[closest_idx] = self._subclusters[
            closest_idx
        ].packed_centroid
        return False


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
    n_samples : int
        Number of samples that belong to each subcluster.

    linear_sum : ndarray
        Linear sum of all the samples in a subcluster. Prevents holding
        all sample data in memory.

    packed_centroid : ndarray of shape (branching_factor + 1, n_features)
        Centroid of the subcluster. Prevent recomputing of centroids when
        ``BFNode.packed_centroids`` is called.

    mol_indices : list, default=[]
        List of indices of molecules included in the given cluster.

    child : _BFNode
        Child Node of the subcluster. Once a given _BFNode is set as the child
        of the _BFNode, it is set to ``self.child``.
    """

    # NOTE: Slots deactivates __dict__, and thus reduces memory usage of python objects
    __slots__ = ("_buffer", "packed_centroid", "child", "mol_indices")

    def __init__(
        self,
        *,
        linear_sum: NDArray[np.integer] | None = None,
        mol_indices: tp.Sequence[int] = (),
        n_features: int = 2048,
        buffer: NDArray[np.integer] | None = None,
    ):
        # NOTE: Internally, _buffer holds both "linear_sum" and "n_samples" It is
        # guaranteed to always have the minimum required uint dtype It should not be
        # accessed by external classes, only used internally. The individual parts can
        # be accessed in a read-only way using the linear_sum and n_samples
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
            self.packed_centroid = calc_centroid(buffer[:-1], buffer[-1], pack=True)
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
                self.packed_centroid = pack_fingerprints(
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
                self.packed_centroid = np.empty(
                    0, dtype=np.uint8
                )  # Will be overwritten
        self.mol_indices = list(mol_indices)
        self.child: tp.Optional["_BFNode"] = None

    @property
    def unpacked_centroid(self) -> NDArray[np.uint8]:
        return _unpack_fingerprints(self.packed_centroid, self.n_features)

    @property
    def n_features(self) -> int:
        return len(self._buffer) - 1

    @property
    def dtype_name(self) -> str:
        return self._buffer.dtype.name

    @property
    def linear_sum(self) -> NDArray[np.integer]:
        read_only_view = self._buffer[:-1]
        read_only_view.flags.writeable = False
        return read_only_view

    @property
    def n_samples(self) -> int:
        # Returns a python int, which is guaranteed to never overflow in sums, so
        # n_samples can always be safely added when accessed through this property
        return self._buffer.item(-1)

    # NOTE: Part of the contract is that all elements of linear sum must always be
    # less or equal to n_samples. This function does not check this
    def replace_n_samples_and_linear_sum(
        self, n_samples: int, linear_sum: NDArray[np.integer]
    ) -> None:
        # Cast to the minimum uint that can hold the inputs
        self._buffer = self._buffer.astype(min_safe_uint(n_samples), copy=False)
        # NOTE: Assignments are safe and do not recast the buffer
        self._buffer[:-1] = linear_sum
        self._buffer[-1] = n_samples
        self.packed_centroid = calc_centroid(linear_sum, n_samples, pack=True)

    # NOTE: Part of the contract is that all elements of linear sum must always be
    # less or equal to n_samples. This function does not check this
    def add_to_n_samples_and_linear_sum(
        self, n_samples: int, linear_sum: NDArray[np.integer]
    ) -> None:
        # Cast to the minimum uint that can hold the inputs
        new_n_samples = self.n_samples + n_samples
        self._buffer = self._buffer.astype(min_safe_uint(new_n_samples), copy=False)
        # NOTE: Assignment and inplace add are safe and do not recast the buffer
        self._buffer[:-1] += linear_sum
        self._buffer[-1] = new_n_samples
        self.packed_centroid = calc_centroid(
            self._buffer[:-1], new_n_samples, pack=True
        )

    def update(self, subcluster: "_BFSubcluster") -> None:
        self.add_to_n_samples_and_linear_sum(
            subcluster.n_samples, subcluster.linear_sum
        )
        self.mol_indices.extend(subcluster.mol_indices)

    def merge_subcluster(
        self,
        nominee_cluster: "_BFSubcluster",
        threshold: float,
        merge_accept_fn: MergeAcceptFunction,
    ) -> bool:
        """Check if a cluster is worthy enough to be merged. If yes, merge."""
        old_n = self.n_samples
        nom_n = nominee_cluster.n_samples
        new_n = old_n + nom_n
        old_ls = self.linear_sum
        nom_ls = nominee_cluster.linear_sum
        # np.add with explicit dtype is safe from overflows, e.g. :
        # np.add(np.uint8(255), np.uint8(255), dtype=np.uint16) = np.uint16(510)
        new_ls = np.add(old_ls, nom_ls, dtype=min_safe_uint(new_n))
        if merge_accept_fn(threshold, new_ls, new_n, old_ls, nom_ls, old_n, nom_n):
            self.replace_n_samples_and_linear_sum(new_n, new_ls)
            self.mol_indices.extend(nominee_cluster.mol_indices)
            return True
        return False


class _CentroidsMolIds(tp.TypedDict):
    centroids: list[NDArray[np.uint8]]
    mol_ids: list[list[int]]


class BitBirch:
    r"""Implements the BitBIRCH clustering algorithm, 'Lean' version

    Memory and time efficient, online-learning algorithm. It constructs a tree data
    structure with the cluster centroids being read off the leaf.

    If you find this software useful please cite the following articles:
    - BitBIRCH: efficient clustering of large molecular libraries:
      https://doi.org/10.1039/D5DD00030K
    - BitBIRCH Clustering Refinement Strategies:
      https://doi.org/10.1021/acs.jcim.5c00627
    - BitBIRCH-Lean: TO-BE-ADDED

    Parameters
    ----------

    threshold : float = 0.65
        The similarity radius of the subcluster obtained by merging a new sample and the
        closest subcluster should be greater than the threshold. Otherwise a new
        subcluster is started. Setting this value to be very low promotes splitting and
        vice-versa.

    branching_factor : int = 50
        Maximum number of 'BitFeatures' subclusters in each node. If a new sample
        enters such that the number of subclusters exceed the branching_factor then that
        node is split into two nodes with the subclusters redistributed in each. The
        parent subcluster of that node is removed and two new subclusters are added as
        parents of the 2 split nodes.

    merge_criterion: str
        radius, diameter or tolerance
        - radius: merge subcluster based on comparison to centroid of the cluster
        - diameter: merge subcluster based on instant Tanimoto similarity of cluster
        - tolerance: applies tolerance threshold to diameter merge criteria, which
            will merge subcluster with stricter threshold for newly added molecules

    tolerance: float
        Penalty value for similarity threshold of the 'tolerance' merge criteria

    Notes
    -----

    The tree data structure consists of nodes with each node holdint a number of
    subclusters ('BitFeatures'). The maximum number of subclusters in a node is
    determined by the branching factor. Each subcluster maintains a linear sum,
    mol_indices and the number of samples in that subcluster. In addition, each
    subcluster can also have a node as its child, if the subcluster is not a member of a
    leaf node.

    Each time a new fingerprint is fitted, it is merged with the subcluster closest to
    it and the linear sum, mol_indices and the number of samples int the corresponding
    subcluster are updated. This is done recursively untils the properties of a leaf
    node are updated.
    """

    def __init__(
        self,
        *,
        threshold: float = 0.65,
        branching_factor: int = 50,
        merge_criterion: str | MergeAcceptFunction | None = None,
        tolerance: float | None = None,
    ):
        # Criterion for merges
        self._threshold = threshold
        self._branching_factor = branching_factor
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
            if isinstance(merge_criterion, MergeAcceptFunction):
                if tolerance is not None:
                    raise ValueError(
                        "'tolerance' arg is disregarded for custom merge functions"
                    )
                self._merge_accept_fn = merge_criterion
            else:
                self._merge_accept_fn = get_merge_accept_fn(merge_criterion, tolerance)

        # Tree state
        self._num_fitted_fps = 0
        self._root: _BFNode | None = None
        self._dummy_leaf = _BFNode(branching_factor=2, n_features=0)

        # For backwards compatibility, weak-register in global state This is used to
        # update the merge_accept function if the global set_merge() is called
        # (discouraged)
        _BITBIRCH_INSTANCES.add(self)

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def branching_factor(self) -> int:
        return self._branching_factor

    @property
    def merge_criterion(self) -> str:
        return self._merge_accept_fn.name

    @property
    def tolerance(self) -> float | None:
        fn = self._merge_accept_fn
        if hasattr(fn, "tolerance"):
            return fn.tolerance
        return None

    @property
    def is_init(self) -> bool:
        r"""Whether the tree has been initialized (True after first call to `fit()`)"""
        return self._dummy_leaf._next_leaf is not None

    @property
    def num_fitted_fps(self) -> int:
        r"""Total number of fitted fingerprints"""
        return self._num_fitted_fps

    def set_merge(
        self,
        criterion: str | MergeAcceptFunction | None = None,
        *,
        tolerance: float = 0.05,
        threshold: float | None = None,
        branching_factor: int | None = None,
    ) -> None:
        r"""Changes the criteria for merging subclusters in this BitBirch tree

        For an explanation of the parameters see the `BitBirch` class docstring.
        """
        if _global_merge_accept is not None:
            raise ValueError(
                "BitBirch.set_merge() can only called if "
                "the global set_merge() function has *not* been used"
            )
        if criterion is not None:
            if isinstance(criterion, MergeAcceptFunction):
                if tolerance is not None:
                    raise ValueError(
                        "'tolerance' arg is disregarded for custom merge functions"
                    )
                self._merge_accept_fn = criterion
            else:
                self._merge_accept_fn = get_merge_accept_fn(criterion, tolerance)
            self._merge_accept_fn = get_merge_accept_fn(criterion, tolerance)
        if threshold is not None:
            self._threshold = threshold
        if branching_factor is not None:
            self._branching_factor = branching_factor

    def fit(
        self,
        X: _Input | Path | str,
        /,
        reinsert_indices: tp.Iterable[int] | None = None,
        input_is_packed: bool = True,
        n_features: int | None = None,
        max_fps: int | None = None,
    ) -> tpx.Self:
        r"""Build a BF Tree for the input data.

        Parameters
        ----------

        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Input data.

        reinsert_indices: Iterable[int]
            if `reinsert_indices` is passed, X corresponds only to the molecules that
            will be reinserted into the tree, and `reinsert_indices` are the indices
            associated with these molecules.

        input_is_packed: bool
            Whether the input fingerprints are packed

        n_features: int
            Number of featurs of input fingerprints. Only required for packed inputs if
            it is not a multiple of 8, otherwise it is redundant.

        Returns
        -------

        self
            Fitted estimator.
        """
        if isinstance(X, (Path, str)):
            X = _mmap_file_and_madvise_sequential(Path(X), max_fps=max_fps)
            mmanager = _ArrayMemPagesManager.from_bb_input(X)
        else:
            X = X[:max_fps]
            mmanager = _ArrayMemPagesManager.from_bb_input(X, can_release=False)

        n_features = _validate_n_features(X, input_is_packed, n_features)
        # Start a new tree the first time this function is called
        if self._only_has_leaves:
            raise ValueError("Internal nodes were released, call reset() before fit()")
        if not self.is_init:
            self._initialize_tree(n_features)
        self._root = cast("_BFNode", self._root)  # After init, this is not None

        # The array iterator either copies, un-sparsifies, or does nothing
        # with the array rows, depending on the kind of X passed
        arr_iterable = _get_array_iterable(X, input_is_packed, n_features)
        arr_iterable = cast(tp.Iterable[NDArray[np.uint8]], arr_iterable)
        iterable: tp.Iterable[tuple[int, NDArray[np.uint8]]]
        if reinsert_indices is None:
            iterable = enumerate(arr_iterable, self.num_fitted_fps)
        else:
            iterable = zip(reinsert_indices, arr_iterable)

        threshold = self._threshold
        branching_factor = self._branching_factor
        merge_accept_fn = self._merge_accept_fn

        arr_idx = 0
        for idx, fp in iterable:
            subcluster = _BFSubcluster(
                linear_sum=fp, mol_indices=[idx], n_features=n_features
            )
            split = self._root.insert_bf_subcluster(
                subcluster, merge_accept_fn, threshold
            )

            if split:
                new_subcluster1, new_subcluster2 = _split_node(self._root)
                self._root = _BFNode(branching_factor, n_features)
                self._root.append_subcluster(new_subcluster1)
                self._root.append_subcluster(new_subcluster2)

            self._num_fitted_fps += 1
            arr_idx += 1
            if mmanager.can_release and mmanager.should_release_curr_page(arr_idx):
                mmanager.release_curr_page_and_update_addr()
        return self

    def _fit_np(
        self,
        X: _Input | Path | str,
        reinsert_index_seqs: tp.Iterable[tp.Sequence[int]] | None = None,
    ) -> tpx.Self:
        r"""Build a BF Tree starting from buffers

        Buffers are arrays of the form:
            - buffer[0:-1] = linear_sum
            - buffer[-1] = n_samples
        And X is either an array or a list of such buffers

        if `reinsert_index_seqs` is passed, X corresponds only to the buffers to be
        reinserted into the tree, and `reinsert_index_seqs` are the sequences
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
        if isinstance(X, (Path, str)):
            X = _mmap_file_and_madvise_sequential(Path(X))
            mmanager = _ArrayMemPagesManager.from_bb_input(X)
        else:
            mmanager = _ArrayMemPagesManager.from_bb_input(X, can_release=False)

        n_features = _validate_n_features(X, input_is_packed=False) - 1
        # Start a new tree the first time this function is called
        if self._only_has_leaves:
            raise ValueError("Internal nodes were released, call reset() before fit()")
        if not self.is_init:
            self._initialize_tree(n_features)
        self._root = cast("_BFNode", self._root)  # After init, this is not None

        # The array iterator either copies, un-sparsifies, or does nothing with the
        # array rows, depending on the kind of X passed
        arr_iterable = _get_array_iterable(X, input_is_packed=False, dtype=X[0].dtype)
        merge_accept_fn = self._merge_accept_fn
        threshold = self._threshold
        branching_factor = self._branching_factor
        idx_provider: tp.Iterable[tp.Sequence[int]]
        arr_idx = 0
        if reinsert_index_seqs is None:
            idx_provider = ([idx] for idx in range(self.num_fitted_fps))
        else:
            idx_provider = reinsert_index_seqs
        for idxs, buf in zip(idx_provider, arr_iterable):
            subcluster = _BFSubcluster(
                buffer=buf, mol_indices=idxs, n_features=n_features
            )
            split = self._root.insert_bf_subcluster(
                subcluster, merge_accept_fn, threshold
            )

            if split:
                new_subcluster1, new_subcluster2 = _split_node(self._root)
                self._root = _BFNode(branching_factor, n_features)
                self._root.append_subcluster(new_subcluster1)
                self._root.append_subcluster(new_subcluster2)

            self._num_fitted_fps += len(idxs)
            arr_idx += 1
            if mmanager.can_release and mmanager.should_release_curr_page(arr_idx):
                mmanager.release_curr_page_and_update_addr()
        return self

    # Provided for backwards compatibility
    def fit_reinsert(
        self,
        X: _Input | Path | str,
        reinsert_indices: tp.Iterable[int],
        input_is_packed: bool = True,
        n_features: int | None = None,
        max_fps: int | None = None,
    ) -> tpx.Self:
        r""":meta private:"""
        return self.fit(X, reinsert_indices, input_is_packed, n_features, max_fps)

    def _initialize_tree(self, n_features: int) -> None:
        # Initialize the root (and a dummy node to get back the subclusters
        self._root = _BFNode(self.branching_factor, n_features)
        self._dummy_leaf._next_leaf = self._root
        self._root._prev_leaf = self._dummy_leaf

    def _get_leaves(self) -> tp.Iterator[_BFNode]:
        r"""Yields all leaf nodes"""
        if not self.is_init:
            raise ValueError("The model has not been fitted yet.")
        leaf = self._dummy_leaf._next_leaf
        while leaf is not None:
            yield leaf
            leaf = leaf._next_leaf

    def get_centroids_mol_ids(
        self, sort: bool = True, packed: bool = True
    ) -> _CentroidsMolIds:
        """Get a dict with centroids and mol indices of the leaves"""
        # NOTE: This is different from the original bitbirch, here outputs are sorted by
        # default
        centroids = []
        mol_ids = []
        attr = "packed_centroid" if packed else "unpacked_centroid"
        for subcluster in self._get_leaf_bfs(sort=sort):
            centroids.append(getattr(subcluster, attr))
            mol_ids.append(subcluster.mol_indices)
        return {"centroids": centroids, "mol_ids": mol_ids}

    def get_centroids(
        self, sort: bool = True, packed: bool = True
    ) -> list[NDArray[np.uint8]]:
        r"""Get a list of arrays with the centroids' fingerprints"""
        # NOTE: This is different from the original bitbirch, here outputs are sorted by
        # default
        attr = "packed_centroid" if packed else "unpacked_centroid"
        return [getattr(s, attr) for s in self._get_leaf_bfs(sort=sort)]

    def get_cluster_mol_ids(self, sort: bool = True) -> list[list[int]]:
        r"""Get the indices of the molecules in each cluster"""
        return [s.mol_indices for s in self._get_leaf_bfs(sort=sort)]

    def get_assignments(
        self, n_mols: int | None = None, sort: bool = True, check_valid: bool = True
    ) -> NDArray[np.uint64]:
        r"""Get an array with the cluster labels associated with each fingerprint idx"""
        if n_mols is not None:
            warnings.warn("The n_mols argument is redundant", DeprecationWarning)
        if n_mols is not None and n_mols != self.num_fitted_fps:
            raise ValueError(
                f"Provided n_mols {n_mols} is different"
                f" from the number of fitted fingerprints {self.num_fitted_fps}"
            )

        if check_valid:
            assignments = np.full(self.num_fitted_fps, 0, dtype=np.uint64)
        else:
            assignments = np.empty(self.num_fitted_fps, dtype=np.uint64)

        iterator: tp.Iterable[list[int]]
        if sort:
            iterator = self.get_cluster_mol_ids(sort=True)
        else:
            iterator = (
                s.mol_indices for leaf in self._get_leaves() for s in leaf._subclusters
            )
        for i, mol_ids in enumerate(iterator, 1):
            assignments[mol_ids] = i

        # Check that there are no unassigned molecules
        if check_valid and (assignments == 0).any():
            raise ValueError("There are unasigned molecules")
        return assignments

    def dump_assignments(
        self,
        path: Path | str,
        smiles: tp.Iterable[str] = (),
        sort: bool = True,
        check_valid: bool = True,
    ) -> None:
        r"""Dump the cluster assignments to a *.csv file"""
        import pandas as pd  # Hide pandas import since it is heavy

        path = Path(path)
        if isinstance(smiles, str):
            smiles = [smiles]
        smiles = np.asarray(smiles, dtype=np.str_)
        # Dump cluster assignments to *.csv
        assignments = self.get_assignments(sort=sort, check_valid=check_valid)
        if smiles.size and (len(assignments) != len(smiles)):
            raise ValueError(
                f"Len of the provided smiles {len(smiles)}"
                f" must match the number of fitted fingerprints {self.num_fitted_fps}"
            )
        if not smiles:
            df = pd.DataFrame({"assignments": assignments})
        if smiles.size:
            df["smiles"] = smiles
        df.to_csv(path, index=False)

    def reset(self) -> None:
        r"""Reset the tree state

        Delete *all internal nodes and leafs*, does not reset the merge criterion or
        other merge parameters.
        """
        # Reset the whole tree
        if self._root is not None:
            self._root._prev_leaf = None
            self._root._next_leaf = None
        self._dummy_leaf._next_leaf = None
        self._root = None
        self._num_fitted_fps = 0

    def delete_internal_nodes(self) -> None:
        r"""Delete all nodes in the tree that are not leaves

        This function is for advanced usage only. It should be called if there is need
        to use the BitBirch leaf clusters, but you need to release the memory held by
        the internal nodes. After calling this function, no more fingerprints can be fit
        into the tree, unless a call to `.reset()` afterwards releases the *whole tree*,
        including the leaf clusters.
        """
        if not tp.cast(_BFNode, self._root).is_leaf:
            # release all nodes that are not leaves,
            # they are kept alive by references from dummy_leaf
            self._root = None

    @property
    def _only_has_leaves(self) -> bool:
        return (self._root is None) and (self._dummy_leaf._next_leaf is not None)

    def refine_inplace(
        self,
        X: _Input | Path | str,
        initial_mol: int = 0,
        input_is_packed: bool = True,
        n_largest: int = 1,
    ) -> tpx.Self:
        r"""Refine the tree: break the largest clusters in singletons and re-fit"""
        if not self.is_init:
            raise ValueError("The model has not been fitted yet.")
        # Release the memory held by non-leaf nodes
        self.delete_internal_nodes()

        # Extract the BitFeatures of the leaves, breaking the largest cluster apart into
        # singleton subclusters
        fps_bfs, mols_bfs = self._bf_to_np_refine(  # This function takes a bunch of mem
            X,
            initial_mol=initial_mol,
            input_is_packed=input_is_packed,
            n_largest=n_largest,
        )
        # Reset the tree
        self.reset()

        # Rebuild the tree again from scratch, reinserting all the subclusters
        for bufs, mol_idxs in zip(fps_bfs.values(), mols_bfs.values()):
            self._fit_np(bufs, reinsert_index_seqs=mol_idxs)
        return self

    def _get_leaf_bfs(self, sort: bool = True) -> list[_BFSubcluster]:
        r"""Get the BitFeatures of the leaves"""
        bfs = [s for leaf in self._get_leaves() for s in leaf._subclusters]
        if sort:
            # Sort the BitFeatures by the number of samples in the cluster
            bfs.sort(key=lambda x: x.n_samples, reverse=True)
        return bfs

    def _bf_to_np_refine(
        self,
        X: _Input | Path | str | tp.Sequence[Path],
        initial_mol: int = 0,
        input_is_packed: bool = True,
        n_largest: int = 1,
    ) -> tuple[dict[str, list[NDArray[np.integer]]], dict[str, list[list[int]]]]:
        """Prepare numpy bufs ('np') for BitFeatures, splitting the biggest n clusters

        The largest clusters are split into singletons. In order to perform this split,
        the *original* fingerprint array used to fit the tree (X) has to be provided,
        together with the index associated with the first fingerprint.

        The split is only performed for the returned 'np' buffers, the clusters in the
        tree itself are not modified
        """
        if n_largest < 1:
            raise ValueError("n_largest must be >= 1")

        bfs = self._get_leaf_bfs()
        largest = bfs[:n_largest]
        rest = bfs[n_largest:]
        n_features = largest[0].n_features

        dtypes_to_fp, dtypes_to_mols = self._prepare_bf_to_buffer_dicts(rest)
        # Add X and mol indices of the "big" cluster
        if input_is_packed:
            unpack_or_copy = lambda x: _unpack_fingerprints(
                cast(NDArray[np.uint8], x), n_features
            )
        else:
            unpack_or_copy = lambda x: x.copy()

        for big_bf in largest:
            full_arr_idxs = [(idx - initial_mol) for idx in big_bf.mol_indices]
            _X: _Input
            if isinstance(X, (Path, str)):
                # Only load the specific required mol idxs
                _X = cast(NDArray[np.integer], np.load(X, mmap_mode="r"))[full_arr_idxs]
                arr_idxs = list(range(len(_X)))
                mol_idxs = big_bf.mol_indices
            elif isinstance(X[0], Path):
                # Only load the specific required mol idxs
                sort_idxs = np.argsort(full_arr_idxs)
                _X = _get_fingerprints_from_file_seq(
                    cast(tp.Sequence[Path], X),
                    [full_arr_idxs[i] for i in sort_idxs],
                )
                arr_idxs = list(range(len(_X)))
                mol_idxs = big_bf.mol_indices
                mol_idxs = [mol_idxs[i] for i in sort_idxs]
            else:
                # Index the full array / list
                _X = cast(_Input, X)
                arr_idxs = full_arr_idxs
                mol_idxs = big_bf.mol_indices

            for mol_idx, arr_idx in zip(mol_idxs, arr_idxs):
                buffer = np.empty(n_features + 1, dtype=np.uint8)
                buffer[:-1] = unpack_or_copy(_X[arr_idx])
                buffer[-1] = 1
                dtypes_to_fp["uint8"].append(buffer)
                dtypes_to_mols["uint8"].append([mol_idx])
        return dtypes_to_fp, dtypes_to_mols

    def _bf_to_np(
        self,
    ) -> tuple[dict[str, list[NDArray[np.integer]]], dict[str, list[list[int]]]]:
        """Prepare numpy buffers ('np') for BitFeatures of all clusters"""
        return self._prepare_bf_to_buffer_dicts(self._get_leaf_bfs())

    @staticmethod
    def _prepare_bf_to_buffer_dicts(
        bfs: list["_BFSubcluster"],
    ) -> tuple[dict[str, list[NDArray[np.integer]]], dict[str, list[list[int]]]]:
        # Helper function used when returning lists of subclusters
        dtypes_to_fp = defaultdict(list)
        dtypes_to_mols = defaultdict(list)
        for bf in bfs:
            dtypes_to_fp[bf.dtype_name].append(bf._buffer)
            dtypes_to_mols[bf.dtype_name].append(bf.mol_indices)
        return dtypes_to_fp, dtypes_to_mols

    def __repr__(self) -> str:
        fn = self._merge_accept_fn
        parts = [
            f"threshold={self.threshold}",
            f"branching_factor={self.branching_factor}",
            f"merge_criterion='{fn.name if fn.name in BUILTIN_MERGES else fn}'",
        ]
        if self.tolerance is not None:
            parts.append(f"tolerance={self.tolerance}")
        return f"{self.__class__.__name__}({', '.join(parts)})"


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
def _get_array_iterable(
    X: _Input,
    input_is_packed: bool = True,
    n_features: int | None = None,
    dtype: DTypeLike = np.uint8,
) -> tp.Iterable[NDArray[np.integer]]:
    if input_is_packed:
        # Unpacking copies the fingerprints, so no extra copy required
        # NOTE: cast seems to have a very small overhead in this loop for some reason
        return (_unpack_fingerprints(a, n_features) for a in X)  # type: ignore
    if isinstance(X, list):
        # No copy is required here unless the dtype is not uint8
        return (a.astype(dtype, copy=False) for a in X)
    if isinstance(X, np.ndarray):
        # A copy is required here to avoid keeping a ref to the full array alive
        return (a.astype(dtype, copy=True) for a in X)
    return _iter_sparse(X)


# NOTE: In practice this branch is never used, it could probably safely be deleted
def _iter_sparse(X: tp.Any) -> tp.Iterator[NDArray[np.uint8]]:
    import scipy.sparse  # Hide this import since scipy is heavy

    if not scipy.sparse.issparse(X):
        raise ValueError(f"Input of type {type(X)} is not supported")
    n_samples, n_features = X.shape
    X_indices = X.indices
    X_data = X.data
    X_indptr = X.indptr
    for i in range(n_samples):
        a = np.zeros(n_features, dtype=np.uint8)
        startptr, endptr = X_indptr[i], X_indptr[i + 1]
        nonzero_indices = X_indices[startptr:endptr]
        a[nonzero_indices] = X_data[startptr:endptr].astype(np.uint8, copy=False)
        yield a
