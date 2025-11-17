r"""BitBirch 'Lean' classes that fully respects the sklearn API contract.

Use these classes as a drop-in replacement of `sklearn.cluster.Birch` if you are used to
the `sklearn` way of doing things, with the caveat that global clustering is not
currently supported.
"""

import typing as tp
from numpy.typing import NDArray
import numpy as np
import typing_extensions as tpx

from sklearn.utils.validation import check_is_fitted, validate_data
from sklearn.metrics import pairwise_distances_argmin, pairwise_distances
from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    ClusterMixin,
    TransformerMixin,
    _fit_context,
)

from bblean.fingerprints import unpack_fingerprints
from bblean.bitbirch import BitBirch as _BitBirch
from bblean._merges import MergeAcceptFunction

__all__ = ["BitBirch", "UnpackedBitBirch"]

# Required functions for sklearn API:
# - fit() *must be defined*
# - transform()  *must be defined*
# - fit_predict()  (ClusterMixin) default implementation is to fit and then return lbls
# - predict()  # overloaded to use jt instead of euclidean
# - fit_transform()  (TransformerMixin, delegates to *fit* and *transform*)
# - set_output()  (TransformerMixin via _SetOutputMixin)
# set_output(transform="pandas") or transform="default" (numpy array) (or "polars",
# if polars is installed)

#  The following requires _n_features_out after fitting
# - get_feature_names_out() ["bitbirch0", "bitbirch1", ...] (ClassNamePrefix...)

# - get_metadata_routing() () (!?) New feature, unclear what this is and unnecessary
# - partial_fit() ()  Same as fit() for BitBirch

# These require that the parameters are specified in __init__, and are assigned
# to names (or attributes) with the convention self.<param>.
# - get_params() (BaseEstimator)
# - set_params() (BaseEstimator)


class BitBirch(
    ClassNamePrefixFeaturesOutMixin,
    ClusterMixin,
    TransformerMixin,
    BaseEstimator,
    _BitBirch,
):
    r"""Implements the BitBIRCH clustering algorithm, 'Lean' version.

    Inputs to this estimator are *packed* fingerprints by default. If you need get a
    class that always accepts an unpacked input use `bblean.sklearn.UnpackedBitBirch`

    See `bblean.bitbirch.BitBirch` for more details"""

    _parameter_constraints: dict[str, list[tp.Any]] = {}

    def __init__(
        self,
        *,
        threshold: float = 0.65,
        branching_factor: int = 50,
        merge_criterion: str | MergeAcceptFunction | None = None,
        tolerance: float | None = None,
        compute_labels: bool = True,
    ):
        super().__init__(
            threshold=threshold,
            branching_factor=branching_factor,
            merge_criterion=merge_criterion,
            tolerance=tolerance,
        )
        self.compute_labels = compute_labels

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(  # type: ignore
        self, X, y=None, input_is_packed: bool = True, n_features: int | None = None
    ) -> tpx.Self:
        super().fit(X, input_is_packed=input_is_packed, n_features=n_features)
        centroids = np.stack(
            [bf.unpacked_centroid for bf in self._get_leaf_bfs(sort=True)]
        )
        self.subcluster_centers_ = centroids
        self.subcluster_labels_ = np.arange(1, len(centroids) + 1)
        self._n_features_out = centroids.shape[0]
        if self.compute_labels:
            self.labels_ = self.get_assignments()
        return self

    @_fit_context(prefer_skip_nested_validation=True)
    def partial_fit(  # type: ignore
        self,
        X=None,
        y=None,
        input_is_packed: bool = True,
        n_features: int | None = None,
    ) -> tpx.Self:
        if X is None:
            raise ValueError()
        self.fit(X, input_is_packed=input_is_packed, n_features=n_features)
        if self.compute_labels:
            self.labels_ = self.get_assignments()
        return self

    # Overloaded since self.labels_ may not be set
    def fit_predict(  # type: ignore
        self, X, y=None, input_is_packed: bool = True, n_features: int | None = None
    ) -> NDArray[np.integer]:
        self.fit(X, input_is_packed=input_is_packed, n_features=n_features)
        if not self.compute_labels:
            self.labels_ = self.get_assignments()
        return self.labels_

    def predict(  # type: ignore
        self, X, input_is_packed: bool = True, n_features: int | None = None
    ) -> NDArray[np.integer]:
        """Predict data using the ``centroids`` of subclusters."""
        check_is_fitted(self)
        X = validate_data(self, X, accept_sparse="csr", reset=False)
        X = (
            (unpack_fingerprints(X, n_features=n_features) if input_is_packed else X)
            .astype(np.uint8, copy=False)
            .view(np.bool)
        )
        # TODO: Due to a sklearn bug this performs unnecessary casts
        centers = self.subcluster_centers_.astype(np.uint8, copy=False).view(np.bool)
        argmin = pairwise_distances_argmin(X, centers, metric="jaccard")
        return self.subcluster_labels_[argmin]

    def transform(  # type: ignore
        self,
        X,
        input_is_packed: bool = True,
        n_features: int | None = None,
    ):
        check_is_fitted(self)
        X = validate_data(self, X, accept_sparse="csr", reset=False)
        X = (
            (unpack_fingerprints(X, n_features=n_features) if input_is_packed else X)
            .astype(np.uint8, copy=False)
            .view(np.bool)
        )
        centers = self.subcluster_centers_.astype(np.uint8, copy=False).view(np.bool)
        return pairwise_distances(X, centers, metric="jaccard")

    def __sklearn_tags__(self):  # type: ignore
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        return tags


class UnpackedBitBirch(BitBirch):
    r"""Implements the BitBIRCH clustering algorithm, 'Lean' version.

    Inputs to this estimator are *unpacked* fingerprints always

    See `bblean.bitbirch.BitBirch` for more details"""

    def fit(  # type: ignore
        self, X, y=None, input_is_packed: bool = False, n_features: int | None = None
    ) -> tpx.Self:
        return super().fit(X, y, input_is_packed=input_is_packed, n_features=n_features)

    def partial_fit(  # type: ignore
        self, X, y=None, input_is_packed: bool = False, n_features: int | None = None
    ):
        return super().partial_fit(
            X, y, input_is_packed=input_is_packed, n_features=n_features
        )

    def fit_predict(  # type: ignore
        self, X, y=None, input_is_packed: bool = False, n_features: int | None = None
    ):
        return super().fit_predict(
            X, y, input_is_packed=input_is_packed, n_features=n_features
        )

    def predict(  # type: ignore
        self,
        X,
        input_is_packed: bool = False,
        n_features: int | None = None,
    ):
        return super().predict(
            X, input_is_packed=input_is_packed, n_features=n_features
        )

    def transform(  # type: ignore
        self, X, input_is_packed: bool = False, n_features: int | None = None
    ):
        return super().transform(
            X, input_is_packed=input_is_packed, n_features=n_features
        )
