r"""Merging criteria for BitBIRCH clustering

The functionality in this module is advanced and not needed for normal usage of the
library. Make sure you understand the features here before applying them.
"""

import re
import typing as tp
import numpy as np
from numpy.typing import NDArray

# NOTE: jt_isim_from_sum is equivalent to jt_isim_diameter_compl_from_sum
from bblean.similarity import jt_isim_from_sum, jt_isim_radius_compl_from_sum

_BUILTIN_MERGES = [
    "radius",
    "diameter",
    "flexible-tolerance-diameter",
    "tolerance-diameter",
    "tolerance-radius",
    "tolerance",
    "never",
]


class DiscardSubcluster(Exception):
    r"""If raised in hooks, immediatly exit the merge discarding the incident subcluster

    Discarded subclusters will not be stored in the final tree, and will only show
    up if calling `bblean.BitBirch.get_assigments` (or the `labels_` attribute if
    using `bblean.sklearn`) with a cluster label of 0.
    """


class RejectMerge(Exception):
    r"""If raised in hooks, immediatly exit the merge and reject it"""


class MergeAcceptFunction:
    r"""Base class for user defined merges

    If you want to implement a custom BitBirch merge you can subclass this and
    pass an instance of this function to a `bblean.BitBirch` class upon creation
    as ``BitBirch(..., merge_criterion=instance)``.

    .. warning::

        This is an advanced feature, make sure you fully understand what you are doing!
    """

    # For the merge functions, although outputs of jt_isim_from_sum f64, directly using
    # f64 is *not* faster than starting with uint64

    def __call__(
        self,
        thresh: float,
        new_ls: NDArray[np.integer],
        new_n: int,
        old_ls: NDArray[np.integer],
        nom_ls: NDArray[np.integer],
        old_n: int,
        nom_n: int,
        old_idxs: tp.Sequence[int],
        nom_idxs: tp.Sequence[int],
    ) -> bool:
        try:
            thresh = self.on_check_merge_start(
                thresh, new_ls, new_n, old_ls, nom_ls, old_n, nom_n, old_idxs, nom_idxs
            )
            accepted = self.check_merge(
                thresh, new_ls, new_n, old_ls, nom_ls, old_n, nom_n, old_idxs, nom_idxs
            )
            self.on_check_merge_end(accepted, old_idxs, nom_idxs)
        except RejectMerge:
            return False
        return accepted

    def on_check_merge_start(
        self,
        threshold: float,
        new_sum: NDArray[np.integer],
        new_n: int,
        old_sum: NDArray[np.integer],
        nominee_sum: NDArray[np.integer],
        old_n: int,
        nominee_n: int,
        old_idxs: tp.Sequence[int],
        nominee_idxs: tp.Sequence[int],
    ) -> float:
        r"""Hook called before a merge is checked (meant to be overriden)

        See `MergeAcceptFunction.check_merge` for an explanation of the different args

        .. warning::

            Numpy arrays passed to this function may use uint types, watch out for
            pitfalls of unsigned integer arithmetic.

        This function must return the threshold, unchanged. If the threshold is modified
        by this function, the new threshold will be used for this specific merge check.
        """
        return threshold

    def check_merge(
        self,
        threshold: float,
        new_sum: NDArray[np.integer],
        new_n: int,
        old_sum: NDArray[np.integer],
        nominee_sum: NDArray[np.integer],
        old_n: int,
        nominee_n: int,
        old_idxs: tp.Sequence[int],
        nominee_idxs: tp.Sequence[int],
    ) -> bool:
        r"""Check if a merge should be accepted.

        All user-defined merges should subclass this function.

        threshold: Threshold for the merge
        new_sum: old_sum + nominee_sum
        new_n: old_n + new_n
        old_sum: col-wise sum of all fingerprints in this cluster
        nominee_sum: col-wise sum of all fingerprints in the nominee cluster
        old_n: size of this cluster
        nominee_n: size of the nominee cluster
        old_idxs: Indices of the mols in cluster before the nominee cluster is merged
        nominee_idxs: Nominee indices to merge.

        If merging a single molecule (the most common case, when calling
        `bblean.BitBirch.fit`), ``nominee_idxs`` will be a list with a *single index*,
        ``new_n = 1``, and ``nominee_sum`` will be the molecule fingerprint.

        This function must return a boolean that determines whether the merge was
        accepted

        .. warning::

            Numpy arrays passed to this function may use uint types, watch out for
            pitfalls of unsigned integer arithmetic.
        """
        raise NotImplementedError

    def on_check_merge_end(
        self,
        accepted: bool,
        old_idxs: tp.Sequence[int],
        nominee_idxs: tp.Sequence[int],
    ) -> None:
        r"""Hook called after a merge is checked (meant to be overriden)

        accept: Whether the merge was accepted
        old_idxs: Indices of the mols in cluster before the nominee cluster is merged
        nominee_idxs: Nominee indices to merge.

        If merging a single molecule (the most common case, when calling
        `bblean.BitBirch.fit`, ``nominee_idxs`` will be a list with a *single index*)

        This function must not return a value
        """

    @property
    def name(self) -> str:
        return "-".join(
            s.lower()
            for s in re.split(r"(?=[A-Z])", self.__class__.__name__)[1:]
            if s != "Merge"
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class RadiusMerge(MergeAcceptFunction):

    def check_merge(
        self,
        threshold: float,
        new_sum: NDArray[np.integer],
        new_n: int,
        old_sum: NDArray[np.integer],
        nominee_sum: NDArray[np.integer],
        old_n: int,
        nominee_n: int,
        old_idxs: tp.Sequence[int] | None = None,
        nominee_idxs: tp.Sequence[int] | None = None,
    ) -> bool:
        r""":meta private:"""
        return jt_isim_radius_compl_from_sum(new_sum, new_n) >= threshold


class DiameterMerge(MergeAcceptFunction):

    def check_merge(
        self,
        threshold: float,
        new_sum: NDArray[np.integer],
        new_n: int,
        old_sum: NDArray[np.integer],
        nominee_sum: NDArray[np.integer],
        old_n: int,
        nominee_n: int,
        old_idxs: tp.Sequence[int] | None = None,
        nominee_idxs: tp.Sequence[int] | None = None,
    ) -> bool:
        r""":meta private:"""
        return jt_isim_from_sum(new_sum, new_n) >= threshold


class FlexibleToleranceDiameterMerge(MergeAcceptFunction):
    # NOTE: Equivalent to tolerance-diameter but uses min(old_dc, threshold) as the
    # criteria

    def __init__(
        self,
        tolerance: float = 0.05,
        n_max: int = 1000,
        decay: float = 1e-3,
        adaptive: bool = True,
    ) -> None:
        self.tolerance = tolerance
        self.decay = decay
        self.offset = np.exp(-decay * n_max)
        if not adaptive:
            self.decay = 0.0
            self.offset = 0.0

    def check_merge(
        self,
        threshold: float,
        new_sum: NDArray[np.integer],
        new_n: int,
        old_sum: NDArray[np.integer],
        nominee_sum: NDArray[np.integer],
        old_n: int,
        nominee_n: int,
        old_idxs: tp.Sequence[int] | None = None,
        nominee_idxs: tp.Sequence[int] | None = None,
    ) -> bool:
        r""":meta private:"""
        new_dc = jt_isim_from_sum(new_sum, new_n)
        if new_dc < threshold:
            return False
        if old_n == 1:
            return True
        old_dc = jt_isim_from_sum(old_sum, old_n)
        tol = max(self.tolerance * (np.exp(-self.decay * old_n) - self.offset), 0.0)
        return new_dc >= min(old_dc, threshold) - tol

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.tolerance})"


class ToleranceDiameterMerge(MergeAcceptFunction):
    # NOTE: The reliability of the estimate of the cluster should be a function of the
    # size of the old cluster, so in this metric, tolerance is larger for small clusters
    # tolerance = max{ alpha * (exp(-decay * N_old) - offset), 0}

    def __init__(
        self,
        tolerance: float = 0.05,
        n_max: int = 1000,
        decay: float = 1e-3,
        adaptive: bool = True,
    ) -> None:
        self.tolerance = tolerance
        self.decay = decay
        self.offset = np.exp(-decay * n_max)
        if not adaptive:
            self.decay = 0.0
            self.offset = 0.0

    def check_merge(
        self,
        threshold: float,
        new_sum: NDArray[np.integer],
        new_n: int,
        old_sum: NDArray[np.integer],
        nominee_sum: NDArray[np.integer],
        old_n: int,
        nominee_n: int,
        old_idxs: tp.Sequence[int] | None = None,
        nominee_idxs: tp.Sequence[int] | None = None,
    ) -> bool:
        r""":meta private:"""
        new_dc = jt_isim_from_sum(new_sum, new_n)
        if new_dc < threshold:
            return False
        # If the old n is 1 then merge directly (infinite tolerance), since the
        # old_d is undefined for a single fp
        if old_n == 1:
            return True
        # Only merge if the new_dc is greater or equal to the old, up to some tolerance,
        # which decays with N
        old_dc = jt_isim_from_sum(old_sum, old_n)
        tol = max(self.tolerance * (np.exp(-self.decay * old_n) - self.offset), 0.0)
        return new_dc >= old_dc - tol

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.tolerance})"


class ToleranceRadiusMerge(ToleranceDiameterMerge):

    def check_merge(
        self,
        threshold: float,
        new_sum: NDArray[np.integer],
        new_n: int,
        old_sum: NDArray[np.integer],
        nominee_sum: NDArray[np.integer],
        old_n: int,
        nominee_n: int,
        old_idxs: tp.Sequence[int] | None = None,
        nominee_idxs: tp.Sequence[int] | None = None,
    ) -> bool:
        r""":meta private:"""
        new_rc = jt_isim_radius_compl_from_sum(new_sum, new_n)
        if new_rc < threshold:
            return False
        if old_n == 1:
            return True
        old_rc = jt_isim_radius_compl_from_sum(old_sum, old_n)
        tol = max(self.tolerance * (np.exp(-self.decay * old_n) - self.offset), 0.0)
        return new_rc >= old_rc - tol


class NeverMerge(MergeAcceptFunction):
    r""":meta private:"""

    def check_merge(
        self,
        threshold: float,
        new_sum: NDArray[np.integer],
        new_n: int,
        old_sum: NDArray[np.integer],
        nominee_sum: NDArray[np.integer],
        old_n: int,
        nominee_n: int,
        old_idxs: tp.Sequence[int] | None = None,
        nominee_idxs: tp.Sequence[int] | None = None,
    ) -> bool:
        return False


class ToleranceLegacyMerge(MergeAcceptFunction):
    r""":meta private:"""

    def __init__(self, tolerance: float = 0.05) -> None:
        self.tolerance = tolerance

    def check_merge(
        self,
        threshold: float,
        new_sum: NDArray[np.integer],
        new_n: int,
        old_sum: NDArray[np.integer],
        nominee_sum: NDArray[np.integer],
        old_n: int,
        nominee_n: int,
        old_idxs: tp.Sequence[int] | None = None,
        nominee_idxs: tp.Sequence[int] | None = None,
    ) -> bool:
        # First two branches are equivalent to 'diameter'
        new_dc = jt_isim_from_sum(new_sum, new_n)
        if new_dc < threshold:
            return False
        if old_n == 1 or nominee_n != 1:
            return True
        # 'new_dc >= threshold' and 'new_n == old_n + 1' are guaranteed here
        old_dc = jt_isim_from_sum(old_sum, old_n)
        return (new_dc * new_n - old_dc * (old_n - 1)) / 2 >= old_dc - self.tolerance

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.tolerance})"


# Make these ones leaner with less calls so they don't take up any extra time at all in
# the default case
class _FastMerge(MergeAcceptFunction):
    def __call__(
        self,
        thresh: float,
        new_ls: NDArray[np.integer],
        new_n: int,
        old_ls: NDArray[np.integer],
        nom_ls: NDArray[np.integer],
        old_n: int,
        nom_n: int,
        old_idxs: tp.Sequence[int],
        nom_idxs: tp.Sequence[int],
    ) -> bool:
        return self.check_merge(
            thresh, new_ls, new_n, old_ls, nom_ls, old_n, nom_n, old_idxs, nom_idxs
        )

    @property
    def name(self) -> str:
        return "-".join(
            s.lower()
            for s in re.split(r"(?=[A-Z])", self.__class__.__name__)[1:]
            if s not in ["Merge", "Fast"]
        )


class _FastDiameterMerge(_FastMerge, DiameterMerge):
    pass


class _FastRadiusMerge(_FastMerge, RadiusMerge):
    pass


class _FastToleranceDiameterMerge(_FastMerge, ToleranceDiameterMerge):
    pass


class _FastFlexibleToleranceDiameterMerge(_FastMerge, FlexibleToleranceDiameterMerge):
    pass


class _FastToleranceRadiusMerge(_FastMerge, ToleranceRadiusMerge):
    pass


class _FastToleranceLegacyMerge(_FastMerge, ToleranceLegacyMerge):
    pass


class _FastNeverMerge(_FastMerge, NeverMerge):
    pass


class _FastCappedMerge(_FastMerge, MergeAcceptFunction):
    r""":meta private:"""

    def __init__(self, max_size: int) -> None:
        if max_size < 1:
            raise ValueError("Max size must be greater or equal to 1")
        self._max_size = max_size
        # Total size starts with 1 since the first insertion doesn't go through a merge
        self._total_size = 1

    def check_merge(
        self,
        threshold: float,
        new_sum: NDArray[np.integer],
        new_n: int,
        old_sum: NDArray[np.integer],
        nominee_sum: NDArray[np.integer],
        old_n: int,
        nominee_n: int,
        old_idxs: tp.Sequence[int] | None = None,
        nominee_idxs: tp.Sequence[int] | None = None,
    ) -> bool:
        if self._total_size >= self._max_size:
            return True
        self._total_size += 1
        return False

    def reset(self) -> None:
        self._total_size = 1

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._max_size})"


class _FastCappedToleranceDiameterMerge(_FastToleranceDiameterMerge):
    r""":meta private:"""

    def __init__(self, max_size: int, tolerance: float = 0.05) -> None:
        super().__init__(tolerance)
        if max_size < 1:
            raise ValueError("Max size must be greater or equal to 1")
        self._max_size = max_size
        # Total size starts with 1 since the first insertion doesn't go through a merge
        self._total_size = 1

    def check_merge(
        self,
        threshold: float,
        new_sum: NDArray[np.integer],
        new_n: int,
        old_sum: NDArray[np.integer],
        nominee_sum: NDArray[np.integer],
        old_n: int,
        nominee_n: int,
        old_idxs: tp.Sequence[int] | None = None,
        nominee_idxs: tp.Sequence[int] | None = None,
    ) -> bool:
        if self._total_size >= self._max_size:
            return True
        accept = super().check_merge(
            threshold,
            new_sum,
            new_n,
            old_sum,
            nominee_sum,
            old_n,
            nominee_n,
            old_idxs,
            nominee_idxs,
        )
        if not accept:
            self._total_size += 1
        return accept

    def reset(self) -> None:
        self._total_size = 1

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.tolerance}, {self._max_size})"


class _FastCappedDiameterMerge(_FastDiameterMerge):
    r""":meta private:"""

    def __init__(self, max_size: int) -> None:
        if max_size < 1:
            raise ValueError("Max size must be greater or equal to 1")
        self._max_size = max_size
        # Total size starts with 1 since the first insertion doesn't go through a merge
        self._total_size = 1

    def check_merge(
        self,
        threshold: float,
        new_sum: NDArray[np.integer],
        new_n: int,
        old_sum: NDArray[np.integer],
        nominee_sum: NDArray[np.integer],
        old_n: int,
        nominee_n: int,
        old_idxs: tp.Sequence[int] | None = None,
        nominee_idxs: tp.Sequence[int] | None = None,
    ) -> bool:
        if self._total_size >= self._max_size:
            return True
        accept = super().check_merge(
            threshold,
            new_sum,
            new_n,
            old_sum,
            nominee_sum,
            old_n,
            nominee_n,
            old_idxs,
            nominee_idxs,
        )
        if not accept:
            self._total_size += 1
        return accept

    def reset(self) -> None:
        self._total_size = 1

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._max_size})"


def _get_merge_accept_fn(
    merge_criterion: str, tolerance: float = 0.05
) -> MergeAcceptFunction:
    if merge_criterion == "radius":
        return _FastRadiusMerge()
    elif merge_criterion == "diameter":
        return _FastDiameterMerge()
    elif merge_criterion == "never":
        return _FastNeverMerge()
    elif merge_criterion == "tolerance-legacy":
        return _FastToleranceLegacyMerge(tolerance)
    elif merge_criterion == "tolerance-diameter":
        return _FastToleranceDiameterMerge(tolerance)
    elif merge_criterion == "flexible-tolerance-diameter":
        return _FastFlexibleToleranceDiameterMerge(tolerance)
    elif merge_criterion == "tolerance-radius":
        return _FastToleranceRadiusMerge(tolerance)
    elif re.match("diameter-capped-[0-9]+", merge_criterion):
        size = int(merge_criterion.split("-")[-1])
        return _FastCappedDiameterMerge(size)
    elif re.match("tolerance-diameter-capped-[0-9]+", merge_criterion):
        size = int(merge_criterion.split("-")[-1])
        return _FastCappedToleranceDiameterMerge(size, tolerance)
    elif re.match("capped-[0-9]+", merge_criterion):
        size = int(merge_criterion.split("-")[-1])
        return _FastCappedMerge(size)
    raise ValueError(
        f"Unknown merge criterion {merge_criterion} "
        "Valid criteria are: radius|diameter|tolerance-diameter|tolerance-radius"
    )
