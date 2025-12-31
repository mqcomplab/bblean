r"""Merging criteria for BitBIRCH clustering"""

import numpy as np
from numpy.typing import NDArray

# NOTE: jt_isim_from_sum is equivalent to jt_isim_diameter_compl_from_sum
from bblean.similarity import jt_isim_from_sum, jt_isim_radius_compl_from_sum

BUILTIN_MERGES = [
    "radius",
    "diameter",
    "tolerance-diameter",
    "tolerance-radius",
    "tolerance-legacy",
    "never-merge",
]


class MergeAcceptFunction:
    # For the merge functions, although outputs of jt_isim_from_sum f64, directly using
    # f64 is *not* faster than starting with uint64
    name: str = ""

    def __call__(
        self,
        threshold: float,
        new_ls: NDArray[np.integer],
        new_n: int,
        old_ls: NDArray[np.integer],
        nom_ls: NDArray[np.integer],
        old_n: int,
        nom_n: int,
    ) -> bool:
        raise NotImplementedError("Must be implemented by subclasses")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class RadiusMerge(MergeAcceptFunction):
    name = "radius"

    def __call__(
        self,
        threshold: float,
        new_ls: NDArray[np.integer],
        new_n: int,
        old_ls: NDArray[np.integer],
        nom_ls: NDArray[np.integer],
        old_n: int,
        nom_n: int,
    ) -> bool:
        return jt_isim_radius_compl_from_sum(new_ls, new_n) >= threshold


class DiameterMerge(MergeAcceptFunction):
    name = "diameter"

    def __call__(
        self,
        threshold: float,
        new_ls: NDArray[np.integer],
        new_n: int,
        old_ls: NDArray[np.integer],
        nom_ls: NDArray[np.integer],
        old_n: int,
        nom_n: int,
    ) -> bool:
        return jt_isim_from_sum(new_ls, new_n) >= threshold


class ToleranceDiameterMerge(MergeAcceptFunction):
    name = "tolerance-diameter"
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

    def __call__(
        self,
        threshold: float,
        new_ls: NDArray[np.integer],
        new_n: int,
        old_ls: NDArray[np.integer],
        nom_ls: NDArray[np.integer],
        old_n: int,
        nom_n: int,
    ) -> bool:
        new_dc = jt_isim_from_sum(new_ls, new_n)
        if new_dc < threshold:
            return False
        # If the old n is 1 then merge directly (infinite tolerance), since the
        # old_d is undefined for a single fp
        if old_n == 1:
            return True
        # Only merge if the new_dc is greater or equal to the old, up to some tolerance,
        # which decays with N
        old_dc = jt_isim_from_sum(old_ls, old_n)
        tol = max(self.tolerance * (np.exp(-self.decay * old_n) - self.offset), 0.0)
        return new_dc >= old_dc - tol

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.tolerance})"


class ToleranceRadiusMerge(ToleranceDiameterMerge):
    name = "tolerance-radius"

    def __call__(
        self,
        threshold: float,
        new_ls: NDArray[np.integer],
        new_n: int,
        old_ls: NDArray[np.integer],
        nom_ls: NDArray[np.integer],
        old_n: int,
        nom_n: int,
    ) -> bool:
        new_rc = jt_isim_radius_compl_from_sum(new_ls, new_n)
        if new_rc < threshold:
            return False
        if old_n == 1:
            return True
        old_rc = jt_isim_radius_compl_from_sum(old_ls, old_n)
        tol = max(self.tolerance * (np.exp(-self.decay * old_n) - self.offset), 0.0)
        return new_rc >= old_rc - tol

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.tolerance})"


class NeverMerge(ToleranceDiameterMerge):
    name = "never-merge"

    def __call__(
        self,
        threshold: float,
        new_ls: NDArray[np.integer],
        new_n: int,
        old_ls: NDArray[np.integer],
        nom_ls: NDArray[np.integer],
        old_n: int,
        nom_n: int,
    ) -> bool:
        return False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class ToleranceMerge(MergeAcceptFunction):
    name = "tolerance-legacy"

    def __init__(self, tolerance: float = 0.05) -> None:
        self.tolerance = tolerance

    def __call__(
        self,
        threshold: float,
        new_ls: NDArray[np.integer],
        new_n: int,
        old_ls: NDArray[np.integer],
        nom_ls: NDArray[np.integer],
        old_n: int,
        nom_n: int,
    ) -> bool:
        # First two branches are equivalent to 'diameter'
        new_dc = jt_isim_from_sum(new_ls, new_n)
        if new_dc < threshold:
            return False
        if old_n == 1 or nom_n != 1:
            return True
        # 'new_dc >= threshold' and 'new_n == old_n + 1' are guaranteed here
        old_dc = jt_isim_from_sum(old_ls, old_n)
        return (new_dc * new_n - old_dc * (old_n - 1)) / 2 >= old_dc - self.tolerance

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.tolerance})"


def get_merge_accept_fn(
    merge_criterion: str, tolerance: float = 0.05
) -> MergeAcceptFunction:
    if merge_criterion == "radius":
        return RadiusMerge()
    elif merge_criterion == "diameter":
        return DiameterMerge()
    elif merge_criterion == "tolerance-legacy":
        return ToleranceMerge(tolerance)
    elif merge_criterion == "tolerance-diameter":
        return ToleranceDiameterMerge(tolerance)
    elif merge_criterion == "tolerance-radius":
        return ToleranceRadiusMerge(tolerance)
    elif merge_criterion == "never-merge":
        return NeverMerge(tolerance)
    raise ValueError(
        f"Unknown merge criterion {merge_criterion} "
        "Valid criteria are: radius|diameter|tolerance-diameter|tolerance-radius"
    )
