r"""Merging criteria for BitBIRCH clustering"""

from numpy.typing import NDArray
import numpy as np

from bblean.fingerprints import centroid_from_sum
from bblean.similarity import jt_isim_from_sum

BUILTIN_MERGES = [
    "radius",
    "diameter",
    "tolerance",
    "tolerance-adapt",
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
        # NOTE: Use uint64 sum since jt_isim_from_sum casts to uint64 internally
        # This prevents multiple casts
        new_unpacked_centroid = centroid_from_sum(new_ls, new_n, pack=False)
        new_ls_1 = np.add(new_ls, new_unpacked_centroid, dtype=np.uint64)
        new_n_1 = new_n + 1
        new_jt = jt_isim_from_sum(new_ls, new_n)
        new_jt_1 = jt_isim_from_sum(new_ls_1, new_n_1)
        return new_jt_1 * new_n_1 - new_jt * (new_n - 1) >= threshold * 2


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


class ToleranceDiameterAdaptiveMerge(MergeAcceptFunction):
    name = "tolerance-adapt"
    # NOTE: The reliability of the estimate of the cluster should be a function of the
    # size of the old cluster, so in this metric, tolerance is larger for small clusters
    # tolerance = max{ alpha * (exp(-decay * N_old) - offset), 0}

    def __init__(
        self, tolerance: float = 0.05, n_max: int = 1000, decay: float = 1e-3
    ) -> None:
        self.tolerance = tolerance
        self.decay = decay
        self.offset = np.exp(-decay * n_max)

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
        new_d = jt_isim_from_sum(new_ls, new_n)
        if new_d < threshold:
            return False

        # If the old n is 1 then merge directly (infinmin{ ite tolerance), since the
        # old_d is undefined for a single fp
        if old_n == 1:
            return True
        # Only merge if the new_d is greater or equal to the old, up to some tolerance,
        # which decays with N
        old_d = jt_isim_from_sum(old_ls, old_n)
        tol = max(self.tolerance * (np.exp(-self.decay * old_n) - self.offset), 0.0)
        return new_d >= old_d - tol

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.tolerance})"


class ToleranceMerge(MergeAcceptFunction):
    name = "tolerance"

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
        new_jt = jt_isim_from_sum(new_ls, new_n)
        if new_jt < threshold:
            return False
        if old_n == 1 or nom_n != 1:
            return True
        # 'new_jt >= threshold' and 'new_n == old_n + 1' are guaranteed here
        old_jt = jt_isim_from_sum(old_ls, old_n)
        return (new_jt * new_n - old_jt * (old_n - 1)) / 2 >= old_jt - self.tolerance

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.tolerance})"


def get_merge_accept_fn(
    merge_criterion: str, tolerance: float = 0.05
) -> MergeAcceptFunction:
    if merge_criterion == "radius":
        return RadiusMerge()
    elif merge_criterion == "diameter":
        return DiameterMerge()
    elif merge_criterion == "tolerance":
        return ToleranceMerge(tolerance)
    elif merge_criterion in "tolerance-adapt":
        return ToleranceDiameterAdaptiveMerge(tolerance)
    raise ValueError(
        f"Unknown merge criterion {merge_criterion} "
        "Valid criteria are: radius|diameter|tolerance|tolerance-adapt"
    )
