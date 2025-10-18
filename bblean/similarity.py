"""Optimized molecular similarity calculators"""

import os
import warnings

from numpy.typing import NDArray
import numpy as np

# NOTE: The most expensive calculation is *jt_sim_packed*, followed by _popcount_2d,
# centroid_from_sum, packing and unpacking
# TODO: Packing and unpacking *should be done in C++ using a lookup table*
__all__ = [
    # JT sim between two (sets of) fingerprints, and average tanimoto (using iSIM)
    "jt_isim_from_sum",
    "jt_isim",
    "jt_sim_packed",
    "jt_most_dissimilar_packed",
    # Radius and diameter from sum
    "jt_isim_radius_from_sum",
    "jt_isim_radius_compl_from_sum",
    "jt_isim_diameter_from_sum",
    # Radius and diameter from fps (packed and unpacked)
    "jt_isim_radius",
    "jt_isim_radius_compl",
    "jt_isim_diameter",
    # Centroid and medoid
    # Radius and diameter unpacked / packed
    "centroid_from_sum",
    "jt_isim_medoid",
    # Complementary similarity
    "jt_compl_isim",
]

# jt_isim_packed and jt_isim_unpacked are not exposed, only used within functions for
# speed
from bblean._py_similarity import centroid_from_sum, jt_compl_isim, jt_isim_medoid

if os.getenv("BITBIRCH_NO_EXTENSIONS"):
    from bblean._py_similarity import (
        jt_isim_from_sum,
        jt_isim_unpacked,
        jt_isim_packed,
        jt_sim_packed,
        jt_most_dissimilar_packed,
    )
else:
    try:
        from bblean._cpp_similarity import (  # type: ignore
            jt_isim_from_sum,
            jt_sim_packed,
            jt_isim_unpacked_u8,
            jt_isim_packed_u8,
            jt_most_dissimilar_packed,
            unpack_fingerprints,
        )

        # Wrap these two since doing
        def jt_isim_unpacked(arr: NDArray[np.integer]) -> float:
            # Wrapping like this is slightly faster than letting pybind11 autocast
            if arr.dtype == np.uint64:
                return jt_isim_from_sum(
                    np.sum(arr, axis=0, dtype=np.uint64), len(arr)  # type: ignore
                )
            return jt_isim_unpacked_u8(arr)

        # Probably a mypy bug
        def jt_isim_packed(  # type: ignore
            arr: NDArray[np.integer], n_features: int | None = None
        ) -> float:
            # Wrapping like this is slightly faster than letting pybind11 autocast
            if arr.dtype == np.uint64:
                return jt_isim_from_sum(
                    np.sum(
                        unpack_fingerprints(arr, n_features),  # type: ignore
                        axis=0,
                        dtype=np.uint64,
                    ),
                    len(arr),
                )
            return jt_isim_packed_u8(arr)

    except ImportError:
        from bblean._py_similarity import (  # type: ignore
            jt_isim_from_sum,
            jt_isim_unpacked,
            jt_isim_packed,
            jt_sim_packed,
            jt_most_dissimilar_packed,
        )

        warnings.warn(
            "C++ optimized similarity calculations not available,"
            " falling back to python implementation"
        )


def jt_isim(
    fps: NDArray[np.integer],
    input_is_packed: bool = True,
    n_features: int | None = None,
) -> float:
    r"""Average Tanimoto, using iSIM

    iSIM Tanimoto was first propsed in:
    https://pubs.rsc.org/en/content/articlelanding/2024/dd/d4dd00041b

    :math:`iSIM_{JT}(X)` is an excellent :math:`O(N)` approximation of the average
    Tanimoto similarity of a set of fingerprints.

    Also equivalent to the complement of the Tanimoto diameter
    :math:`iSIM_{JT}(X) = 1 - D_{JT}(X)`.

    Parameters
    ----------
    arr : np.ndarray
        2D fingerprint array

    input_is_packed : bool
        Whether the input array has packed fingerprints

    n_features: int | None
        Number of features when unpacking fingerprints. Only required if
        not a multiple of 8

    Returns
    ----------
    isim : float
           iSIM Jaccard-Tanimoto value
    """
    if input_is_packed:
        return jt_isim_packed(fps, n_features)
    return jt_isim_unpacked(fps)


def jt_isim_diameter(
    arr: NDArray[np.integer],
    input_is_packed: bool = True,
    n_features: int | None = None,
) -> float:
    r"""Calculate the Tanimoto diameter of a set of fingerprints"""
    return jt_isim_diameter_from_sum(
        np.sum(
            unpack_fingerprints(arr, n_features) if input_is_packed else arr,
            axis=0,
            dtype=np.uint64,
        ),  # type: ignore
        len(arr),
    )


def jt_isim_radius(
    arr: NDArray[np.integer],
    input_is_packed: bool = True,
    n_features: int | None = None,
) -> float:
    r"""Calculate the Tanimoto radius of a set of fingerprints"""
    return jt_isim_radius_from_sum(
        np.sum(
            unpack_fingerprints(arr, n_features) if input_is_packed else arr,
            axis=0,
            dtype=np.uint64,
        ),  # type: ignore
        len(arr),
    )


def jt_isim_radius_compl(
    arr: NDArray[np.integer],
    input_is_packed: bool = True,
    n_features: int | None = None,
) -> float:
    r"""Calculate the complement of the Tanimoto radius of a set of fingerprints"""
    return jt_isim_radius_compl_from_sum(
        np.sum(
            unpack_fingerprints(arr, n_features) if input_is_packed else arr,
            axis=0,
            dtype=np.uint64,
        ),  # type: ignore
        len(arr),
    )


def jt_isim_radius_compl_from_sum(ls: NDArray[np.integer], n: int) -> float:
    r"""Calculate the complement of the Tanimoto radius of a set of fingerprints"""
    #  Calculates 1 - R = Rc
    # NOTE: Use uint64 sum since jt_isim_from_sum casts to uint64 internally
    # This prevents multiple casts
    new_unpacked_centroid = centroid_from_sum(ls, n, pack=False)
    new_ls_1 = np.add(ls, new_unpacked_centroid, dtype=np.uint64)
    new_n_1 = n + 1
    new_jt = jt_isim_from_sum(ls, n)
    new_jt_1 = jt_isim_from_sum(new_ls_1, new_n_1)
    return (new_jt_1 * new_n_1 - new_jt * (n - 1)) / 2


def jt_isim_radius_from_sum(ls: NDArray[np.integer], n: int) -> float:
    r"""Calculate the Tanimoto radius of a set of fingerprints"""
    return 1 - jt_isim_radius_compl_from_sum(ls, n)


def jt_isim_diameter_from_sum(ls: NDArray[np.integer], n: int) -> float:
    r"""Calculate the Tanimoto diameter of a set of fingerprints.

    Equivalent to ``1 - jt_isim_from_sum(ls, n)``"""
    return 1 - jt_isim_from_sum(ls, n)
