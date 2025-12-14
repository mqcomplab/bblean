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
    "centroid",
    "jt_isim_medoid",
    # Complementary similarity
    "jt_compl_isim",
    "jt_stratified_sampling",
    "jt_sim_matrix_packed",
]

from bblean._py_similarity import centroid_from_sum, centroid
from bblean.fingerprints import pack_fingerprints, unpack_fingerprints

# jt_isim_packed and jt_isim_unpacked are not exposed, only used within functions for
# speed

if os.getenv("BITBIRCH_NO_EXTENSIONS"):
    from bblean._py_similarity import (
        jt_isim_from_sum,
        jt_isim_unpacked,
        jt_isim_packed,
        jt_compl_isim,
        _jt_sim_arr_vec_packed,
        jt_most_dissimilar_packed,
    )
else:
    try:
        from bblean._cpp_similarity import (  # type: ignore
            jt_isim_from_sum,
            jt_isim_unpacked_u8,
            jt_isim_packed_u8,
            jt_compl_isim,  # TODO: Does it need wrappers for non-uint8?
            _jt_sim_arr_vec_packed,
            jt_most_dissimilar_packed,
            # Needed for wrappers
            unpack_fingerprints as _unpack_fingerprints,
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
                        _unpack_fingerprints(arr, n_features),  # type: ignore
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
            jt_compl_isim,
            _jt_sim_arr_vec_packed,
            jt_most_dissimilar_packed,
        )

        warnings.warn(
            "C++ optimized similarity calculations not available,"
            " falling back to python implementation"
        )


def jt_isim_medoid(
    fps: NDArray[np.uint8],
    input_is_packed: bool = True,
    n_features: int | None = None,
    pack: bool = True,
) -> tuple[int, NDArray[np.uint8]]:
    r"""Calculate the (Tanimoto) medoid of a set of fingerprints, using iSIM

    Returns both the index of the medoid in the input array and the medoid itself

    .. note::
        Returns the first (or only) fingerprint for array of size 2 and 1 respectively.
        Raises ValueError for arrays of size 0

    """
    if not fps.size:
        raise ValueError("Size of fingerprints set must be > 0")
    if input_is_packed:
        fps = unpack_fingerprints(fps, n_features)
    if len(fps) < 3:
        idx = 0  # Medoid undefined for sets of 3 or more fingerprints
    else:
        idx = np.argmin(jt_compl_isim(fps, input_is_packed, n_features)).item()
    m = fps[idx]
    if pack:
        return idx, pack_fingerprints(m)
    return idx, m


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
            (
                unpack_fingerprints(arr.astype(np.uint8, copy=False), n_features)
                if input_is_packed
                else arr
            ),
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
            (
                unpack_fingerprints(arr.astype(np.uint8, copy=False), n_features)
                if input_is_packed
                else arr
            ),
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
            (
                unpack_fingerprints(arr.astype(np.uint8, copy=False), n_features)
                if input_is_packed
                else arr
            ),
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


# General wrapper that works both in C++ and python
def jt_sim_packed(
    x: NDArray[np.uint8],
    y: NDArray[np.uint8],
) -> NDArray[np.float64]:
    r"""Tanimoto similarity between packed fingerprints

    Either both inputs are vectors of shape (F,) (Numpy scalar is returned), or
    one is an vector (F,) and the other an array of shape (N, F) (Numpy array of
    shape (N,) is returned).
    """
    if x.ndim == 1 and y.ndim == 1:
        return _jt_sim_arr_vec_packed(x.reshape(1, -1), y)[0]
    if x.ndim == 2:
        return _jt_sim_arr_vec_packed(x, y)
    if y.ndim == 2:
        return _jt_sim_arr_vec_packed(y, x)
    raise ValueError(
        "Expected either two 1D vectors, or one 1D vector and one 2D array"
    )


def jt_sim_matrix_packed(arr: NDArray[np.uint8]) -> NDArray[np.float64]:
    r"""Tanimoto similarity matrix between all pairs of packed fps in arr"""
    matrix = np.ones((len(arr), len(arr)), dtype=np.float64)
    for i in range(len(arr)):
        # Set the similarities for each row
        matrix[i, i + 1 :] = jt_sim_packed(arr[i], arr[i + 1 :])
        # Set the similarities for each column (symmetric)
        matrix[i + 1 :, i] = matrix[i, i + 1 :]
    return matrix


def estimate_jt_std(
    fps: NDArray[np.uint8],
    n_samples: int | None = None,
    input_is_packed: bool = True,
    n_features: int | None = None,
    min_samples: int = 1_000_000,
) -> float:
    r"""Estimate the std of all pairwise Tanimoto.

    Returns
    -------
    std : float
        The standard deviation of all pairwise Tanimoto among the sampled fingerprints.
    """
    num_fps = len(fps)
    if num_fps > min_samples:
        np.random.seed(42)
        random_choices = np.random.choice(num_fps, size=min_samples, replace=False)
        fps = fps[random_choices]
        num_fps = len(fps)
    if n_samples is None:
        # Heuristic: use at least 50 samples, or 1 per 10,000 fingerprints,
        # to balance statistical representativeness and computational efficiency
        n_samples = max(num_fps // 10_000, 50)
    sample_idxs = jt_stratified_sampling(fps, n_samples, input_is_packed, n_features)

    # Work with only the sampled fingerprints
    fps = fps[sample_idxs]
    num_fps = len(fps)
    pairs = np.empty(num_fps * (num_fps - 1) // 2, dtype=np.float64)
    # NOTE: Calc upper triangular part of pairwise matrix only, slightly more efficient,
    # but difference is negligible in tests
    offset = 0
    for i in range(len(fps)):
        num = num_fps - i - 1
        pairs[offset : offset + num] = jt_sim_packed(fps[i], fps[i + 1 :])
        offset += num
    return np.std(pairs).item()


def jt_stratified_sampling(
    fps: NDArray[np.uint8],
    n_samples: int,
    input_is_packed: bool = True,
    n_features: int | None = None,
) -> NDArray[np.int64]:
    r"""Sample from a set of fingerprints according to their complementary similarity

    Given a group of fingerprints, calculate all complementary similarities, order, and
    sample the first element from consecutive groups of length ``num_fps // n_samples +
    1``.

    ..  note ::

        This is not true statistical stratified sampling, it is not random, and the
        strata are not homogeneous. It is meant as a reliable, deterministic method to
        obtain a representative sample from a set of fingerprints.
    """
    # Stratified sampling without replacement
    if n_samples == 0:
        return np.array([], dtype=np.int64)
    if n_samples > len(fps):
        raise ValueError("n_samples must be <= len(fps)")
    # Get the indices that would sort the complementary similarities
    sorted_indices = np.argsort(jt_compl_isim(fps, input_is_packed, n_features))
    # Split into n_samples strata
    strata = np.array_split(sorted_indices, n_samples)
    # Get first index of each strata
    return np.array([s[0] for s in strata])
