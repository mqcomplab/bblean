r"""Optimized molecular similarity calculators"""

import warnings

from numpy.typing import NDArray
import numpy as np

# from bblean.utils import min_safe_uint
# from bblean.fingerprints import unpack_fingerprints, calc_centroid

__all__ = ["jt_isim", "jt_sim_packed", "jt_most_dissimilar_packed"]


def min_safe_uint(nmax: int) -> np.dtype:
    r"""Returns the min uint dtype that holds a (positive) py int, excluding "object".

    Input must be a positive python integer.
    """
    out = np.min_scalar_type(nmax)
    # Check if the dtype is a pointer to a python bigint
    if out.hasobject:
        raise ValueError(f"n_samples: {nmax} is too large to hold in a uint64 array")
    return out


def unpack_fingerprints(
    a: NDArray[np.uint8], n_features: int | None = None
) -> NDArray[np.uint8]:
    r"""Unpack packed uint8 arrays into binary uint8 arrays (with only 0s and 1s)

    .. note::

        If `n_features` is not passed, unpacking will only recover the correct number of
        features if it is a multiple of 8, otherwise fingerprints will be padded with
        zeros to the closest multiple of 8. This is generally not an issue since most
        common fingerprints feature sizes (2048, 1024, etc) are multiples of 8, but if
        you are using a non-standard number of features you should pass `n_features`
        explicitly.
    """
    # n_features is required to discard padded zeros if it is not a multiple of 8
    return np.unpackbits(a, axis=-1, count=n_features)


def calc_centroid(
    linear_sum: NDArray[np.integer], n_samples: int, *, pack: bool = True
) -> NDArray[np.uint8]:
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
    # NOTE: Numpy guarantees bools are stored as 0xFF -> True and 0x00 -> False,
    # so this view is fully safe
    if n_samples <= 1:
        centroid = linear_sum.astype(np.uint8, copy=False)
    else:
        centroid = (linear_sum >= n_samples * 0.5).view(np.uint8)
    if pack:
        return np.packbits(centroid, axis=-1)
    return centroid


# Requires numpy >= 2.0
def _popcount(a: NDArray[np.uint8]) -> NDArray[np.uint32]:
    # a is packed uint8 array with last axis = bytes
    # Sum bit-counts across bytes to get per-object totals

    # If the array has columns that are a multiple of 8, doing a bitwise count
    # over the buffer reinterpreted as uint64 is slightly faster.
    # This is zero cost if the exception is not triggered. Not having a be a multiple of
    # 8 is a very unlikely scenario, since fps are typically 1024 or 2048
    b: NDArray[np.integer]
    try:
        b = a.view(np.uint64)
    except ValueError:
        b = a
    return np.bitwise_count(b).sum(axis=-1, dtype=np.uint32)


# O(N) approximation to obtain "most dissimilar fingerprints" within an array
def jt_most_dissimilar_packed(
    Y: NDArray[np.uint8], n_features: int | None = None
) -> tuple[np.integer, np.integer, NDArray[np.float64], NDArray[np.float64]]:
    """Finds two fps in a packed fp array that are the most Tanimoto-dissimilar

    This is not guaranteed to find the most dissimilar fps, it is
    a robust O(N) approximation that doesn't affect final cluster quality.
    First find centroid of Y, then find fp_1, the most dissimilar molecule
    to the centroid. Finally find fp_2, the most dissimilar molecule to fp_1

    Returns
    -------
    fp_1 : int
        index of the first fingerprint
    fp_2 : int
        index of the second fingerprint
    sims_fp_1 : np.ndarray
        Tanimoto similarities of Y to fp_1
    sims_fp_2: np.ndarray
        Tanimoto similarities of Y to fp_2
    """
    # Get the centroid of the fps
    n_samples = len(Y)
    Y_unpacked = unpack_fingerprints(Y, n_features)
    # np.sum() automatically promotes to uint64 unless forced to a smaller dtype
    linear_sum = np.sum(Y_unpacked, axis=0, dtype=min_safe_uint(n_samples))
    packed_centroid = calc_centroid(linear_sum, n_samples, pack=True)

    cardinalities = _popcount(Y)

    # Get similarity of each fp to the centroid, and the least similar fp idx (fp_1)
    sims_cent = jt_sim_packed(Y, packed_centroid, cardinalities)
    fp_1 = np.argmin(sims_cent)

    # Get similarity of each fp to fp_1, and the least similar fp idx (fp_2)
    sims_fp_1 = jt_sim_packed(Y, Y[fp_1], cardinalities)
    fp_2 = np.argmin(sims_fp_1)

    # Get similarity of each fp to fp_2
    sims_fp_2 = jt_sim_packed(Y, Y[fp_2], cardinalities)
    return fp_1, fp_2, sims_fp_1, sims_fp_2


def jt_sim_packed(
    arr: NDArray[np.uint8],
    vec: NDArray[np.uint8],
    _cardinalities: NDArray[np.integer] | None = None,
) -> NDArray[np.float64]:
    r"""Tanimoto similarity between a matrix of packed fps and a single packed fp"""
    # NOTE: If _cardinalities is passed, it must be the result of calling _popcount(arr)

    # Maximum value in the denominator sum is the 2 * n_features (which is typically
    # uint16, but we use uint32 for safety)
    intersection = _popcount(np.bitwise_and(arr, vec))
    if _cardinalities is None:
        _cardinalities = _popcount(arr)
    # Return value requires an out-of-place operation since it casts uints to f64
    #
    # There may be NaN in the similarity array if the both the cardinality
    # and the vector are just zeros, in which case the intersection is 0 -> 0 / 0
    #
    # In these cases the fps are equal so the similarity *should be 1*, so we
    # clamp the denominator, which is A | B (zero only if A & B is zero too).
    return intersection / np.maximum(_cardinalities + _popcount(vec) - intersection, 1)


def jt_isim(c_total: NDArray[np.integer], n_objects: int) -> float:
    r"""iSIM Tanimoto calculation

    iSIM Tanimoto was first propsed in:
    https://pubs.rsc.org/en/content/articlelanding/2024/dd/d4dd00041b

    Parameters
    ----------
    c_total : np.ndarray
              Sum of the elements from an array of fingerprints X, column-wise
              c_total = np.sum(X, axis=0)

    n_objects : int
                Number of elements
                n_objects = X.shape[0]

    Returns
    ----------
    isim : float
           iSIM Jaccard-Tanimoto value
    """
    if n_objects < 2:
        warnings.warn(
            f"Invalid n_objects = {n_objects} in isim. Expected n_objects >= 2",
            RuntimeWarning,
        )
        return np.nan

    x = c_total.astype(np.uint64, copy=False)
    sum_kq = np.sum(x)
    # isim of fingerprints that are all zeros should be 1 (they are all equal)
    if sum_kq == 0:
        return 1
    sum_kqsq = np.dot(x, x)  # *dot* conserves dtype
    a = (sum_kqsq - sum_kq) / 2  # 'a' is scalar f64
    return a / (a + n_objects * sum_kq - sum_kqsq)
