r"""Fallback python implementation of molecular similarity calculators"""

import warnings

from numpy.typing import NDArray
import numpy as np

from bblean.utils import min_safe_uint
from bblean.fingerprints import unpack_fingerprints, pack_fingerprints


def centroid_from_sum(
    linear_sum: NDArray[np.integer], n_samples: int, *, pack: bool = True
) -> NDArray[np.uint8]:
    r"""Calculates the majority vote centroid from a sum of fingerprint values

    The majority vote centroid is an good approximation of the Tanimoto centroid.

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


def centroid(
    fps: NDArray[np.uint8],
    input_is_packed: bool = True,
    n_features: int | None = None,
    *,
    pack: bool = True,
) -> NDArray[np.uint8]:
    r"""Calculates the majority vote centroid from a set of fingerprints

    The majority vote centroid is an good approximation of the Tanimoto centroid.
    """
    if input_is_packed:
        fps = unpack_fingerprints(fps, n_features)
    return centroid_from_sum(
        np.sum(fps, axis=0, dtype=np.uint64),  # type: ignore
        len(fps),
        pack=pack,
    )


def jt_compl_isim(
    fps: NDArray[np.uint8], input_is_packed: bool = True, n_features: int | None = None
) -> NDArray[np.float64]:
    """Get all complementary (Tanimoto) similarities of a set of fps, using iSIM"""
    if input_is_packed:
        fps = unpack_fingerprints(fps, n_features)
    # Vectorized calculation of all compl isim
    # For all compl isim N is N_total - 1
    n_objects = len(fps) - 1
    if n_objects < 2:
        msg = "Invalid fps. len(fps) must be >= 3"
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        return np.full(len(fps), fill_value=np.nan, dtype=np.float64)
    linear_sum = np.sum(fps, axis=0)
    comp_sims = [jt_isim_from_sum(linear_sum - fp, n_objects) for fp in fps]
    return np.array(comp_sims, dtype=np.float64)


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
    packed_centroid = centroid_from_sum(linear_sum, n_samples, pack=True)

    cardinalities = _popcount(Y)

    # Get similarity of each fp to the centroid, and the least similar fp idx (fp_1)
    sims_cent = _jt_sim_packed_precalc_cardinalities(Y, packed_centroid, cardinalities)
    fp_1 = np.argmin(sims_cent)

    # Get similarity of each fp to fp_1, and the least similar fp idx (fp_2)
    sims_fp_1 = _jt_sim_packed_precalc_cardinalities(Y, Y[fp_1], cardinalities)
    fp_2 = np.argmin(sims_fp_1)

    # Get similarity of each fp to fp_2
    sims_fp_2 = _jt_sim_packed_precalc_cardinalities(Y, Y[fp_2], cardinalities)
    return fp_1, fp_2, sims_fp_1, sims_fp_2


def _jt_sim_arr_vec_packed(
    x: NDArray[np.uint8],
    y: NDArray[np.uint8],
) -> NDArray[np.float64]:
    r"""Tanimoto similarity between packed fingerprints

    Either both inputs are vectors of shape (F,) (Numpy scalar is returned), or one is
    an vector (F,) and the other an array of shape (N, F) (Numpy array of shape (N,) is
    returned).
    """
    if x.ndim != 2 or y.ndim != 1:
        raise ValueError("Expected a 2D array and a 1D vector as inputs")
    return _jt_sim_packed_precalc_cardinalities(x, y, _popcount(x))


def _jt_sim_packed_precalc_cardinalities(
    x: NDArray[np.uint8],
    y: NDArray[np.uint8],
    cardinalities: NDArray[np.integer],
) -> NDArray[np.float64]:
    # _cardinalities must be the result of calling _popcount(arr)

    # Maximum value in the denominator sum is the 2 * n_features (which is typically
    # uint16, but we use uint32 for safety)
    intersection = _popcount(np.bitwise_and(x, y))

    # Return value requires an out-of-place operation since it casts uints to f64
    #
    # There may be NaN in the similarity array if the both the cardinality
    # and the vector are just zeros, in which case the intersection is 0 -> 0 / 0
    #
    # In these cases the fps are equal so the similarity *should be 1*, so we
    # clamp the denominator, which is A | B (zero only if A & B is zero too).
    return intersection / np.maximum(cardinalities + _popcount(y) - intersection, 1)


def jt_isim_unpacked(arr: NDArray[np.integer]) -> float:
    # cast is slower
    return jt_isim_from_sum(
        np.sum(arr, axis=0, dtype=np.uint64), len(arr)  # type: ignore
    )


def jt_isim_packed(fps: NDArray[np.integer], n_features: int | None = None) -> float:
    # cast is slower
    return jt_isim_from_sum(
        np.sum(
            unpack_fingerprints(fps, n_features),  # type: ignore
            axis=0,
            dtype=np.uint64,
        ),
        len(fps),
    )


def jt_isim_from_sum(linear_sum: NDArray[np.integer], n_objects: int) -> float:
    r"""iSIM Tanimoto, from sum of rows of a fingerprint array and number of rows

    iSIM Tanimoto was first propsed in:
    https://pubs.rsc.org/en/content/articlelanding/2024/dd/d4dd00041b

    :math:`iSIM_{JT}(X)` is an excellent :math:`O(N)` approximation of the average
    Tanimoto similarity of a set of fingerprints.

    Also equivalent to the complement of the Tanimoto diameter
    :math:`iSIM_{JT}(X) = 1 - D_{JT}(X)`.

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
            stacklevel=2,
        )
        return np.nan

    x = linear_sum.astype(np.uint64, copy=False)
    sum_kq = np.sum(x)
    # isim of fingerprints that are all zeros should be 1 (they are all equal)
    if sum_kq == 0:
        return 1
    sum_kqsq = np.dot(x, x)  # *dot* conserves dtype
    a = (sum_kqsq - sum_kq) / 2  # 'a' is scalar f64
    return a / (a + n_objects * sum_kq - sum_kqsq)
