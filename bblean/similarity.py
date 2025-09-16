r"""Optimized molecular similarity calculators"""

import warnings

from numpy.typing import NDArray
import numpy as np

from bblean.utils import _popcount


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
