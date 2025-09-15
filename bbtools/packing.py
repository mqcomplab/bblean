import numpy as np
from numpy.typing import NDArray


# Requires numpy >= 2.0
def popcount(a: NDArray[np.uint8]) -> NDArray[np.uint32]:
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


def pack_fingerprints(a: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Packs binary uint8 arrays (only 0s and 1s) to uint8 arrays"""
    # packbits may pad with zeros if n_features is not a multiple of 8
    return np.packbits(a, axis=-1)


def unpack_fingerprints(a: NDArray[np.uint8], n_features: int) -> NDArray[np.uint8]:
    """Unpacks packed uint8 arrays into binary uint8 arrays (with only 0s and 1s)"""
    # n_features is required to discard padded zeros if it is not a multiple of 8
    return np.unpackbits(a, axis=-1, count=n_features)


def jt_sim_packed(
    arr: NDArray[np.uint8],
    vec: NDArray[np.uint8],
    cardinalities: NDArray[np.integer] | None = None,
) -> NDArray[np.float64]:
    r"""Tanimoto similarity between a matrix of packed fingerprints and a single packed
    fingerprint.

    If "cardinalities" is passed, it must be the result of calling popcount(arr).
    """
    # Maximum value in the denominator sum is the 2 * n_features (which is typically
    # uint16, but we use uint32 for safety)
    intersection = popcount(np.bitwise_and(arr, vec))
    if cardinalities is None:
        cardinalities = popcount(arr)
    # Return value requires an out-of-place operation since it casts uints to f64
    #
    # There may be NaN in the similarity array if the both the cardinality
    # and the vector are just zeros, in which case the intersection is 0 -> 0 / 0
    #
    # In these cases the fps are equal so the similarity *should be 1*, so we
    # clamp the denominator, which is A | B (zero only if A & B is zero too).
    return intersection / np.maximum(cardinalities + popcount(vec) - intersection, 1)
