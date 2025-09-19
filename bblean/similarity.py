r"""Optimized molecular similarity calculators"""

import warnings
# NOTE: The most expensive calculation is *jt_sim_packed*, followed by _popcount_2d,
# calc_centroid, packing and unpacking
# TODO: Packing and unpacking *should be done in C++ using a lookup table*
__all__ = ["jt_isim", "jt_sim_packed", "jt_most_dissimilar_packed"]


try:
    from bblean._cpp_similarity import (
        jt_isim,
        jt_sim_packed,
        jt_most_dissimilar_packed_also_requiring_unpacked,
    )

    from numpy.typing import NDArray
    import numpy as np

    from bblean.fingerprints import unpack_fingerprints

    def jt_most_dissimilar_packed(
        Y: NDArray[np.uint8], n_features: int | None = None
    ) -> tuple[np.integer, np.integer, NDArray[np.float64], NDArray[np.float64]]:
        # Unpacking is done in python since Numpy's implementation is good enough,
        # and its not worth it to redo it in C++
        return jt_most_dissimilar_packed_also_requiring_unpacked(
            Y, unpack_fingerprints(Y, n_features)
        )

except ImportError:
    from bblean._py_similarity import jt_isim, jt_sim_packed, jt_most_dissimilar_packed

    warnings.warn(
        "C++ optimized similarity calculations not available,"
        " falling back to python implementation"
    )
