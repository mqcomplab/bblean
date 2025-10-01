r"""Optimized molecular similarity calculators"""

import os
import warnings

from numpy.typing import NDArray
import numpy as np

# NOTE: The most expensive calculation is *jt_sim_packed*, followed by _popcount_2d,
# calc_centroid, packing and unpacking
# TODO: Packing and unpacking *should be done in C++ using a lookup table*
__all__ = ["jt_isim_from_sum", "jt_sim_packed", "jt_most_dissimilar_packed"]


if os.getenv("BITBIRCH_NO_EXTENSIONS"):
    from bblean._py_similarity import (
        jt_isim_from_sum,
        jt_sim_packed,
        jt_most_dissimilar_packed,
    )
else:
    try:
        from bblean._cpp_similarity import (  # type: ignore
            jt_isim_from_sum,
            jt_sim_packed,
            jt_most_dissimilar_packed,
        )
    except ImportError:
        from bblean._py_similarity import (  # type: ignore
            jt_isim_from_sum,
            jt_sim_packed,
            jt_most_dissimilar_packed,
        )

        warnings.warn(
            "C++ optimized similarity calculations not available,"
            " falling back to python implementation"
        )


def jt_isim(c_total: NDArray[np.integer], n_objects: int) -> float:
    warnings.warn(
        "Please use jt_isim_from_sum(...) instead", DeprecationWarning, stacklevel=2
    )
    return jt_isim_from_sum(c_total, n_objects)
