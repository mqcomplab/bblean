r"""Optimized molecular similarity calculators"""

import warnings

__all__ = ["jt_isim", "jt_sim_packed", "jt_most_dissimilar_packed"]


try:
    # Overwrite definitions with cpp extensions if possible
    from bblean.cpp_similarity import jt_isim, jt_sim_packed, jt_most_dissimilar_packed
except ImportError:
    from bblean._py_similarity import jt_isim, jt_sim_packed, jt_most_dissimilar_packed
    warnings.warn(
        "C++ optimized similarity calculations not available,"
        " falling back to python implementation"
    )
