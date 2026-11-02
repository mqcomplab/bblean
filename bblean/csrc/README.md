# C++ extensions for accellerated similarity calculations and clustering

- `kernels.hpp`: popcounts, Tanimoto similarity, iSIM, majority centroids and the
  most dissimilar pair of a set of fingerprints, shared by both extensions.
- `similarity.cpp`: `bblean._cpp_similarity`, used by `bblean.similarity`.
- `bitbirch.cpp`: `bblean._cpp_bitbirch`, the BitBirch tree used by `bblean.BitBirch`
  (same results as the python tree), and the builtin merge criteria.
