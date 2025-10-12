.. _bblean-parameters:

.. currentmodule:: bblean

Tuning BitBIRCH parameters
==========================

BitBIRCH has a few parameters parameters that can be adjusted to modify the quality of
the resulting clustering.

Merge criterion and tolerance
-----------------------------

The ``merge_criterion`` is used to determine whether two clusters can be merged inside a
node in the BitBIRCH tree. The criteria may be asymetric (they consider differently
clusters already in the tree, *old clusters*, and clusters that are being inserted, or
*nominee clusters*). There are three main merge criteria implemented:

- **radius** (symmetric): The radius of the *resulting cluster*
    must be less or equal than the ``threshold`` value.
- **diameter** (symmetric): The diameter (equivalently, the average similarity) of the
    *resulting cluster* must be less or equal than the ``threshold`` value.
- **tolerance-diameter** (asymmetric): The *diameter* criteria must be satisfied **and**
    the diameter of the *resulting cluster* must be larger or equal to that of the *old
    cluster* (unless the *old cluster* has a single fingerprint). Some slack can be
    provided with a value of ``tolerance``.
- **tolerance-radius** (asymmetric): The *radius* criterion must be satisfied **and**
    the radius of the *resulting cluster* must be larger or equal to that of the *old
    cluster* (unless the *old cluster* has a single fingerprint). Some slack can be
    provided with a value of ``tolerance``.

Both **tolerance-diameter** and **tolerance-radius** reduce the ``tolerance`` slack
exponentially as the cluster gets larger. This behaviour is usually desirable, but can
be turned off with ``adaptive=False``.

Currently we recommend the *diameter* criteria for the initial build of the tree, and
the corresponding *tolerance-diameter* criteria for refinement and tree-combining. The
default slack value for tolerance (0.05) is good for most purposes, although you may
want no slack (tolerance=0) if it is important to maintain the average Tanimoto values
after refinement. Using a very large value for tolerance will flatten the diameter
distribution of the clusters.

Threshold
---------

The ``threshold`` determines the minimum metric acceptable within a given cluster.
If adding a new molecule to a cluster would result in a lower average similarity than
``threshold``, BitBIRCH will instead create a new cluster. High threshold values may
result in *many small, compact clusters*. Low threshold may result in *few large,
diffuse clusters*.

The clustering results for a given threshold value will depend **on the kind of
fingerprint used**. Sparse fingerprints (e. g. ECFPs) typically have lower pairwise
Jaccard-Tanimoto similarities, which means you will want a low threshold to recover
meaningful structure. Denser fingerprints (e. g. the default `rdkit`
fingerprints) require larger threshold.

A typical recommendation is to use a threshold in the range of *0.2-0.35* for ECFP4 or
ECFP6, and a threshold in the range of **0.5-0.65** for `rdkit` fingerprints. Within
these ranges the method is not very sensitive to the threshold value chosen, but
choosing the wrong range for a given fingerprint kind may be **very disadvantageous**.

Branching factor
----------------

The ``branching_factor`` determines how many clusters each node of the BitBIRCH tree
can hold before splitting into new nodes. A high branching factor will result in fewer
nodes, which means tree insertions will better approximate a thorough search over the
full fingerprint set, and memory usage will be lower. However, a very high branching
factor, may also incurr in a higher computational cost.

A recommended branching factor that performs well in terms of memory use and compute
cost is 254. Higher branching factors may be useful to reduce memory usage when
clustering hundreds of millions of molecules, at the cost of some speed (for example you
may want 1000 for 100M-200M molecules).

The clustering results depend on the ``branching_factor``, but only very weakly. Most of
the effect is limited to performance and memory usage.
