<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/mqcomplab/bblean/main/docs/src/_static/logo-dark-bw.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/mqcomplab/bblean/main/docs/src/_static/logo-light-bw.svg">
  <img alt="BitBIRCH-Lean logo" src="https://raw.githubusercontent.com/mqcomplab/bblean/main/docs/src/_static/logo-light-bw.svg">
</picture>
<br>
<br>

[![DOI](https://zenodo.org/badge/1051268662.svg)](https://doi.org/10.5281/zenodo.17139445)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![CI](https://github.com/mqcomplab/bblean/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/mqcomplab/bblean/actions/workflows/ci.yaml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![Code coverage](https://img.shields.io/endpoint?url=https%3A%2F%2Fmqcomplab.github.io%2Fbblean%2Fcoverage%2Fcoverage-badge.json) 

## Overview

BitBIRCH-Lean is a high-throughput implementation of the BitBIRCH clustering
algorithm designed for very large molecular libraries.

If you find this software useful please cite the following articles:

- *BitBIRCH: efficient clustering of large molecular libraries*:
    https://doi.org/10.1039/D5DD00030K
- *BitBIRCH Clustering Refinement Strategies*:
    https://doi.org/10.1021/acs.jcim.5c00627
- *BitBIRCH-Lean*:
    (preprint) https://www.biorxiv.org/content/10.1101/2025.10.22.684015v1

**NOTE**: BitBirch-Lean is currently beta software, expect minor breaking changes until
we hit version 1.0

The [documentation](https://mqcomplab.github.io/bblean/devdocs) of the developer version is a work in progress. Please let us know if you find any issues.

⚠️  **Important**: The default `threshold` is 0.3 and the default fingerprint kind to
*ecfp4*. We recommend setting `threshold` to 0.5-0.65 for *rdkit* fingerprints and
0.3-0.4 for *ecfp4* or *ecfp6* fingerprints (although you may need further tuning for
your specific library / fingerprint set). For more information on tuning these
parameters see [the best
practices](https://mqcomplab.github.io/bblean/devdocs/user-guide/notebooks/bitbirch_best_practices.html)
and [parameter
tuning](https://mqcomplab.github.io/bblean/devdocs/user-guide/parameters.html) guides.

## Installation

BitBIRCH-Lean requires Python 3.11 or higher, and can be installed in Windows, Linux or
macOS via pip, which automatically includes C++ extensions:

```bash
pip install bblean
# Alternatively you can use 'uv pip install'
bb --help
```

We recommend installing `bblean` in a conda environment or a `venv`.

Memory usage and C++ extensions are most optimized for Linux / macOS. We support windows
on a best-effort basis, some releases may not have Windows support.

### From source

To build from source instead (editable mode):

```bash
git clone git@github.com:mqcomplab/bblean,
cd bblean

conda env create --file ./environment.yaml
conda activate bblean

BITBIRCH_BUILD_CPP=1 pip install -e .

# If you want to build without the C++ extensions run this instead:
pip install -e .

bb --help
```

If the extensions install successfully, they will be automatically used each time
BitBirch-Lean or its classes are used. No need to do anything else.

If you run into any issues when installing the extensions, please open a GitHub issue
and tag it with `C++`.

## CLI Quickstart

<div align="center">
<img src="bblean-demo-v2.gif" width="600" />
</div>

BitBIRCH-Lean provides a convenient CLI interface, `bb`. The CLI can be used to convert
SMILES files into compact fingerprint arrays, and cluster them in parallel or serial
mode with a single command, making it straightforward to triage collections with
millions of molecules. The CLI prints a run banner with the parameters used, memory
usage (when available), and elapsed timings so you can track each job at a glance.

The most important commands you need are:

- `bb fps-from-smiles`: Generate fingerprints from a `*.smi` file.
- `bb run` or `bb multiround`: Cluster the fingerprints
- `bb plot-summary` or `bb plot-tsne`: Analyze the clusters

An example usual workflow is as follows:

1. **Generate fingerprints from SMILES**: The repository ships with a ChEMBL
   sample that you can use right away for testing:

   ```bash
   bb fps-from-smiles examples/chembl-33-natural-products-sample.smi
   ```

   This writes a packed fingerprint array to the current working directory (use
   `--out-dir <dir>` for a different location). The naming convention is
   `packed-fps-uint8-508e53ef.npy`, where `508e53ef` is a unique identifier (use `--name
   <name>` if you prefer a different name). The packed `uint8` format is required for
   maximum memory-efficient, so keep the default
   `--pack` and `--dtype` values unless you have a very good reason to change them.
   You can optionally split over multiple files for parallel parallel processing with `--num-parts <num>`.

3. **Cluster the fingerprints**: To cluster in serial mode, point `bb run` at the
   generated array (or a directory with multiple `*.npy` files):

   ```bash
   bb run ./packed-fps-uint8-508e53ef.npy
   ```

   The outputs are stored in directory such as `bb_run_outputs/504e40ef/`, where
   `504e40ef` is a unique identifier (use `--out-dir <dir>` for a different location).
   Additional flags can be set to control the BitBIRCH `--branching`, `--threshold`,
   and merge criterion. Optionally, cluster refinement can be performed with `--refine-num 1`.
   `bb run --help ` for details.

   To cluster in parallel mode, use `bb multiround ./file-or-dir` instead. If pointed to
   a directory with multiple `*.npy` files, files will be clustered in parallel and
   sub-trees will be merged iteratively in intermediate rounds. For more information:
   `bb multiround --help`. Outputs are written by default to
   `bb_multiround_outputs/<unique-id>/`.

4. **Visualize the results**: You can plot a summary of the largest clusters with
   `bb plot-summary <output-path> --top 20` (largest 20 clusters). Passing the optional `--smiles <path-to-file.smi>` argument
   additionally generates Murcko scaffold analysis. For a t-SNE
   visualization try `bb plot-tsne <output-path> -- top 20`.
   t-SNE plots use [openTSNE](https://opentsne.readthedocs.io/en/latest/) as a backend,
   which is a parallel, extremely fast implementation. We recommend you consult the corresponding
   documentation for info on the available parameters.
   Still, expect t-SNE plots to be slow for very large datasets (more than 1M molecules).

### Manually exploring clustering results

Every run directory contains a raw `clusters.pkl` file with the molecule indices for each
cluster, plus metadata in `*.json` files that captures the exact settings and
performance characteristics. A quick Python session is all you need to get started:

```python
import pickle

clusters = pickle.load(open("bb_run_outputs/504e40ef/clusters.pkl", "rb"))
clusters[:2]
# [[321, 323, 326, 328, 337, ..., 9988, 9989],
#  [5914, 5915, 5916, 5917, 5918, ..., 9990, 9991, 9992, 9993]]
```

The indices refer to the position of each molecule in the order they were read from the
fingerprint files, making it easy to link back to your original SMILES records.

## Python Quickstart and Examples

For an example of how to use the main `bblean` classes and functions consult
`examples/bitbirch_quickstart.ipynb`. The `examples/dataset_splitting.ipynb` notebook
contains an adapted notebook by Pat Walters ([Some Thoughts on Splitting Chemical
Datasets](https://practicalcheminformatics.blogspot.com/2024/11/some-thoughts-on-splitting-chemical.html)).
More examples will be added soon!

A quick summary:

```python
import pickle

import matplotlib.pyplot as plt
import numpy as np

import bblean
import bblean.plotting as plotting
import bblean.analysis as analysis

# Create the fingerprints and pack them into a numpy array, starting from a *.smi file
smiles = bblean.load_smiles("./examples/chembl-33-natural-products-sample.smi")
fps = bblean.fps_from_smiles(smiles, pack=True, n_features=2048, kind="rdkit")

# Fit the figerprints (by default all bblean functions take *packed* fingerprints)
# A threhsold of 0.5-0.65 is good for rdkit fingerprints, a threshold of 0.3-0.4
# is better for ECFPs
tree = bblean.BitBirch(branching_factor=50, threshold=0.65, merge_criterion="diameter")
tree.fit(fps)

# Refine the tree (if needed)
tree.set_merge("tolerance-diameter", tolerance=0.0)
tree.refine_inplace(fps)

# Visualize the results
clusters = tree.get_cluster_mol_ids()
ca = analysis.cluster_analysis(clusters, fps, smiles)
plotting.summary_plot(ca, title="ChEMBL Sample")
plt.show()

# Save the resulting clusters, metrics, and fps
with open("./clusters.pkl", "wb") as f:
    pickle.dump(clusters, f)
ca.dump_metrics("./metrics.csv")
np.save("./fps-packed-2048.npy", fps)
```

## Public Python API and Documentation

By default all functions take *packed* fingerprints of dtype `uint8`. Many functions
support an `input_is_packed: bool` flag, which you can toggle to `False` in case for
some reason you want to pass unpacked fingerprints (not recommended).

- Functions and classes that *end in an underscore* are considered private (such as
  `_private_function(...)`) and should not be used, since they can be removed or
  modified without warning.
- All functions and classes that are in *modules that end with an underscore* are also
  considered private (such as `bblean._private_module.private_function(...)`) and should
  not be used, since they can be removed or modified without warning.
- All other functions and classes are part of the stable public API and can be used.
    However, expect minor breaking changes before we hit version 1.0

## Contributing

If you find a bug in BitBIRCH-Lean or have an issue with the usage
or documentation please open an issue in the GitHub issue tracker.

If you want to contribute to BitBIRCH-Lean with a bug fix, improving the documentation,
with usability, maintainability, or performance, please open an issue with your
idea/request (or directly open a PR from a fork if you prefer).

Currently we don't directly accept PRs with new features that have not been extensively
validated, but if you have an idea to improve the BitBIRCH algorithm you may want to
contact the Miranda-Quintana Lab, we are open to collaborations.

To contribute, first create a
[fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo),
then clone your fork (`git clone git@github.com:<user>/bblean`. We recommend you install
`pre-commit` (`pre-commit install --hook-type pre-push`), which will run some checks
before you push to your branch. After you have finished work on your branch, [open a
pull
request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request).
