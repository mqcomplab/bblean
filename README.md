# BitBIRCH-Lean

[![DOI](https://zenodo.org/badge/1051268662.svg)](https://doi.org/10.5281/zenodo.17139445)
[![CI](https://github.com/mqcomplab/bblean/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/mqcomplab/bblean/actions/workflows/ci.yaml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Overview

BitBIRCH-Lean is a high-throughput implementation of the BitBIRCH clustering
algorithm designed for very large molecular libraries.

If you find this software useful please cite the following articles:

- *BitBIRCH: efficient clustering of large molecular libraries*:
    https://doi.org/10.1039/D5DD00030K
- *BitBIRCH Clustering Refinement Strategies*:
    https://doi.org/10.1021/acs.jcim.5c00627
- *BitBIRCH-Lean*: TO-BE-ADDED

## Installation

From source, editable mode, using a conda environment:

```bash
conda env create --file ./environment.yaml
conda activate bblean

pip install -e .

bb --help
```

BitBirch-Lean has optional C++ extensions. These have been currently tested
on Linux x86 only. You should expect a speedup of ~1.8-2.0x on Linux. To install the
extensions from source run the following command:

```bash
BITBIRCH_BUILD_CPP=1 pip install -e .
```

If you run into any issues when installing the extensions, please open a GitHub issue
and tag it with `C++`.


<img src="bb-demo.gif" width="600" />

## CLI Quickstart

BitBIRCH-Lean provides a convenient CLI interface, `bb`. The CLI can be used to convert
SMILES files into compact fingerprint arrays, and cluster them in parallel or serial
mode with a single command, making it straightforward to triage collections with
millions of molecules.

1. **Generate fingerprints from SMILES.** The repository ships with a ChEMBL
   sample that you can use right away for testing:

   ```bash
   bb fps-from-smiles examples/chembl-sample.smi
   ```

   This command writes a packed fingerprint array to the current working directory (use
   `--out-dir <dir>` for a different location). The naming convention is
   `packed-fps-uint8-508e53ef.npy`, where `508e53ef` is a unique identifier (use `--name
   <name>` if you prefer a different name). The packed `uint8` format is required for
   maximum memory-efficient in the BitBIRCH Lean implementation, so keep the default
   `--pack` and `--dtype` values unless you have a very good reason to change them.

2. **Cluster the fingerprints.** To cluster in serial mode, point `bb run` at the
   generated array (or a directory with multiple `.npy` files):

   ```bash
   bb run ./packed-fps-uint8-508e53ef.npy
   ```

   The outputs are stored in directory such as `bb_run_outputs/504e40ef/`, where
   `504e40ef` is a unique identifier (use `--out-dir <dir>` for a different location).

   To cluster in parallel mode, use `bb multiround ./file-or-dir` instead. If pointed to
   a directory with multiple `*.npy` files, files will be clustered in parallel and
   sub-trees will be merged iteratively in intermediate rounds. For more information:
   `bb multiround --help`. In this case the outputs are written by default to
   `bb_multiround_outputs/<unique-id>/`. *Note that currently intermediate numpy files
   are saved but please do not rely on this, it may change in the near future.*

3. **Inspect the log.** The CLI prints a run banner with the parameters used, memory
   usage (when available), and elapsed timings so you can track each job at a glance.

## Exploring clustering results

Every run directory contains a `clusters.pkl` file with the molecule indices for each
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

## Helpful CLI commands

- `bb fps-info`: Summaries for one or more packed fingerprint `.npy` files.
- `bb fps-from-smiles`: Convert SMILES files to fingerprint arrays in `uint8`
  format (optionally split across multiple files for very large datasets).
- `bb run`: Serial BitBIRCH clustering over packed fingerprint arrays.
  Additional options let you tune the branching factor, threshold, and
  tolerance; run `bb run --help` for details.

## Python Quickstart

For an example of how to use the main `bblean` classes and functions consult
`examples/bitbirch_quickstart.ipynb`. More examples will be added soon!

A quick summary:

```python
import pickle

import numpy as np

import bblean
import bblean.plotting as plotting
import bblean.analysis as analysis

# Create the fingerprints and pack them into a numpy array, starting from a *.smi file
smiles = bblean.load_smiles("./examples/chembl-smiles.smi")
fps = bblean.fps_from_smiles(smiles, pack=True, n_features=2048)

# Fit the figerprints (by default all bblean functions take *packed* fingerprints)
tree = bblean.BitBirch(branching_factor=50, threshold=0.65, merge_criterion="diameter")
tree.fit(fps)

# Refine the tree (if needed)
tree.set_merge(threshold=0.70, merge_criterion="tolerance", tolerance=0.05)
tree.refine_inplace(fps)

# Visualize the results
clusters = tree.get_cluster_mol_ids()
ca = analysis.cluster_analysis(clusters, smiles, fps)
plotting.summary_plot(ca, title="ChEMBL Sample")
plt.show()

# Save the resulting clusters, metrics, and fps
with open("./clusters.pkl", "wb") as f:
    pickle.dump(clusters, f)
ca.dump_metrics("./metrics.csv")
np.save("./fps-packed-2048.npy", fps)
```

## Python API and Documentation

By default all functions take *packed* fingerprints of dtype `uint8`. Many functions
support an `input_is_packed: bool` flag, which you can toggle to `False` in case for
some reason you want to pass unpacked fingerprints (not recommended).

Documentation is currently a work in progress, for the time being you can consult
functions and classes `"""docstrings"""` for info on usage, or the Jupyter notebook
examples under `./examples` (More to be added soon!).

- Functions and classes that *end in an underscore* are considered private (such as
  `_private_function(...)`) and should not be used, since they can be removed or
  modified without warning.
- All functions and classes that are in *modules that end with an underscore* are also
  considered private (such as `bblean._private_module.private_function(...)`) and should
  not be used, since they can be removed or modified without warning.
- All other functions and classes are part of the stable public API and can be safely used.

## Building C++ extensions

## Contributing

TODO: Add some info about how to contribute to the repo / open issues
