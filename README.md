# BitBIRCH-Lean

## Overview

BitBIRCH-Lean is a high-throughput implementation of the BitBIRCH clustering
algorithm designed for very large molecular libraries.  The `bb` CLI converts
SMILES files into compact fingerprint arrays and clusters them with a single
command, making it straightforward to triage collections with millions of
molecules.

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

<img src="bb-demo.gif" width="600" />

## CLI Quick start

1. **Generate fingerprints from SMILES.** The repository ships with a ChEMBL
   sample that you can use right away:

   ```bash
   bb fps-from-smiles examples/chembl-sample.smi -o output
   ```

   This command writes a packed fingerprint array to
   `output/packed-fps-uint8.npy`.  The packed `uint8` format is required by the
   memory-efficient BitBIRCH implementation, so keep the default `--pack` and
   `--dtype` values unless you have a specific reason to change them.

2. **Cluster the fingerprints.** Point `bb run` at the generated array (or a
   directory with multiple `.npy` files):

   ```bash
   bb run output/packed-fps-uint8.npy
   ```

   The run outputs are stored in a timestamped directory such as
   `bb_run_outputs/504e40ef/`.  Use `--output-dir` if you prefer a fixed
   location.

3. **Inspect the log.** The CLI prints a run banner with the parameters used,
   memory usage (when available), and elapsed timings so you can track each job
   at a glance.

## Exploring clustering results

Every run directory contains a `clusters.pkl` file with the molecule indices for
each cluster, plus metadata (`config.json`, `timings.json`, `peak-rss.json`) that
captures the exact settings and performance characteristics.  A quick Python
session is all you need to get started:

```python
import pickle

clusters = pickle.load(open("bb_run_outputs/504e40ef/clusters.pkl", "rb"))
clusters[:2]
# [[321, 323, 326, 328, 337, ..., 9988, 9989],
#  [5914, 5915, 5916, 5917, 5918, ..., 9990, 9991, 9992, 9993]]
```

The indices refer to the position of each molecule in the order they were read
from the fingerprint files, making it easy to link back to your original SMILES
records.

## Helpful CLI commands

- `bb fps-info`: Summaries for one or more packed fingerprint `.npy` files.
- `bb fps-from-smiles`: Convert SMILES files to fingerprint arrays in `uint8`
  format (optionally split across multiple files for very large datasets).
- `bb run`: Serial BitBIRCH clustering over packed fingerprint arrays.
  Additional options let you tune the branching factor, threshold, and
  tolerance; run `bb run --help` for details.

## Python Quickstart

For example of how to use the main `bblean` classes and functions consult
`examples/bitbirch_quickstart.ipynb`. More examples will be added soon!

A quick summary:

```python
import bblean
import pickle
import numpy as np

# Create and pack the fingerprints
smiles = bblean.smiles.load_smiles("./examples/chembl-smiles.smi")
fps = bblean.fingerprints.fps_from_smiles(smiles, pack=True, n_features=2048)

# Fit the figerprints
tree = bblean.BitBirch(branching_factor=50, threshold=0.65, merge_criterion="diameter")
tree.fit(fps, n_features=2048)

# Refine the tree (if needed)
tree.set_merge(threshold=0.70, merge_criterion="tolerance", tolerance=0.05)
tree.refine_inplace(fps)

# Visualize the results
clusters = tree.get_cluster_mol_ids()
ca = bblean.analysis.cluster_analysis(
    clusters, smiles, fps, n_features=2048, input_is_packed=True
)
bblean.plotting.summary_plot(ca, title="ChEMBL Sample")
plt.show()

# Save the results
ca.dump_metrics("./metrics.csv")
np.save("./fps-packed-2048.npy", fps)
with open("./clusters.pkl", "wb") as f:
    pickle.dump(clusters, f)
```

## API and Documentation

Documentation is currently a work in progress, for the time being you can consult
functions and classes `"""docstrings"""` for info on usage, or the Jupyter notebook
examples under `./examples`.

- Functions and classes that *end in an underscore* are considered private (such as
  `_private_function(...)`) and should not be used, since they can be removed or
  modified without warning.
- All functions and classes that are in *modules that end with an underscore* are also
  considered private (such as `bblean._private_module.private_function(...)`) and should
  not be used, since they can be removed or modified without warning.
- All other functions and classes are part of the stable public API and can be safely used.
