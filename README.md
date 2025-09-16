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

From source, using a conda environment:

```bash
conda env create --file ./environment.yaml
conda activate bblean
pip install -e .
bb --help
```

<img src="bb-demo.gif" width="600" />

## Quick start

1. **Generate fingerprints from SMILES.** The repository ships with a ChEMBL
   sample that you can use right away:

   ```bash
   bb fps-from-smiles examples/chembl_33_10K.smi -o output
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
