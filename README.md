# BitBIRCH-Lean

## Overview

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
```

## Basic usage

<img src="bb-demo.gif" width="600" />

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
