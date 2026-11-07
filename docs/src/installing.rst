.. _bblean-installing:

.. currentmodule:: bblean

Installing
==========

BitBIRCH-Lean requires Python 3.11 or newer. To install from source in editable mode using a conda environment:

.. code-block:: bash

    conda env create --file ./environment.yaml
    conda activate bblean

    pip install -e .

    bb --help

BitBIRCH-Lean has optional C++ extensions. These have been thoroughly tested on Linux x86,
and partially on Linux and macOS aarch64.
You should expect a speedup of up to 4x vs plain python on Linux x86, depending on your CPU.
The extensions are automatically built and installed, with flags tuning for the native
CPU arch, when installing from source. To build a more portable binary, with
conservative compilation flags (expect less speedup in this case) set
``BITBIRCH_BUILD_PORTABLE=1`` before installing. To disable the extensions set
``BITBIRCH_PY_ONLY=1``.

If the extensions install successfully, they will be automatically used each time
BitBIRCH-Lean or its classes are used. No need to do anything else.

If you run into any issues with the extensions, please open a GitHub issue in
the `issue Tracker <https://github.com/mqcomplab/bblean/issues>`_ and tag it with the
``C++`` label.

Setting up memory management for Linux
======================================

Memory compression is not enabled by default on Linux. If you are running a system with
limited RAM, enabling memory compression can help improve performance by compressing
memory pages. See :ref:`linux-memory-setup` for instructions on how to enable memory
compression.
