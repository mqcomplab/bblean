.. _bblean-installing:

.. currentmodule:: bblean

Installing
==========

BitBIRCH-Lean requires Python 3.11 or newer. To install from source in editable mode using a conda environment:

.. code-block:: bash

    conda env create --file ./environment.yaml
    conda activate bblean

    BITBIRCH_BUILD_NATIVE=1 pip install -e .

    bb --help

BitBIRCH-Lean has optional C++ extensions. These have been thoroughly tested on Linux x86,
and partially on Linux aarch64 and macOS arm64.
You should expect a speedup of up to 4x vs plain python on Linux x86, depending on your CPU.
The extensions are automatically built and installed when installing from source. By
default they are built with conservative, portable compilation flags.
``BITBIRCH_BUILD_NATIVE=1`` (recommended, as in the command above) tunes them for
the native CPU arch instead, which gives a larger speedup, but the resulting binary may
not run on other machines with older CPUs. To disable the extensions set
``BITBIRCH_PY_ONLY=1``.
On Windows PowerShell, set the variables in a separate command before running ``pip``,
for example:

.. code-block:: powershell

    $env:BITBIRCH_BUILD_NATIVe=1
    pip install -e .

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
