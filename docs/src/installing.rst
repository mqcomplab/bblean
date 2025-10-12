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

BitBIRCH-Lean has optional C++ extensions. These have been currently tested on Linux x86
only. You should expect a speedup of ~1.8-2.0x on Linux. To install the extensions from
source run the following command:

.. code-block:: bash

    BITBIRCH_BUILD_CPP=1 pip install -e .

If the extensions install successfully, they will be automatically used each time
BitBIRCH-Lean or its classes are used. No need to do anything else.

If you run into any issues when installing the extensions, please open a GitHub issue in
the `issue Tracker <https://github.com/mqcomplab/bblean/issues>`_ and tag it with the
``C++`` label.

Setting up memory management for Linux
======================================

Memory compression is not enabled by default on Linux. If you are running a system with
limited RAM, enabling memory compression can help improve performance by compressing
memory pages. See :ref:`linux-memory-setup` for instructions on how to enable memory
compression.
