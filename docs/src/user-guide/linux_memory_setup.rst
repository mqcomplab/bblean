.. _linux-memory-setup:

Linux Memory Setup
==================

On MacOS and Windows memory compression is enabled by default. However on Linux, memory
compression (e.g. ``zram`` and ``zswap``) is not enabled. Usually this is not an issue,
but if you are running a system with limited RAM, enabling these features can help
improve performance by compressing memory pages. This document will provide a
step-by-step guide to enable ``zram`` and ``zswap`` on a Linux system.

``Zram`` setup 
--------------
This quick start guide follows the ``zram`` article on the ArchLinux wiki. See the `article <https://wiki.archlinux.org/title/Zram>`_ for more information. 

#. Step 1: Install Required Packages
    You can install ``zram`` using your package manager. For example, on Debian-based systems, you can run: 

    .. code-block:: bash

        sudo apt install zram-tools

#. Step 2: Set up swap area
    To create a swap device for ``zram``, you can use the following command:

    .. code-block:: bash

        sudo mkswap /dev/zram0

#. Step 3: Configure zram settings 
    Open ``/etc/sysctl.d/99-zram.conf`` with nano:

    .. code-block:: bash

        sudo nano /etc/sysctl.d/99-zram.conf

    Add the following lines to the file:

    .. code-block:: bash

        vm.swappiness = 180
        vm.watermark_boost_factor = 0
        vm.watermark_scale_factor = 125
        vm.page-cluster = 0

    These are the recommended settings according to the ArchLinux Wiki. Feel free to adjust any other ``zram`` settings as needed.

#. Step 4: Activate ``zram``
    You can enable ``zram`` by running the following command:

    .. code-block:: bash

        sudo swapon --discard --priority 100 /dev/zram0

    It is important that the priority for ``zram`` is the highest of all swapfiles. This way, Linux will first use ``zram`` and then use other swapfiles.

#. Step 5: Verify ``zram`` is active
    To check if ``zram`` is enabled and working correctly, you can run:

    .. code-block:: bash

        swapon

    You should see an entry for ``zram0`` in the output, indicating that ``zram`` is active.

    Example output:

    .. code-block:: bash

        NAME       TYPE       SIZE USED PRIO
        /swapfile  file         2G   0B   -2
        /dev/zram0 partition   16G   0B  100

    A higher numeric value for priority (PRIO) means that Linux will use the ``zram0`` swap before the ``swapfile``.  

#. Step 6: Disabling ``zram``
    If you need to disable ``zram``, you can stop the service and disable it:

    .. code-block:: bash

        sudo swapoff /dev/zram0

``Zswap`` setup
---------------

Zswap is usually preinstalled with Debian distributions. To check whether ``zswap`` is enabled, run:

.. code-block:: bash

    cat /sys/module/zswap/parameters/enabled

This will display ``Y`` if zswap is enabled and ``N`` if it is disabled. 
The next section shows how to enable ``zswap`` and change its parameters for the current session.

To enable zswap, run:

.. code-block:: bash

    echo 1 | sudo tee /sys/module/zswap/parameters/enabled

Changing ``zswap`` parameters is similar to enabling ``zswap`` For example, you can change the algorithm by:

.. code-block:: bash

    echo zstd | sudo tee /sys/module/zswap/parameters/compressor

and the maximum pool percent by:

.. code-block:: bash

    echo 30 | sudo tee /sys/module/zswap/parameters/max_pool_percent

The max pool percent is the amount of memory zswap is allowed to use for memory compression. 
