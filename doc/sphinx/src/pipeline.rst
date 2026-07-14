.. _singularity-eos: https://lanl.github.io/singularity-eos

The CCSNe Pipeline
==================

.. note::

   These docs are under active development. If you encounter a bug or add a new feature, let the developers know!

If you're interested in exploring the core-collapse supernovae (e.g. ``progenitor``) problem in ``Phoebus``, 
you'll need to take a stellar progenitor from a Lagrangian code (e.g. MESA, KEPLER, GR1D) and convert it to
a ``Phoebus``-readable input file.

This Python pipeline offers a variety of options to make that conversion as simple as possible. 


How do I use this pipeline?
----------------------------

Like ``Phoebus``, this pipeline works through a master input file. You can run the entire process using the following command:

.. code-block:: bash

    python3 ccsne.py --file {input file name} -v

Within the input file, there are a variety of flags and options that can be used to process your stellar progenitor.

.. add table here later....

The full parameter documentation can also be found in ``params.in``, which can be copied and used as a starting template.


Dependencies
````````````

For this pipeline to run, you'll need the following libraries:
- ``numpy``
- ``astropy``
- ``scipy``
- ``pandas``

You'll also need an installation of `singularity-eos`_ that's been compiled with the Python bindings enabled. Something similar to 
the following (adjusted for your HPC environment) should suffice.

.. code-block:: bash

    git clone --recursive git@github.com:lanl/singularity-eos.git
    cd singularity-eos

    # load python and hdf5/phdf5 here!

    cmake -B builddir -S . \
    -DSINGULARITY_FORCE_SUBMODULE_MODE=ON \
    -DSINGULARITY_USE_FORTRAN=OFF \
    -DSINGULARITY_USE_STELLAR_COLLAPSE=ON \
    -DSINGULARITY_BUILD_PYTHON=ON \
    -DSINGULARITY_USE_SPINER=ON

    cmake --build builddir --parallel
    mkdir install # << or install directory of your choice
    cmake --install builddir --prefix install


Documentation
-------------

This entire pipeline can be used as a "black-box" process. However, there are many functions that may be helpful for detailed analysis, 
processing, progenitor I/O, troubleshooting, and more. Detailed documentation for all classes, methods, and functions can be found below.

.. toctree::

   :glob:
    .. check to see that this works...

   ../../../scripts/python/ccsne/*