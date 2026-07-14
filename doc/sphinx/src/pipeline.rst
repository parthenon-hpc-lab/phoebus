.. _singularity-eos: https://lanl.github.io/singularity-eos

The CCSNe Pipeline
==================

.. note::

   These docs (and this pipeline!) are under active development. If you encounter a bug or add a new feature, let the developers know!

If you're interested in exploring the core-collapse supernovae (e.g. ``progenitor`` ) problem in ``Phoebus``, 
you'll need to take a stellar progenitor from a Lagrangian code (e.g. MESA, KEPLER, GR1D) and convert it to
a ``Phoebus`` -readable input file.

This Python pipeline offers a variety of options to make that conversion as simple as possible. 


How do I use this pipeline?
----------------------------

Like ``Phoebus``, this pipeline works through a master input file. You can run the entire process using the following command:

.. code-block:: bash

    python3 ccsne.py --file {input file name} -v

Within the input file, there are a variety of flags and options that can be used to process your stellar progenitor.

.. list-table:: Parameters
   :header-rows: 1
   :widths: 18 22 10 50

   * - Parameter
     - Default
     - Optional?
     - Description
   * - ``--model-path``
     - ``../lawhite/mesa/15m_at_cc.dat``
     - 
     - Absolute or relative path to input model/data.
   * - ``--model-name``
     - ``aag21``
     - 
     - Desired name/id for your model.
   * - ``--model-type``
     - ``MESA``
     - 
     - Type of model. Options: ``MESA``, ``GR1D``, ``KEPLER``.
   * - ``--model-header``
     - ``-1``
     - x
     - Number of header lines in model file. If < 0, goes to defaults (MESA: 4, KEPLER: 1).
   * - ``--atbounce``
     - 
     - x
     - If using a GR1D model, pulls data files at bounce (true/false).
   * - ``--timestamp``
     - ``0.3``
     - x
     - If using a GR1D model and ``atbounce = false``, gets data from timestep (in seconds).
   * - ``--wma-vel``
     - 
     - x
     - Enables weighted mean averaging for velocity (true/false).
   * - ``--wma-all``
     - 
     - x
     - Enables weighted mean averaging for all primitives in profile (true/false).
   * - ``--wma-n``
     - ``100``
     - x
     - Window for weighted mean average.
   * - ``--wma-exp``
     - ``3``
     - x
     - Exponent to raise wma weights to.
   * - ``--factor-interp``
     - ``10``
     - x
     - Factor to increase profile radial grid resolution by.
   * - ``--use-drad-interp``
     - 
     - x
     - Enables radial spacing criteria for interpolation instead of factor. Recommended for lagrangian (mass coordinate) input models (true/false).
   * - ``--drad-interp``
     - ``1e7``
     - x
     - If using radial spacing for interpolation, sets minimum spacing requirement in cm.
   * - ``--use-uniform``
     - 
     - x
     - Determines if the higher resolution profile uses a uniform grid (true/false).
   * - ``--unif-max``
     - ``5.0e9``
     - x
     - If above disabled, sets outer boundary of inner uniform grid (uniform > log).
   * - ``--adm-problem``
     - ``StellarTable``
     - x
     - Type of ADM interpolation to use. Default: ``StellarCollapse``. Options: ``StellarCollapse``, ``tov``, ``homologous``.
   * - ``--eos-path``
     - ``../grsolver/SFHo.h5``
     - 
     - Absolute or relative path to eos table (must be ``*.h5`` file).
   * - ``--eos-type``
     - ``StellarCollapse``
     - x
     - Type of tabulated eos. Default: ``StellarCollapse``. Options: ``StellarCollapse``, ``Helmholtz``.
   * - ``--use-rhocut``
     - 
     - x
     - Enables a density-only cut for the new grid (true/false).
   * - ``--rhocut``
     - ``2.0e3``
     - x
     - If using rhocut, the density to truncate at in g/cm^3.
   * - ``--use-radcut``
     - 
     - x
     - Enables a radius-only cut for the new grid; keeps original inner radius (true/false).
   * - ``--radcut``
     - ``1.0e9``
     - x
     - If using radcut, the radius to truncate at in cm.
   * - ``--use-custom``
     - 
     - x
     - Lets user input custom grid (radial) in cm (true/false).
   * - ``--custom-min``
     - ``5.0e5``
     - x
     - Inner radial limit for custom grid in cm.
   * - ``--custom-max``
     - ``5.0e9``
     - x
     - Outer radial limit for custom grid in cm.
   * - ``--use-def-rad``
     - 
     - x
     - Uses the default input radial grid of the progenitor (best used for GR1D models at or after bounce) (true/false).
   * - ``--zones``
     - ``2048``
     - x
     - Desired number of grid zones for new profile. Recommended to be 2^x zones for phoebus AMR.
   * - ``--interp-method``
     - ``cubic``
     - x
     - Method of interpolation for ADM solver, non-ADM primitives. Options: ``linear``, ``cubic``, ``akima``, ``makima``.
   * - ``--bc-type``
     - ``not-a-knot``
     - x
     - Boundary conditions if using cubic interpolation. Options: ``clamped``, ``not-a-knot``, ``periodic``.
   * - ``--interp-method-adm``
     - ``piecewise``
     - x
     - Method of interpolation for ADM solver, ADM quantities. Options: ``linear``, ``piecewise``.
   * - ``--iterations``
     - ``100``
     - x
     - Number of iterations the ADM solver will try when converging.
   * - ``--dalpha-eps``
     - ``1.0e-12``
     - x
     - TODO:: docs????
   * - ``--extrapolate``
     - 
     - x
     - Extrapolate the output grid to r = 0 (true/false).
   * - ``--do-fixup``
     - 
     - x
     - Enables a FLASH-style fixup to radius (face --> cell centered) and first (inner) zone velocity (true/false).
   * - ``--save-path``
     - ``test-profiles``
     - 
     - Absolute or relative path to desired directory to save all output.
   * - ``--save-info``
     - 
     - x
     - Saves info file for cgs -> phb unit conversions and input (true/false).
   * - ``--save-unconverted``
     - 
     - x
     - Saves output for phoebus before conversion to phb units (true/false).
   * - ``--save-input``
     - 
     - x
     - Saves input parameter file for future use (true/false).
   * - ``--save-raw``
     - 
     - x
     - Saves the processed, pre ADM solver progenitor (true/false).
   * - ``--depr``
     - 
     - x
     - FOR TESTING ONLY. Uses original grsolver pipeline (true/false).

The full parameter documentation can also be found in ``phoebus/scripts/python/ccsne/params.in``, which can be copied and used as a starting template.


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

.. automodule:: ccsne.progenitor
    :caption: Progenitor Processing
    :members:
