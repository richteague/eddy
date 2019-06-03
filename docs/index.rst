eddy - Extracting Disk Dynamics
===============================

*eddy* provides a suite of tools to extract kinematical information from
spatially and spectrally resolved line data.

The :class:`fit_cube.rotationmap` class enables the fitting of a Keplerian
rotation pattern to a rotation map to infer geometrical properties of the disk,
a dynamical mass of the star, or to infer the presence of a flared emission
surface or warp in the disk.

The :class:`fit_annulus.annulus` class contains the functionality to infer rotational and radial
velocities based on an annulus of spectra extracted from an azimuthally
symmetric disk.


Installation
------------

Installation is as easy as cloning the git respository then,

.. code-block:: bash

    pip install .

Note that you'll need ``v3.0`` or higher of `emcee`_ in order for this to run
which must be installed from GitHub rather than through pip or conda.


Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Documentation

   user/annulus
   user/rotationmap

.. toctree::
   :maxdepth: 3
   :caption: Tutorials

   tutorials/fitannulus_1
   tutorials/fitannulus_2
   tutorials/fitcube_1
   tutorials/fitcube_2


Support
-------

If you are having issues, please open a `issue`_ on the GitHub page.


License
-------

The project is licensed under the `MIT license`_.


.. _emcee: https://github.com/dfm/emcee
.. _issue: https://github.com/richteague/eddy/issues
.. _MIT license: https://github.com/richteague/eddy/blob/master/LICENSE.md
