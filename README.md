# eddy - Extracting Disk DYnamics

Python tools to recover precise rotation profiles of disks. These methods were
initially published in Teague et al. (2018) and Teague et al. (in prep.).

## Example Notebooks

Annotated examples of how to use the functions and demonstrations of the
limitations of the method can be found in the `examples` folder. These Jupyter
Notebooks can be downloaded and run locally so to play with the variables.

## Functions

`eddy.py` contains an `ensemble` class which has all the functionality described
in the Python Notebook. This should be relatively easy to use, for example to
use the method by minimizing the width of the stacked line profile:

```python
from eddy import ensemble
annulus = ensemble(spectra=spectra, theta=theta, velax=velax)
vrot = annulus.get_vrot_dV()
```

or for the Gaussian Process approach:

```python
vrot = annulus.get_vrot_GP()
```

There are also functions
