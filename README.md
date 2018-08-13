# eddy - Extracting Disk DYnamics

Python tools to recover precise rotation profiles of disks. These methods were initially published in [Teague et al. (2018)](https://ui.adsabs.harvard.edu/#abs/2018ApJ...860L..12T) and Teague et al. (in prep.). The idea is that by using all the spectra in an annulus we can gain a significant advantage in determining the line center and thus the line-of-sight velocity for the line.

## Usage

A more thorough description of the functions available and how they work can be found in the Jupyter Notebooks in the Examples folder. In brief, `eddy` contains an `ensemble` class which takes an array of spectra extracted from an annulus of constant radius, the polar coordinates of where the spectra were taken from (where `theta = 0` is the red-shifted major axis), and the velocity axis.

```python
from eddy.eddy import ensemble
annulus = ensemble(spectra=spectra, theta=theta, velax=velax)
```

One then has the option to use the `dV` method, where we minimize the linewidth of the stacked line profile,

```python
vrot = annulus.get_vrot_dV()
```

or the `GP` method which relaxes the assumption about an intrinsic Gaussian line profile and aims to find the smoothest line profile.

```python
vrot = annulus.get_vrot_GP()
```

The former will not return uncertainties (only the value which minimizes the linewidth), while the `GP` approach will.

## Model Spectra

There are also functions to build model spectra to test these methods in the `modelling` file, returning the spectra, the polar angles and the velocity axis for a given set of parameters.

```python
from eddy.modelling import gaussian_ensemble
spectra, theta, velax = gaussian_ensemble(vrot=1500., Tb=35., dV=350.,
                                          dV_chan=50., rms=3.)
```

## Discussion of Methodology

Also within the Examples folder are two Jupyter Notebooks which explain how the two methods work and provides a description of what the functions are doing under the hood.
