# eddy - Extracting Disk DYnamics

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/fc14f5eca31c418388424d4ec38a604b)](https://www.codacy.com/app/richteague/eddy?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=richteague/eddy&amp;utm_campaign=Badge_Grade)

Python tools to recover precise rotation profiles of (predominantly azimuthally symmetric) disks. As the the rotation of the disk Doppler shifts the lines different amounts as a function of azimuth, shifting them back to a common systemic velocity allows us to sample the intrinsic line profile at a much higher effective resolution than the observations.

## Usage

To start, install the `eddy` package

```
pip install .
```

In brief, `eddy` contains an `ensemble` class which takes an array of spectra extracted from an annulus of constant radius, the polar coordinates of where the spectra were taken from (where `theta = 0` is the red-shifted major axis), and the velocity axis.

```python
from eddy.eddy import ensemble
annulus = ensemble(spectra=spectra, theta=theta, velax=velax)
```

One then has the option to use the `dV` method, where we minimize the line width of the stacked line profile,

```python
vrot = annulus.get_vrot_dV()
```

or the `GP` method which relaxes the assumption about an intrinsic Gaussian line profile and aims to find the smoothest line profile.

```python
vrot = annulus.get_vrot_GP()
```

The former will not return uncertainties (only the value which minimizes the line width), while the `GP` approach will. A more thorough description of the functions available and how they work can be found in the Jupyter Notebooks in the Examples folder.

## Model Spectra

There are also functions to build model spectra to test these methods in the `modelling.py`, returning the spectra, the polar angles and the velocity axis for a given set of parameters.

```python
from eddy.modelling import gaussian_ensemble
spectra, theta, velax = gaussian_ensemble(vrot=1500., Tb=35., dV=350., dV_chan=50., rms=3.)
```

More example can be found in the example notebooks.
