# eddy - Extracting Disk DYnamics

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1440052.svg)](https://doi.org/10.5281/zenodo.1440052)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/fc14f5eca31c418388424d4ec38a604b)](https://www.codacy.com/app/richteague/eddy?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=richteague/eddy&amp;utm_campaign=Badge_Grade)

Python tools to recover precise rotation profiles of (predominantly azimuthally symmetric) disks. As the the rotation of the disk Doppler shifts the lines different amounts as a function of azimuth, shifting them back to a common systemic velocity allows us to sample the intrinsic line profile at a much higher effective resolution than the observations.

## Usage

To start, install the `eddy` package

```
pip install --user .
```

In brief, `eddy.annulus` contains an `ensemble` class which takes an array of spectra extracted from an annulus of constant radius, the polar coordinates of where the spectra were taken from (where `theta = 0` is the red-shifted major axis), and the velocity axis.

```python
from eddy.annulus import ensemble
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

## Idea

For this approach we exploit the fact that most disks are predominantly azimuthally symmetric. This means that if we take an annulus of constant radius (after accounting for the flaring of the disk) all the spectra in the annulus should share the same line peak and width. They all have differing line centres due to the Doppler shift arising from the rotation of the disk. The figure below shows an example of this.

---

<p align='center'>
  <img src="https://github.com/richteague/eddy/blob/master/Examples/Images/first_moment_and_spectra.png" width="713.2" height="383"><br/>
  <b>Figure 1:</b> The left-hand panel shows the rotation pattern for a flared disk, with both near and far sides visible. The right hand panels show spectra taken from an annulus of constant radius. For an azimuthally symmetric disk these should be identical in terms of peak and width, however their centers will be offset due ot the projected component of the rotation.
</p>

---

With an annulus of spectra and some rotation velocity, we can shift all the lines back to a common systemic velocity. Rather than assuming a velocity _a priori_, we can use the data to infer the correct value. `eddy` will search for the rotation velocity which either

* minimizes the width of the stacked profile of shifted lines.

or 

* minimizes the variance in the residuals from a smooth model.

The former method is very fast, but implicity assumes a Gaussian line profile and does not return an uncertainty. The second method is more robust and returns a reasonable measure of the uncertainty, however is slower due to the MCMC needed to sample the posterior distributions.

## Model Spectra

There are also functions to build model spectra to test these methods in the `modelling.py`, returning the spectra, the polar angles and the velocity axis for a given set of parameters.

```python
from eddy.modelling import gaussian_ensemble
spectra, theta, velax = gaussian_ensemble(vrot=1500., Tb=35., dV=350., dV_chan=50., rms=3.)
```

More example can be found in the example notebooks.
