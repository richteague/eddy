# eddy - Extracting Disk DYnamics

[![status](http://joss.theoj.org/papers/2868c5ad4b6405eba1aaf1cd8ea53274/status.svg)](http://joss.theoj.org/papers/2868c5ad4b6405eba1aaf1cd8ea53274)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1440051.svg)](https://doi.org/10.5281/zenodo.1440051)
<a href="http://ascl.net/1901.010"><img src="https://img.shields.io/badge/ascl-1901.010-blue.svg?colorB=262255" alt="ascl:1901.010" /></a>
[![Documentation Status](https://readthedocs.org/projects/eddy/badge/?version=latest)](https://eddy.readthedocs.io/en/latest/?badge=latest)

`eddy` is a suite of Python tools to recover precise velocity profiles of protoplanetary disks from Doppler shifted line emission. `eddy` makes fitting of first moment maps and the inference of a rotation velocity from an annulus of spectra a breeze.

## Installation

The most simple method is with `pip`,

```
pip install astro-eddy
```

The only real dependencies for this are `numpy`, `scipy`, `matplotlib`, and [`emcee`](https://github.com/dfm/emcee), If you want to run the Gaussian Process method you will also need [`celerite`](https://github.com/dfm/celerite) which can be easily installed if you follow their [installation guide](https://celerite.readthedocs.io/en/stable/python/install/).

If things have installed correctly you should be able to run the [Jupyter Notebooks](https://github.com/richteague/eddy/tree/master/docs/tutorials) with no errors. If something goes wrong, please [open an issue](https://github.com/richteague/eddy/issues/new).

## Useage

For guides on how to use `eddy` you will find extensive examples in the [documents](https://github.com/richteague/eddy/tree/master/docs). We shamelessly recommend [bettermoments](https://github.com/richteague/bettermoments) to make the moment maps required for the fitting.

## Citations

If you use `eddy` as part of your research, please cite the [JOSS article](http://joss.theoj.org/papers/10.21105/joss.01220):

```latex
@article{eddy,
    doi = {10.21105/joss.01220},
    url = {https://doi.org/10.21105/joss.01220},
    year = {2019},
    month = {feb},
    publisher = {The Open Journal},
    volume = {4},
    number = {34},
    pages = {1220},
    author = {Richard Teague},
    title = {eddy},
    journal = {The Journal of Open Source Software}
}
```

A full list of citations including dependencies can be found on the [citations](./docs/citations.md) page.

## Works using `eddy`

A list of the works using (or at least citing) `eddy` can be found on [ADS](https://ui.adsabs.harvard.edu/abs/2019JOSS....4.1220T/citations).
