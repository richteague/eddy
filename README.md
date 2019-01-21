# eddy - Extracting Disk DYnamics

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1440052.svg)](https://doi.org/10.5281/zenodo.1440052)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/fc14f5eca31c418388424d4ec38a604b)](https://www.codacy.com/app/richteague/eddy?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=richteague/eddy&amp;utm_campaign=Badge_Grade)

Python tools to recover precise rotation profiles of protoplanetary disks from Doppler shifted line emission. `eddy` makes fitting of first moment maps and the inference of a rotation velocity from an annulus of spectra a breeze.

### Installation

To install the `eddy` packge, first clone the directory, then in the main directory,

```
pip install --user .
```

Direct installation from `pip` is work in progress.

### Useage

For guides on how to use `eddy` you will find extensive examples in the [docs](https://github.com/richteague/eddy/tree/master/docs).

In brief, `eddy.fit_annulus` contains the functionality to infer the rotation profile from an annulus of Doppler shifted spectra, as discussed in [Teague et al. (2018a,](https://ui.adsabs.harvard.edu/#abs/2018ApJ...860L..12T/abstract)[b)](https://ui.adsabs.harvard.edu/#abs/2018ApJ...868..113T/abstract). Functions to fit the rotation map (we shameless recommend [bettermoments](https://github.com/richteague/bettermoments) to make these), including a flared surface geometry can be found in `eddy.fit_cube`.

### Attribution

Coming soon.
