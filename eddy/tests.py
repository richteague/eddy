from __future__ import division, print_function

import pytest
import numpy as np
from modelling import gaussian_ensemble
from modelling import flared_disk_ensemble


@pytest.fixture
def gaussian_annulus(N=10):
    """Returns annulus with Gaussian lines."""
    return gaussian_ensemble(vrot=1500., N=N, return_ensemble=True)


@pytest.fixture
def flared_annulus(N=10):
    """Returns annulus with optically thick lines."""
    return flared_disk_ensemble(N=N, return_ensemble=True)


def test_number_of_spectra():
    for N in [2, 5, 10]:
        assert gaussian_annulus(N).spectra.shape[0] == N
        assert flared_annulus(N).spectra.shape[0] == N


def test_channel_width():
    annulus = gaussian_annulus()
    assert annulus.channel == 30.
    assert np.diff(annulus.velax).mean() == 30.
    annulus = flared_annulus()
    assert annulus.channel == 30.
    assert np.diff(annulus.velax).mean() == 30.


def test_number_of_position_angles():
    annulus = gaussian_annulus()
    assert annulus.spectra.shape[0] == annulus.theta.size
    annulus = flared_annulus()
    assert annulus.spectra.shape[0] == annulus.theta.size


def test_no_empty_spectra():
    annulus = gaussian_annulus()
    assert int(np.any(np.all(annulus.spectra == 0, axis=1))) == 0
    annulus = flared_annulus()
    assert int(np.any(np.all(annulus.spectra == 0, axis=1))) == 0
