"""
Functions to apply a simply minimization to the data.
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import minimize


def deproject(params, velax, spectra, angles):
    """Return the average deprojected spectrum."""
    vrot, PA = params
    zipped = zip(spectra, angles)
    deproj = [np.interp(velax, velax + vrot * np.cos(t + PA), s)
              for s, t in zipped]
    return np.average(deproj, axis=0)


def estimate_vrot(velax, spectra):
    """Estimate the rotation velocity for the spectra."""
    centers = np.take(velax, np.argmax(spectra, axis=1))
    vmin, vmax = centers.min(), centers.max()
    vlsr = np.average([vmin, vmax])
    return np.average([vlsr - vmin, vmax - vlsr])


def estimate_rms(velax, spectrum):
    """Estimate the RMS of the spectrum."""
    Tb = spectrum.max()
    x0 = velax[spectrum.argmax()]
    dV = np.trapz(spectrum, velax) / Tb / np.sqrt(2. * np.pi)
    p0 = [x0, dV, Tb]
    x0, dV, _ = curve_fit(gaussian, velax, spectrum, maxfev=10000, p0=p0)[0]
    return np.nanstd(spectrum[abs(velax - x0) > 3. * dV])


def gaussian(x, x0, dx, A):
    """Gaussian function with Doppler width."""
    return A * np.exp(-np.power((x-x0)/dx, 2))


def fit_width(velax, spectrum):
    """Return the width of the spectrum."""
    Tb = spectrum.max()
    x0 = velax[spectrum.argmax()]
    dV = np.trapz(spectrum, velax) / Tb / np.sqrt(2. * np.pi)
    p0 = [x0, dV, Tb]
    try:
        popt, _ = curve_fit(gaussian, velax, spectrum, maxfev=10000, p0=p0)
        return popt[1]
    except:
        return 1e20


def deprojected_width(params, velax, spectra, angles):
    """Return the width of the deprojected values."""
    return fit_width(velax, deproject(params, velax, spectra, angles))


def minimize_wrapper(rpnts, velax, ridxs, pixels, tpix):
    """Return the minimized {vrot, dPA} profiles."""
    p0, rms, deprojected = [], [], []
    for idx, radius in enumerate(rpnts):
        spectra = pixels[ridxs == idx + 1]
        angles = tpix[ridxs == idx + 1]
        vrot = estimate_vrot(velax, spectra)
        res = minimize(deprojected_width, (vrot, 0.0),
                       args=(velax, spectra, angles), method='Nelder-Mead')
        spectrum = deproject(res.x, velax, spectra, angles)
        p0 += [res.x]
        rms += [estimate_rms(velax, spectrum)]
        deprojected += [spectrum]
    p0, rms = np.squeeze(p0), np.squeeze(rms)
    deprojected = np.squeeze(deprojected)
    return p0.T, rms, deprojected
