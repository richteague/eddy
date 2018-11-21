"""
Collapse a FITS image cube down to a velocity map using one of several methods.
This can be run from the command line using the default parameters, but also
within a Python script to have access to more of the fine tuning of the
functions.
"""

import argparse
import numpy as np
import scipy.constants as sc


from astropy.io import fits


def _get_cube(path):
    """Return the data and velocity axis from the cube."""
    return _get_data(path), _get_velax(path)


def _get_data(path):
    """Read the FITS cube. Returning in [Jy/beam]."""
    return np.squeeze(fits.getdata(path))


def _get_velax(path):
    """Read the velocity axis information. Returning in [m/s]."""
    return _read_velocity_axis(fits.getheader(path))


def _read_rest_frequency(header):
    """Read the rest frequency in [Hz]."""
    try:
        nu = header['restfreq']
    except KeyError:
        try:
            nu = header['restfrq']
        except KeyError:
            nu = header['crval3']
    return nu


def _read_velocity_axis(header):
    """Wrapper for _velocityaxis and _spectralaxis."""
    if 'freq' in header['ctype3'].lower():
        specax = _read_spectral_axis(header)
        nu = _read_rest_frequency(header)
        velax = (nu - specax) * sc.c / nu
    else:
        velax = _read_spectral_axis(header)
    return velax


def _read_spectral_axis(header):
    """Returns the spectral axis in [Hz] or [m/s]."""
    specax = (np.arange(header['naxis3']) - header['crpix3'] + 1.0)
    return header['crval3'] + specax * header['cdelt3']


def _save_array(original_path, new_path, array):
    """Use the header from `original_path` to save a new FITS file."""
    header = fits.getheader(original_path)

    # Remove pesky values.
    ['naxis3', 'cdelt3', 'crval3', 'crpix3', 'history']
    for key in ['history']:
        try:
            header.pop(key, None)
        except KeyError:
            pass

    # Update the spectral axis.
    header['naxis3'] = 1
    header['cdelt3'] = np.nanstd(array)
    header['crval3'] = np.nanmean(array)
    header['crpix3'] = 1e0

    fits.writeto(new_path, array, header, overwrite=True)


def collapse_quadratic(velax, data, dV=None, rms=None, N=5):
    """
    Collapse the cube using the quadratic method from Teague & Foreman-Mackey
    (2018) (https://github.com/richteague/bettermoments).

    Args:
        velax (ndarray): -
        data (ndarray): -
        linewidth (Optional[float]): -
        rms (Optional[float]): -
        N (Optional[int]): -

    Returns:
        v0 (ndarray): -
        dV0 (ndarray): -
        Fnu (ndarray): -
        dFnu (ndarray): -
    """

    try:
        from bettermoments import quadratic
    except ImportError:
        raise ImportError("Cannot find `bettermoments`.")

    # Estimate the uncertainty.
    if rms is None:
        rms = np.nanstd([data[:int(N)], data[-int(N):]])

    # Convert linewidth units.
    chan = np.diff(velax).mean()
    if dV is not None:
        dV /= chan

    return quadratic(data, x0=velax[0], dx=chan, uncertainty=rms, linewidth=dV)


if __name__ == '__main__':

    # Parse the methods.
    parser = argparse.ArgumentParser()
    parser.add_argument('path',
                        help='Path to the FITS cube.')
    parser.add_argument('-method', default='quadratic',
                        help='Method used to collapse cube.')
    parser.add_argument('-clip', default=5.0,
                        help='Mask values below this SNR.')
    parser.add_argument('-fill', default=np.nan,
                        help='Fill value for masked pixels.')
    parser.add_argument('--silent', action='store_true',
                        help='Run silently.')
    args = parser.parse_args()

    # Read in the cube [Jy/beam] and velocity axis [m/s].
    data, velax = _get_cube(args.path)

    # Collapse the cube.
    if args.method.lower() == 'quadratic':

        # Calculate values.
        if not args.silent:
            print("Calculating maps...")
        v0, dv0, Fnu, dFnu = collapse_quadratic(velax, data)

        # Mask the data.
        if args.clip is not None:

            mask = abs(1e-10 * np.random.randn(dFnu.size).reshape(dFnu.shape))
            mask = np.where(np.logical_or(dFnu == 0.0, np.isnan(dFnu)),
                            mask, dFnu)
            mask = np.where(np.isfinite(Fnu), Fnu, 0.0) / mask >= args.clip

            v0 = np.where(mask, v0, args.fill)
            dv0 = np.where(mask, dv0, args.fill)
            Fnu = np.where(mask, Fnu, args.fill)
            dFnu = np.where(mask, dFnu, args.fill)

        # Save the values.
        if not args.silent:
            print("Saving maps...")
        _save_array(args.path, args.path.replace('.fits', '_v0.fits'), v0)
        _save_array(args.path, args.path.replace('.fits', '_dv0.fits'), dv0)
        _save_array(args.path, args.path.replace('.fits', '_Fnu.fits'), Fnu)
        _save_array(args.path, args.path.replace('.fits', '_dFnu.fits'), dFnu)

    else:
        raise ValueError("Unknown method.")
