"""
Collapse a FITS image cube down to a velocity map using one of several methods.
This can be run from the command line using the default parameters, but also
within a Python script to have access to more of the fine tuning of the
functions. The available methods are:

'quadradtic' -  Using the bettermoments package from Teague & Foreman-Mackey
                (2018) to return line center and peak values, both with
                uncertainties.

'maximum'    -  The velocity of the peak intensity for each pixel (ninth moment
                if using CASA's immoments). Will return the peak value and the
                channel.

'first'      -  Traditional intensity weighted average velocity. Will only
                return the calculated line center.
"""

import argparse
import numpy as np
import scipy.constants as sc
from astropy.io import fits


def _H3(x):
    """Fourth Hermite polynomial."""
    return (np.sqrt(8.) * x**3 - np.sqrt(18.) * x) / np.sqrt(6.)


def _H4(x):
    """Fifth Hermite polynomial."""
    return (16. * x**4 - 12. * x**2 + 3.) / np.sqrt(24.)


def GH_gaussian(x, x0, dx, A, h3, h4):
    """Gauss-Hermite expansion of a Gaussian."""
    w = (x - x0) / dx
    return A * np.exp(-w**2.) * (1.0 + h3 * _H3(w) + h4 * _H4(w))


def gaussian(x, x0, dx, A):
    """Gaussian function. dx is Doppler width."""
    return A * np.exp(-np.power((x-x0)/dx, 2.0))


def thick_gaussian(x, x0, dx, Tex, tau):
    """Optically thick line profile. dx is Doppler width."""
    return Tex * (1. - np.exp(-gaussian(x, x0, dx, tau)))


def _get_cube(path):
    """Return the data and velocity axis from the cube."""
    return _get_data(path), _get_velax(path)


def _get_data(path):
    """Read the FITS cube."""
    return np.squeeze(fits.getdata(path))


def _get_velax(path):
    """Read the velocity axis information."""
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


def _estimate_RMS(data, N=5):
    """Return the estimated RMS in the first and last N channels."""
    rms = np.nanstd([data[:int(N)], data[-int(N):]])
    return rms * np.ones(data[0].shape)


def _save_array(original_path, new_path, array):
    """Use the header from `original_path` to save a new FITS file."""
    header = fits.getheader(original_path)

    # Remove pesky values.
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
        velax (ndarray): Velocity axis of the cube.
        data (ndarray): Flux density or brightness temperature array. Assumes
            that the first axis is the velocity axis.
        dV (Optional[float]): Doppler width of the line. If specified will be
            used to smooth the data prior to the quadratic fit.
        rms (Optional[float]): Noise per pixel. If none is specified, will be
            calculated from the first and last N channels.
        N (Optional[int]): Number of channels to use in the estimation of the
            noise.

    Returns:
        v0 (ndarray): Line center in the same units as velax.
        dv0 (ndarray): Uncertainty on v0 in the same units as velax.
        Fnu (ndarray): Line peak in the same units as the data.
        dFnu (ndarray): Uncertainty in Fnu in the same units as the data.
    """

    try:
        from bettermoments import quadratic
    except ImportError:
        raise ImportError("Cannot find `bettermoments`.")

    # Check the shape.
    if data.shape[0] != velax.size:
        raise ValueError("Must have the zeroth axis as velocity.")

    # Estimate the uncertainty.
    if rms is None:
        rms = _estimate_RMS(data, N=N)
        rms = np.nanmean(rms)

    # Convert linewidth units.
    chan = np.diff(velax).mean()
    if dV is not None:
        dV /= chan

    return quadratic(data, x0=velax[0], dx=chan, uncertainty=rms, linewidth=dV)


def collapse_maximum(velax, data, return_peak=True, rms=None, N=5):
    """
    Coallapses the cube by taking the velocity of the maximum intensity pixel
    along the spectral axis. This is the 'ninth' moment in CASA's immoments
    task. Can additionally return the peak value ('eighth moment').

    Args:
        velax (ndarray): Velocity axis of the cube.
        data (ndarray): Flux densities or brightness temperature array. Assumes
            that the first axis is the velocity axis.
        return_peak (Optional[bool]): Also return the peak value.
        rms (Optional[float]): Noise per pixel. If none is specified, will be
            calculated from the first and last N channels.
        N (Optional[int]): Number of channels to use in the estimation of the
            noise.

    Returns:
        v0 (ndarray): Line center in the same units as velax.
        dv0 (ndarray): Uncertainty on v0 in the same units as velax. Will be
            the channel width.
        Fnu (ndarray): If return_peak=True. Line peak in the same units as
            the data.
        dFnu (ndarray): If return_peak=True. Uncertainty in Fnu in the same
            units as the data. Will be the RMS calculate from the first and
            last channels.
    """
    v0 = np.take(velax, np.argmax(data, axis=0))
    dv0 = np.ones(v0.shape) * abs(np.diff(velax).mean())
    if not return_peak:
        return v0, dv0, None, None
    Fnu = np.nanmax(data, axis=0)
    if rms is None:
        dFnu = _estimate_RMS(data, N=N)
    else:
        dFnu = rms * np.ones(v0.shape)
    return v0, dv0, Fnu, dFnu


def collapse_first(velax, data, clip=None, rms=None, N=5, mask=None):
    """
    Collapses the cube using the intensity weighted average velocity (or first
    moment map).

    Args:
        velax (ndarray): Velocity axis of the cube.
        data (ndarray): Flux densities or brightness temperature array. Assumes
            that the first axis is the velocity axis.
        clip (Optional[float]): Clip any pixels below this SNR level.
        rms (Optional[float]): Noise per pixel. If none is specified, will be
            calculated from the first and last N channels.
        N (Optional[int]): Number of channels to use in the estimation of the
            noise.
        mask (Optional[ndarray]): A boolean or integeter array masking certain
            pixels to be excluded in the fitting. Can either be a full 3D mask,
            a 2D channel mask, or a 1D spectrum mask.

    Returns:
        v0 (ndarray): Line center in the same units as velax.
    """

    # Calculate the weights.
    weight_noise = 1e-20 * np.random.randn(data.size).reshape(data.shape)
    weights = np.where(np.isfinite(data), data, weight_noise)
    no_signal = np.sum(data, axis=0) > 0.0
    weights = np.where(no_signal[None, :, :], weights, weight_noise)

    # Apply the SNR clipping.
    if clip is not None:
        Fnu, dFnu = collapse_maximum(velax, data, return_peak=True, N=N)[2:]
        if rms is None:
            rms = _estimate_RMS(data, N=N)
        dFnu = rms * np.ones(Fnu.shape)
        snrmask = (Fnu / dFnu >= clip)
        weights = np.where(snrmask, weights, weight_noise)

    # Mask the data.
    if mask is not None:
        if mask.shape == data.shape:
            weights = np.where(mask, weights, weight_noise)
        elif mask.shape == data[0].shape:
            weights = np.where(mask[None, :, :], weights, weight_noise)
        elif mask.shape == velax.shape:
            weights = np.where(mask[:, None, None], weights, weight_noise)
        else:
            raise ValueError("Unknown shape for the mask.")

    # Calculate the intensity weighted average velocity. Apply the mask for
    # regions with no signal in them, or clipped.
    velax_cube = np.ones(data.shape) * velax[:, None, None]
    v0 = np.average(velax_cube, weights=weights, axis=0)
    mask = np.logical_and(no_signal, np.sum(weights) > 1e-10)
    return np.where(mask, v0, np.nan)


def collapse_gaussian(velax, data, clip=None, rms=None, N=5, thick=False):
    """
    Collapse the cube by fitting a Gaussian to the line profile. Make take a
    while to fit for large cubes or if using the optically thick line profile.

    Args:
        velax (ndarray): Velocity axis of the cube.
        data (ndarray): Flux density or brightness temperature array. Assumes
            that the first axis is the velocity axis.
        clip (Optional[flat]): Skips pixels which have a SNR less than this.
        rms (Optional[float]): Noise per pixel. If none is specified, will be
            calculated from the first and last N channels.
        N (Optional[int]): Number of channels to use in the estimation of the
            noise.
        thick (Optional[bool]): Fit an optically thick Gaussian line profile
            rather than just a single Gaussian component.

    Returns:
        popt (ndarray): If thick is False, returns the spatial maps of
            [x0, dV, Tb]. Instead if thick is True, returns the spatial maps of
            [x0, dV, Tex, tau].
        copt (ndarray): The uncertainties, calculated from the diagonal of the
            covariance matrix, for the returned values.
    """

    # Use scipy for the fitting.
    from scipy.optimize import curve_fit

    #  Estimate the starting positions.
    v0, _, Tb, _ = collapse_quadratic(velax, data, rms=rms, N=N)
    dV = np.trapz(data * velax[:, None, None], axis=0)
    dV /= Tb * np.sqrt(np.pi)

    # Calculate the signal to noise ratio of each pixel.
    clip = clip if clip is not None else -100.0
    rms = rms if rms is not None else _estimate_RMS(data, N).mean()
    snr = Tb / rms
    rms *= np.ones(velax.size)

    # Fit the Gaussians, pixel by pixel. NaN as holder for no fit.
    popt = np.zeros((4, data.shape[1], data.shape[2])) * np.nan
    copt = np.zeros(popt.shape) * np.nan
    ndim = 4 if thick else 3

    # Define the bounds for fitting.
    if thick:
        bounds = ((velax.min(), 0., 0., 0.), (velax.max(), 1e4, 1e5, 10.))
    else:
        bounds = ((velax.min(), 0., 0.), (velax.max(), 1e4, 1e5))

    # Loop through all the pixels in the image.
    # TODO: include a progress bar if I'm feeling fancy.
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            if clip > snr[i, j]:
                continue

            # Initial parameters.
            p0 = [v0[i, j], dV[i, j], Tb[i, j]]
            if thick:
                p0 += [1.0]

            # Fit the pixels.
            try:
                if thick:
                    p, c = curve_fit(thick_gaussian, velax, data[:, i, j],
                                     p0=p0, sigma=rms, absolute_sigma=True,
                                     maxfev=10000, bounds=bounds)
                else:
                    p, c = curve_fit(gaussian, velax, data[:, i, j], p0=p0,
                                     sigma=rms, absolute_sigma=True,
                                     maxfev=10000, bounds=bounds)
                popt[:ndim, i, j] = p
                copt[:ndim, i, j] = np.sqrt(np.diag(c))
            except:
                continue

    return popt[:ndim], copt[:ndim]


def collapse_gausshermite(velax, data, clip=None, rms=None, N=5):
    """
    Collapse the cube by fittingHermit expansion of a Gaussian to allow for a
    parametrised skewness and kurtosis. Note that h3 and h4 are not exact
    representations of these two but all for a more complex profile.

    Args:
        velax (ndarray): Velocity axis of the cube.
        data (ndarray): Flux density or brightness temperature array. Assumes
            that the first axis is the velocity axis.
        clip (Optional[flat]): Skips pixels which have a SNR less than this.
        rms (Optional[float]): Noise per pixel. If none is specified, will be
            calculated from the first and last N channels.
        N (Optional[int]): Number of channels to use in the estimation of the
            noise.

    Returns:
        popt (ndarray): Returns the spatial map of [x0, dx, Tb, h3, h4].
    """

    # Use scipy for the fitting.
    from scipy.optimize import curve_fit

    #  Estimate the starting positions.
    v0, _, Tb, _ = collapse_quadratic(velax, data, rms=rms, N=N)
    dV = np.trapz(data * velax[:, None, None], axis=0)
    dV /= Tb * np.sqrt(np.pi)

    # Calculate the signal to noise ratio of each pixel.
    clip = clip if clip is not None else -100.0
    rms = rms if rms is not None else _estimate_RMS(data, N).mean()
    snr = Tb / rms
    rms *= np.ones(velax.size)

    # Fit the Gaussians, pixel by pixel. NaN as holder for no fit.
    popt = np.zeros((5, data.shape[1], data.shape[2])) * np.nan
    copt = np.zeros(popt.shape) * np.nan
    bounds = ((velax.min(), 0.0, 0.0, -1.0, -1.0),
              (velax.max(), np.inf, np.inf, 1.0, 1.0))

    # Loop through all the pixels in the image.
    # TODO: include a progress bar if I'm feeling fancy.
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            if clip > snr[i, j]:
                continue

            # Initial parameters.
            p0 = [v0[i, j], dV[i, j], Tb[i, j], 0.0, 0.0]

            # Fit the pixels.
            try:
                p, c = curve_fit(GH_gaussian, velax, data[:, i, j], p0=p0,
                                 sigma=rms, absolute_sigma=True, maxfev=10000,
                                 bounds=bounds)
                popt[:, i, j] = p
                copt[:, i, j] = np.sqrt(np.diag(c))
            except:
                continue

    return popt, copt


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
    v0, dv0, Fnu, dFnu = None, None, None, None

    # Collapse the cube with the approrpriate method.

    if args.method.lower() == 'quadratic':
        v0, dv0, Fnu, dFnu = collapse_quadratic(velax, data)
    elif args.method.lower() == 'maximum':
        v0, dv0, Fnu, dFnu = collapse_maximum(velax, data, return_peak=True)
    elif args.method.lower() == 'first':
        v0 = collapse_first(velax, data, clip=2.0)
    else:
        raise ValueError("Unknown method.")

    # Mask the data. If no uncertainties are found for dFnu, use the RMS.
    # TODO: only use the central 1/2 of the image to avoid primary beam issues.

    if args.clip is not None:

        if not args.silent:
            print("Masking maps.")

        if Fnu is None:
            signal = collapse_maximum(velax, data, return_peak=True)[2]
        else:
            signal = Fnu
        if dFnu is None:
            noise = collapse_maximum(velax, data, return_peak=True)[3]
        else:
            noise = dFnu

        rndm = abs(1e-10 * np.random.randn(noise.size).reshape(noise.shape))
        mask = np.where(noise != 0.0, noise, rndm)
        mask = np.where(np.isfinite(mask), mask, rndm)
        mask = np.where(np.isfinite(signal), signal, 0.0) / mask >= args.clip

        v0 = np.where(mask, v0, args.fill)
        if dv0 is not None:
            dv0 = np.where(mask, dv0, args.fill)
        if Fnu is not None:
            Fnu = np.where(mask, Fnu, args.fill)
        if dFnu is not None:
            dFnu = np.where(mask, dFnu, args.fill)

    # Save the files.

    if not args.silent:
        print("Saving maps.")
    if v0 is not None:
        _save_array(args.path, args.path.replace('.fits', '_v0.fits'), v0)
    if dv0 is not None:
        _save_array(args.path, args.path.replace('.fits', '_dv0.fits'), dv0)
    if Fnu is not None:
        _save_array(args.path, args.path.replace('.fits', '_Fnu.fits'), Fnu)
    if dFnu is not None:
        _save_array(args.path, args.path.replace('.fits', '_dFnu.fits'), dFnu)
