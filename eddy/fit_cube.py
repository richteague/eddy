"""
Class to load up a velocity map and fit a Keplerian profile to it. The main
functions of interest are:

    disk_coords: Given geometrical properties of the disk and the emission
        surface, will deproject the data into a face-on view in either polar or
        cartesian coordaintes.

    keplerian: Builds a Keplerian rotation pattern with the provided
        geometrical properties and emission surface. Does not account for any
        deviations due to pressure gradients or self gravity.

    fit_keplerian: Fits a Keplerian profile to the data. It is possible to hold
        various parameters constant or let them vary. Also allows for a flared
        emission surface which can be constrained with good quality data.

TODO:

    1) Include bounds for the initial optimization.
    2) More robust plotting for the residual maps. A separate function maybe?
    3) Can we do the deprojection analytically rather than iteratively?
"""

import numpy as np
from astropy.io import fits
import scipy.constants as sc
import warnings

warnings.filterwarnings("ignore")


class rotationmap:

    msun = 1.988e30
    fwhm = 2. * np.sqrt(2 * np.log(2))

    def __init__(self, path, uncertainty=None, clip=None, downsample=None):
        """Initialize the class."""

        # Read in the data and position axes.
        self.data = np.squeeze(fits.getdata(path))
        self.header = fits.getheader(path)
        if uncertainty is not None:
            self.error = np.squeeze(fits.getdata(uncertainty))
        else:
            print("No uncertainties found, assuming uncertainties of 10%.")
            self.error = 0.1 * self.data
        self.error = np.where(np.isnan(self.error), 0.0, self.error)

        # Make sure this is in [km/s].
        if np.nanmedian(self.data) > 10.0:
            self.data /= 1e3
            self.error /= 1e3

        self.xaxis = self._read_position_axis(a=1)
        self.yaxis = self._read_position_axis(a=2)
        self.dpix = abs(np.diff(self.xaxis)).mean()

        # Clip and downsample the cube to speed things up.
        if clip is not None:
            self._clip_cube(clip)
        if downsample is not None:
            self._downsample_cube(downsample)
            self.dpix = abs(np.diff(self.xaxis)).mean()
        self.mask = np.isfinite(self.data)

        # Estimate the systemic velocity.
        self.vlsr = np.nanmedian(self.data)

        # Beam parameters. TODO: Make sure CASA beam tables work.
        try:
            self.bmaj = self.header['bmaj'] * 3600.
            self.bmin = self.header['bmin'] * 3600.
            self.bpa = self.header['bpa']
        except KeyError:
            self.bmaj = None
            self.bmin = None
            self.bpa = None

    # -- Fitting functions. -- #

    def fit_keplerian(self, p0, params, r_min=None, r_max=None, optimize=True,
                      nwalkers=None, nburnin=300, nsteps=100, scatter=1e-3,
                      plot_walkers=True, plot_corner=True, plot_bestfit=True,
                      plot_residual=True, return_samples=False):
        """
        Fit a Keplerian rotation profile to the data.

        Args:
            p0 (list): List of the free parameters to fit.
            params (dictionary): Dictionary of the parameters used for the
                Keplerian model. If the value is fixed, specify the values:

                    params['x0'] = 0.45

                while if it is a free parameter, provide the index in p0:

                    params['x0'] = 0

                making sure the value is an integer. The values needed are:
                'x0', 'y0', 'inc', 'PA', 'mstar', 'vlsr', 'dist'. If a flared
                emission surface is wanted then you can add 'z0', 'psi' and
                'tilt' with their descriptions found in disk_coords(). To
                include a convolution with the beam stored in the header use
                params['beam'] = True, where this must be a boolean, not an
                integer.
            r_min (Optional[float]): Inner radius to fit in (arcsec).
            r_max (Optional[float]): Outer radius to fit in (arcsec). Note that
                for the masking the default p0 and params values are used for
                the deprojection, or those found from the optimization.
            optimize (Optional[bool]): Use scipy.optimize to find the p0 values
                which maximize the likelihood. Better results will likely be
                found.
            nwalkers (Optional[int]): Number of walkers to use for the MCMC.
            scatter (Optional[float]): Scatter used in distributing walker
                starting positions around the initial p0 values.
            plot_walkers (Optional[bool]): Plot the samples taken by the
                walkers.
            plot_corner (Optional[bool]): Plot the covariances of the
                posteriors.
            plot_bestfit (Optional[bool]): Plot the best fit model.
            plot_residual (Optional[bool]): Plot the residual from the data and
                the best fit model.
            return_samples (Optional[bool]): If true, return all the samples of
                the posterior distribution after the burn in period, otherwise
                just return the 16th, 50th and 84th percentiles of each
                posterior distribution.

        Returns:
            samples (ndarray): If return_sample = True, return all the samples
                of the posterior distribution after the burn in period,
                otherwise just return the 16th, 50th and 84th percentiles of
                each posterior distribution.
        """

        # Load up emcee.
        try:
            import emcee
        except ImportError:
            raise ImportError("Cannot find emcee.")

        # Check the dictionary. May need some more work.
        params = self._verify_dictionary(params)

        # Calculate the inverse variance mask.
        r_min = r_min if r_min is not None else 0.0
        r_max = r_max if r_max is not None else 1e5

        temp = rotationmap._populate_dictionary(p0, params)
        self.ivar = self._calc_ivar(x0=temp['x0'], y0=temp['y0'],
                                    inc=temp['inc'], PA=temp['PA'],
                                    z0=temp['z0'], psi=temp['psi'],
                                    tilt=temp['tilt'], r_min=r_min,
                                    r_max=r_max)

        # Check what the parameters are.
        labels = rotationmap._get_labels(params)
        if len(labels) != len(p0):
            raise ValueError("Mismatch in labels and p0. Check for integers.")
        print("Assuming:\n\tp0 = [%s]." % (', '.join(labels)))

        # Run an initial optimization using scipy.minimize. Recalculate the
        # inverse variance mask.
        if optimize:
            p0 = self._optimize_p0(p0, params)
            temp = rotationmap._populate_dictionary(p0, params)
            self.ivar = self._calc_ivar(x0=temp['x0'], y0=temp['y0'],
                                        inc=temp['inc'], PA=temp['PA'],
                                        z0=temp['z0'], psi=temp['psi'],
                                        tilt=temp['tilt'], r_min=r_min,
                                        r_max=r_max)

        # Plot the data to show where the mask is.
        self.plot_data(ivar=self.ivar)

        # Make sure all starting positions are valid.
        # COMING SOON.

        # Set up and run the MCMC.
        ndim = len(p0)
        nwalkers = 4 * ndim if nwalkers is None else nwalkers
        p0 = rotationmap._random_p0(p0, scatter, nwalkers)

        sampler = emcee.EnsembleSampler(nwalkers, ndim, self._ln_probability,
                                        args=[params, np.nan])
        sampler.run_mcmc(p0, nburnin + nsteps)
        samples = sampler.chain[:, -int(nsteps):]
        samples = samples.reshape(-1, samples.shape[-1])

        bestfit = np.median(samples, axis=0)
        bestfit = rotationmap._populate_dictionary(bestfit, params)

        # Diagnostic plots.
        if plot_walkers:
            rotationmap.plot_walkers(sampler.chain.T, nburnin, labels)
        if plot_corner:
            rotationmap.plot_corner(samples, labels)
        if plot_bestfit:
            self.plot_bestfit(bestfit, ivar=self.ivar)
        if plot_residual:
            self.plot_residual(bestfit, ivar=self.ivar)

        # Return the posterior distributions.
        if return_samples:
            return samples
        return np.percentile(samples, [16, 60, 84], axis=0)

    def _optimize_p0(self, theta, params):
        """Optimize the initial starting positions."""
        from scipy.optimize import minimize

        # Negative log-likelihood function.
        def nlnL(theta):
            return -self._ln_probability(theta, params)

        # TODO: think of a way to include bounds.

        res = minimize(nlnL, x0=theta, method='TNC',
                       options={'maxiter': 100000, 'ftol': 1e-3})
        theta = res.x
        if res.success:
            print("Optimized starting positions:")
        else:
            print("WARNING: scipy.optimize did not converge.")
            print("Starting positions:")
        print('\tp0 =', ['%.4e' % t for t in theta])
        return theta

    @staticmethod
    def _random_p0(p0, scatter, nwalkers):
        """Get the starting positions."""
        p0 = np.squeeze(p0)
        dp0 = np.random.randn(nwalkers * len(p0)).reshape(nwalkers, len(p0))
        dp0 = np.where(p0 == 0.0, 1.0, p0)[None, :] * (1.0 + scatter * dp0)
        return np.where(p0[None, :] == 0.0, dp0 - 1.0, dp0)

    def _ln_likelihood(self, params):
        """Log-likelihood function. Simple chi-squared likelihood."""
        model = self._make_model(params) * 1e-3
        lnx2 = np.where(self.mask, np.power((self.data - model), 2), 0.0)
        lnx2 = -0.5 * np.sum(lnx2 * self.ivar)
        return lnx2 if np.isfinite(lnx2) else -np.inf

    def _ln_probability(self, theta, *params_in):
        """Log-probablility function."""
        model = rotationmap._populate_dictionary(theta, params_in[0])
        if np.isfinite(self._ln_prior(model)):
            return self._ln_likelihood(model)
        return -np.inf

    def _ln_prior(self, params):
        """Log-priors. Uniform and uninformative."""
        if abs(params['x0']) > 0.5:
            return -np.inf
        if abs(params['y0']) > 0.5:
            return -np.inf
        if not 0. < params['inc'] < 90.:
            return -np.inf
        if not -360. < params['PA'] < 360.:
            return -np.inf
        if not 0.0 < params['mstar'] < 5.0:
            return -np.inf
        if abs(self.vlsr - params['vlsr'] / 1e3) > 1.0:
            return -np.inf
        if not 0.0 <= params['z0'] < 1.0:
            return -np.inf
        if not 0.0 < params['psi'] < 2.0:
            return -np.inf
        if not -1.0 < params['tilt'] < 1.0:
            return -np.inf
        return 0.0

    def _calc_ivar(self, x0=0.0, y0=0.0, inc=0.0, PA=0.0, z0=0.0, psi=0.0,
                   tilt=0.0, r_min=0.0, r_max=1e5):
        """Calculate the inverse variance including radius mask."""
        try:
            assert self.error.shape == self.data.shape
        except AttributeError:
            self.error = self.error * np.ones(self.data.shape)
        rvals = self.disk_coords(x0=x0, y0=y0, inc=inc, PA=PA,
                                 z0=z0, psi=psi, tilt=tilt)[0]
        mask = np.logical_and(rvals >= r_min, rvals <= r_max)
        mask = np.logical_and(mask, self.error > 0.0)
        return np.where(mask, np.power(self.error, -2.0), 0.0)

    @staticmethod
    def _get_labels(params):
        """Return the labels of the parameters to fit."""
        idxs, labs = [], []
        for k in params.keys():
            if isinstance(params[k], int):
                if not isinstance(params[k], bool):
                    idxs.append(params[k])
                    labs.append(k)
        return np.array(labs)[np.argsort(idxs)]

    @staticmethod
    def _populate_dictionary(theta, dictionary_in):
        """Populate the dictionary of free parameters."""
        dictionary = dictionary_in.copy()
        for key in dictionary.keys():
            if isinstance(dictionary[key], int):
                if not isinstance(dictionary[key], bool):
                    dictionary[key] = theta[dictionary[key]]
        return dictionary

    def _verify_dictionary(self, params):
        """Check there are the the correct keys."""
        if params.get('x0') is None:
            params['x0'] = 0.0
        if params.get('y0') is None:
            params['y0'] = 0.0
        if params.get('z0') is None:
            params['z0'] = 0.0
        if params.get('psi') is None:
            params['psi'] = 1.0
        if params.get('dist') is None:
            params['dist'] = 100.
        if params.get('tilt') is None:
            params['tilt'] = 0.0
        if params.get('vlsr') is None:
            params['vlsr'] = self.vlsr
        if params.get('beam') is None:
            params['beam'] = False
        elif params.get('beam'):
            if self.bmaj is None:
                params['beam'] = False
        return params

    # -- Deprojection functions. -- #

    def disk_coords(self, x0=0.0, y0=0.0, inc=0.0, PA=0.0, z0=0.0, psi=0.0,
                    tilt=0.0, frame='polar'):
        """
        Get the disk coordinates given certain geometrical parameters and an
        emission surface. The emission surface is parameterized as a powerlaw
        profile: z(r) = z0 * (r / 1")^psi. For a razor thin disk, z0 = 0.0,
        while for a conical disk, as described in Rosenfeld et al. (2013),
        psi = 1.0.

        Args:
            x0 (Optional[float]): Source right ascension offset (arcsec).
            y0 (Optional[float]): Source declination offset (arcsec).
            inc (Optional[float]): Source inclination (degrees).
            PA (Optional[float]): Source position angle (degrees). Measured
                between north and the red-shifted semi-major axis in an
                easterly direction.
            z0 (Optional[float]): Aspect ratio at 1" for the emission surface.
                To get the far side of the disk, make this number negative.
            psi (Optional[float]): Flaring angle for the emission surface.
            tilt (Optional[float]): Value between -1 and 1, positive values
                result in the north side of the disk being closer to the
                observer; negative values the south.
            frame (Optional[str]): Frame of reference for the returned
                coordinates. Either 'polar' or 'cartesian'.

        Returns:
            c1 (ndarryy): Either r (cylindrical) or x depending on the frame.
            c2 (ndarray): Either theta or y depending on the frame.
            c3 (ndarray): Height above the midplane, z.
        """

        # Check the input variables.

        frame = frame.lower()
        if frame not in ['cartesian', 'polar']:
            raise ValueError("frame must be 'cartesian' or 'polar'.")

        # Define the emission surface function. This approach should leave
        # some flexibility for more complex emission surface parameterizations.

        def func(r):
            return z0 * np.power(r, psi)

        # Calculate the pixel values.

        if frame == 'cartesian':
            c1, c2 = self._get_flared_cart_coords(x0, y0, inc, PA, func, tilt)
            c3 = func(np.hypot(c1, c2))
        else:
            c1, c2 = self._get_flared_polar_coords(x0, y0, inc, PA, func, tilt)
            c3 = func(c1)
        return c1, c2, c3

    def deproject_image(self, x0=0.0, y0=0.0, inc=0.0, PA=0.0, z0=0.0, psi=1.0,
                        tilt=0.0, image=None):
        """
        Deproject the image given the geometrical parameters. If no image is
        given, will used the attached rotation map.

        Args:
            x0 (Optional[float]): Source right ascension offset (arcsec).
            y0 (Optional[float]): Source declination offset (arcsec).
            inc (Optional[float]): Source inclination (degrees).
            PA (Optional[float]): Source position angle (degrees). Measured
                between north and the red-shifted semi-major axis in an
                easterly direction.
            z0 (Optional[float]): Aspect ratio at 1" for the emission surface.
                To get the far side of the disk, make this number negative.
            psi (Optional[float]): Flaring angle for the emission surface.
            tilt (Optional[float]): Value between -1 and 1, positive values
                result in the north side of the disk being closer to the
                observer; negative values the south.
            image (Optional[ndarray]): Image to deproject (must be the same
                shape as the original velocity map). If non is specified then
                it will default to the attached velocity map.
        Returns:
            vep (ndarray): Geometrically deprojected image on the attached x-
                and y-axis.
        """

        from scipy.interpolate import griddata

        # Deproject the pixels into cartesians coordinates.
        xpix, ypix, _ = self.disk_coords(x0=x0, y0=y0, inc=inc, PA=PA, z0=z0,
                                         psi=psi, tilt=tilt, frame='cartesian')
        xpix, ypix = xpix.flatten(), ypix.flatten()
        if image is not None:
            dpix = image.flatten()
        else:
            dpix = self.data.flatten()
        if xpix.shape != dpix.shape:
            raise ValueError("Unknown shaped input image.")

        # Deproject the mask.
        isnan = griddata((xpix, ypix), np.where(self.mask, 1, 0).flatten(),
                         (self.xaxis[None, :], self.yaxis[:, None]),
                         method='nearest')

        # Mask any NaN values and regrid the data.
        mask = np.isfinite(dpix)
        xpix, ypix, dpix = xpix[mask], ypix[mask], dpix[mask]
        depr = griddata((xpix, ypix), dpix,
                        (self.xaxis[None, :], self.yaxis[:, None]),
                        method='linear')
        return np.where(isnan, depr, np.nan)

    @staticmethod
    def _rotate_coords(x, y, PA):
        """Rotate (x, y) by PA [deg]."""
        x_rot = x * np.cos(np.radians(PA)) - y * np.sin(np.radians(PA))
        y_rot = y * np.cos(np.radians(PA)) + x * np.sin(np.radians(PA))
        return x_rot, y_rot

    @staticmethod
    def _deproject_coords(x, y, inc):
        """Deproject (x, y) by inc [deg]."""
        return x, y / np.cos(np.radians(inc))

    def _get_cart_sky_coords(self, x0, y0):
        """Return caresian sky coordinates in [arcsec, arcsec]."""
        return np.meshgrid(self.xaxis - x0, self.yaxis - y0)

    def _get_polar_sky_coords(self, x0, y0):
        """Return polar sky coordinates in [arcsec, radians]."""
        x_sky, y_sky = self._get_cart_sky_coords(x0, y0)
        return np.hypot(y_sky, x_sky), np.arctan2(x_sky, y_sky)

    def _get_midplane_cart_coords(self, x0, y0, inc, PA):
        """Return cartesian coordaintes of midplane in [arcsec, arcsec]."""
        x_sky, y_sky = self._get_cart_sky_coords(x0, y0)
        x_rot, y_rot = rotationmap._rotate_coords(y_sky, x_sky, -PA)
        return rotationmap._deproject_coords(x_rot, y_rot, inc)

    def _get_midplane_polar_coords(self, x0, y0, inc, PA):
        """Return the polar coordinates of midplane in [arcsec, radians]."""
        x_mid, y_mid = self._get_midplane_cart_coords(x0, y0, inc, PA)
        return np.hypot(y_mid, x_mid), np.arctan2(y_mid, x_mid)

    def _get_flared_polar_coords(self, x0, y0, inc, PA, func, tilt):
        """Return polar coordinates of surface in [arcsec, radians]."""
        x_mid, y_mid = self._get_midplane_cart_coords(x0, y0, inc, PA)
        r_mid, t_mid = self._get_midplane_polar_coords(x0, y0, inc, PA)
        for _ in range(5):
            y_tmp = func(r_mid) * np.sign(tilt) * np.tan(np.radians(inc))
            y_tmp = y_mid - y_tmp
            r_mid = np.hypot(y_tmp, x_mid)
            t_mid = np.arctan2(y_tmp, x_mid)
        return r_mid, t_mid

    def _get_flared_cart_coords(self, x0, y0, inc, PA, func, tilt):
        """Return cartesian coordinates of surface in [arcsec, arcsec]."""
        r_mid, t_mid = self._get_flared_polar_coords(x0, y0, inc,
                                                     PA, func, tilt)
        return r_mid * np.cos(t_mid), r_mid * np.sin(t_mid)

    # -- Functions to build Keplerian rotation profiles. -- #

    def keplerian(self, x0=0.0, y0=0.0, inc=0.0, PA=0.0, z0=0.0, psi=0.0,
                  tilt=0.0, mstar=1.0, dist=100., vlsr=0.0):
        """
        Return a Keplerian rotation profile (not including pressure) in [m/s].
        This includes the deivation due to non-zero heights above the midplane,
        see Teague et al. (2018a,c) for a thorough description.

        Args:
            x0 (Optional[float]): Source right ascension offset (arcsec).
            y0 (Optional[float]): Source declination offset (arcsec).
            inc (Optional[float]): Source inclination (degrees).
            PA (Optional[float]): Source position angle (degrees). Measured
                between north and the red-shifted semi-major axis in an
                easterly direction.
            z0 (Optional[float]): Aspect ratio at 1" for the emission surface.
                To get the far side of the disk, make this number negative.
            psi (Optional[float]): Flaring angle for the emission surface.
            tilt (Optional[float]): Value between -1 and 1, positive values
                result in the north side of the disk being closer to the
                observer; negative values the south.
            mstar (Optional[float]): Mass of the star in (solar masses).
            dist (Optional[float]): Distance to the source in (parsec).
            vlsr (Optional[floar]): Systemic velocity in (m/s).

        Returns:
            vproj (ndarray): Projected Keplerian rotation at each pixel (m/s).
        """
        coords = self.disk_coords(x0=x0, y0=y0, inc=inc, PA=PA, z0=z0, psi=psi,
                                  tilt=tilt, frame='polar')
        rvals = coords[0] * sc.au * dist
        zvals = coords[2] * sc.au * dist
        vkep = sc.G * mstar * self.msun * np.power(rvals, 2.0)
        vkep = np.sqrt(vkep * np.power(np.hypot(rvals, zvals), -3.0))
        return vkep * np.sin(np.radians(inc)) * np.cos(coords[1]) + vlsr

    def _make_model(self, params):
        """Build the Keplerian model from the dictionary of parameters."""
        vkep = self.keplerian(x0=params['x0'], y0=params['y0'],
                              inc=params['inc'], PA=params['PA'],
                              vlsr=params['vlsr'], z0=params['z0'],
                              psi=params['psi'], mstar=params['mstar'],
                              dist=params['dist'], tilt=params['tilt'])
        if params['beam']:
            vkep = rotationmap._convolve_image(vkep, self._beamkernel())
        return vkep

    # -- Helper functions for loading up the data. -- #

    def _clip_cube(self, radius):
        """Clip the cube to  clip arcseconds from the origin."""
        xa = abs(self.xaxis - radius).argmin()
        xb = abs(self.xaxis + radius).argmin()
        ya = abs(self.yaxis - radius).argmin()
        yb = abs(self.yaxis + radius).argmin()
        self.data = self.data[yb:ya, xa:xb]
        self.error = self.error[yb:ya, xa:xb]
        self.xaxis = self.xaxis[xa:xb]
        self.yaxis = self.yaxis[yb:ya]

    def _downsample_cube(self, N):
        """Downsample the cube to make faster calculations."""
        N0 = int(N / 2)
        self.xaxis = self.xaxis[N0::N]
        self.yaxis = self.yaxis[N0::N]
        self.data = self.data[N0::N, N0::N]
        self.error = self.error[N0::N, N0::N]

    def _read_position_axis(self, a=1):
        """Returns the position axis in [arcseconds]."""
        if a not in [1, 2]:
            raise ValueError("'a' must be in [0, 1].")
        a_len = self.header['naxis%d' % a]
        a_del = self.header['cdelt%d' % a]
        a_pix = self.header['crpix%d' % a]
        return 3600 * ((np.arange(a_len) - a_pix + 1.5) * a_del)

    # -- Convolution functions. -- #

    def _beamkernel(self):
        """Returns the 2D Gaussian kernel for convolution."""
        from astropy.convolution import Kernel
        bmaj = self.bmaj / self.dpix / self.fwhm
        bmin = self.bmin / self.dpix / self.fwhm
        bpa = np.radians(self.bpa)
        return Kernel(self._gaussian2D(bmin, bmaj, bpa + 90.).T)

    def _gaussian2D(self, dx, dy, PA=0.0):
        """2D Gaussian kernel in pixel coordinates."""
        xm = np.arange(-4*np.nanmax([dy, dx]), 4*np.nanmax([dy, dx])+1)
        x, y = np.meshgrid(xm, xm)
        x, y = self._rotate_coords(x, y, PA)
        k = np.power(x / dx, 2) + np.power(y / dy, 2)
        return np.exp(-0.5 * k) / 2. / np.pi / dx / dy

    @staticmethod
    def _convolve_image(image, kernel, fast=True):
        """Convolve the image with the provided kernel."""
        if fast:
            from astropy.convolution import convolve_fft
            return convolve_fft(image, kernel)
        from astropy.convolution import convolve
        return convolve(image, kernel)

    # -- Plotting functions. -- #

    def plot_data(self, levels=None, ivar=None):
        """Plot the first moment map."""
        import matplotlib.cm as cm
        import matplotlib.pyplot as plt
        ax = plt.subplots()[1]
        if levels is None:
            levels = np.nanpercentile(self.data, [2, 98]) - self.vlsr
            levels = max(abs(levels[0]), abs(levels[1]))
            levels = self.vlsr + np.linspace(-levels, levels, 30)
        im = ax.contourf(self.xaxis, self.yaxis, self.data, levels,
                         cmap=cm.RdBu_r, extend='both')
        cb = plt.colorbar(im, pad=0.02)
        cb.set_label(r'${\rm  v_{obs} \quad (km\,s^{-1})}$',
                     rotation=270, labelpad=15)
        if ivar is not None:
            ax.contour(self.xaxis, self.yaxis, ivar, [0], colors='k')
        self._gentrify_plot(ax)

    def plot_bestfit(self, params, ivar=None, residual=False):
        """Plot the best-fit model."""
        import matplotlib.cm as cm
        import matplotlib.pyplot as plt
        ax = plt.subplots()[1]
        vkep = self._make_model(params) * 1e-3
        levels = np.where(self.ivar != 0.0, vkep, np.nan)
        levels = np.nanpercentile(levels, [2, 98])
        levels = np.linspace(levels[0], levels[1], 30)
        im = ax.contourf(self.xaxis, self.yaxis, vkep, levels,
                         cmap=cm.RdBu_r, extend='both')
        if ivar is not None:
            ax.contour(self.xaxis, self.yaxis, ivar, [0], colors='k')
        cb = plt.colorbar(im, pad=0.02)
        cb.set_label(r'${\rm  v_{Kep} \quad (km\,s^{-1})}$',
                     rotation=270, labelpad=15)
        self._gentrify_plot(ax)

    def plot_residual(self, params, ivar=None):
        """Plot the residual from the provided model."""
        import matplotlib.cm as cm
        import matplotlib.pyplot as plt
        ax = plt.subplots()[1]
        vres = self.data - self._make_model(params) * 1e-3
        levels = np.where(self.ivar != 0.0, vres, np.nan)
        levels = np.nanpercentile(levels, [10, 90])
        levels = np.linspace(levels[0], levels[1], 30)
        im = ax.contourf(self.xaxis, self.yaxis, vres, levels,
                         cmap=cm.RdBu_r, extend='both')
        if ivar is not None:
            ax.contour(self.xaxis, self.yaxis, ivar, [0], colors='k')
        cb = plt.colorbar(im, pad=0.02)
        cb.set_label(r'${\rm  v_{Obs} - v_{Kep} \quad (km\,s^{-1})}$',
                     rotation=270, labelpad=15)
        self._gentrify_plot(ax)

    @staticmethod
    def plot_walkers(samples, nburnin=None, labels=None):
        """Plot the walkers to check if they are burning in."""

        # Import matplotlib.
        import matplotlib.pyplot as plt

        # Check the length of the label list.
        if labels is None:
            if samples.shape[0] != len(labels):
                raise ValueError("Not correct number of labels.")

        # Cycle through the plots.
        for s, sample in enumerate(samples):
            ax = plt.subplots()[1]
            for walker in sample.T:
                ax.plot(walker, alpha=0.1, color='k')
            ax.set_xlabel('Steps')
            if labels is not None:
                ax.set_ylabel(labels[s])
            if nburnin is not None:
                ax.axvline(nburnin, ls=':', color='r')

    @staticmethod
    def plot_corner(samples, labels=None, quantiles=None):
        """Plot the corner plot to check for covariances."""
        import corner
        quantiles = [0.16, 0.5, 0.84] if quantiles is None else quantiles
        corner.corner(samples, labels=labels, title_fmt='.4f',
                      quantiles=quantiles, show_titles=True)

    def plot_beam(self, ax, dx=0.125, dy=0.125, **kwargs):
        """Plot the sythensized beam on the provided axes."""
        from matplotlib.patches import Ellipse
        beam = Ellipse(ax.transLimits.inverted().transform((dx, dy)),
                       width=self.bmin, height=self.bmaj, angle=-self.bpa,
                       fill=False, hatch=kwargs.get('hatch', '////////'),
                       lw=kwargs.get('linewidth', kwargs.get('lw', 1)),
                       color=kwargs.get('color', kwargs.get('c', 'k')),
                       zorder=kwargs.get('zorder', 1000))
        ax.add_patch(beam)

    def _gentrify_plot(self, ax):
        """Gentrify the plot."""
        from matplotlib.ticker import MultipleLocator
        ax.set_aspect(1)
        ax.grid(ls=':', color='k', alpha=0.3)
        ax.tick_params(which='both', right=True, top=True)
        ax.set_xlim(self.xaxis.max(), self.xaxis.min())
        ax.set_ylim(self.yaxis.min(), self.yaxis.max())
        ax.xaxis.set_major_locator(MultipleLocator(1.0))
        ax.yaxis.set_major_locator(MultipleLocator(1.0))
        ax.set_xlabel('Offset (arcsec)')
        ax.set_ylabel('Offset (arcsec)')
        if self.bmaj is not None:
            self.plot_beam(ax=ax)
