# -*- coding: utf-8 -*-

import time
import emcee
import numpy as np
from astropy.io import fits
import scipy.constants as sc
import warnings

warnings.filterwarnings("ignore")

class rotationmap:
    """
    Read in the velocity maps and initialize the class. To make the fitting
    quicker, we can clip the cube to a smaller region, or downsample the
    image to get a quicker initial fit.

    Args:
        path (str): Relative path to the rotation map you want to fit.
        uncertainty (optional[srt]): Relative path to the map of v0
            uncertainties. Must be a FITS file with the same shape as the
            data. If nothing is specified, will assume a 10% on all pixels.
        clip (optional[float]): If specified, clip the data down to a
            square with sides 2 x clip centered on the image center.
        downsample (optional[int]): Downsample the image by this factor for
            quicker fitting. For example, using ``downsample=4`` on an image
            which is 1000 x 1000 pixels will reduce it to 250 x 250 pixels.
            If you use ``downsample='beam'`` it will sample roughly
            spatially independent pixels using the beam major axis as the
            spacing.
        x0 (optional[float]): Recenter the image to this offset.
        y0 (optional[float]): Recenter the image to this offset.
        unit (optional[str]): Unit of the input data cube. By deafult
            assumed to be [m/s].
    """

    msun = 1.988e30
    fwhm = 2. * np.sqrt(2 * np.log(2))
    disk_coords_niter = 5
    priors = {}

    def __init__(self, path, uncertainty=None, clip=None, downsample=None,
                 x0=0.0, y0=0.0, unit='m/s'):
        # Read in the data and position axes.
        self.data = np.squeeze(fits.getdata(path))
        self.header = fits.getheader(path)
        if uncertainty is not None:
            self.error = np.squeeze(fits.getdata(uncertainty))
        else:
            print("No uncertainties found, assuming uncertainties of 10%.")
            print("You can change this at any time with rotationmap.error.")
            self.error = 0.1 * self.data
        self.error = np.where(np.isnan(self.error), 0.0, self.error)

        # Convert the data to [km/s].
        if unit.lower() == 'm/s':
            self.data *= 1e-3
            self.error *= 1e-3
        elif unit.lower() != 'km/s':
            raise ValueError("unit must me 'm/s' or 'km/s'.")

        # Read the position axes.
        # TODO: Check half-pixel offsets.
        self.xaxis = self._read_position_axis(a=1)
        self.yaxis = self._read_position_axis(a=2)
        self.dpix = abs(np.diff(self.xaxis)).mean()

        # Recenter the image if provided.
        if (x0 != 0.0) or (y0 != 0.0):
            self._shift_center(dx=x0, dy=y0)

        # Beam parameters.
        self._readbeam()

        # Clip and downsample the cube to speed things up.
        if downsample is not None:
            if downsample == 'beam':
                downsample = int(self.bmaj / self.dpix)
            self._downsample_cube(downsample)
            self.dpix = abs(np.diff(self.xaxis)).mean()
        if clip is not None:
            self._clip_cube(clip)
        self.nypix = self.yaxis.size
        self.nxpix = self.xaxis.size
        self.mask = np.isfinite(self.data)

        # Estimate the systemic velocity.
        self.vlsr = np.nanmedian(self.data)

        # Set priors.
        self._set_default_priors()

    def fit_keplerian(self, p0, params, r_min=None, r_max=None, optimize=True,
                      nwalkers=None, nburnin=300, nsteps=100, scatter=1e-3,
                      plots=None, returns=None, pool=None, emcee_kwargs=None,
                      niter=1):
        """
        Fit a Keplerian rotation profile to the data. Note that for a disk with
        a non-zero height, the sign of the inclination dictates the direction
        of the tilt: a positive tilt rotates the disk about around the x-axis
        such that the southern side of the disk is closer to the observer. The
        same definition is used for the warps.

        The function must be provided with a dictionary of parameters,
        ``params`` where key is the parameter and its value is either the fixed
        value this will take,

            ``params['PA'] = 45.0``

        would fix position angle to 45 degrees. Or, if an integeter, the index
        in the starting positions, ``p0``, such that,

            ``params['mstar'] = 0``

        would mean that the first value in ``p0`` is the guess for ``'mstar'``.

        For a list of the geometrical parameters, included flared emission
        surfaces or warps, see :func:`disk_coords`. In addition to these
        geometrical properties ``params`` also requires ``'mstar'``,
        ``'vlsr'``, ``'dist'``.

        To include a spatial convolution of the model rotation map you may also
        set ``params['beam']=True``, however this is only really necessary for
        low spatial resolution data, or for rotation maps made via the
        intensity-weighted average velocity.

        Args:
            p0 (list): List of the free parameters to fit.
            params (dictionary): Dictionary of the model parameters.
            r_min (optional[float]): Inner radius to fit in [arcsec].
            r_max (optional[float]): Outer radius to fit in [arcsec].
            optimize (optional[bool]): Use ``scipy.optimize`` to find the
                ``p0`` values which maximize the likelihood. Better results
                will likely be found. Note that for the masking the default
                ``p0`` and ``params`` values are used for the deprojection, or
                those foundvfrom the optimization. If this results in a poor
                initial mask, try with ``optimise=False`` or with a ``niter``
                value larger than 1.
            niter (optional[int]): Number of iterations to perform using the
                median PDF values from the previous MCMC run as starting
                positions. This is probably only useful if you have no idea
                about the starting positions for the emission surface or if you
                want to remove walkers stuck in local minima.
            nwalkers (optional[int]): Number of walkers to use for the MCMC.
            scatter (optional[float]): Scatter used in distributing walker
                starting positions around the initial p0 values.
            plots (optional[list]): List of the diagnostic plots to make. This
                can include ``'mask'``, ``'walkers'``, ``'corner'``,
                ``'bestfit'``, ``'residual'``, or ``'none'`` if no plots are to
                be plotted. By default, all are plotted.
            returns (optional[list]): List of items to return. Can contain
                ``'samples'``, ``'percentiles'``, ``'dict'`` or ``'none'``. By
                default only ``'percentiles'`` are returned.
            pool (optional): An object with a `map` method.
            emcee_kwargs (Optional[dict]): Dictionary to pass to the emcee
                ``EnsembleSampler``.

        Returns:
            to_return (list): Depending on the returns list provided.
                ``'samples'`` will be all the samples of the posteriors (with
                the burn-in periods removed). ``'percentiles'`` will return the
                16th, 50th and 84th percentiles of each posterior functions.
                ``'dict'`` will return a dictionary of the median parameters
                which can be directly input to other functions.

        .. _Rosenfeld et al. (2013): https://ui.adsabs.harvard.edu/abs/2013ApJ...774...16R/abstract

        """

        # Check the dictionary. May need some more work.
        params = self._verify_dictionary(params)

        # Calculate the inverse variance mask.
        r_min = r_min if r_min is not None else 0.0
        r_max = r_max if r_max is not None else 1e5

        p0 = np.squeeze(p0).astype(float)
        temp = rotationmap._populate_dictionary(p0, params)
        self.ivar = self._calc_ivar(x0=temp['x0'], y0=temp['y0'],
                                    inc=temp['inc'], PA=temp['PA'],
                                    z0=temp['z0'], psi=temp['psi'],
                                    z1=temp['z1'], phi=temp['phi'],
                                    w_i=temp['w_i'], w_r=temp['w_r'],
                                    w_t=temp['w_t'], r_min=r_min,
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
                                        z1=temp['z1'], phi=temp['phi'],
                                        w_i=temp['w_i'], w_r=temp['w_r'],
                                        w_t=temp['w_t'], r_min=r_min,
                                        r_max=r_max)

        # Set up and run the MCMC with emcee.
        time.sleep(0.5)
        nwalkers = 2 * p0.size if nwalkers is None else nwalkers
        emcee_kwargs = {} if emcee_kwargs is None else emcee_kwargs
        emcee_kwargs['scatter'], emcee_kwargs['pool'] = scatter, pool
        for n in range(int(niter)):

            # Recalculate the uncertainties for each iteration.
            if n > 0:
                temp = rotationmap._populate_dictionary(p0, params)
                temp = self._verify_dictionary(temp)
                self.ivar = self._calc_ivar(x0=temp['x0'], y0=temp['y0'],
                                            inc=temp['inc'], PA=temp['PA'],
                                            z0=temp['z0'], psi=temp['psi'],
                                            z1=temp['z1'], phi=temp['phi'],
                                            w_i=temp['w_i'], w_r=temp['w_r'],
                                            w_t=temp['w_t'], r_min=r_min,
                                            r_max=r_max)

            # Run the sampler.
            sampler = self._run_mcmc(p0=p0, params=params, nwalkers=nwalkers,
                                     nburnin=nburnin, nsteps=nsteps,
                                     **emcee_kwargs)
            if type(params['PA']) is int:
                sampler.chain[:, :, params['PA']] %= 360.0

            # Split off the samples.
            samples = sampler.chain[:, -int(nsteps):]
            samples = samples.reshape(-1, samples.shape[-1])
            p0 = np.median(samples, axis=0)
            medians = rotationmap._populate_dictionary(p0, params)
            medians = self._verify_dictionary(medians)

        # Diagnostic plots.
        if plots is None:
            plots = ['mask', 'walkers', 'corner', 'bestfit', 'residual']
        plots = np.atleast_1d(plots)
        if 'none' in plots:
            plots = []
        if 'mask' in plots:
            self.plot_data(ivar=self.ivar)
        if 'walkers' in plots:
            rotationmap._plot_walkers(sampler.chain.T, nburnin, labels)
        if 'corner' in plots:
            rotationmap._plot_corner(samples, labels)
        if 'bestfit' in plots:
            self._plot_bestfit(medians, ivar=self.ivar)
        if 'residual' in plots:
            self._plot_residual(medians, ivar=self.ivar)

        # Generate the output.
        if returns is None:
            returns = ['percentiles']
        returns = np.atleast_1d(returns)
        if 'none' in returns:
            return None
        to_return = []
        if 'samples' in returns:
            to_return += [samples]
        if 'lnprob' in returns:
            to_return += [sampler.lnprobability[nburnin:]]
        if 'percentiles' in returns:
            to_return += [np.percentile(samples, [16, 60, 84], axis=0)]
        if 'dict' in returns:
            to_return += [medians]
        return to_return if len(to_return) > 1 else to_return[0]

    def set_prior(self, param, args, type='flat'):
        """
        Set the prior for the given parameter. There are two types of priors
        currently usable, ``'flat'`` which requires ``args=[min, max]`` while
        for ``'gaussian'`` you need to specify ``args=[mu, sig]``.

        Args:
            param (str): Name of the parameter.
            args (list): Values to use depending on the type of prior.
            type (optional[str]): Type of prior to use.
        """
        type = type.lower()
        if type not in ['flat', 'gaussian']:
            raise ValueError("type must be 'flat' or 'gaussian'.")
        if type == 'flat':
            def prior(p):
                if not min(args) <= p <= max(args):
                    return -np.inf
                return np.log(1.0 / (args[1] - args[0]))
        else:
            def prior(p):
                lnp = np.exp(-0.5 * ((args[0] - p) / args[1])**2)
                return np.log(lnp / np.sqrt(2. * np.pi) / args[1])
        rotationmap.priors[param] = prior

    def disk_coords(self, x0=0.0, y0=0.0, inc=0.0, PA=0.0, z0=0.0, psi=0.0,
                    z1=0.0, phi=0.0, w_i=0.0, w_r=1.0, w_t=0.0,
                    frame='cylindrical', shadowed=False, shadowed_kwargs=None,
                    **kwargs):
        r"""
        Get the disk coordinates given certain geometrical parameters and an
        emission surface. The emission surface is parameterized as a powerlaw
        profile:

        .. math::

            z(r) = z_0 \times \left(\frac{r}{1^{\prime\prime}}\right)^{\psi} +
            z_1 \times \left(\frac{r}{1^{\prime\prime}}\right)^{\varphi}

        Where both ``z0`` and ``z1`` are given in [arcsec]. For a razor thin
        disk, ``z0=0.0``, while for a conical disk, as described in `Rosenfeld
        et al. (2013)`_, ``psi=1.0``. We can also include a warp which is
        parameterized by,

        .. math::

            z_{\rm warp}(r,\, t) = r \times \tan \left(w_i \times \exp\left(-
            \frac{r^2}{2 w_r^2} \right) \times \sin(t - w_t)\right)

        where ``w_i`` is the inclination in [radians] describing the warp at
        the disk center. The width of the warp is given by ``w_r`` [arcsec] and
        ``w_t`` in [radians] is the angle of nodes (where the warp is zero),
        relative to the position angle of the disk, measured east of north.

        .. WARNING::

            The use of warps is largely untested. Use with caution!
            If you are using a warp, increase the number of iterations
            for the inference through ``self.disk_coords_niter`` (by default at
            5). For high inclinations, also set ``shadowed=True``.

        Args:
            x0 (optional[float]): Source right ascension offset [arcsec].
            y0 (optional[float]): Source declination offset [arcsec].
            inc (optional[float]): Source inclination [degrees].
            PA (optional[float]): Source position angle [degrees]. Measured
                between north and the red-shifted semi-major axis in an
                easterly direction.
            z0 (optional[float]): Aspect ratio at 1" for the emission surface.
                To get the far side of the disk, make this number negative.
            psi (optional[float]): Flaring angle for the emission surface.
            z1 (optional[float]): Correction term for ``z0``.
            phi (optional[float]): Flaring angle correction term for the
                emission surface.
            w_i (optional[float]): Warp inclination in [degrees] at the disk
                center.
            w_r (optional[float]): Scale radius of the warp in [arcsec].
            w_t (optional[float]): Angle of nodes of the warp in [degrees].
            frame (optional[str]): Frame of reference for the returned
                coordinates. Either ``'cartesian'`` or ``'cylindrical'``.
            shadowed (optional[bool]): If true, use a more robust, however
                slower deprojection routine which accurately takes into account
                shadowing at high inclinations.
            shadowed_kwargs (optional[dict]): Optional kwargs to be passed to
                ``_get_shadowed_coords`` including ``_get_diskframe_coords``
                and ``scipy.interpolate.griddata``.

        Returns:
            Either the ``(r, phi, z)`` coordinates if ``frame='cylindrical'``
            or ``(x, y, z)`` coordinates if ``frame='cartesian'``.
        """

        # Check the input variables.

        frame = frame.lower()
        if frame not in ['cylindrical', 'cartesian']:
            raise ValueError("frame must be 'cylindrical' or 'cartesian'.")

        # Define the emission surface, z_func, and the warp function, w_func.

        def z_func(r):
            z = z0 * np.power(r, psi) + z1 * np.power(r, phi)
            return np.where(z > 0.0, z, 0.0)

        def w_func(r, t):
            warp = np.radians(w_i) * np.exp(-0.5 * (r / w_r)**2)
            return r * np.tan(warp * np.sin(t - np.radians(w_t)))

        # Calculate the pixel values.
        if shadowed:
            if shadowed_kwargs is None:
                shadowed_kwargs = {}
            r, t, z = self._get_shadowed_coords(x0, y0, inc, PA, z_func,
                                                w_func, **shadowed_kwargs)
        else:
            r, t, z = self._get_flared_coords(x0, y0, inc, PA, z_func, w_func)
        if frame == 'cylindrical':
            return r, t, z
        return r * np.cos(t), r * np.sin(t), z

    def plot_data(self, levels=None, ivar=None, return_fig=False):
        """
        Plot the first moment map. By default will clip the velocity contours
        such that the velocities around the systemic velocity are highlighted.

        Args:
            levels (optional[list]): List of contour levels to use.
            ivar (optional[ndarray]): Inverse variances for each pixel. Will
                draw a solid contour around the regions with finite ``ivar``
                values.
            return_fig (optional[bool]): Return the figure.
        """
        import matplotlib.cm as cm
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        if levels is None:
            levels = np.nanpercentile(self.data, [2, 98]) - self.vlsr
            levels = max(abs(levels[0]), abs(levels[1]))
            levels = self.vlsr + np.linspace(-levels, levels, 30)
        im = ax.contourf(self.xaxis, self.yaxis, self.data, levels,
                         cmap=rotationmap.colormap(), extend='both', zorder=-9)
        cb = plt.colorbar(im, pad=0.03, format='%.2f')
        cb.set_label(r'${\rm v_{0} \quad (km\,s^{-1})}$',
                     rotation=270, labelpad=15)
        if ivar is not None:
            ax.contour(self.xaxis, self.yaxis, ivar, [0], colors='k')
        self._gentrify_plot(ax)
        if return_fig:
            return fig

    # -- MCMC Functions -- #

    def _optimize_p0(self, theta, params, **kwargs):
        """Optimize the initial starting positions."""
        from scipy.optimize import minimize

        # TODO: implement bounds.
        # TODO: cycle through parameters one at a time for a better fit.

        # Negative log-likelihood function.
        def nlnL(theta):
            return -self._ln_probability(theta, params)

        method = kwargs.pop('method', 'TNC')
        options = kwargs.pop('options', {})
        options['method'] = options.pop('maxiter', 10000)
        options['ftol'] = options.pop('ftol', 1e-3)
        res = minimize(nlnL, x0=theta, method=method, options=options)

        if res.success:
            theta = res.x
            print("Optimized starting positions:")
        else:
            print("WARNING: scipy.optimize did not converge.")
            print("Starting positions:")
        print('\tp0 =', ['%.2e' % t for t in theta])
        time.sleep(.3)
        return theta

    def _run_mcmc(self, p0, params, nwalkers, nburnin, nsteps, **kwargs):
        """Run the MCMC sampling. Returns the sampler."""
        p0 = self._random_p0(p0, kwargs.pop('scatter', 1e-3), nwalkers)
        progress = kwargs.pop('progress', True)
        sampler = emcee.EnsembleSampler(nwalkers, p0.shape[1],
                                        self._ln_probability,
                                        args=[params, np.nan],
                                        **kwargs)
        if emcee.__version__ >= '3':
            sampler.run_mcmc(p0, nburnin + nsteps, progress=progress)
        else:
            sampler.run_mcmc(p0, nburnin + nsteps)
        return sampler

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
        lnp = self._ln_prior(model)
        if np.isfinite(lnp):
            return lnp + self._ln_likelihood(model)
        return -np.inf

    def _set_default_priors(self):
        """Set the default priors."""
        self.set_prior('x0', [-0.5, 0.5], 'flat')
        self.set_prior('y0', [-0.5, 0.5], 'flat')
        self.set_prior('inc', [-90., 90.0], 'flat')
        self.set_prior('PA', [-360., 360.], 'flat')
        self.set_prior('mstar', [0.1, 5.], 'flat')
        self.set_prior('vlsr', [np.nanmin(self.data) * 1e3,
                                np.nanmax(self.data) * 1e3], 'flat')
        self.set_prior('z0', [0., 5.], 'flat')
        self.set_prior('psi', [0., 5.], 'flat')
        self.set_prior('z1', [-5., 5.], 'flat')
        self.set_prior('phi', [0., 5.], 'flat')
        self.set_prior('w_i', [0., 90.], 'flat')
        self.set_prior('w_r', [self.dpix, 10.0], 'flat')
        self.set_prior('w_t', [-90., 90.], 'flat')

    def _ln_prior(self, params):
        """Log-priors."""
        lnp = 0.0
        for key in params.keys():
            if key in rotationmap.priors.keys():
                lnp += rotationmap.priors[key](params[key])
        return lnp

    def _calc_ivar(self, x0=0.0, y0=0.0, inc=0.0, PA=0.0, z0=0.0, psi=0.0,
                   z1=0.0, phi=1.0, w_i=0.0, w_r=1.0, w_t=0.0, r_min=0.0,
                   r_max=1e5):
        """Calculate the inverse variance including radius mask."""
        try:
            assert self.error.shape == self.data.shape
        except AttributeError:
            self.error = self.error * np.ones(self.data.shape)
        rvals = self.disk_coords(x0=x0, y0=y0, inc=inc, PA=PA, z0=z0,
                                 psi=psi, z1=z1, phi=phi)[0]
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
                    labs.append(r'${{\rm {}}}$'.format(k))
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
        if params.get('z1') is None:
            params['z1'] = 0.0
        if params.get('phi') is None:
            params['phi'] = 1.0
        if params.get('w_i') is None:
            params['w_i'] = 0.0
        if params.get('w_r') is None:
            params['w_r'] = 1.0
        if params.get('w_t') is None:
            params['w_t'] = 0.0
        if params.get('dist') is None:
            params['dist'] = 100.
        if params.get('vlsr') is None:
            params['vlsr'] = self.vlsr
        return params

    # -- Deprojection Functions -- #

    @staticmethod
    def _rotate_coords(x, y, PA):
        """Rotate (x, y) by PA [deg]."""
        x_rot = y * np.cos(np.radians(PA)) + x * np.sin(np.radians(PA))
        y_rot = x * np.cos(np.radians(PA)) - y * np.sin(np.radians(PA))
        return x_rot, y_rot

    @staticmethod
    def _deproject_coords(x, y, inc):
        """Deproject (x, y) by inc [deg]."""
        return x, y / np.cos(np.radians(inc))

    def _get_cart_sky_coords(self, x0, y0):
        """Return caresian sky coordinates in [arcsec, arcsec]."""
        return np.meshgrid(self.xaxis - x0, self.yaxis - y0)

    def _get_midplane_cart_coords(self, x0, y0, inc, PA):
        """Return cartesian coordaintes of midplane in [arcsec, arcsec]."""
        x_sky, y_sky = self._get_cart_sky_coords(x0, y0)
        x_rot, y_rot = self._rotate_coords(x_sky, y_sky, PA)
        return rotationmap._deproject_coords(x_rot, y_rot, inc)

    def _get_midplane_polar_coords(self, x0, y0, inc, PA):
        """Return the polar coordinates of midplane in [arcsec, radians]."""
        x_mid, y_mid = self._get_midplane_cart_coords(x0, y0, inc, PA)
        return np.hypot(y_mid, x_mid), np.arctan2(y_mid, x_mid)

    def _get_flared_coords(self, x0, y0, inc, PA, z_func, w_func):
        """Return cyclindrical coords of surface in [arcsec, rad, arcsec]."""
        x_mid, y_mid = self._get_midplane_cart_coords(x0, y0, inc, PA)
        r_tmp, t_tmp = np.hypot(x_mid, y_mid), np.arctan2(y_mid, x_mid)
        for _ in range(self.disk_coords_niter):
            z_tmp = z_func(r_tmp) + w_func(r_tmp, t_tmp)
            y_tmp = y_mid + z_tmp * np.tan(np.radians(inc))
            r_tmp = np.hypot(y_tmp, x_mid)
            t_tmp = np.arctan2(y_tmp, x_mid)
        return r_tmp, t_tmp, z_func(r_tmp)

    def _get_shadowed_coords(self, x0, y0, inc, PA, z_func, w_func, **kwargs):
        """Return cyclindrical coords of surface in [arcsec, rad, arcsec]."""

        # Make the disk-frame coordinates.
        extend = kwargs.pop('extend', 2.0)
        oversample = kwargs.pop('oversample', 2.0)
        xdisk, ydisk, rdisk, tdisk = self._get_diskframe_coords(extend,
                                                                oversample)
        zdisk = z_func(rdisk) + w_func(rdisk, tdisk)

        # Incline the disk.
        inc = np.radians(inc)
        x_dep = xdisk
        y_dep = ydisk * np.cos(inc) - zdisk * np.sin(inc)
        z_dep = ydisk * np.sin(inc) + zdisk * np.cos(inc)

        # Remove shadowed pixels.
        if inc < 0.0:
            y_dep = np.maximum.accumulate(y_dep, axis=0)
        else:
            y_dep = np.minimum.accumulate(y_dep[::-1], axis=0)[::-1]

        # Rotate and the disk.
        x_rot, y_rot = self._rotate_coords(x_dep, y_dep, PA)
        x_rot, y_rot = x_rot + x0, y_rot + y0

        # Grid the disk.
        from scipy.interpolate import griddata
        disk = (x_rot.flatten(), y_rot.flatten())
        grid = (self.xaxis[None, :], self.yaxis[:, None])
        kwargs['method'] = kwargs.pop('method', 'nearest')
        r_obs = griddata(disk, rdisk.flatten(), grid, **kwargs)
        t_obs = griddata(disk, tdisk.flatten(), grid, **kwargs)
        #z_obs = griddata(disk, zdisk.flatten(), grid, **kwargs)
        return r_obs, t_obs, z_func(r_obs)
        #return np.hypot(x_obs, y_obs), np.arctan2(y_obs, x_obs), z_obs

    def _get_diskframe_coords(self, extend=2.0, oversample=0.5):
        """Disk-frame coordinates based on the cube axes."""
        x_disk = np.linspace(extend * self.xaxis[0], extend * self.xaxis[-1],
                             int(self.nxpix * oversample))[::-1]
        y_disk = np.linspace(extend * self.yaxis[0], extend * self.yaxis[-1],
                             int(self.nypix * oversample))
        x_disk, y_disk = np.meshgrid(x_disk, y_disk)
        r_disk = np.hypot(x_disk, y_disk)
        t_disk = np.arctan2(y_disk, x_disk)
        return x_disk, y_disk, r_disk, t_disk

    # -- Functions to build Keplerian rotation profiles. -- #

    def _keplerian(self, x0=0.0, y0=0.0, inc=0.0, PA=0.0, z0=0.0, psi=0.0,
                   z1=0.0, phi=1.0, w_i=0.0, w_r=1.0, w_t=0.0, mstar=1.0,
                   dist=100., vlsr=0.0, shadowed=False, **kwargs):
        """
        Return a Keplerian rotation profile (not including pressure) in [m/s].
        This includes the deivation due to non-zero heights above the midplane,
        see the documents in disk_coords() for more details of the parameters.

        Args:
            x0 (Optional[float]): Source right ascension offset [arcsec].
            y0 (Optional[float]): Source declination offset [arcsec].
            inc (Optional[float]): Source inclination [degrees].
            PA (Optional[float]): Source position angle [degrees]. Measured
                between north and the red-shifted semi-major axis in an
                easterly direction.
            z0 (Optional[float]): Aspect ratio at 1" for the emission surface.
                To get the far side of the disk, make this number negative.
            psi (Optional[float]): Flaring angle for the emission surface.
            z1 (Optional[float]): Aspect ratio correction term at 1" for the
                emission surface. In general, should be opposite sign to z0.
            phi (Optional[float]): Flaring angle correction term for the
                emission surface.
            w_i (Optional[float]): Warp inclination in [degrees] at the disk
                center.
            w_r (Optional[float]): Scale radius of the warp in [arcsec].
            w_t (Optional[float]): Angle of nodes of the warp in [degrees].
            mstar (Optional[float]): Mass of the star in [Msun].
            dist (Optional[float]): Distance to the source in [parsec].
            vlsr (Optional[floar]): Systemic velocity in [m/s].
            shadowed (Optional[bool]): If true, use a more robust, however
                slower deprojection routine which accurately takes into account
                shadowing at high inclinations.

        Returns:
            vproj (ndarray): Projected Keplerian rotation at each pixel (m/s).
        """
        coords = self.disk_coords(x0=x0, y0=y0, inc=inc, PA=PA, z0=z0, psi=psi,
                                  z1=z1, phi=phi, w_i=w_i, w_r=w_r, w_t=w_t,
                                  shadowed=shadowed, frame='cylindrical')
        rvals = coords[0] * sc.au * dist
        zvals = coords[2] * sc.au * dist
        vkep = sc.G * mstar * self.msun * np.power(rvals, 2.0)
        vkep = np.sqrt(vkep * np.power(np.hypot(rvals, zvals), -3.0))

        # Include a radially changing inclination to account for the warp.
        inc += w_i * np.exp(-0.5 * coords[0]**2 * w_r**-2)
        return vkep * abs(np.sin(np.radians(inc))) * np.cos(coords[1]) + vlsr

    def _make_model(self, params):
        """Build the Keplerian model from the dictionary of parameters."""
        vkep = self._keplerian(x0=params['x0'], y0=params['y0'],
                               inc=params['inc'], PA=params['PA'],
                               vlsr=params['vlsr'], z0=params['z0'],
                               psi=params['psi'], z1=params['z1'],
                               phi=params['phi'], w_i=params['w_i'],
                               w_r=params['w_r'], w_t=params['w_t'],
                               mstar=params['mstar'], dist=params['dist'],
                               shadowed=params.pop('shadowed', False))
        if params.pop('beam', False):
            vkep = rotationmap._convolve_image(vkep, self._beamkernel())
        return vkep

    # -- Functions to help determine the emission height. -- #

    def _get_peak_pix(self, PA, inc=None, x0=None, y0=None, frame='sky',
                      ycut=20.0):
        """
        Get the coordinates of the peak velocities. This allows you to quickly
        see if the disk is flared or have an idea of the emssision surface.

        Args:
            PA (float): Position angle of the disk in [degrees]. This is
            measured from North to the red-shifted major axis Eastwards.
            inc (Optional[float]): Disk inclination in [degrees], needed to
                deproject into true height values.
            frame (Optional[str]): Frame for the returned coordinates. If
                frame='sky', return x- and y-offset values. Otherwise use the
                rotated disk coordiantes.
            ycut (Optional[float]): Only use the `ycut` fraction of the minor
                axis for fitting.

        Returns [if frame='sky']:
            x (ndarray): x-positions of the peak values [arcsec].
            y (ndarray): y-positions of the peak values [arcsec].

        [if frame='disk']:
            r (ndarray): Radial offset in [arcsec].
            z (ndarray): Height in [arcsec].
        """

        if frame not in ['sky', 'disk']:
            raise ValueError("'frame' must be 'sky' or 'disk'.")
        if frame == 'disk' and inc is None:
            raise ValueError("Must provide disk inclination to deproject.")

        mask = np.percentile(self.yaxis, [50 + ycut / 2.0])
        mask = abs(self.yaxis)[:, None] > mask

        if x0 is not None or y0 is not None:
            data = self.shift_center(dx0=x0, dy0=y0, save=False)
        else:
            data = self.data.copy()
        x0 = 0.0 if x0 is None else x0
        y0 = 0.0 if y0 is None else y0

        data = self.rotate_image(PA, data=data, save=False)
        data = np.where(mask, np.nan, data)

        x = self.xaxis.copy()
        ymax = np.take(self.yaxis, np.nanargmax(data, axis=0))
        ymin = np.take(self.yaxis, np.nanargmin(data, axis=0))
        y = np.where(x >= 0, ymax, ymin)

        if frame == 'sky':
            # TODO: Fix this mess.
            if PA <= 180.0:
                x, y = rotationmap._rotate_coords(x, y, 90. - PA)
            else:
                x, y = rotationmap._rotate_coords(y, x, PA - 90.)
            return x + x0, y + y0
        idxs = np.argsort(abs(x))
        return abs(x)[idxs], y[idxs] / np.sin(np.radians(inc))

    def _fit_surface(self, inc, PA, x0=None, y0=None, r_min=None, r_max=None,
                     fit_z1=False, ycut=20.0, **kwargs):
        """
        Fit the emission surface with the parametric model based on the pixels
        of peak velocity.

        Args:
            inc (float): Disk inclination in [degrees].
            PA (float): Disk position angle in [degrees].
            x0 (Optional[float]): Disk center x-offset in [arcsec].
            y0 (Optional[float]): Disk center y-offset in [arcsec].
            r_min (Optional[float]): Minimum radius to fit in [arcsec].
            r_max (Optional[float]): Maximum radius to fit in [arcsec].
            fit_z1 (Optional[bool]): Include (z1, phi) in the fit.
            y_cut (Optional[float]): Only use the `ycut` fraction of the minor
                axis for fitting.
            kwargs (Optional[dict]): Additional kwargs for curve_fit.

        Returns:
            popt (list): Best-fit parameters of (z0, psi[, z1, phi]).
        """

        from scipy.optimize import curve_fit
        def z_func(x, z0, psi, z1=0.0, phi=1.0):
            return z0 * x**psi + z1 * x**phi

        # Get the coordinates to fit.
        r, z = self.get_peak_pix(PA=PA, inc=inc, x0=x0, y0=y0,
                                 frame='disk', ycut=ycut)

        # Mask the coordinates.
        r_min = r_min if r_min is not None else r[0]
        r_max = r_max if r_max is not None else r[-1]
        m1 = np.logical_and(np.isfinite(r), np.isfinite(z))
        m2 = (r >= r_min) & (r <= r_max)
        mask = m1 & m2
        r, z = r[mask], z[mask]

        # Initial guesses for the free params.
        p0 = [z[abs(r - 1.0).argmin()], 1.0]
        if fit_z1:
            p0 += [-0.05, 3.0]
        maxfev = kwargs.pop('maxfev', 100000)

        # First fit the inner half to get a better p0 value.
        mask = r <= 0.5 * r.max()
        p_in = curve_fit(z_func, r[mask], z[mask], p0=p0, maxfev=maxfev,
                         **kwargs)[0]

        # Remove the pixels which are 'negative' depending on the inclination.
        mask = z > 0.0 if np.sign(p_in[0]) else z < 0.0
        popt = curve_fit(z_func, r[mask], z[mask], p0=p_in, maxfev=maxfev,
                         **kwargs)[0]
        return popt

    # -- Axes Functions -- #

    def _clip_cube(self, radius):
        """Clip the cube to +/- radius arcseconds from the origin."""
        xa = abs(self.xaxis - radius).argmin()
        if self.xaxis[xa] < radius:
            xa -= 1
        xb = abs(self.xaxis + radius).argmin()
        if -self.xaxis[xb] < radius:
            xb += 1
        xb += 1
        ya = abs(self.yaxis + radius).argmin()
        if -self.yaxis[ya] < radius:
            ya -= 1
        yb = abs(self.yaxis - radius).argmin()
        if self.yaxis[yb] < radius:
            yb += 1
        yb += 1
        self.xaxis = self.xaxis[xa:xb]
        self.yaxis = self.yaxis[ya:yb]
        self.data = self.data[ya:yb, xa:xb]
        self.error = self.error[ya:yb, xa:xb]

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

    def _shift_center(self, dx=0.0, dy=0.0, data=None, save=True):
        """
        Shift the center of the image.

        Args:
            dx (optional[float]): shift along x-axis [arcsec].
            dy (optional[float]): Shift along y-axis [arcsec].
            data (optional[ndarray]): Data to shift. If nothing is provided,
                will shift the attached ``rotationmap.data``.
            save (optional[bool]): If True, overwrite ``rotationmap.data``.
        """
        from scipy.ndimage import shift
        data = self.data.copy() if data is None else data
        to_shift = np.where(np.isfinite(data), data, 0.0)
        data = shift(to_shift, [-dy / self.dpix, dx / self.dpix])
        if save:
            self.data = data
        return data

    def _rotate_image(self, PA, data=None, save=True):
        """
        Rotate the image anticlockwise about the center.

        Args:
            PA (float): Rotation angle in [degrees].
            data (optional[ndarray]): Data to shift. If nothing is provided,
                will shift the attached ``rotationmap.data``.
            save (optional[bool]): If True, overwrite ``rotationmap.data``.
        """
        from scipy.ndimage import rotate
        data = self.data.copy() if data is None else data
        to_rotate = np.where(np.isfinite(data), data, 0.0)
        data = rotate(to_rotate, PA - 90.0, reshape=False)
        if save:
            self.data = data
        return data

    # -- Convolution functions. -- #

    def _readbeam(self):
        """Reads the beam properties from the header."""
        try:
            if self.header.get('CASAMBM', False):
                beam = fits.open(self.path)[1].data
                beam = np.median([b[:3] for b in beam.view()], axis=0)
                self.bmaj, self.bmin, self.bpa = beam
            else:
                self.bmaj = self.header['bmaj'] * 3600.
                self.bmin = self.header['bmin'] * 3600.
                self.bpa = self.header['bpa']
        except:
            self.bmaj = self.dpix
            self.bmin = self.dpix
            self.bpa = 0.0
            self.beamarea = self.dpix**2.0

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

    @staticmethod
    def colormap():
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        c2 = plt.cm.Reds(np.linspace(0, 1, 32))
        c1 = plt.cm.Blues_r(np.linspace(0, 1, 32))
        c1 = np.vstack([c1, [1, 1, 1, 1]])
        colors = np.vstack((c1, c2))
        return mcolors.LinearSegmentedColormap.from_list('eddymap', colors)

    @property
    def extent(self):
        return [self.xaxis[0], self.xaxis[-1], self.yaxis[0], self.yaxis[-1]]

    def _plot_bestfit(self, params, ivar=None, residual=False, return_ax=False):
        """Plot the best-fit model."""
        import matplotlib.pyplot as plt
        ax = plt.subplots()[1]
        vkep = self._make_model(params) * 1e-3
        levels = np.where(self.ivar != 0.0, vkep, np.nan)
        levels = np.nanpercentile(levels, [2, 98])
        levels = np.linspace(levels[0], levels[1], 30)
        im = ax.contourf(self.xaxis, self.yaxis, vkep, levels,
                         cmap=rotationmap.colormap(), extend='both')
        if ivar is not None:
            ax.contour(self.xaxis, self.yaxis, ivar, [0], colors='k')
        cb = plt.colorbar(im, pad=0.02, format='%.2f')
        cb.set_label(r'${\rm v_{mod} \quad (km\,s^{-1})}$',
                     rotation=270, labelpad=15)
        self._gentrify_plot(ax)
        if return_ax:
            return ax

    def _plot_residual(self, params, ivar=None, return_ax=False):
        """Plot the residual from the provided model."""
        import matplotlib.cm as cm
        import matplotlib.pyplot as plt
        ax = plt.subplots()[1]
        vres = self.data * 1e3 - self._make_model(params)
        levels = np.where(self.ivar != 0.0, vres, np.nan)
        levels = np.nanpercentile(levels, [2, 98])
        levels = max(abs(levels[0]), abs(levels[1]))
        levels = np.linspace(-levels, levels, 30)
        im = ax.contourf(self.xaxis, self.yaxis, vres, levels,
                         cmap=cm.RdBu_r, extend='both')
        if ivar is not None:
            ax.contour(self.xaxis, self.yaxis, ivar, [0], colors='k')
        cb = plt.colorbar(im, pad=0.02, format='%d')
        cb.set_label(r'${\rm  v_{0} - v_{mod} \quad (m\,s^{-1})}$',
                     rotation=270, labelpad=15)
        self._gentrify_plot(ax)
        if return_ax:
            return ax

    def plot_surface(self, ax=None, x0=0.0, y0=0.0, inc=0.0, PA=0.0, z0=0.0,
                     psi=0.0, z1=0.0, phi=1.0, w_i=0.0, w_r=1.0, w_t=0.0,
                     r_min=0.0, r_max=None, ntheta=16, nrad=10,
                     check_mask=True, **kwargs):
        """
        Overplot the emission surface onto the provided axis.

        Args:
            ax (Optional[AxesSubplot]): Axis to plot onto.
            x0 (Optional[float]): Source right ascension offset [arcsec].
            y0 (Optional[float]): Source declination offset [arcsec].
            inc (Optional[float]): Source inclination [deg].
            PA (Optional[float]): Source position angle [deg]. Measured
                between north and the red-shifted semi-major axis in an
                easterly direction.
            z0 (Optional[float]): Aspect ratio at 1" for the emission surface.
                To get the far side of the disk, make this number negative.
            psi (Optional[float]): Flaring angle for the emission surface.
            z1 (Optional[float]): Aspect ratio correction term at 1" for the
                emission surface. Should be opposite sign to z0.
            phi (Optional[float]): Flaring angle correction term for the
                emission surface.
            w_i (Optional[float]): Warp inclination in [degrees] at the disk
                center.
            w_r (Optional[float]): Scale radius of the warp in [arcsec].
            w_t (Optional[float]): Angle of nodes of the warp in [degrees].
            r_min (Optional[float]): Inner radius to plot, default is 0.
            r_max (Optional[float]): Outer radius to plot.
            ntheta (Optional[int]): Number of theta contours to plot.
            nrad (Optional[int]): Number of radial contours to plot.
            check_mask (Optional[bool]): Mask regions which are like projection
                errors for highly flared surfaces.

        Returns:
            ax (AxesSubplot): Axis with the contours overplotted.
        """

        # Dummy axis to overplot.
        if ax is None:
            import matplotlib.pyplot as plt
            ax = plt.subplots()[1]

        # Front half of the disk.
        rf, tf, zf = self.disk_coords(x0=x0, y0=y0, inc=inc, PA=PA, z0=z0,
                                      psi=psi, z1=z1, phi=phi)
        rf = np.where(zf >= 0.0, rf, np.nan)
        tf = np.where(zf >= 0.0, tf, np.nan)

        # Rear half of the disk.
        rb, tb, zb = self.disk_coords(x0=x0, y0=y0, inc=inc, PA=PA, z0=-z0,
                                      psi=psi, z1=-z1, phi=phi)
        rb = np.where(zb <= 0.0, rb, np.nan)
        tb = np.where(zb <= 0.0, tb, np.nan)

        # Flat disk for masking.
        rr, _, _ = self.disk_coords(x0=x0, y0=y0, inc=inc, PA=PA)

        # Make sure the bounds are OK.
        r_min = 0.0 if r_min is None else r_min
        r_max = rr.max() if r_max is None else r_max

        # Make sure the front side hides the rear.
        mf = np.logical_and(rf >= r_min, rf <= r_max)
        mb = np.logical_and(rb >= r_min, rb <= r_max)
        tf = np.where(mf, tf, np.nan)
        rb = np.where(~mf, rb, np.nan)
        tb = np.where(np.logical_and(np.isfinite(rb), mb), tb, np.nan)

        # For some geometries we want to make sure they're not doing funky
        # things in the outer disk when psi is large.
        if check_mask:
            mm = rr <= check_mask * r_max
            rf = np.where(mm, rf, np.nan)
            rb = np.where(mm, rb, np.nan)
            tf = np.where(mm, tf, np.nan)
            tb = np.where(mm, tb, np.nan)

        # Popluate the kwargs with defaults.
        lw = kwargs.pop('lw', kwargs.pop('linewidth', 0.5))
        zo = kwargs.pop('zorder', 10000)
        c = kwargs.pop('colors', kwargs.pop('c', 'k'))

        radii = np.linspace(0, r_max, int(nrad + 1))[1:]
        theta = np.linspace(-np.pi, np.pi, int(ntheta + 1))[:-1]

        # Do the plotting.
        ax.contour(self.xaxis, self.yaxis, rf, levels=radii, colors=c,
                   linewidths=lw, linestyles='-', zorder=zo, **kwargs)
        ax.contour(self.xaxis, self.yaxis, tf, levels=theta, colors=c,
                   linewidths=lw, linestyles='-', zorder=zo, **kwargs)
        ax.contour(self.xaxis, self.yaxis, rb, levels=radii, colors=c,
                   linewidths=lw, linestyles='--', zorder=zo, **kwargs)
        ax.contour(self.xaxis, self.yaxis, tb, levels=theta, colors=c,
                   linewidths=lw, linestyles='--', zorder=zo)
        return ax

    def _plot_axes(self, ax, x0=0., y0=0., inc=0., PA=0., major=1.0, **kwargs):
        """
        Plot the major and minor axes on the provided axis.

        Args:
            ax (Matplotlib axes): Axes instance to plot onto.
            x0 (optional[float]): Relative x-location of the center [arcsec].
            y0 (optional[float]): Relative y-location of the center [arcsec].
            inc (optional[float]): Inclination of the disk in [degrees].
            PA (optional[float]): Position angle of the disk in [degrees].
            major (optional[float]): Size of the major axis line in [arcsec].
        """

        # Default parameters plotting values.
        ls = kwargs.pop('ls', kwargs.pop('linestyle', '--'))
        lw = kwargs.pop('lw', kwargs.pop('linewidth', 0.5))
        lc = kwargs.pop('c', kwargs.pop('color', 'k'))
        zo = kwargs.pop('zorder', -2)

        # Plotting.
        PA = np.radians(PA)
        ax.plot([major * np.cos(PA) - x0, major * np.cos(PA + np.pi) - x0],
                [major * np.sin(PA) - y0, major * np.sin(PA + np.pi) - y0],
                ls=ls, lw=lw, color=lc, zorder=zo)
        minor = major * np.cos(np.radians(inc))
        ax.plot([minor * np.cos(PA + 0.5 * np.pi) - x0,
                 minor * np.cos(PA + 1.5 * np.pi) - x0],
                [minor * np.sin(PA + 0.5 * np.pi) - y0,
                 minor * np.sin(PA + 1.5 * np.pi) - y0],
                ls=ls, lw=lw, color=lc, zorder=zo)

    def _plot_maxima(self, x0=0.0, y0=0.0, inc=0.0, PA=0.0, r_max=None,
                    plot_axes_kwargs=None, scatter_kwargs=None,
                    return_ax=False):
        """
        Mark the position of the maximum velocity as a function of radius. This
        can help demonstrate if there is an appreciable bend in velocity
        contours indicative of a flared emission surface. This function pulls
        the results from fit_cube.get_peak_pix() for plotting.

        Args:
            x0 (Optional[float]): Source center x-offset in [arcsec].
            y0 (Optional[float]): Source center y-offset in [arcsec].
            inc (Optional[float]): Disk inclination in [degrees].
            PA (Optional[float]): Disk position angle in [degrees].
            r_max (Optional[float]): Maximum offset along the major axis to
                plot in [arcsec]. This can never be larger than the span of the
                x-axis.
            plot_axes_kwargs (Optional[dict]): Dictionary of kwargs for the
                plot_axes function.
            scatter_kwargs (Optional[dict]): Dictionary of kwargs for the
                scatter plot of the maximum pixels.
            return_ax (Optional[bool]): If True, return the axis.

        Returns:
            ax (Matplotlib axis): If return_ax is True.
        """

        # Background figure.
        ax = self.plot_data(return_ax=True)
        if plot_axes_kwargs is None:
            plot_axes_kwargs = dict()
        self.plot_axes(ax=ax, x0=x0, y0=y0, inc=inc, PA=PA,
                       major=r_max if r_max is not None else 1.0,
                       **plot_axes_kwargs)

        # Get the pixels.
        x, y = self.get_peak_pix(PA=PA, inc=inc, x0=x0, y0=y0, frame='sky')
        r = np.hypot(x, y)
        mask = r <= r_max
        x, y, r = x[mask], y[mask], r[mask]

        # Scatter plot of the pixels.
        if scatter_kwargs is None:
            scatter_kwargs = dict()
        cm = scatter_kwargs.pop('cmap', 'bone_r')
        ms = scatter_kwargs.pop('s', scatter_kwargs.pop('size', 5))
        ec = scatter_kwargs.pop('ec', scatter_kwargs.pop('edgecolor', 'k'))
        lw = scatter_kwargs.pop('lw', scatter_kwargs.pop('linewidth', 1.25))
        if lw != 0.0:
            ax.scatter(x, y, c=np.hypot(x, y), s=ms, lw=lw, cmap=cm,
                       edgecolor=ec, vmin=0.0, vmax=r.max(), **scatter_kwargs)
        ax.scatter(x, y, c=np.hypot(x, y), s=ms, lw=0., cmap=cm, edgecolor=ec,
                   vmin=0.0, vmax=r.max(), **scatter_kwargs)
        if return_ax:
            return ax

    @staticmethod
    def _plot_walkers(samples, nburnin=None, labels=None, histogram=True):
        """Plot the walkers to check if they are burning in."""

        # Import matplotlib.
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

        # Check the length of the label list.
        if labels is None:
            if samples.shape[0] != len(labels):
                raise ValueError("Not correct number of labels.")

        # Cycle through the plots.
        for s, sample in enumerate(samples):
            fig, ax = plt.subplots()
            for walker in sample.T:
                ax.plot(walker, alpha=0.1, color='k')
            ax.set_xlabel('Steps')
            if labels is not None:
                ax.set_ylabel(labels[s])
            if nburnin is not None:
                ax.axvline(nburnin, ls=':', color='r')

            # Include the histogram.
            if histogram:
                fig.set_size_inches(1.37 * fig.get_figwidth(),
                                    fig.get_figheight(), forward=True)
                ax_divider = make_axes_locatable(ax)
                bins = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 50)
                hist, _ = np.histogram(sample[nburnin:].flatten(), bins=bins,
                                       density=True)
                bins = np.average([bins[1:], bins[:-1]], axis=0)
                ax1 = ax_divider.append_axes("right", size="35%", pad="2%")
                ax1.fill_betweenx(bins, hist, np.zeros(bins.size), step='mid',
                                  color='darkgray', lw=0.0)
                ax1.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1])
                ax1.set_xlim(0, ax1.get_xlim()[1])
                ax1.set_yticklabels([])
                ax1.set_xticklabels([])
                ax1.tick_params(which='both', left=0, bottom=0, top=0, right=0)
                ax1.spines['right'].set_visible(False)
                ax1.spines['bottom'].set_visible(False)
                ax1.spines['top'].set_visible(False)

    @staticmethod
    def _plot_corner(samples, labels=None, quantiles=None):
        """Plot the corner plot to check for covariances."""
        import corner
        quantiles = [0.16, 0.5, 0.84] if quantiles is None else quantiles
        corner.corner(samples, labels=labels, title_fmt='.4f', bins=30,
                      quantiles=quantiles, show_titles=True)

    def _plot_beam(self, ax, dx=0.125, dy=0.125, **kwargs):
        """Plot the sythensized beam on the provided axes."""
        from matplotlib.patches import Ellipse
        beam = Ellipse(ax.transLimits.inverted().transform((dx, dy)),
                       width=self.bmin, height=self.bmaj, angle=-self.bpa,
                       fill=kwargs.get('fill', False),
                       hatch=kwargs.get('hatch', '//////////'),
                       lw=kwargs.get('linewidth', kwargs.get('lw', 1)),
                       color=kwargs.get('color', kwargs.get('c', 'k')),
                       zorder=kwargs.get('zorder', 1000))
        ax.add_patch(beam)

    def _gentrify_plot(self, ax):
        """Gentrify the plot."""
        from matplotlib.ticker import MaxNLocator
        ax.set_aspect(1)
        ax.grid(ls=':', color='k', alpha=0.1, lw=0.5)
        ax.tick_params(which='both', right=True, top=True)
        ax.set_xlim(self.xaxis.max(), self.xaxis.min())
        ax.set_ylim(self.yaxis.min(), self.yaxis.max())
        ax.xaxis.set_major_locator(MaxNLocator(5, min_n_ticks=3))
        ax.yaxis.set_major_locator(MaxNLocator(5, min_n_ticks=3))
        ax.set_xlabel('Offset (arcsec)')
        ax.set_ylabel('Offset (arcsec)')
        if self.bmaj is not None:
            self._plot_beam(ax=ax)
