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
            data. If nothing is specified, will tried to find an uncerainty
            file following the ``bettermoments`` format. If this is not found,
            it will assume a 10% on all pixels.
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
        self.path = path
        self.data = np.squeeze(fits.getdata(path))
        self.header = fits.getheader(path)
        if uncertainty is not None:
            self.error = abs(np.squeeze(fits.getdata(uncertainty)))
        else:
            try:
                uncertainty = '_'.join(self.path.split('_')[:-1])
                uncertainty += '_d' + self.path.split('_')[-1]
                print("Assuming uncertainties in {}".format(uncertainty))
                self.error = abs(np.squeeze(fits.getdata(uncertainty)))
            except FileNotFoundError:
                print("No uncertainties found, assuming uncertainties of 10%.")
                print("Change this at any time with rotationmap.error.")
                self.error = 0.1 * abs((self.data - np.nanmedian(self.data)))
        self.error = np.where(np.isnan(self.error), 0.0, self.error)

        # Convert the data to [km/s].
        if unit.lower() == 'm/s':
            self.data *= 1e-3
            self.error *= 1e-3
        elif unit.lower() != 'km/s':
            raise ValueError("unit must me 'm/s' or 'km/s'.")

        # Read the position axes.
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
                downsample = int(np.ceil(self.bmaj / self.dpix))
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

        # Set the defaults for 'shadowed' emission surfaces.
        self.shadowed = False
        self.shadowed_extend = 1.5
        self.shadowed_oversample = 2.0
        self.shadowed_method = 'nearest'

    def fit_map(self, p0, params, r_min=None, r_max=None, optimize=True,
                nwalkers=None, nburnin=300, nsteps=100, scatter=1e-3,
                plots=None, returns=None, pool=None, emcee_kwargs=None,
                niter=1, shadowed=False):
        """
        Fit a rotation profile to the data. Note that for a disk with
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

        .. _Rosenfeld et al. (2013): https://ui.adsabs.harvard.edu/abs/2013ApJ...774...16R/abstract

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
            nwalkers (optional[int]): Number of walkers to use for the MCMC.
            nburnin (optional[int]): Number of steps to discard for burn-in.
            nsteps (optional[int]): Number of steps to use to sample the
                posterior distributions.
            scatter (optional[float]): Scatter used in distributing walker
                starting positions around the initial ``p0`` values.
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
            niter (optional[int]): Number of iterations to perform using the
                median PDF values from the previous MCMC run as starting
                positions. This is probably only useful if you have no idea
                about the starting positions for the emission surface or if you
                want to remove walkers stuck in local minima.

        Returns:
            to_return (list): Depending on the returns list provided.
                ``'samples'`` will be all the samples of the posteriors (with
                the burn-in periods removed). ``'percentiles'`` will return the
                16th, 50th and 84th percentiles of each posterior functions.
                ``'dict'`` will return a dictionary of the median parameters
                which can be directly input to other functions.
        """

        # Check the dictionary. May need some more work.
        if r_min is not None:
            if 'r_min' in params.keys():
                print("Found `r_min` in `params`. Overwriting value.")
            params['r_min'] = r_min
        if r_max is not None:
            if 'r_max' in params.keys():
                print("Found `r_max` in `params`. Overwriting value.")
            params['r_max'] = r_max
        params = self.verify_params_dictionary(params)
        self.shadowed = shadowed

        # Generate the mask for fitting based on the params.
        p0 = np.squeeze(p0).astype(float)
        temp = rotationmap._populate_dictionary(p0, params)
        self.ivar = self._calc_ivar(temp)

        # Check what the parameters are.
        labels = rotationmap._get_labels(params)
        labels_raw = []
        for label in labels:
            label_raw = label.replace('$', '').replace('{', '')
            label_raw = label_raw.replace(r'\rm ', '').replace('}', '')
            labels_raw += [label_raw]
        if len(labels) != len(p0):
            raise ValueError("Mismatch in labels and p0. Check for integers.")
        print("Assuming:\n\tp0 = [%s]." % (', '.join(labels_raw)))

        # Run an initial optimization using scipy.minimize. Recalculate the
        # inverse variance mask.
        if optimize:
            p0 = self._optimize_p0(p0, params)

        # Make the mask for fitting.
        temp = rotationmap._populate_dictionary(p0, params)
        temp = self.verify_params_dictionary(temp)

        self.ivar = self._calc_ivar(temp)

        # Set up and run the MCMC with emcee.
        time.sleep(0.5)
        nwalkers = 2 * p0.size if nwalkers is None else nwalkers
        emcee_kwargs = {} if emcee_kwargs is None else emcee_kwargs
        emcee_kwargs['scatter'], emcee_kwargs['pool'] = scatter, pool
        for n in range(int(niter)):

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
            medians = self.verify_params_dictionary(medians)

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
            to_return += [np.percentile(samples, [16, 50, 84], axis=0)]
        if 'dict' in returns:
            to_return += [medians]
        self.shadowed = False
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
                    r_cavity=0.0, r_taper=None, q_taper=None, w_i=0.0, w_r=1.0,
                    w_t=0.0, frame='cylindrical', **_):
        r"""
        Get the disk coordinates given certain geometrical parameters and an
        emission surface. The emission surface is most simply described as a
        power law profile,

        .. math::

            z(r) = z_0 \times \left(\frac{r}{1^{\prime\prime}}\right)^{\psi}

        where ``z0`` and ``psi`` can be provided by the user. With the increase
        in spatial resolution afforded by interferometers such as ALMA there
        are a couple of modifications that can be used to provide a better
        match to the data.

        An inner cavity can be included with the ``r_cavity`` argument which
        makes the transformation:

        .. math::

            \tilde{r} = {\rm max}(0, r - r_{\rm cavity})

        Note that the inclusion of a cavity will mean that other parameters,
        such as ``z0``, would need to change as the radial axis has effectively
        been shifted.

        To account for the drop in emission surface in the outer disk where the
        gas surface density decreases there are two descriptions. The preferred
        way is to include an exponential taper to the power law profile,

        .. math::

            z_{\rm tapered}(r) = z(r) \times \exp\left( -\left[
            \frac{r}{r_{\rm taper}} \right]^{q_{\rm taper}} \right)

        where both ``r_taper`` and ``q_taper`` values must be set.

        We can also include a warp which is parameterized by,

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
            r_cavity (optional[float]): Outer radius of a cavity. Within this
                region the emission surface is taken to be zero.
            w_i (optional[float]): Warp inclination in [degrees] at the disk
                center.
            w_r (optional[float]): Scale radius of the warp in [arcsec].
            w_t (optional[float]): Angle of nodes of the warp in [degrees].
            frame (optional[str]): Frame of reference for the returned
                coordinates. Either ``'cartesian'`` or ``'cylindrical'``.

        Returns:
            array, array, array: Three coordinate arrays with ``(r, phi, z)``,
            in units of [arcsec], [radians], [arcsec], if
            ``frame='cylindrical'`` or ``(x, y, z)``, all in units of [arcsec]
            if ``frame='cartesian'``.
        """

        # Check the input variables.

        frame = frame.lower()
        if frame not in ['cylindrical', 'cartesian']:
            raise ValueError("frame must be 'cylindrical' or 'cartesian'.")

        # Apply the inclination concention.

        inc = inc if inc < 90.0 else inc - 180.0

        # Check that the necessary pairs are provided.

        msg = "Must specify either both or neither of `{}` and `{}`."
        if (r_taper is not None) != (q_taper is not None):
            raise ValueError(msg.format('r_taper', 'q_taper'))

        # Set the defaults.

        z0 = 0.0 if z0 is None else z0
        psi = 1.0 if psi is None else psi
        r_cavity = 0.0 if r_cavity is None else r_cavity
        r_taper = np.inf if r_taper is None else r_taper
        q_taper = 1.0 if q_taper is None else q_taper

        # Define the emission surface, z_func, and the warp function, w_func.

        def z_func(r_in):
            r = np.clip(r_in - r_cavity, a_min=0.0, a_max=None)
            z = z0 * np.power(r, psi) * np.exp(-np.power(r / r_taper, q_taper))
            return np.clip(z, a_min=0.0, a_max=None)

        def w_func(r_in, t):
            r = np.clip(r_in - r_cavity, a_min=0.0, a_max=None)
            warp = np.radians(w_i) * np.exp(-0.5 * (r / w_r)**2)
            return r * np.tan(warp * np.sin(t - np.radians(w_t)))

        # Calculate the pixel values.

        if self.shadowed:
            coords = self._get_shadowed_coords(x0, y0, inc, PA, z_func, w_func)
        else:
            coords = self._get_flared_coords(x0, y0, inc, PA, z_func, w_func)
        if frame == 'cylindrical':
            return coords
        r, t, z = coords
        return r * np.cos(t), r * np.sin(t), z

    def plot_data(self, levels=None, ivar=None, return_fig=False):
        """
        Plot the first moment map. By default will clip the velocity contours
        such that the velocities around the systemic velocity are highlighted.

        Args:
            levels (optional[list]): List of contour levels to use.
            ivar (optional[ndarray]): Inverse variances for each pixel. Will
                draw a solid contour around the regions with finite ``ivar``
                values and fill regions not considered.
            return_fig (optional[bool]): Return the figure.
        Returns:
            fig (Matplotlib figure): If ``return_fig`` is ``True``. Can access
                the axes through ``fig.axes`` for additional plotting.
        """
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        if levels is None:
            levels = np.nanpercentile(self.data, [2, 98]) - self.vlsr
            levels = max(abs(levels[0]), abs(levels[1]))
            levels = self.vlsr + np.linspace(-levels, levels, 30)
        im = ax.contourf(self.xaxis, self.yaxis, self.data, levels,
                         cmap=rotationmap.colormap(), extend='both', zorder=-9)
        cb = plt.colorbar(im, pad=0.03, format='%.2f')
        cb.minorticks_on()
        cb.set_label(r'${\rm v_{0} \quad (km\,s^{-1})}$',
                     rotation=270, labelpad=15)
        if ivar is not None:
            ax.contour(self.xaxis, self.yaxis, ivar,
                       [0.0], colors='k')
            ax.contourf(self.xaxis, self.yaxis, ivar,
                        [-1.0, 0.0], colors='k', alpha=0.5)
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
        options['maxiter'] = options.pop('maxiter', 10000)
        options['maxfun'] = options.pop('maxfun', 10000)
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
        sampler = emcee.EnsembleSampler(nwalkers, p0.shape[1],
                                        self._ln_probability,
                                        args=[params, np.nan],
                                        **kwargs)
        if emcee.__version__ >= '3':
            progress = kwargs.pop('progress', True)
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

        # Basic Geometry.
        self.set_prior('x0', [-0.5, 0.5], 'flat')
        self.set_prior('y0', [-0.5, 0.5], 'flat')
        self.set_prior('inc', [-90.0, 90.0], 'flat')
        self.set_prior('PA', [-360.0, 360.0], 'flat')
        self.set_prior('mstar', [0.1, 5.0], 'flat')
        self.set_prior('vlsr', [np.nanmin(self.data) * 1e3,
                                np.nanmax(self.data) * 1e3], 'flat')

        # Emission Surface
        self.set_prior('z0', [0.0, 5.0], 'flat')
        self.set_prior('psi', [0.0, 5.0], 'flat')
        self.set_prior('r_cavity', [0.0, 1e30], 'flat')
        self.set_prior('r_taper', [0.0, 1e30], 'flat')
        self.set_prior('q_taper', [0.0, 5.0], 'flat')

        # Warp
        self.set_prior('w_i', [-90.0, 90.0], 'flat')
        self.set_prior('w_r', [0.0, self.xaxis.max()], 'flat')
        self.set_prior('w_t', [-180.0, 180.0], 'flat')

        # Velocity Profile
        self.set_prior('vp_100', [0.0, 1e4], 'flat')
        self.set_prior('vp_q', [-2.0, 0.0], 'flat')
        self.set_prior('vp_rtaper', [0.0, 1e30], 'flat')
        self.set_prior('vp_qtaper', [0.0, 5.0], 'flat')
        self.set_prior('vr_100', [-1e3, 1e3], 'flat')
        self.set_prior('vr_q', [-2.0, 2.0], 'flat')

    def _ln_prior(self, params):
        """Log-priors."""
        lnp = 0.0
        for key in params.keys():
            if key in rotationmap.priors.keys():
                lnp += rotationmap.priors[key](params[key])
                if not np.isfinite(lnp):
                    return lnp
        return lnp

    def _calc_ivar(self, params):
        """Calculate the inverse variance including radius mask."""

        # Check the error array is the same shape as the data.
        try:
            assert self.error.shape == self.data.shape
        except AttributeError:
            self.error = self.error * np.ones(self.data.shape)

        # Deprojected coordinates.
        r, t = self.disk_coords(**params)[:2]
        t = abs(t) if params['abs_PA'] else t

        # Radial mask.
        mask_r = np.logical_and(r >= params['r_min'], r <= params['r_max'])
        mask_r = ~mask_r if params['exclude_r'] else mask_r

        # Azimuthal mask.
        mask_t = np.logical_and(t >= params['PA_min'], t <= params['PA_max'])
        mask_t = ~mask_t if params['exclude_PA'] else mask_t

        # Finite value mask.
        mask_f = np.logical_and(np.isfinite(self.data), self.error > 0.0)

        # Velocity mask.
        v_min, v_max = params['v_min'], params['v_max']
        mask_v = np.logical_and(self.data >= v_min, self.data <= v_max)
        mask_v = ~mask_v if params['exclude_v'] else mask_v

        # Combine.
        mask = np.logical_and(np.logical_and(mask_v, mask_f),
                              np.logical_and(mask_r, mask_t))
        return np.where(mask, np.power(self.error, -2.0), 0.0)

    @staticmethod
    def _get_labels(params):
        """Return the labels of the parameters to fit."""
        idxs, labs = [], []
        for k in params.keys():
            if isinstance(params[k], int):
                if not isinstance(params[k], bool):
                    idxs.append(params[k])
                    try:
                        idx = k.index('_') + 1
                        label = k[:idx] + '{{' + k[idx:] + '}}'
                    except ValueError:
                        label = k
                    label = r'${{\rm {}}}$'.format(label)
                    labs.append(label)
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

    def verify_params_dictionary(self, params):
        """
        Check that the minimum number of parameters are provided for the
        fitting. For non-essential parameters a default value is set.
        """

        # Basic geometrical properties.
        params['x0'] = params.pop('x0', 0.0)
        params['y0'] = params.pop('y0', 0.0)
        params['dist'] = params.pop('dist', 100.0)
        params['vlsr'] = params.pop('vlsr', self.vlsr)
        if params.get('inc', None) is None:
            raise KeyError("Must provide `'inc'`.")
        if params.get('PA', None) is None:
            raise KeyError("Must provide `'PA'`.")

        # Rotation profile.
        has_vp_100 = params.get('vp_100', False)
        has_mstar = params.get('mstar', False)
        if has_mstar and has_vp_100:
            raise KeyError("Only provide either `'mstar'` or `'vp_100'`.")
        if not has_mstar and not has_vp_100:
            raise KeyError("Must provide either `'mstar'` or `'vp_100'`.")
        if has_mstar:
            params['vfunc'] = self._proj_vkep
        else:
            params['vfunc'] = self._proj_vpow
        params['vp_q'] = params.pop('vp_q', -0.5)
        params['vp_rtaper'] = params.pop('vp_rtaper', 1e10)
        params['vp_qtaper'] = params.pop('vp_qtaper', 1.0)
        params['vr_100'] = params.pop('vr_100', 0.0)
        params['vr_q'] = params.pop('vr_q', 0.0)

        # Flared emission surface.
        params['z0'] = params.pop('z0', 0.0)
        params['psi'] = params.pop('psi', 1.0)
        params['r_taper'] = params.pop('r_taper', 1e10)
        params['q_taper'] = params.pop('q_taper', 1.0)
        params['r_cavity'] = params.pop('r_cavity', 0.0)

        # Warp parameters.
        params['w_i'] = params.pop('w_i', 0.0)
        params['w_r'] = params.pop('w_r', 1.0)
        params['w_t'] = params.pop('w_t', 0.0)
        params['shadowed'] = params.pop('shadowed', False)

        # Masking parameters.
        params['r_min'] = params.pop('r_min', 0.0)
        params['r_max'] = params.pop('r_max', 1e10)
        params['exclude_r'] = params.pop('exclude_r', False)
        params['PA_min'] = params.pop('PA_min', -np.pi)
        params['PA_max'] = params.pop('PA_max', np.pi)
        params['exclude_PA'] = params.pop('exclude_PA', False)
        params['abs_PA'] = params.pop('abs_PA', False)
        params['v_min'] = params.pop('v_min', np.nanmin(self.data))
        params['v_max'] = params.pop('v_max', np.nanmax(self.data))
        params['exclude_v'] = params.pop('exclude_v', False)

        if params['r_min'] >= params['r_max']:
            raise ValueError("`r_max` must be greater than `r_min`.")
        if params['PA_min'] >= params['PA_max']:
            raise ValueError("`PA_max` must be great than `PA_min`.")
        if params['v_min'] >= params['v_max']:
            raise ValueError("`v_max` must be greater than `v_min`.")

        # Beam convolution.
        params['beam'] = False

        return params

    def evaluate_models(self, samples, params, draws=0.5,
                        collapse_func=np.mean, coords_only=False):
        """
        Evaluate models based on the samples provided and the parameter
        dictionary. If ``draws`` is an integer, it represents the number of
        random draws from ``samples`` which are then averaged to return the
        model. Alternatively ``draws`` can be a float between 0 and 1,
        representing the percentile of the samples to use to evaluate the model
        at.

        Args:
            samples (ndarray): An array of samples returned from ``fit_map``.
            params (dict): The parameter dictionary passed to ``fit_map``.
            draws (Optional[int/float]): If an integer, describes the number of
                random draws averaged to form the returned model. If a float,
                represents the percentile used from the samples. Must be
                between 0 and 1 if a float.
            collapse_func (Optional[callable]): How to collapse the random
                number of samples. Must be a function which allows an ``axis``
                argument (as with most Numpy functions).
            coords_only (Optional[bool]): Return the deprojected coordinates
                rather than the v0 model. Default is False.

        Returns:
            model (ndarray): The sampled model, either the v0 model, or, if
                ``coords_only`` is True, the deprojected cylindrical
                coordinates, (r, t, z).
        """

        # Check the input.
        nparam = np.sum([isinstance(params[k], int) for k in params.keys()])
        if samples.shape[1] != nparam:
            warning = "Invalid number of free parameters in 'samples': {:d}."
            raise ValueError(warning.format(nparam))
        if not callable(collapse_func):
            raise ValueError("'collapse_func' must be callable.")
        verified_params = self.verify_params_dictionary(params.copy())

        # Avearge over a random draw of models.
        if isinstance(draws, int):
            models = []
            for idx in np.random.randint(0, samples.shape[0], draws):
                tmp = self._populate_dictionary(samples[idx], verified_params)
                if coords_only:
                    models += [self.disk_coords(**tmp)]
                else:
                    models += [self._make_model(tmp)]
            return collapse_func(models, axis=0)

        # Take a percentile of the samples.
        elif isinstance(draws, float):
            tmp = np.percentile(samples, draws, axis=0)
            tmp = self._populate_dictionary(tmp, verified_params)
            if coords_only:
                return self.disk_coords(**tmp)
            else:
                return self._make_model(tmp)

        else:
            raise ValueError("'draws' must be a float or integer.")

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
        """Return cartesian sky coordinates in [arcsec, arcsec]."""
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

    def _get_shadowed_coords(self, x0, y0, inc, PA, z_func, w_func):
        """Return cyclindrical coords of surface in [arcsec, rad, arcsec]."""

        # Make the disk-frame coordinates.
        extend = self.shadowed_extend
        oversample = self.shadowed_oversample
        diskframe_coords = self._get_diskframe_coords(extend, oversample)
        xdisk, ydisk, rdisk, tdisk = diskframe_coords
        zdisk = z_func(rdisk) + w_func(rdisk, tdisk)

        # Incline the disk.
        inc = np.radians(inc)
        x_dep = xdisk
        y_dep = ydisk * np.cos(inc) - zdisk * np.sin(inc)

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
        r_obs = griddata(disk, rdisk.flatten(), grid,
                         method=self.shadowed_method)
        t_obs = griddata(disk, tdisk.flatten(), grid,
                         method=self.shadowed_method)
        return r_obs, t_obs, z_func(r_obs)

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

    # -- Functions to build projected velocity profiles. -- #

    def _vphi(self, params):
        """
        Return a rotation profile in [m/s]. If ``mstar`` is set, assume a
        Keplerian profile including corrections for non-zero emission heights.
        Otherwise, assume a powerlaw profile. See the documents in
        ``disk_coords`` for more details of the parameters.

        Args:
            params (dict): Dictionary of parameters describing the model.

        Returns:
            vproj (ndarray): Projected Keplerian rotation at each pixel (m/s).
        """
        rvals, tvals, zvals = self.disk_coords(**params)
        v_phi = params['vfunc'](rvals, tvals, zvals, params)
        return v_phi + params['vlsr']

    def _proj_vkep(self, rvals, tvals, zvals, params):
        """Projected Keplerian rotational velocity profile."""
        rvals *= sc.au * params['dist']
        zvals *= sc.au * params['dist']
        v_phi = sc.G * params['mstar'] * self.msun * np.power(rvals, 2)
        v_phi = np.sqrt(v_phi * np.power(np.hypot(rvals, zvals), -3))
        return self._proj_vphi(v_phi, tvals, params)

    def _proj_vpow(self, rvals, tvals, zvals, params):
        """Projected power-law rotational velocity profile."""
        v_phi = (rvals * params['dist'] / 100.)**params['vp_q']
        v_phi *= np.exp(-(rvals / params['vp_rtaper'])**params['vp_qtaper'])
        v_phi = self._proj_vphi(params['vp_100'] * v_phi, tvals, params)
        v_rad = (rvals * params['dist'] / 100.)**params['vr_q']
        v_rad = self._proj_vrad(params['vr_100'] * v_rad, tvals, params)
        return v_phi + v_rad

    def _proj_vphi(self, v_phi, tvals, params):
        """Project the rotational velocity."""
        return v_phi * np.cos(tvals) * abs(np.sin(np.radians(params['inc'])))

    def _proj_vrad(self, v_rad, tvals, params):
        """Project the radial velocity."""
        return v_rad * np.sin(tvals) * abs(np.sin(np.radians(params['inc'])))

    def _make_model(self, params):
        """Build the velocity model from the dictionary of parameters."""
        v_phi = self._vphi(params)
        if params['beam']:
            v_phi = rotationmap._convolve_image(v_phi, self._beamkernel())
        return v_phi

    # -- Functions to help determine the emission height. -- #

    def find_maxima(self, x0=0.0, y0=0.0, PA=0.0, vlsr=None, r_max=None,
                    r_min=None, smooth=False, through_center=True):
        """
        Find the node of velocity maxima along the major axis of the disk.

        Args:
            x0 (Optional[float]): Source center offset along x-axis in
                [arcsec].
            y0 (Optional[float]): Source center offset along y-axis in
                [arcsec].
            PA (Optioanl[float]): Source position angle in [deg].
            vlsr (Optional[float]): Systemic velocity in [m/s].
            r_max (Optional[float]): Maximum offset to consider in [arcsec].
            r_min (Optional[float]): Minimum offset to consider in [arcsec].
            smooth (Optional[bool/float]): Smooth the line of nodes. If
                ``True``, smoth with the beam kernel, otherwise ``smooth``
                describes the FWHM of the Gaussian convolution kernel in
                [arcsec].
            through_center (Optional[bool]): If ``True``, force the central
                pixel to go through ``(0, 0)``.

        Returns:
            array, array: Arrays of ``x_sky`` and ``y_sky`` values of the
            maxima.
        """

        # Default parameters.
        vlsr = np.nanmedian(self.data) if vlsr is None else vlsr
        r_max = 0.5 * self.xaxis.max() if r_max is None else r_max
        r_min = 0.0 if r_min is None else r_min

        # Shift and rotate the image.
        data = self._shift_center(dx=x0, dy=y0, save=False)
        data = self._rotate_image(PA=PA, data=data, save=False)

        # Find the maximum values. Apply some clipping to help.
        mask = np.maximum(0.3 * abs(self.xaxis), self.bmaj)
        mask = abs(self.yaxis)[:, None] > mask[None, :]
        resi = np.where(mask, 0.0, abs(data - vlsr))
        resi = np.take(self.yaxis, np.argmax(resi, axis=0))

        # Gentrification.
        if through_center:
            resi[abs(self.xaxis).argmin()] = 0.0
        if smooth:
            if isinstance(smooth, bool):
                kernel = np.hanning(self.bmaj / self.dpix)
                kernel /= kernel.sum()
            else:
                from astropy.convolution import Gaussian1DKernel
                kernel = Gaussian1DKernel(smooth / self.fwhm / self.dpix)
            resi = np.convolve(resi, kernel, mode='same')
        mask = np.logical_and(abs(self.xaxis) <= r_max,
                              abs(self.xaxis) >= r_min)
        x, y = self.xaxis[mask], resi[mask]

        # Rotate back to sky-plane and return.
        x, y = self._rotate_coords(x, -y, PA)
        return x + x0, y + y0

    def find_minima(self, x0=0.0, y0=0.0, PA=0.0, vlsr=None, r_max=None,
                    r_min=None, smooth=False, through_center=True):
        """
        Find the line of nodes where v0 = v_LSR along the minor axis.

        Args:
            x0 (Optional[float]): Source center offset along x-axis in
                [arcsec].
            y0 (Optional[float]): Source center offset along y-axis in
                [arcsec].
            PA (Optioanl[float]): Source position angle in [deg].
            vlsr (Optional[float]): Systemic velocity in [m/s].
            r_max (Optional[float]): Maximum offset to consider in [arcsec].
            r_min (Optional[float]): Minimum offset to consider in [arcsec].
            smooth (Optional[bool/float]): Smooth the line of nodes. If
                ``True``, smoth with the beam kernel, otherwise ``smooth``
                describes the FWHM of the Gaussian convolution kernel in
                [arcsec].
            through_center (Optional[bool]): If ``True``, force the central
                pixel to go through ``(0, 0)``.

        Returns:
            array, array: Arrays of ``x_sky`` and ``y_sky`` values of the
            velocity minima.
        """

        # Default parameters.
        vlsr = np.nanmedian(self.data) if vlsr is None else vlsr
        r_max = 0.5 * self.xaxis.max() if r_max is None else r_max
        r_min = 0.0 if r_min is None else r_min

        # Shift and rotate the image.
        data = self._shift_center(dx=x0, dy=y0, save=False)
        data = self._rotate_image(PA=PA, data=data, save=False)

        # Find the maximum values. Apply some clipping to help.
        mask = np.maximum(0.3 * abs(self.yaxis), self.bmaj)
        mask = abs(self.xaxis)[None, :] > mask[:, None]
        resi = np.where(mask, 1e10, abs(data - vlsr))
        resi = np.take(-self.yaxis, np.argmin(resi, axis=1))

        # Gentrification.
        if through_center:
            resi[abs(self.yaxis).argmin()] = 0.0
        if smooth:
            if isinstance(smooth, bool):
                kernel = np.hanning(self.bmaj / self.dpix)
                kernel /= kernel.sum()
            else:
                from astropy.convolution import Gaussian1DKernel
                kernel = Gaussian1DKernel(smooth / self.fwhm / self.dpix)
            resi = np.convolve(resi, kernel, mode='same')
        mask = np.logical_and(abs(self.yaxis) <= r_max,
                              abs(self.yaxis) >= r_min)
        x, y = resi[mask], self.yaxis[mask]

        # Rotate back to sky-plane and return.
        x, y = self._rotate_coords(x, -y, PA)
        return x + x0, y + y0

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
        except KeyError:
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

    def _plot_bestfit(self, params, ivar=None, residual=False,
                      return_ax=False):
        """Plot the best-fit model."""
        import matplotlib.pyplot as plt
        ax = plt.subplots()[1]
        vkep = self._make_model(params) * 1e-3
        levels = np.nanpercentile(vkep, [2, 98])
        levels = np.linspace(levels[0], levels[1], 30)
        im = ax.contourf(self.xaxis, self.yaxis, vkep, levels,
                         cmap=rotationmap.colormap(), extend='both')
        if ivar is not None:
            ax.contour(self.xaxis, self.yaxis, ivar,
                       [0.0], colors='k')
            ax.contourf(self.xaxis, self.yaxis, ivar,
                        [-1.0, 0.0], colors='k', alpha=0.5)
        cb = plt.colorbar(im, pad=0.02, format='%.2f')
        cb.set_label(r'${\rm v_{mod} \quad (km\,s^{-1})}$',
                     rotation=270, labelpad=15)
        cb.minorticks_on()
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
            ax.contour(self.xaxis, self.yaxis, ivar,
                       [0.0], colors='k')
            ax.contourf(self.xaxis, self.yaxis, ivar,
                        [-1.0, 0.0], colors='k', alpha=0.5)
        cb = plt.colorbar(im, pad=0.02, format='%d', ax=ax)
        cb.set_label(r'${\rm  v_{0} - v_{mod} \quad (m\,s^{-1})}$',
                     rotation=270, labelpad=15)
        cb.minorticks_on()
        self._gentrify_plot(ax)
        if return_ax:
            return ax

    def plot_surface(self, ax=None, x0=0.0, y0=0.0, inc=0.0, PA=0.0, z0=0.0,
                     psi=0.0, r_cavity=0.0, r_taper=None, q_taper=None,
                     w_i=0.0, w_r=1.0, w_t=0.0, r_min=0.0, r_max=None,
                     ntheta=24, nrad=15, check_mask=True, **kwargs):
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
                                      psi=psi, r_cavity=r_cavity,
                                      r_taper=r_taper, q_taper=q_taper)
        mf = np.logical_and(zf >= 0.0, np.isfinite(self.data))
        rf = np.where(mf, rf, np.nan)
        tf = np.where(mf, tf, np.nan)

        # Rear half of the disk.
        rb, tb, zb = self.disk_coords(x0=x0, y0=y0, inc=inc, PA=PA, z0=-z0,
                                      psi=psi, r_cavity=r_cavity,
                                      r_taper=r_taper, q_taper=q_taper)
        mb = np.logical_and(zb <= 0.0, np.isfinite(self.data))
        rb = np.where(mb, rb, np.nan)
        tb = np.where(mb, tb, np.nan)

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
        lw = kwargs.pop('lw', kwargs.pop('linewidth', 1.0))
        zo = kwargs.pop('zorder', 10000)
        c = kwargs.pop('colors', kwargs.pop('c', 'k'))

        radii = np.linspace(0, r_max, int(nrad + 1))[1:]
        theta = np.linspace(-np.pi, np.pi, int(ntheta))
        theta += np.diff(theta)[0]

        # Do the plotting.
        ax.contour(self.xaxis, self.yaxis, rf, levels=radii, colors=c,
                   linewidths=lw, linestyles='-', zorder=zo, **kwargs)
        for tt in theta:
            tf_tmp = np.where(abs(tf - tt) <= 0.5, tf, np.nan)
            ax.contour(self.xaxis, self.yaxis, tf_tmp, levels=[tt], colors=c,
                       linewidths=lw, linestyles='-', zorder=zo, **kwargs)
        ax.contour(self.xaxis, self.yaxis, rb, levels=radii, colors=c,
                   linewidths=lw, linestyles='--', zorder=zo, **kwargs)
        for tt in theta:
            tb_tmp = np.where(abs(tb - tt) <= 0.5, tb, np.nan)
            ax.contour(self.xaxis, self.yaxis, tb_tmp, levels=theta, colors=c,
                       linewidths=lw, linestyles='--', zorder=zo)

        return ax

    def _plot_axes(self, ax, x0=0.0, y0=0.0, inc=0.0, PA=0.0, major=1.0,
                   plot_kwargs=None):
        """
        Plot the major and minor axes on the provided axis.

        Args:
            ax (Matplotlib axes): Axes instance to plot onto.
            x0 (Optional[float]): Relative x-location of the center [arcsec].
            y0 (Optional[float]): Relative y-location of the center [arcsec].
            inc (Optional[float]): Inclination of the disk in [degrees].
            PA (Optional[float]): Position angle of the disk in [degrees].
            major (Optional[float]): Size of the major axis line in [arcsec].
            plot_kwargs (Optional[dict]): Dictionary of parameters to pass to
                ``matplotlib.plot``.

        Returns:
            matplotlib axis: Matplotlib ax with axes drawn.
        """
        x = np.array([major, -major, 0.0, 0.0])
        y = np.array([0.0, 0.0, major * np.cos(np.radians(inc)),
                      -major * np.cos(np.radians(inc))])
        x, y = self._rotate_coords(x, y, PA=PA)
        x, y = x + x0, y + y0
        plot_kwargs = {} if plot_kwargs is None else plot_kwargs
        c = plot_kwargs.pop('c', plot_kwargs.pop('color', 'k'))
        ls = plot_kwargs.pop('ls', plot_kwargs.pop('linestyle', ':'))
        lw = plot_kwargs.pop('lw', plot_kwargs.pop('linewidth', 1.0))
        ax.plot(x[:2], y[:2], c=c, ls=ls, lw=lw)
        ax.plot(x[2:], y[2:], c=c, ls=ls, lw=lw)
        return ax

    def plot_maxima(self, x0=0.0, y0=0.0, inc=0.0, PA=0.0, vlsr=None,
                    r_max=1.0, r_min=None, smooth=False, through_center=True,
                    plot_axes_kwargs=None, plot_kwargs=None, return_fig=False):
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
            vlsr (Optional[float]): Systemic velocity in [m/s].
            r_max (Optional[float]): Maximum offset to consider in [arcsec].
            r_min (Optional[float]): Minimum offset to consider in [arcsec].
                This is useful to skip large regions where beam convolution
                dominates the map.
            smooth (Optional[bool/float]): Smooth the line of nodes. If
                ``True``, smoth with the beam kernel, otherwise ``smooth``
                describes the FWHM of the Gaussian convolution kernel in
                [arcsec].
            through_center (Optional[bool]): If ``True``, force the central
                pixel to go through ``(0, 0)``.
            plot_axes_kwargs (Optional[dict]): Dictionary of kwargs for the
                plot_axes function.
            plot_kwargs (Optional[dict]): Dictionary of kwargs for the
                plot of the maximum and minimum pixels.
            return_fig (Optional[bool]): If True, return the figure.

        Returns:
            (Matplotlib figure): If return_fig is ``True``. Can access the axis
                through ``fig.axes[0]``.
        """

        # Background figure.
        fig = self.plot_data(return_fig=True)
        ax = fig.axes[0]
        if plot_axes_kwargs is None:
            plot_axes_kwargs = dict()
        self._plot_axes(ax=ax, x0=x0, y0=y0, inc=inc, PA=PA,
                        major=plot_axes_kwargs.pop('major', r_max),
                        **plot_axes_kwargs)

        # Get the pixels for the maximum and minimum values.
        x_maj, y_maj = self.find_maxima(x0=x0, y0=y0, PA=PA, vlsr=vlsr,
                                        r_max=r_max, r_min=r_min,
                                        smooth=smooth,
                                        through_center=through_center)
        r_max = r_max * np.cos(np.radians(inc))
        x_min, y_min = self.find_minima(x0=x0, y0=y0, PA=PA, vlsr=vlsr,
                                        r_max=r_max, r_min=r_min,
                                        smooth=smooth,
                                        through_center=through_center)

        # Plot the lines.
        plot_kwargs = {} if plot_kwargs is None else plot_kwargs
        c = plot_kwargs.pop('c', plot_kwargs.pop('color', 'k'))
        lw = plot_kwargs.pop('lw', plot_kwargs.pop('linewidth', 1.0))

        ax.plot(x_maj, y_maj, c=c, lw=lw, **plot_kwargs)
        ax.plot(x_min, y_min, c=c, lw=lw, **plot_kwargs)

        if return_fig:
            return fig

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

    def plot_beam(self, ax, dx=0.125, dy=0.125, **kwargs):
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
            self.plot_beam(ax=ax)
