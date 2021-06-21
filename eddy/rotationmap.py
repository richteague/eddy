# -*- coding: utf-8 -*-

import time
import zeus
import emcee
import numpy as np
import scipy.constants as sc
from .datacube import datacube
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


class rotationmap(datacube):
    """
    Read in the velocity maps and initialize the class. To make the fitting
    quicker, we can clip the cube to a smaller region, or downsample the
    image to get a quicker initial fit.

    Args:
        path (str): Relative path to the rotation map you want to fit.
        FOV (Optional[float]): If specified, clip the data down to a
            square field of view with sides of `FOV` [arcsec].
        uncertainty (Optional[str]): Relative path to the map of v0
            uncertainties. Must be a FITS file with the same shape as the
            data. If nothing is specified, will tried to find an uncerainty
            file following the ``bettermoments`` format. If this is not found,
            it will assume a 10% on all pixels.
        downsample (Optional[int]): Downsample the image by this factor for
            quicker fitting. For example, using ``downsample=4`` on an image
            which is 1000 x 1000 pixels will reduce it to 250 x 250 pixels.
            If you use ``downsample='beam'`` it will sample roughly
            spatially independent pixels using the beam major axis as the
            spacing.
    """

    priors = {}

    def __init__(self, path, FOV=None, uncertainty=None, downsample=None):
        datacube.__init__(self, path=path, FOV=FOV, fill=None)
        self._readuncertainty(uncertainty=uncertainty, FOV=self.FOV)
        if downsample is not None:
            self._downsample_cube(downsample)
        self.mask = np.isfinite(self.data)
        self.vlsr = np.nanmedian(self.data)
        self.vlsr_kms = self.vlsr / 1e3
        self._set_default_priors()

    # -- FITTING FUNCTIONS -- #

    def fit_map(self, p0, params, r_min=None, r_max=None, optimize=True,
                nwalkers=None, nburnin=300, nsteps=100, scatter=1e-3,
                plots=None, returns=None, pool=None, mcmc='emcee',
                mcmc_kwargs=None, niter=1):
        """
        Fit a rotation profile to the data. Note that for a disk with
        a non-zero height, the sign of the inclination dictates the direction
        of rotation: a positive inclination denotes a clockwise rotation, while
        a negative inclination denotes an anti-clockwise rotation.

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
                ``'samples'``, ``'sampler'``, ``'percentiles'``, ``'dict'``,
                ``'model'``, ``'residuals'`` or ``'none'``. By default only
                ``'percentiles'`` are returned.
            pool (optional): An object with a `map` method.
            mcmc_kwargs (Optional[dict]): Dictionary to pass to the MCMC
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

        params_tmp = self.verify_params_dictionary(params.copy())

        # Generate the mask for fitting based on the params.

        p0 = np.squeeze(p0).astype(float)
        temp = rotationmap._populate_dictionary(p0, params_tmp)
        self.ivar = self._calc_ivar(temp)

        # Check what the parameters are.

        labels = rotationmap._get_labels(params_tmp)
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
            p0 = self._optimize_p0(p0, params_tmp)

        # Set up and run the MCMC with emcee.

        time.sleep(0.5)
        nsteps = np.atleast_1d(nsteps)
        nburnin = np.atleast_1d(nburnin)
        nwalkers = np.atleast_1d(nwalkers)

        mcmc_kwargs = {} if mcmc_kwargs is None else mcmc_kwargs
        mcmc_kwargs['scatter'], mcmc_kwargs['pool'] = scatter, pool

        for n in range(int(niter)):

            # Make the mask for fitting.

            temp = rotationmap._populate_dictionary(p0, params_tmp)
            temp = self.verify_params_dictionary(temp)
            self.ivar = self._calc_ivar(temp)

            # Run the sampler.

            sampler = self._run_mcmc(p0=p0, params=params_tmp,
                                     nwalkers=nwalkers[n % nwalkers.size],
                                     nburnin=nburnin[n % nburnin.size],
                                     nsteps=nsteps[n % nsteps.size],
                                     mcmc=mcmc, **mcmc_kwargs)

            if type(params_tmp['PA']) is int:
                sampler.chain[:, :, params_tmp['PA']] %= 360.0

            # Split off the samples.

            if mcmc == 'emcee':
                samples = sampler.chain[:, -int(nsteps[n % nsteps.size]):]
            else:
                samples = sampler.chain[-int(nsteps[n % nsteps.size]):]
            samples = samples.reshape(-1, samples.shape[-1])
            p0 = np.median(samples, axis=0)
            medians = rotationmap._populate_dictionary(p0, params.copy())
            medians = self.verify_params_dictionary(medians)

        # Diagnostic plots.

        if plots is None:
            plots = ['walkers', 'corner', 'bestfit', 'residual']
        plots = np.atleast_1d(plots)
        if 'none' in plots:
            plots = []
        if 'walkers' in plots:
            if mcmc == 'emcee':
                walkers = sampler.chain.T
            else:
                walkers = np.rollaxis(sampler.chain.copy(), 2)
            rotationmap._plot_walkers(walkers, nburnin[-1], labels)
        if 'corner' in plots:
            rotationmap._plot_corner(samples, labels)
        if 'bestfit' in plots:
            self.plot_model(samples=samples,
                            params=params,
                            mask=self.ivar)
        if 'residual' in plots:
            self.plot_model_residual(samples=samples,
                                     params=params,
                                     mask=self.ivar)

        # Generate the output.

        to_return = []

        if returns is None:
            returns = ['samples']
        returns = np.atleast_1d(returns)

        if 'none' in returns:
            return None
        if 'samples' in returns:
            to_return += [samples]
        if 'sampler' in returns:
            to_return += [sampler]
        if 'lnprob' in returns:
            to_return += [sampler.lnprobability[nburnin:]]
        if 'percentiles' in returns:
            to_return += [np.percentile(samples, [16, 50, 84], axis=0)]
        if 'dict' in returns:
            to_return += [medians]
        if 'model' in returns or 'residual' in returns:
            model = self.evaluate_models(samples, params)
            if 'model' in returns:
                to_return += [model]
            if 'residual' in returns:
                to_return += [self.data * 1e3 - model]

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

    def plot_data(self, vmin=None, vmax=None, ivar=None, return_fig=False):
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

        vmin = np.nanpercentile(self.data, [2]) if vmin is None else vmin
        vmax = np.nanpercentile(self.data, [98]) if vmax is None else vmax
        vmax = max(abs(vmin - self.vlsr), abs(vmax - self.vlsr))
        im = ax.imshow(self.data / 1e3, origin='lower', extent=self.extent,
                       vmin=(self.vlsr-vmax)/1e3, vmax=(self.vlsr+vmax)/1e3,
                       cmap=rotationmap.cmap(), zorder=-9)
        cb = plt.colorbar(im, pad=0.03, extend='both', format='%.2f')
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

    def _run_mcmc(self, p0, params, nwalkers, nburnin, nsteps, mcmc, **kwargs):
        """Run the MCMC sampling. Returns the sampler."""

        if mcmc == 'zeus':
            EnsembleSampler = zeus.EnsembleSampler
        else:
            EnsembleSampler = emcee.EnsembleSampler

        p0 = self._random_p0(p0, kwargs.pop('scatter', 1e-3), nwalkers)
        moves = kwargs.pop('moves', None)
        pool = kwargs.pop('pool', None)

        sampler = EnsembleSampler(nwalkers,
                                  p0.shape[1],
                                  self._ln_probability,
                                  args=[params, np.nan],
                                  moves=moves,
                                  pool=pool)

        progress = kwargs.pop('progress', True)

        sampler.run_mcmc(p0, nburnin + nsteps, progress=progress, **kwargs)

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
        model = self._make_model(params)
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
        self.set_prior('vlsr', [np.nanmin(self.data),
                                np.nanmax(self.data)], 'flat')

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
            if key in rotationmap.priors.keys() and params[key] is not None:
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
        t = abs(t) if params['abs_phi'] else t

        # Radial mask.

        mask_r = np.logical_and(r >= params['r_min'], r <= params['r_max'])
        mask_r = ~mask_r if params['exclude_r'] else mask_r

        # Azimuthal mask.

        mask_p = np.logical_and(t >= np.radians(params['phi_min']),
                                t <= np.radians(params['phi_max']))
        mask_p = ~mask_p if params['exclude_phi'] else mask_p

        # Finite value mask.

        mask_f = np.logical_and(np.isfinite(self.data), self.error > 0.0)

        # Velocity mask.

        v_min, v_max = params['v_min'], params['v_max']
        mask_v = np.logical_and(self.data >= v_min, self.data <= v_max)
        mask_v = ~mask_v if params['exclude_v'] else mask_v

        # Combine.

        mask = np.logical_and(np.logical_and(mask_v, mask_f),
                              np.logical_and(mask_r, mask_p))
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

        has_vp_100 = False if params.get('vp_100') is None else True
        has_mstar = False if params.get('mstar') is None else True
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

        params['z0'] = params.pop('z0', None)
        params['psi'] = params.pop('psi', None)
        params['r_taper'] = params.pop('r_taper', None)
        params['q_taper'] = params.pop('q_taper', None)
        params['r_cavity'] = params.pop('r_cavity', None)
        params['z_func'] = params.pop('z_func', None)

        # Warp parameters.

        params['w_i'] = params.pop('w_i', 0.0)
        params['w_r'] = params.pop('w_r', 0.5 * max(self.xaxis))
        params['w_t'] = params.pop('w_t', 0.0)
        params['shadowed'] = params.pop('shadowed', False)

        # Masking parameters.

        params['r_min'] = params.pop('r_min', 0.0)
        params['r_max'] = params.pop('r_max', 1e10)
        params['exclude_r'] = params.pop('exclude_r', False)
        params['phi_min'] = params.pop('phi_min', -180.0)
        params['phi_max'] = params.pop('phi_max', 180.0)
        params['exclude_phi'] = params.pop('exclude_phi', False)
        params['abs_phi'] = params.pop('abs_phi', False)
        params['v_min'] = params.pop('v_min', np.nanmin(self.data))
        params['v_max'] = params.pop('v_max', np.nanmax(self.data))
        params['exclude_v'] = params.pop('exclude_v', False)
        params['user_mask'] = params.pop('user_mask', np.ones(self.data.shape))

        if params['r_min'] >= params['r_max']:
            raise ValueError("`r_max` must be greater than `r_min`.")
        if params['phi_min'] >= params['phi_max']:
            raise ValueError("`phi_max` must be great than `phi_min`.")
        if params['v_min'] >= params['v_max']:
            raise ValueError("`v_max` must be greater than `v_min`.")

        # Beam convolution.

        params['beam'] = bool(params.pop('beam', False))

        return params

    def evaluate_models(self, samples=None, params=None, draws=0.5,
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

        if params is None:
            raise ValueError("Must provide model parameters dictionary.")

        # Model is fully specified.

        if samples is None:
            if coords_only:
                return self.disk_coords(**params.copy())
            else:
                return self._make_model(params.copy())

        nparam = np.sum([isinstance(params[k], int) for k in params.keys()])
        if samples.shape[1] != nparam:
            warning = "Invalid number of free parameters in 'samples': {:d}."
            raise ValueError(warning.format(nparam))
        if not callable(collapse_func):
            raise ValueError("'collapse_func' must be callable.")
        verified_params = self.verify_params_dictionary(params.copy())

        # Avearge over a random draw of models.

        if isinstance(int(draws) if draws > 1.0 else draws, int):
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

    def save_model(self, samples=None, params=None, model=None, filename=None,
                   overwrite=True):
        """
        Save the model as a FITS file.
        """
        from astropy.io import fits
        if model is None:
            model = self.evaluate_models(samples, params)
        if self.header['naxis1'] > self.nypix:
            canvas = np.ones(self._original_shape) * np.nan
            canvas[self._ya:self._yb, self._xa:self._xb] = model
            model = canvas.copy()
        if filename is None:
            filename = self.path.replace('.fits', '_model.fits')
        fits.writeto(filename, model, self.header, overwrite=overwrite)

    def mirror_residual(self, samples, params, mirror_velocity_residual=True,
                        mirror_axis='minor', return_deprojected=False,
                        deprojected_dpix_scale=1.0):
        """
        Return the residuals after subtracting a mirror image of either the
        rotation map, as in _Huang et al. 2018, or the residuals, as in
        _Izquierdo et al. 2021.

        .. _Huang et al. 2018: https://ui.adsabs.harvard.edu/abs/2018ApJ...867....3H/abstract
        .. _Izquierdo et al. 2021: https://ui.adsabs.harvard.edu/abs/2021arXiv210409530V/abstract

        Args:
            samples (ndarray): An array of samples returned from ``fit_map``.
            params (dict): The parameter dictionary passed to ``fit_map``.
            mirror_velocity_residual (Optional[bool]): If ``True``, the
                default, mirror the velocity residuals, otherwise use the line
                of sight velocity corrected for the systemic velocity.
            mirror_axis (Optional[str]): Which axis to mirror the image along,
                either ``'minor'`` or ``'major'``.
            return_deprojected (Optional[bool]): If ``True``, return the
                deprojected image, or, if ``False``, reproject the data onto
                the sky.

        Returns:
            x, y, residual (array, array, array): The x- and y-axes of the
                residual (either the sky axes or deprojected, depending on the
                chosen arguments) and the residual.
        """

        # Calculate the image to mirror.

        if mirror_velocity_residual:
            to_mirror = self.data * 1e3 - self.evaluate_models(samples, params)
        else:
            to_mirror = self.data.copy() * 1e3
            if isinstance(type(params['vlsr']), int):
                to_mirror -= params['vlsr']
            else:
                to_mirror -= np.median(samples, axis=0)[params['vlsr']]

        # Generate the axes for the deprojected image.

        r, t, _ = self.evaluate_models(samples, params, coords_only=True)
        t += np.pi / 2.0 if mirror_axis.lower() == 'minor' else 0.0
        x = np.nanmax(np.where(np.isfinite(to_mirror), r, np.nan))
        x = np.arange(-x, x, deprojected_dpix_scale * self.dpix)
        x -= 0.5 * (x[0] + x[-1])

        xs = (r * np.cos(t)).flatten()
        ys = (r * np.sin(t)).flatten()

        # Deproject the image.

        from scipy.interpolate import griddata
        d = griddata((xs, ys), to_mirror.flatten(), (x[:, None], x[None, :]))

        # Either subtract or add the mirrored image. Only want to add when
        # mirroring the line-of-sight velocity and mirroring about the minor
        # axis.

        if not mirror_velocity_residual and mirror_axis == 'minor':
            d += d[:, ::-1]
        else:
            d -= d[:, ::-1]

        if return_deprojected:
            return x, x, d

        # Reproject the residuals onto the sky plane.

        dd = d.flatten()
        mask = np.isfinite(dd)
        xx, yy = np.meshgrid(x, x)
        xx, yy = xx.flatten(), yy.flatten()
        xx, yy, dd = xx[mask], yy[mask], dd[mask]

        from scipy.interpolate import interp2d

        f = interp2d(xx, yy, dd, kind='linear')
        f = np.squeeze([f(xd, yd) for xd, yd in zip(xs, ys)])
        f = f.reshape(to_mirror.shape)
        f = np.where(np.isfinite(to_mirror), f, np.nan)

        return self.xaxis, self.yaxis, f

    # -- VELOCITY PROJECTION -- #

    def _v0(self, params):
        """Projected velocity structre."""
        rvals, tvals, zvals = self.disk_coords(**params)
        v0 = params['vfunc'](rvals, tvals, zvals, params)
        return v0 + params['vlsr']

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
        """Project the rotational velocity onto the sky."""
        return v_phi * np.cos(tvals) * np.sin(abs(np.radians(params['inc'])))

    def _proj_vrad(self, v_rad, tvals, params):
        """Project the radial velocity onto the sky."""
        return v_rad * np.sin(tvals) * np.sin(-np.radians(params['inc']))

    def _make_model(self, params):
        """Build the velocity model from the dictionary of parameters."""
        v0 = self._v0(params)
        if params['beam']:
            v0 = datacube._convolve_image(v0, self._beamkernel())
        return v0

    def deproject_model_residuals(self, samples, params):
        """
        Deproject the residuals into cylindrical velocity components. Takes the
        median value of the samples to determine the model.

        Args:
            samples (ndarray): An array of samples returned from ``fit_map``.
            params (dict): The parameter dictionary passed to ``fit_map``.
        """
        median_samples = np.median(samples, axis=0)
        verified_params = self.verify_params_dictionary(params.copy())
        model = self._populate_dictionary(median_samples, verified_params)
        v_res = self.data - self._make_model(model)
        rvals, tvals, zvals = self.disk_coords(**model)
        v_p = v_res / np.cos(tvals) / np.sin(-np.radians(model['inc']))
        v_r = v_res / np.sin(tvals) / np.sin(abs(np.radians(model['inc'])))
        v_z = v_res / np.cos(abs(np.radians(model['inc'])))
        v_z = np.where(zvals >= 0.0, -v_z, v_z)
        return v_p, v_r, v_z

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

    def _downsample_cube(self, N, randomize=False):
        """Downsample the cube to make faster calculations."""
        N = int(np.ceil(self.bmaj / self.dpix)) if N == 'beam' else N
        if randomize:
            N0x, N0y = np.random.randint(0, N, 2)
        else:
            N0x, N0y = int(N / 2), int(N / 2)
        if N > 1:
            self.xaxis = self.xaxis[N0x::N]
            self.yaxis = self.yaxis[N0y::N]
            self.data = self.data[N0y::N, N0x::N]
            self.error = self.error[N0y::N, N0x::N]

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

    # -- DATA I/O -- #

    def _readuncertainty(self, uncertainty, FOV=None):
        """Reads the uncertainties."""
        if uncertainty is not None:
            self.error = datacube(uncertainty, FOV=FOV, fill=None).data.copy()
        else:
            try:
                uncertainty = '_'.join(self.path.split('_')[:-1])
                uncertainty += '_d' + self.path.split('_')[-1]
                print("Assuming uncertainties in {}.".format(uncertainty))
                self.error = datacube(uncertainty, FOV=FOV, fill=None)
                self.error = self.error.data.copy()
            except FileNotFoundError:
                print("No uncertainties found, assuming uncertainties of 10%.")
                print("Change this at any time with `rotationmap.error`.")
                self.error = 0.1 * (self.data - np.nanmedian(self.data))
        self.error = np.where(np.isnan(self.error), 0.0, abs(self.error))

    # -- PLOTTING -- #

    def plot_model(self, samples=None, params=None, model=None, mask=None,
                   ax=None, imshow_kwargs=None, cb_label=None,
                   return_fig=False):
        """
        Plot a v0 model using the same scalings as the plot_data() function.

        Args:
            samples (Optional[ndarray]): An array of samples returned from
                ``fit_map``.
            params (Optional[dict]): The parameter dictionary passed to
                ``fit_map``.
            model (Optional[ndarry]): A model array from ``evaluate_models``.
            draws (Optional[int/float]): If an integer, describes the number of
                random draws averaged to form the returned model. If a float,
                represents the percentile used from the samples. Must be
                between 0 and 1 if a float.
            collapse_func (Optional[callable]): How to collapse the random
                number of samples. Must be a function which allows an ``axis``
                argument (as with most Numpy functions).
            mask (Optional[ndarray]): The mask used for the fitting to plot as
                a shaded region.
            ax (Optional[AxesSubplot]): Axis to plot onto.
            imshow_kwargs (Optional[dict]): Dictionary of imshow kwargs.
            cb_label (Optional[str]): Colorbar label.
            return_fig (Optional[bool]): Return the figure.

        Returns:
            fig (Matplotlib figure): If ``return_fig`` is ``True``. Can access
                the axes through ``fig.axes`` for additional plotting.
        """

        # Dummy axis for the plotting.

        if ax is None:
            fig, ax = plt.subplots()

        # Make the model and calculate the plotting limits.

        if model is None:
            model = self.evaluate_models(samples, params.copy())
        vmin, vmax = np.nanpercentile(model, [2, 98])
        vmax = max(abs(vmin - self.vlsr), abs(vmax - self.vlsr))
        vmin = self.vlsr - vmax
        vmax = self.vlsr + vmax

        # Initialize the plotting parameters.

        imshow_kwargs = {} if imshow_kwargs is None else imshow_kwargs
        imshow_kwargs['cmap'] = imshow_kwargs.pop('cmap', datacube.cmap())
        imshow_kwargs['zorder'] = imshow_kwargs.pop('zorder', -9)
        imshow_kwargs['extent'] = self.extent
        imshow_kwargs['origin'] = 'lower'
        imshow_kwargs['vmin'] = imshow_kwargs.pop('vmin', vmin)
        imshow_kwargs['vmax'] = imshow_kwargs.pop('vmax', vmax)
        im = ax.imshow(model, **imshow_kwargs)

        # Overplot the mask if necessary.

        if mask is not None:
            ax.contour(self.xaxis, self.yaxis, mask,
                       [0.0], colors='k')
            ax.contourf(self.xaxis, self.yaxis, mask,
                        [-1.0, 0.0], colors='k', alpha=0.5)

        if cb_label is None:
            cb_label = r'${\rm v_{0} \quad (m\,s^{-1})}$'
        if cb_label != '':
            cb = plt.colorbar(im, pad=0.03, format='%.2f', extend='both')
            cb.set_label(cb_label, rotation=270, labelpad=15)
            cb.minorticks_on()

        self._gentrify_plot(ax)

        if return_fig:
            return fig

    def plot_model_residual(self, samples=None, params=None, model=None,
                            mask=None, ax=None, imshow_kwargs=None,
                            return_fig=False):
        """
        Plot the residual from the provided model.

        Args:
            samples (ndarray): An array of samples returned from ``fit_map``.
            params (dict): The parameter dictionary passed to ``fit_map``.
            mask (Optional[ndarray]): The mask used for the fitting to plot as
                a shaded region.
            ax (Optional[AxesSubplot]): Axis to plot onto.
            imshow_kwargs (Optional[dict]): Dictionary of keyword arguments to
                pass to ``imshow``.
            return_fig (Optional[bool]): Return the figure.

        Returns:
            fig (Matplotlib figure): If ``return_fig`` is ``True``. Can access
                the axes through ``fig.axes`` for additional plotting.
        """

        # Dummy axis to overplot.

        if ax is None:
            fig, ax = plt.subplots()

        # Make the model and calculate the plotting limits.

        if model is None:
            model = self.evaluate_models(samples=samples, params=params.copy())
        vres = self.data - model
        mask = np.ones(vres.shape) if mask is None else mask
        masked_vres = np.where(mask, vres, np.nan)
        vmin, vmax = np.nanpercentile(masked_vres, [2, 98])
        vmax = max(abs(vmin), abs(vmax))

        # Plot the data.

        imshow_kwargs = {} if imshow_kwargs is None else imshow_kwargs
        imshow_kwargs['origin'] = 'lower'
        imshow_kwargs['extent'] = self.extent
        imshow_kwargs['cmap'] = 'RdBu_r'
        imshow_kwargs['vmin'] = imshow_kwargs.pop('vmin', -vmax)
        imshow_kwargs['vmax'] = imshow_kwargs.pop('vmax', vmax)
        im = ax.imshow(vres, **imshow_kwargs)

        # Overplot the mask is necessary.

        if mask is not None:
            ax.contour(self.xaxis, self.yaxis, mask,
                       [0.0], colors='k')
            ax.contourf(self.xaxis, self.yaxis, mask,
                        [-1.0, 0.0], colors='k', alpha=0.5)

        cb = plt.colorbar(im, pad=0.02, format='%d', ax=ax, extend='both')
        cb.set_label(r'${\rm  v_{0} - v_{mod} \quad (m\,s^{-1})}$',
                     rotation=270, labelpad=15)
        cb.minorticks_on()

        self._gentrify_plot(ax)

        if return_fig:
            return fig

    def plot_model_surface(self, samples, params, plot_surface_kwargs=None,
                           mask_with_data=True, return_fig=True):
        """
        Overplot the emission surface onto the provided axis. Takes the median
        value of the samples as the model to plot.

        Args:
            samples (ndarray): An array of samples returned from ``fit_map``.
            params (dict): The parameter dictionary passed to ``fit_map``.
            plot_surface_kwargs (Optional[dict]): Dictionary of kwargs to pass
                to ``plot_surface``.
            mask_with_data (Optional[bool]): If ``True``, mask the surface to
                regions where the data is finite valued.
            return_fig (Optional[bool]): Return the figure.

        Returns:
            fig (Matplotlib figure): If ``return_fig`` is ``True``. Can access
                the axes through ``fig.axes`` for additional plotting.
        """

        # Check the input.

        nparam = np.sum([isinstance(params[k], int) for k in params.keys()])
        if samples.shape[1] != nparam:
            warning = "Invalid number of free parameters in 'samples': {:d}."
            raise ValueError(warning.format(nparam))
        if plot_surface_kwargs is None:
            plot_surface_kwargs = {}
        plot_surface_kwargs['return_fig'] = True

        # Populate the model with the median values.

        model = self.verify_params_dictionary(params.copy())
        model = self._populate_dictionary(np.median(samples, axis=0), model)
        model['mask'] = np.isfinite(self.data) if mask_with_data else None
        model.pop('r_max')
        fig = self.plot_surface(**model, **plot_surface_kwargs)
        self._gentrify_plot(ax=fig.axes[0])

        if return_fig:
            return fig

    def plot_disk_axes(self, x0=0.0, y0=0.0, inc=0.0, PA=0.0, major=1.0,
                       ax=None, plot_kwargs=None, return_ax=True):
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

        # Dummy axis to plot.

        if ax is None:
            fig, ax = plt.subplots()

        x = np.array([major, -major,
                      0.0, 0.0])
        y = np.array([0.0, 0.0,
                      major * np.cos(np.radians(inc)),
                      -major * np.cos(np.radians(inc))])
        x, y = self._rotate_coords(x, y, PA=PA)
        x, y = x + x0, y + y0
        plot_kwargs = {} if plot_kwargs is None else plot_kwargs
        c = plot_kwargs.pop('c', plot_kwargs.pop('color', 'k'))
        ls = plot_kwargs.pop('ls', plot_kwargs.pop('linestyle', ':'))
        lw = plot_kwargs.pop('lw', plot_kwargs.pop('linewidth', 1.0))
        ax.plot(x[:2], y[:2], c=c, ls=ls, lw=lw)
        ax.plot(x[2:], y[2:], c=c, ls=ls, lw=lw)

        if return_ax:
            ax

    def plot_maxima(self, x0=0.0, y0=0.0, inc=0.0, PA=0.0, vlsr=None,
                    r_max=1.0, r_min=None, smooth=False, through_center=True,
                    plot_axes_kwargs=None, plot_kwargs=None,
                    return_fig=False):
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
            levels (Optional[array]): Levels to pass to ``plot_data``.
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
        self.plot_disk_axes(ax=ax, x0=x0, y0=y0, inc=inc, PA=PA,
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
