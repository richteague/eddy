# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from scipy.optimize import curve_fit
from scipy.optimize import minimize

# Check is 'celerite' is installed.
try:
    import celerite
    celerite_installed = True
except ImportError:
    celerite_installed = False

__all__ = ['annulus']

class annulus(object):
    """
    A class containing an annulus of spectra with their associated polar angles
    measured east of north from the redshifted major axis.

    Args:
        spectra (ndarray): Array of shape ``[N, M]`` of spectra to shift and
            fit, where ``N`` is the number of spectra and ``M`` is the length
            of the velocity axis.
        theta (ndarray): Polar angles in [rad] of each of the spectra.
        velax (ndarray): Velocity axis in [m/s] of the spectra.
        remove_empty (optional[bool]): Remove empty spectra.
        sort_spectra (optional[bool]): Sorted the spectra into increasing
            ``theta``.
        suppress_warnings (optional[bool]): If ``True``, suppress all warnings.
    """

    def __init__(self, spectra, theta, velax, remove_empty=True,
                 sort_spectra=True, suppress_warnings=True):

        # Suppress warnings.
        if suppress_warnings:
            import warnings
            warnings.filterwarnings("ignore")

        # Read in the spectra and remove empty values.
        self.theta = theta
        self.spectra = spectra

        # Sort the spectra.
        if sort_spectra:
            idxs = np.argsort(self.theta)
            self.spectra = self.spectra[idxs]
            self.theta = self.theta[idxs]

        # Remove empty pixels.
        if remove_empty:
            idxa = np.sum(self.spectra, axis=-1) > 0.0
            idxb = np.std(self.spectra, axis=-1) != 0.0
            idxs = idxa & idxb
            self.theta = self.theta[idxs]
            self.spectra = self.spectra[idxs]

        # Easier to use variables.
        self.theta_deg = np.degrees(self.theta) % 360.
        self.spectra_flat = self.spectra.flatten()

        # Check there's actually spectra.
        if self.theta.size < 1:
            raise ValueError("No finite spectra. Check for NaNs.")

        # Velocity axis.
        self.velax = velax
        self.channel = np.diff(velax)[0]
        self.velax_range = (self.velax[0] - 0.5 * self.channel,
                            self.velax[-1] + 0.5 * self.channel)
        self.velax_mask = np.array([self.velax[0], self.velax[-1]])

        # Check the shapes are compatible.
        if self.spectra.shape[0] != self.theta.size:
            raise ValueError("Mismatch in number of angles and spectra.")
        if self.spectra.shape[1] != self.velax.size:
            raise ValueError("Mismatch in the spectra and velocity axis.")

    # -- Measure the Velocity -- #

    def get_vlos(self, p0=None, fit_method='GP', fit_vrad=False,
                 resample=False, optimize=True, nwalkers=32, nburnin=500,
                 nsteps=500, scatter=1e-3, signal='int', optimize_kwargs=None,
                 plots=None, returns=None):
        """
        Infer the requested velocities by shifting lines back to a common
        center and stacking. The quality of fit is given by the selected
        method which must be ``'dv'``, ``'GP'`` or ``'SNR'``. The former,
        described in `Teague et al. (2018a)`_, minimizes the line width of the
        resulting spectrum via :func:`deprojected_width`. Similarly,
        ``fit_method='SNR'`` aims to maximize the signal to noise of the
        spectrum, as used in `Yen et al. (2016)`_, with specifics of the method
        described in :func:`deprojected_nSNR`. Finally, ``fit_method='GP'``
        uses a Gaussian Process model which relaxes the assumption of an
        analytical line profile, as used in `Teague et al. (2018b)`_.

        Different types of resampling can be applied to the spectrum. In
        general it is recommended that ``resample=False`` for the Gaussian
        Process approach, while ``resample=True`` for the other two.

        Args:
            fit_method (optional[str]): Method used to define the quality of
                fit. Must be one of ``'GP'``, ``'dV'`` or ``SNR``.
            p0 (optional[list]): Starting positions for the minimization. If
                nothing is provided these will be guessed but this may not
                result in very good starting positions.
            fit_vrad (bool): Include radial motion in the fit.
            resample (optional[bool]): Resampling method to apply. See
                :func:`deprojected_spectrum` for more details.
            optimize (optional[bool]): Optimize the starting positions before
                the MCMC runs. If an integer, the number of iterations to use
                of optimization.
            nwalkers (optional[int]): Number of walkers used for the MCMC runs.
            nburnin (optional[int]): Number of steps used to burn in walkers.
            nsteps (optional[int]): Number of steps taken to sample posteriors.
            scatter (optional[float]): Scatter applied to the starting
                positions before running the MCMC.
            plots (optional[list]):
            returns (optional[list]):

        Returns:
            Either the samples od the posterior for each free parameter, or the
            16th, 50th and 84th pecentiles of the distributions depending on
            what ``returns`` was set to.

        .. _Teague et al. (2018a): https://ui.adsabs.harvard.edu/abs/2018ApJ...860L..12T/abstract
        .. _Teague et al. (2018b): https://ui.adsabs.harvard.edu/abs/2018ApJ...868..113T/abstract
        .. _Yen et al. (2016): https://ui.adsabs.harvard.edu/abs/2016ApJ...832..204Y/abstract

        """

        # Check the input variables.
        fit_method = fit_method.lower()
        if fit_method not in ['dv', 'gp', 'snr']:
            raise ValueError("method must be 'dV', 'GP' or 'SNR'.")
        if fit_method == 'gp' and not celerite_installed:
            raise ImportError("Must install 'celerite' to use GP method.")

        # Run the appropriate methods.
        if fit_method == 'gp':
            if resample:
                print("WARNING: Resampling with the GP method is not advised.")
            return self._fitting_GP(p0=p0, fit_vrad=fit_vrad,
                                    nwalkers=nwalkers, nsteps=nsteps,
                                    nburnin=nburnin, scatter=scatter,
                                    plots=plots, returns=returns,
                                    resample=resample,
                                    optimize_kwargs=optimize_kwargs)

        elif fit_method == 'dv':
            return self._fitting_dV(p0=p0, fit_vrad=fit_vrad,
                                    resample=resample,
                                    optimize_kwargs=optimize_kwargs)

        elif fit_method == 'snr':
            return self._fitting_SNR(p0=p0, fit_vrad=fit_vrad,
                                     resample=resample, signal=signal,
                                     optimize_kwargs=optimize_kwargs)

    # -- Gaussian Processes Approach -- #

    def _fitting_GP(self, p0=None, fit_vrad=False, optimize=False, nwalkers=64,
                    nburnin=50, nsteps=100, resample=False, scatter=1e-3,
                    plots=None, returns=None, optimize_kwargs=None):
        """
        Wrapper for the GP fitting.

        Args:
            p0 (optional[list]): Starting positions.
            optimize (optional[bool]): Run an optimization step prior to the
                MCMC.
            nwalkers (optional[int]): Number of walkers for the MCMC.
            nburnin (optional[int]): Number of steps to discard for burn-in.
            nsteps (optional[int]): Number of steps used to sample the
                posterior distributions.
            resample (optional): Type of resampling to be implemented.
            scatter (optional[float]): Scatter of walkers around ``p0``.
            plots (optional[list]): List of diagnostic plots to make. Can be
                ``'walkers'``, ``'corner'`` or ``'none'``.
            returns (optional[list]) List of values to return. Can be
                ``'samples'``, ``'percentiles'`` or ``'none'``.

        Returns:
            Dependent on what is specified in ``returns``.
        """

        # Starting positions.
        if p0 is None:
            p0 = self._guess_parameters_GP(fit=True)
            if not fit_vrad:
                p0 = np.concatenate([p0[:1], p0[-3:]])
        p0 = np.atleast_1d(p0)

        # Check for NaNs in the starting values.
        if np.any(np.isnan(p0)):
            raise ValueError("WARNING: NaNs in the p0 array.")

        # Optimize the starting positions.
        if optimize:
            if optimize_kwargs is None:
                optimize_kwargs = {}
            p0 = self._optimize_p0_GP(p0, N=int(optimize), resample=resample,
                                      **optimize_kwargs)
        p0 = annulus._randomize_p0(p0, nwalkers, scatter)

        # Run the sampler
        import emcee
        sampler = emcee.EnsembleSampler(nwalkers, p0.shape[1],
                                        self._lnprobability,
                                        args=(p0[:, 0].mean(), resample))
        sampler.run_mcmc(p0, nburnin + nsteps, progress=True)
        samples = sampler.chain[:, -nsteps:]
        samples = samples.reshape(-1, samples.shape[-1])

        # Diagnosis plots if appropriate.
        plots = ['walkers', 'corner'] if plots is None else plots
        plots = [p.lower() for p in np.atleast_1d(plots)]
        if 'walkers' in plots:
            annulus._plot_walkers(sampler, nburnin)
        if 'corner' in plots:
            annulus._plot_corner(samples)

        # Return the requested values.
        returns = ['percentiles'] if returns is None else returns
        returns = [r.lower() for r in np.atleast_1d(returns)]
        if 'none' in returns:
            return None
        if 'percentiles' in returns:
            idx = returns.index('percentiles')
            returns[idx] = np.percentile(samples, [16, 50, 84], axis=0).T
        if 'samples' in returns:
            idx = returns.index('samples')
            returns[idx] = samples
        return returns[0] if len(returns) == 1 else returns

    def _optimize_p0_GP(self, p0, N=1, resample=True, **kwargs):
        """
        Optimize the starting positions, p0. We do this in a slightly hacky way
        because the minimum is not easily found. We first optimize the hyper
        parameters of the GP model, holding the rotation velocity constant,
        then, holding the GP hyperparameters constant, optimizing the rotation
        velocity, before optimizing everything together. This can be run
        multiple times to iteratie to a global optimum. We only update p0 if
        both the minimization converged (res.success == True) and there is an
        improvement in the likelihood.

        One can also pass all the options to optimize.minimize to try different
        minimization techniques. The default values here were based on trial
        and error.

        Args:
            p0 (ndarray): Initial guess of the starting positions.
            N (Optional[int]): Interations of the optimization to run.
            resample (Optional[bool/int]): If true, resample the deprojected
                spectra donw to the original velocity resolution. If an integer
                is given, use this as the bew sampling rate relative to the
                original data.

        Returns:
            p0 (ndarray): Optimized array. If scipy.minimize does not converge
                then p0 will not be updated. No warnings are given, however.
        """

        # Default parameters for the minimization.
        # Bit messy to preserve user chosen values.
        kwargs['method'] = kwargs.get('method', 'L-BFGS-B')
        options = kwargs.pop('options', {})
        kwargs['options'] = {'maxiter': options.pop('maxiter', 100000),
                             'maxfun': options.pop('maxfun', 100000),
                             'ftol': options.pop('ftol', 1e-4)}
        for key in options.keys():
            kwargs['options'][key] = options[key]

        # Starting negative log likelihood to test against.
        fit_vrad = len(p0) == 5
        nlnL = self._nlnL(p0, resample=resample)

        # Cycle through the required number of iterations.
        for _ in range(int(N)):

            # Define the bounds.
            bounds = [(0.8 * p0[0], 1.2 * p0[0])]
            if fit_vrad:
                bounds.append((-0.3 * p0[0], 0.3 * p0[0]))
            bounds += [(0.0, None), (-15.0, 10.0), (0.0, 10.0)]

            # Optimize hyper-parameters, holding vrot and vrad constant.
            res = minimize(self._nlnL_hyper, x0=p0[-3:],
                           args=(p0[0], p0[1] if fit_vrad else 0., resample),
                           bounds=bounds[-3:], **kwargs)
            if res.success:
                p0_temp = p0
                p0_temp[-3:] = res.x
                nlnL_temp = self._nlnL(p0_temp, resample)
                if nlnL_temp < nlnL:
                    p0 = p0_temp
                    nlnL = nlnL_temp
            else:
                if verbose:
                    print('Failed hyper-params mimization: %s' % res.message)

            # Optimize vrot holding the hyper-parameters and vrad constant.
            res = minimize(self._nlnL_vrot, x0=p0[0],
                           args=(p0[-3:], p0[1] if fit_vrad else 0., resample),
                           bounds=[bounds[0]], **kwargs)
            if res.success:
                p0_temp = p0
                p0_temp[0] = res.x
                nlnL_temp = self._nlnL(p0_temp, resample)
                if nlnL_temp < nlnL:
                    p0 = p0_temp
                    nlnL = nlnL_temp
                else:
                    if verbose:
                        print('Failed vrot mimization: %s' % res.message)

            # Optimize vrad holding the hyper-parameters and vrot constant.
            if fit_vrad:
                res = minimize(self._nlnL_vrad, x0=p0[1],
                               args=(p0[0], p0[-3:], resample),
                               bounds=[bounds[1]], **kwargs)
                if res.success:
                    p0_temp = p0
                    p0_temp[1] = res.x
                    nlnL_temp = self._nlnL(p0_temp, resample)
                    if nlnL_temp < nlnL:
                        p0 = p0_temp
                        nlnL = nlnL_temp
                    else:
                        if verbose:
                            print('Failed vrad mimization: %s' % res.message)

            # Final minimization with everything.
            res = minimize(self._nlnL, x0=p0, args=(resample),
                           bounds=bounds, **kwargs)
            if res.success:
                p0_temp = res.x
                nlnL_temp = self._nlnL(p0_temp, resample)
                if nlnL_temp < nlnL:
                    p0 = p0_temp
                    nlnL = nlnL_temp
            else:
                if verbose:
                    print('Failed total mimization: %s' % res.message)

        return p0

    def _guess_parameters_GP(self, fit=True):
        """Guess the starting positions from the spectra."""
        vrot, vrad, _ = self.guess_parameters(fit=fit)
        noise = int(min(10, self.spectra.shape[1] / 3.0))
        noise = np.std([self.spectra[:, :noise], self.spectra[:, -noise:]])
        ln_sig = np.log(np.std(self.spectra))
        ln_rho = np.log(150.)
        return np.array([vrot, vrad, noise, ln_sig, ln_rho])

    @staticmethod
    def _randomize_p0(p0, nwalkers, scatter):
        """Estimate (vrot, noise, lnp, lns) for the spectrum."""
        dp0 = np.random.randn(nwalkers * len(p0)).reshape(nwalkers, len(p0))
        dp0 = np.where(p0 == 0.0, 1.0, p0)[None, :] * (1.0 + scatter * dp0)
        return np.where(p0[None, :] == 0.0, dp0 - 1.0, dp0)

    def _nlnL_vrot(self, vrot, hyperparams, vrad, resample=False):
        """Negative lnlikelihood function with vrot as only argument."""
        theta = np.insert(hyperparams, 0, [vrot, vrad])
        nll = -self._lnlikelihood(theta, resample=resample)
        return nll if np.isfinite(nll) else 1e15

    def _nlnL_hyper(self, hyperparams, vrot, vrad, resample=False):
        """Negative lnlikelihood function with hyperparams as only argument."""
        theta = np.insert(hyperparams, 0, [vrot, vrad])
        nll = -self._lnlikelihood(theta, resample=resample)
        return nll if np.isfinite(nll) else 1e15

    def _nlnL_vrad(self, vrad, vrot, hyperparams, resample=False):
        """Negative lnlikelihood function with vrad as only argument."""
        theta = np.insert(hyperparams, 0, [vrot, vrad])
        nll = -self._lnlikelihood(theta, resample=resample)
        return nll if np.isfinite(nll) else 1e15

    def _nlnL(self, theta, resample=False):
        """Negative log-likelihood function for optimization."""
        nll = -self._lnlikelihood(theta, resample=resample)
        return nll if np.isfinite(nll) else 1e15

    @staticmethod
    def _build_kernel(x, y, hyperparams):
        """Build the GP kernel. Returns None if gp.compute(x) fails."""
        noise, lnsigma, lnrho = hyperparams
        k_noise = celerite.terms.JitterTerm(log_sigma=np.log(noise))
        k_line = celerite.terms.Matern32Term(log_sigma=lnsigma, log_rho=lnrho)
        gp = celerite.GP(k_noise + k_line, mean=np.nanmean(y), fit_mean=True)
        try:
            gp.compute(x)
        except Exception:
            return None
        return gp

    @staticmethod
    def _lnprior(theta, vref):
        """Uninformative log-prior function for MCMC."""
        try:
            vrot, vrad = theta[:-3]
        except ValueError:
            vrot, vrad = theta[0], 0.0
        noise, lnsigma, lnrho = theta[-3:]
        if abs(vrot - vref) / vref > 0.4:
            return -np.inf
        if abs(vrad / vrot) > 1.0:
            return -np.inf
        if vrot <= 0.0:
            return -np.inf
        if noise <= 0.0:
            return -np.inf
        if not -15.0 < lnsigma < 10.:
            return -np.inf
        if not 0.0 <= lnrho <= 10.:
            return -np.inf
        return 0.0

    def _lnlikelihood(self, theta, resample=False):
        """Log-likelihood function for the MCMC."""

        # Unpack the free parameters.
        try:
            vrot, vrad = theta[:-3]
        except ValueError:
            vrot, vrad = theta[0], 0.0
        hyperparams = theta[-3:]

        # Deproject the data and resample if requested.
        x, y = self.deprojected_spectrum(vrot=vrot, vrad=vrad,
                                         resample=resample)
        x, y = self._get_masked_spectrum(x, y)

        # Build the GP model and calculate the log-likelihood.
        gp = annulus._build_kernel(x, y, hyperparams)
        if gp is None:
            return -np.inf
        ll = gp.log_likelihood(y, quiet=True)
        return ll if np.isfinite(ll) else -np.inf

    def _lnprobability(self, theta, vref, resample=False):
        """Log-probability function for the MCMC."""
        if ~np.isfinite(annulus._lnprior(theta, vref)):
            return -np.inf
        return self._lnlikelihood(theta, resample)

    # -- Minimizing Line Width Approach -- #

    def _fitting_dV(self, p0=None, fit_vrad=False, resample=False,
                    optimize_kwargs=None):
        """
        Wrapper for the dV fitting.

        Args:
            Coming Soon.

        Returns:
            Coming Soon.
        """

        # Starting positions.
        if p0 is None:
            p0 = self.guess_parameters(fit=True)[:2]
            if not fit_vrad:
                p0 = p0[:1]
        p0 = np.atleast_1d(p0)

        # Populate the kwargs.
        optimize_kwargs = {} if optimize_kwargs is None else optimize_kwargs
        optimize_kwargs['method'] = optimize_kwargs.get('method', 'L-BFGS-B')
        options = optimize_kwargs.pop('options', {})
        options['maxiter'] = options.pop('maxiter', 10000)
        options['maxfun'] = options.pop('maxfun', 10000)
        options['ftol'] = options.pop('ftol', 1e-4)
        optimize_kwargs['options'] = options

        # Define the bounds.
        bounds = [[0.7 * p0[0], 1.3 * p0[0]]]
        if fit_vrad:
            bounds += [[-0.3 * p0[0], 0.3 * p0[0]]]
        optimize_kwargs['bounds'] = np.array(bounds)

        # Run the minimization.
        res = minimize(self.deprojected_width, x0=p0,
                       args=(fit_vrad, resample),
                       **optimize_kwargs)
        if not res.success:
            print("WARNING: minimize did not converge.")
        return res.x if res.success else np.nan

    def deprojected_width(self, theta, fit_vrad=False, resample=True):
        """
        Return the Gaussian width of the deprojected and stacked spectra.

        Args:
            theta (list): Deprojection velocities, ``(vrot[, vrad])``.
            fit_vrad (optional[bool]): Whether ``vrad`` in is ``theta``.
            resample (optional): How to resample the data.  See
                :func:`deprojected_spectrum` for more details.

        Returns:
            The Doppler width of the average stacked spectrum using the
            velocities to align the individual spectra.
        """
        if fit_vrad:
            vrot, vrad = theta
        else:
            vrot, vrad = theta, 0.0
        x, y = self.deprojected_spectrum(vrot, vrad, resample)
        return annulus._get_gaussian_width(*self._get_masked_spectrum(x, y))

    # -- Rotation Velocity by Maximizing SNR -- #

    def _fitting_SNR(self, p0=None, fit_vrad=False, resample=False,
                     signal='int', optimize_kwargs=None):
        """
        Infer the rotation velocity by finding the rotation velocity which,
        after shifting all spectra to a common velocity, results in the
        maximum signal-to=noise ratio of the stacked profile. This is an
        implementation of the method described in Yen et al. (2016, 2018).

        Args:
            vref (Optional[float]): Predicted rotation velocity, typically the
                Keplerian velocity at that radius. Will be used as the starting
                position for the minimization.
            resample (Optional[bool]): Resample the shifted spectra by this
                factor. For example, resample = 2 will shift and bin the
                spectrum down to sampling rate twice that of the original data.
            signal (Optional[str]): Method used to calculate the signal, either
                'max' for the line peak or 'int' for the integrated intensity.
            weight_SNR (Optional[bool]): If True and sigal == 'int', include
                the Gaussian weighting used to calculate the SNR as decscribed
                in Yen et al. (2018).

        Returns:
            Velocities which maximizese signal to noise of the shifted and
            stacked spectrum.

        .. _Yen et al. (2016): https://ui.adsabs.harvard.edu/abs/2016ApJ...832..204Y/abstract

        """

        # Make sure the signal is defined.
        if signal not in ['max', 'int', 'weighted']:
            raise ValueError("'signal' must be either 'max', 'int', "
                             + "or 'weighted'.")

        # Starting positions.
        if p0 is None:
            p0 = self.guess_parameters(fit=True)[:2]
            if not fit_vrad:
                p0 = p0[:1]
        p0 = np.atleast_1d(p0)

        # Populate the kwargs.
        optimize_kwargs = {} if optimize_kwargs is None else optimize_kwargs
        optimize_kwargs['method'] = optimize_kwargs.get('method', 'L-BFGS-B')
        options = optimize_kwargs.pop('options', {})
        options['maxiter'] = options.pop('maxiter', 10000)
        options['maxfun'] = options.pop('maxfun', 10000)
        options['ftol'] = options.pop('ftol', 1e-4)
        optimize_kwargs['options'] = options

        # Define the bounds.
        bounds = [[0.7 * p0[0], 1.3 * p0[0]]]
        if fit_vrad:
            bounds += [[-0.3 * p0[0], 0.3 * p0[0]]]
        optimize_kwargs['bounds'] = np.array(bounds)

        # Run the minimization.
        res = minimize(self.deprojected_nSNR, x0=p0,
                       args=(fit_vrad, resample, signal),
                       **optimize_kwargs)
        if not res.success:
            print("WARNING: minimize did not converge.")
        return res.x if res.success else np.nan

    def deprojected_nSNR(self, theta, fit_vrad=False, resample=False,
                         signal='int'):
        """
        Return the negative SNR of the deprojected spectrum. There are three
        ways to calculate the signal of the data. ``signal='max'`` will use the
        Gaussian peak relative to the noise, ``signal='int'`` will use the
        integrated spectrum as the signal, while ``signal='weighted'`` will
        additionally use a Gaussian shape weighting so that noise in the line
        wings are minimized, as in `Yen et al. (2016)`_.

        .. warning::

            Currently the noise is calculated based on pixels greater than
            three line widths away from the line center. As both the line
            center and width changes from call to call, the noise will change
            too.

        Args:
            theta (list): Deprojection velocities, ``(vrot[, vrad])``.
            fit_vrad (optional[bool]): Whether ``vrad`` in is ``theta``.
            resample (optional): How to resample the data. See
                :func:`deprojected_spectrum` for more details.
            signal (optional[str]): Definition of SNR to use.

        Returns:
            Negative of the signal-to-noise ratio.
        """
        if fit_vrad:
            vrot, vrad = theta
        else:
            vrot, vrad = theta, 0.0
        x, y = self.deprojected_spectrum(vrot, vrad, resample)
        x0, dx, A = annulus._fit_gaussian(x, y)
        noise = np.std(x[(x - x0) / dx > 3.0])  # Check: noise will vary.
        if signal == 'max':
            SNR = A / noise
        else:
            if signal == 'weighted':
                w = annulus._gaussian(x, x0, dx, (np.sqrt(np.pi) * dx)**-1)
            else:
                w = np.ones(x.size)
            mask = (x - x0) / dx <= 1.0
            SNR = np.trapz((y * w)[mask], x=x[mask])
        return -SNR

    @staticmethod
    def _get_gaussian_width(x, y, fill_value=1e50):
        """Return the absolute width of a Gaussian fit to the spectrum."""
        dV = annulus._fit_gaussian(x, y)[1]
        if np.isfinite(dV):
            return abs(dV)
        return fill_value

    @staticmethod
    def _get_p0_gaussian(x, y):
        """Estimate (x0, dV, Tb) for the spectrum."""
        if x.size != y.size:
            raise ValueError("Mismatch in array shapes.")
        Tb = np.max(y)
        x0 = x[y.argmax()]
        dV = np.trapz(y, x) / Tb / np.sqrt(2. * np.pi)
        return x0, dV, Tb

    @staticmethod
    def _fit_gaussian(x, y, dy=None, return_uncertainty=False):
        """Fit a gaussian to (x, y, [dy])."""
        try:
            popt, cvar = curve_fit(annulus._gaussian, x, y, sigma=dy,
                                   p0=annulus._get_p0_gaussian(x, y),
                                   absolute_sigma=True, maxfev=100000)
            cvar = np.diag(cvar)
        except Exception:
            popt = [np.nan, np.nan, np.nan]
            cvar = popt.copy()
        if return_uncertainty:
            return popt, np.sqrt(cvar)
        return popt

    @staticmethod
    def _fit_gaussian_thick(x, y, dy=None, return_uncertainty=False):
        """Fit an optically thick Gaussian function to (x, y, [dy])."""
        p0 = annulus._fit_gaussian(x=x, y=y, dy=dy, return_uncertainty=False)
        p0 = np.append(p0, 1.0)
        try:
            popt, cvar = curve_fit(annulus._gaussian_thick, x, y, sigma=dy,
                                   p0=p0, absolute_sigma=True, maxfev=100000)
            cvar = np.diag(cvar)
        except Exception:
            popt = [np.nan, np.nan, np.nan, np.nan]
            cvar = popt.copy()
        if return_uncertainty:
            return popt, np.sqrt(cvar)
        return popt

    @staticmethod
    def _get_gaussian_center(x, y):
        """Return the line center from a Gaussian fit to the spectrum."""
        x0 = annulus._fit_gaussian(x, y)[0]
        if np.isfinite(x0):
            return abs(x0)
        return x[np.argmax(y)]

    # -- Line Profile Functions -- #

    @staticmethod
    def _gaussian(x, x0, dV, Tb):
        """Gaussian function."""
        return Tb * np.exp(-np.power((x - x0) / dV, 2.0))

    @staticmethod
    def _gaussian_thick(x, x0, dV, Tex, tau0):
        """Optically thick Gaussian line."""
        tau = annulus._gaussian(x, x0, dV, tau0)
        return Tex * (1. - np.exp(-tau))

    @staticmethod
    def _SHO(x, A, y0):
        """Simple harmonic oscillator."""
        return A * np.cos(x) + y0

    @staticmethod
    def _SHOb(x, A, y0, dx):
        """Simple harmonic oscillator with offset."""
        return A * np.cos(x + dx) + y0

    @staticmethod
    def _dSHO(x, A, B, y0):
        """Two orthogonal simple harmonic oscillators."""
        return A * np.cos(x) + B * np.sin(x) + y0

    # -- Deprojection Functions -- #

    def calc_vlos(self, vrot, vrad=0.0):
        """
        Calculate the line of sight velocity for each spectrum given the
        rotational and radial velocities at the attached polar angles.

        Args:
            vrot (float): Projected rotation velocity in [m/s].
            vrad (optional[float]): Projected radial velocity in [m/s].

        Returns
            Array of projected line of sight velocities at each polar angle.
        """
        _vrot = vrot * np.cos(self.theta)
        _vrad = vrad * np.sin(self.theta)
        return _vrot + _vrad

    def _deprojected_spectra(self, vrot, vrad=0.0):
        """Returns all deprojected points as an ensemble."""
        vlos = self.calc_vlos(vrot=vrot, vrad=vrad)
        return np.array([np.interp(self.velax, self.velax - dv, spectra)
                         for dv, spectra in zip(vlos, self.spectra)])

    def deprojected_spectrum(self, vrot, vrad=0.0, resample=False,
                             scatter=False):
        """
        Returns ``(x, y[, dy])`` of the collapsed and deprojected spectrum
        using the provided velocities to deproject the data. Different methods
        to resample the data can be applied.

            ``reasmple=False`` - returns the unbinned, shifted pixels.

            ``resample=True`` - shifted pixels are binned onto the
            attached velocity axis.

            ``resample=int(N)`` - shifted pixels are binned onto a velocity
            axis which is a factor of N times coarser than the attached
            velocity axis. ``N=1`` is the same as ``resample=True``.

            ``resample=float(N)`` - shifted pixels are binned onto a velocity
            axis with a spacing of N [m/s]. If the velocity spacing is too fine
            this may result in empty bins and thus NaNs in the spectrum.

        It is important to disgintuish between ``float`` and ``int`` arguments
        for ``resample``.

        Args:
            vrot (float): Rotational velocity in [m/s].
            vrad (optional[float]): Radial velocity in [m/s].
            resample (optional): Type of resampling to be applied.
            scatter (optional[bool]): If the spectrum is resampled, whether to
                return the scatter in each velocity bin.

        Returns:
            A deprojected spectrum, resampled using the provided method.

        """
        vlos = self.calc_vlos(vrot=vrot, vrad=vrad)
        vpnts = self.velax[None, :] - vlos[:, None]
        vpnts, spnts = self._order_spectra(vpnts=vpnts.flatten())
        return self._resample_spectra(vpnts, spnts, resample=resample,
                                      scatter=scatter)

    def _line_centroids(self, method='max', spectra=None, velax=None):
        """
        Return the velocities of the peak pixels.

        Args:
            method (str): Method used to determine the line centroid. Must be
                in ['max', 'quadratic', 'gaussian']. The former returns the
                pixel of maximum value, 'quadratic' fits a quadratic to the
                pixel of maximum value and its two neighbouring pixels (see
                Teague & Foreman-Mackey 2018 for details) and 'gaussian' fits a
                Gaussian profile to the line.
            spectra (Optional[ndarray]): The array of spectra to calculate the
                line centroids on. If nothing is given, use self.spectra. Must
                be on the same velocity axis.
            velax (Optional[ndarray]): Must provide the velocity axis if
                different from the attached array.

        Returns:
            vmax (ndarray): Line centroids.
        """
        method = method.lower()
        spectra = self.spectra if spectra is None else spectra
        velax = self.velax if velax is None else velax
        if method == 'max':
            vmax = np.take(velax, np.argmax(spectra, axis=1))
        elif method == 'quadratic':
            try:
                from bettermoments.methods import quadratic
            except ImportError:
                raise ImportError("Please install 'bettermoments'.")
            vmax = [quadratic(spectrum, x0=velax[0], dx=np.diff(velax)[0])[0]
                    for spectrum in spectra]
            vmax = np.array(vmax)
        elif method == 'gaussian':
            vmax = [annulus._get_gaussian_center(velax, spectrum)
                    for spectrum in spectra]
            vmax = np.array(vmax)
        else:
            raise ValueError("method is not 'max', 'gaussian' or 'quadratic'.")
        return vmax

    def _order_spectra(self, vpnts, spnts=None):
        """Return velocity order spectra."""
        spnts = self.spectra_flat if spnts is None else spnts
        if len(spnts) != len(vpnts):
            raise ValueError("Wrong size in 'vpnts' and 'spnts'.")
        idxs = np.argsort(vpnts)
        return vpnts[idxs], spnts[idxs]

    def _resample_spectra(self, vpnts, spnts, resample=False, scatter=False):
        """
        Resample the spectra to a given velocity axis.

        Args:
            vpnts (ndarray): Array of the velocity values.
            spnts (ndarray): Array of the spectrum values.
            resample (bool/int/float): Describes the resampling method. If
                False, no resample is done and the vpnts and spnts are returned
                as is. If an integer (where True = 1), this samples vpnts on a
                velocity grid at a sampling rate 'resample' times that of the
                originally supplied velocity axis. If a float, this will
                describe the spectral resolution of the sampled grid.
            scatter (bool): If True, return the standard deviation in each bin.

        Returns:
            x (ndarray): Velocity bin centers.
            y (ndarray): Mean of the bin.
            dy (ndarray/None): Standard deviation of the bin.
        """
        if not resample:
            if not scatter:
                return vpnts, spnts
            return vpnts, spnts, np.zeros(vpnts.size)
        if isinstance(resample, (int, bool)):
            bins = int(self.velax.size * int(resample) + 1)
            bins = np.linspace(self.velax[0], self.velax[-1], bins)
        elif isinstance(resample, float):
            bins = np.arange(self.velax[0], self.velax[-1], resample)
            bins += 0.5 * (vpnts.max() - bins[-1])
        else:
            raise TypeError("Resample must be a boolean, int or float.")
        y, x, _ = binned_statistic(vpnts, spnts, statistic='mean', bins=bins)
        x = np.average([x[1:], x[:-1]], axis=0)
        if not scatter:
            return x, y
        dy, _, _ = binned_statistic(vpnts, spnts, statistic='std', bins=bins)
        return x, y, dy

    def _get_masked_spectrum(self, x, y):
        """Return the masked spectrum for fitting."""
        mask = np.logical_and(x >= self.velax_mask[0], x <= self.velax_mask[1])
        return x[mask], y[mask]

    def guess_parameters(self, method='quadratic', fit=True):
        """
        Guess the starting positions by fitting the SHO equation to the line
        peaks. These include the rotational and radial velocities and the
        combined systemtic velocities and projected vertical motions.

        Args:
            method (optional[str]): Method used to measure the velocities of
                the line peaks. Must be in ['max', 'quadradtic', 'gaussian'].
            fit (optional[bool]): Use ``scipy.curve_fit`` to fit the line peaks
                as a function of velocity.

        Returns:
            The rotational, radial and systemic velocities all in [m/s].

        """
        vpeaks = self._line_centroids(method=method)
        vlsr = np.mean(vpeaks)

        vrot = vpeaks[abs(self.theta).argmin()]
        vrot -= vpeaks[abs(self.theta - np.pi).argmin()]
        vrot *= 0.5

        vrad = vpeaks[abs(self.theta - 0.5 * np.pi).argmin()]
        vrad -= vpeaks[abs(self.theta + 0.5 * np.pi).argmin()]
        vrad *= 0.5

        if not fit:
            return vrot, vrad, vlsr
        try:
            return curve_fit(annulus._dSHO, self.theta, vpeaks,
                             p0=[vrot, vrad, vlsr], maxfev=10000)[0]
        except Exception:
            return vrot, vrad, vlsr

    # -- River Functions -- #

    def _grid_river(self, tgrid=None, vgrid=None, spnts=None, **kwargs):
        """Grid the data to plot as a river."""
        from scipy.interpolate import griddata
        if tgrid is None:
            tgrid = np.linspace(self.theta[0], self.theta[-1], self.theta.size)
        if vgrid is None:
            vgrid = self.velax
        if spnts is None:
            spnts = self.spectra_flat
        vpnts = (self.velax[None, :] * np.ones(self.spectra.shape)).flatten()
        tpnts = (self.theta[:, None] * np.ones(self.spectra.shape)).flatten()
        sgrid = griddata((vpnts, tpnts), spnts.flatten(),
                         (vgrid[None, :], tgrid[:, None]),
                         **kwargs)
        return vgrid, tgrid, np.where(np.isfinite(sgrid), sgrid, 0.0)

    def plot_river(self, ax=None, vrot=None, vrad=0.0, tgrid=None, vgrid=None,
                   xlims=None, ylims=None, normalize=True, plot_max=True,
                   method='quadratic', imshow=True, residual=False,
                   significance=True, **kwargs):
        """
        Make a river plot, showing how the spectra change around the azimuth.
        This is a nice way to search for structure within the data.

        Args:
            ax (Optional[axes]): Axis to plot onto.
            vrot (Optional[float]): Rotational velocity used to deprojected the
                spectra. If none is provided, no deprojection is used.
            vrad (Optional[float]): Radial velocity used to deproject the
                spectra.
            tgrid (Optional[ndarray]): Theta grid in [rad] used for gridding
                the data.
            vgrid (Optional[ndarray]): Velocity grid in [m/s] used for gridding
                the data.
            xlims (Optional[list]): Minimum and maximum x range for the figure.
            ylims (Optional[list]): Minimum and maximum y range for the figure.
            normalize (Optional[bool]): Normalize each row such that the peak
                of the line is 1.
            plot_max (Optional[bool]): Plot the line centroid for each
                spectrum.
            method (Optional[str]): The method used to calculate the line
                centroid if plot_max == True. Must be 'max', 'gaussian' or
                'quadratic'.
            imshow (Optional[bool]): Plot using imshow rather than contourf.
            residual (Optional[bool]): If true, subtract the azimuthally
                averaged line profile.
            significance (Optional[bool]): Plot the (n * 3) sigma contours for
                the residuals.

        Returns:
            Matplotlib figure.
        """
        if vrot is None:
            toplot = self.spectra
        else:
            toplot = self._deprojected_spectra(self.calc_vlos(vrot, vrad))
        vgrid, tgrid, sgrid = self._grid_river(spnts=toplot,
                                               tgrid=tgrid,
                                               vgrid=vgrid)
        if normalize:
            sgrid /= np.nanmax(sgrid, axis=1)[:, None]

        if residual:
            scatter = np.nanstd(sgrid, axis=0)
            sgrid -= np.nanmean(sgrid, axis=0)[None, :]

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()

        if imshow:
            vmin = kwargs.pop('vmin', 0 if normalize else None)
            vmax = kwargs.pop('vmax', 1 if normalize else None)
            im = ax.imshow(sgrid, origin='lower', interpolation='nearest',
                           extent=[vgrid[0], vgrid[-1], tgrid[0], tgrid[-1]],
                           aspect='auto', vmin=vmin, vmax=vmax, **kwargs)
        else:
            im = ax.contourf(vgrid, tgrid, sgrid, 50, **kwargs)
        plt.colorbar(im, pad=0.02)

        if residual and significance:
            ax.contour(vgrid, tgrid, abs(sgrid) / scatter[None, :],
                       np.arange(3, 300, 3), colors='k', linewidths=1.0)

        if plot_max:
            vmax = self._line_centroids(method=method,
                                       spectra=sgrid, velax=vgrid)
            ax.errorbar(vmax, self.theta, color='k', fmt='o', mew=0.0, ms=2)

        if xlims is not None:
            ax.set_xlim(xlims[0], xlims[1])
        else:
            ax.set_xlim(vgrid[0], vgrid[-1])
        if ylims is not None:
            ax.set_ylim(ylims[0], ylims[1])
        else:
            ax.set_ylim(tgrid[0], tgrid[-1])

        ax.set_xlabel(r'${\rm Velocity \quad (m\,s^{-1})}$')
        ax.set_ylabel(r'${\rm Polar \,\, Angle \quad (rad)}$')
        return fig

    # -- Plotting Functions -- #

    def plot_spectra(self, ax=None, return_fig=False):
        """
        Plot the attached spectra on the same velocity axis.

        Args:
            ax (optional): Matplotlib axis onto which the data will be plotted.
            return_fig (optional[bool]): Return the figure.

        Returns
            Figure with the attached spectra plotted.
        """
        if ax is None:
            fig, ax = plt.subplots()
        for spectrum in self.spectra:
            ax.step(self.velax, spectrum, where='mid', color='k')
        ax.set_xlabel('Velocity')
        ax.set_ylabel('Intensity')
        ax.set_xlim(self.velax[0], self.velax[-1])
        if return_fig:
            return fig

    @staticmethod
    def _make_axes(ax=None):
        """Make an axis to plot on."""
        if ax is None:
            _, ax = plt.subplots()
        return ax

    @staticmethod
    def _plot_corner(samples):
        """Plot the corner plot for the MCMC."""
        import corner
        labels = [r'${\rm v_{rot}}$', r'${\rm v_{rad}}$',
                  r'${\rm \sigma_{rms}}$', r'${\rm ln(\sigma)}$',
                  r'${\rm ln(\rho)}$']
        if samples.shape[1] == 4:
            labels = labels[:1] + labels[-3:]
        corner.corner(samples, labels=labels, quantiles=[0.16, 0.5, 0.84],
                      show_titles=True)

    @staticmethod
    def _plot_walkers(sampler, nburnin):
        """Plot the walkers from the MCMC."""
        labels = [r'${\rm v_{rot}}$', r'${\rm v_{rad}}$',
                  r'${\rm \sigma_{rms}}$', r'${\rm ln(\sigma)}$',
                  r'${\rm ln(\rho)}$']
        if sampler.chain.T.shape[0] == 4:
            labels = labels[:1] + labels[-3:]
        for s, sample in enumerate(sampler.chain.T):
            _, ax = plt.subplots()
            for walker in sample.T:
                ax.plot(walker, alpha=0.1, color='k')
            ax.set_xlabel('Steps')
            ax.set_ylabel(labels[s])
            ax.axvline(nburnin, ls=':', color='k')
