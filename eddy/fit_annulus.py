"""
A class to load an annulus of spectra and ther associated polar angles. By
shifting and stacking the spectra, one can measure extremely precise rotational
velocities, as described in Teague et al. (2018a, 2018c).

The main functions of interest are:

    get_vrot_dV: Find the rotation velocity which minimized the width of the
        stacked spectrum, as used in Teague et al. (2018a).

    get_vrot_SNR: Find the rotation velocity which maximises the
        signal-to-noise ratio of the stacked line profile, as used in Yen et
        al. (2016, 2018) or Matra et al. (2016).

    get_vrot_GP: Find the rotation velocity which returns the smoothest line
        profile when stacked as modelled by a Gaussian Process, as used in
        Teague et al. (2018c). This function works without having to assume a
        line profile and thus works additionally for absorption lines and
        (hyper-)fine structure.

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from scipy.optimize import curve_fit
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d

# Check is 'celerite' is installed.
try:
    import celerite
    celerite_installed = True
except ImportError:
    celerite_installed = False


class annulus(object):

    def __init__(self, spectra, theta, velax, suppress_warnings=True,
                 remove_empty=True, sort_spectra=True):
        """Initialize the class. Assume theta in [rad] and velax in [m/s]."""

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
            idxs = np.sum(self.spectra, axis=-1) > 0.0
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
        self.velax_mask = np.percentile(self.velax, [30, 70])

    # -- Rotation Velocity by Gaussian Process Modelling -- #

    def get_vrot_GP(self, vref=None, p0=None, resample=False, optimize=True,
                    nwalkers=64, nburnin=300, nsteps=300, scatter=1e-3,
                    plot_walkers=True, plot_corner=True, return_all=False,
                    **kwargs):
        """Infer the rotation velocity of the annulus by finding the velocity
        which after deprojecting the spectra to a common velocity produces the
        smoothest spectrum.

        Args:
            vref (Optional[float]): Predicted rotation velocity, typically the
                Keplerian velocity at that radius. Will be used as bounds for
                searching for the true velocity, set as +\- 30%.
            p0 (Optional[list]): Initial parameters for the minimization. Will
                override any guess for vref.
            resample (Optional[bool]): Resample the shifted spectra by this
                factor. For example, resample = 2 will shift and bin the
                spectrum down to sampling rate twice that of the original data.
                Not recommended for the GP approach.
            optimize (Optional[bool]): Optimize the starting positions before
                the MCMC runs. If an integer, the number of iterations to use
                of optimization.
            nwalkers (Optional[int]): Number of walkers used for the MCMC runs.
            nburnin (Optional[int]): Number of steps used to burn in walkers.
            nsteps (Optional[int]): Number of steps taken to sample posteriors.
            scatter (Optional[float]): Scatter applied to the starting
                positions before running the MCMC.
            plot_walkers (Optional[bool]): Plot the trace of the walkers.
            plot_corner (Optional[bool]): Plot the covariances of the
                posteriors using corner.py.
            return_all (Optional[bool]): If True, return the percentiles of the
                posterior distributions for all parameters, otherwise just the
                percentiles for the rotation velocity.
            **kwargs (Optional[dict]): Additional kwargs to pass to minimize.

        Returns:
            percentiles (ndarray): The 16th, 50th and 84th percentiles of the
                posterior distribution for all four parameters if return_all is
                True otherwise just for the rotation velocity.

        """

        # Check is celerite is installed.
        if not celerite_installed:
            raise ImportError("Must install 'celerite' to use GP method.")

        # Import emcee for the MCMC.
        import emcee
        if resample:
            print("WARNING: Resampling with the GP method is not advised.")

        # Initialize the starting positions.
        if vref is not None and p0 is not None:
            print("WARNING: Initial value of p0 (%.2f) " % (p0[0]) +
                  "used in place of vref (%.2f)." % (vref))
        if vref is not None:
            if not isinstance(vref, (int, float)):
                vref = vref[0]
        else:
            vref = self.guess_parameters(fit=True)[0]
        p0 = self._guess_parameters_GP(vref) if p0 is None else p0
        if len(p0) != 4:
            raise ValueError('Incorrect length of p0.')
        vref = p0[0]

        # Optimize if necessary.
        if optimize:
            p0 = self._optimize_p0(p0, N=optimize, resample=resample, **kwargs)
            vref = p0[0]
        p0 = annulus._randomize_p0(p0, nwalkers, scatter)
        if np.any(np.isnan(p0)):
            raise ValueError("WARNING: NaNs in the p0 array.")

        # Set up emcee.
        sampler = emcee.EnsembleSampler(nwalkers, 4, self._lnprobability,
                                        args=(vref, resample))

        # Run the sampler.
        sampler.run_mcmc(p0, nburnin + nsteps)
        samples = sampler.chain[:, -nsteps:]
        samples = samples.reshape(-1, samples.shape[-1])

        # Diagnosis plots if appropriate.
        if plot_walkers:
            annulus._plot_walkers(sampler, nburnin)
        if plot_corner:
            annulus._plot_corner(samples)

        # Return the perncetiles.
        percentiles = np.percentile(samples, [16, 50, 84], axis=0)
        if return_all:
            return percentiles
        return percentiles[:, 0]

    def _optimize_p0(self, p0, N=1, resample=True, **kwargs):
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
        from scipy.optimize import minimize
        verbose = kwargs.pop('verbose', False)

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
        nlnL = self._negative_lnlikelihood(p0, resample=resample)

        # Cycle through the required number of iterations.
        for i in range(int(N)):

            # Define the bounds.
            bounds = (0.8 * p0[0], 1.2 * p0[0])
            bounds = [bounds, (0.0, None), (-15.0, 10.0), (0.0, 10.0)]

            # First minimize hyper parameters, holding vrot constant.
            res = minimize(self._negative_lnlikelihood_hyper, x0=p0[1:],
                           args=(p0[0], resample), bounds=bounds[1:],
                           **kwargs)
            if res.success:
                p0_temp = p0
                p0_temp[1:] = res.x
                nlnL_temp = self._negative_lnlikelihood(p0_temp, resample)
                if nlnL_temp < nlnL:
                    p0 = p0_temp
                    nlnL = nlnL_temp
            else:
                if verbose:
                    print('Failed mimization: %s' % res.message)

            # Second, minimize vrot holding the hyper parameters constant.
            res = minimize(self._negative_lnlikelihood_vrot, x0=p0[0],
                           args=(p0[1:], resample), bounds=[bounds[0]],
                           **kwargs)
            if res.success:
                p0_temp = p0
                p0_temp[0] = res.x
                nlnL_temp = self._negative_lnlikelihood(p0_temp, resample)
                if nlnL_temp < nlnL:
                    p0 = p0_temp
                    nlnL = nlnL_temp
                else:
                    if verbose:
                        print('Failed mimization: %s' % res.message)

            # Final minimization with everything.
            res = minimize(self._negative_lnlikelihood, x0=p0,
                           args=(resample), bounds=bounds,
                           **kwargs)
            if res.success:
                p0_temp = res.x
                nlnL_temp = self._negative_lnlikelihood(p0_temp, resample)
                if nlnL_temp < nlnL:
                    p0 = p0_temp
                    nlnL = nlnL_temp
            else:
                if verbose:
                    print('Failed mimization: %s' % res.message)

        return p0

    def _guess_parameters_GP(self, vref, fit=True):
        """Guess the starting positions from the spectra."""
        vref = self.guess_parameters(fit=fit)[0] if vref is None else vref
        noise = int(min(10, self.spectra.shape[1] / 3.0))
        noise = np.std([self.spectra[:, :noise], self.spectra[:, -noise:]])
        ln_sig = np.log(np.std(self.spectra))
        ln_rho = np.log(150.)
        return np.array([vref, noise, ln_sig, ln_rho])

    @staticmethod
    def _randomize_p0(p0, nwalkers, scatter):
        """Estimate (vrot, noise, lnp, lns) for the spectrum."""
        dp0 = np.random.randn(nwalkers * len(p0)).reshape(nwalkers, len(p0))
        dp0 = np.where(p0 == 0.0, 1.0, p0)[None, :] * (1.0 + scatter * dp0)
        return np.where(p0[None, :] == 0.0, dp0 - 1.0, dp0)

    def _negative_lnlikelihood_vrot(self, vrot, hyperparams, resample=False):
        """Negative log-likelihood function with vrot as only variable."""
        theta = np.insert(hyperparams, 0, vrot)
        nll = -self._lnlikelihood(theta, resample)
        return nll if np.isfinite(nll) else 1e15

    def _negative_lnlikelihood_hyper(self, hyperparams, vrot, resample=False):
        """Negative log-likelihood function with hyperparams as variables."""
        theta = np.insert(hyperparams, 0, vrot)
        nll = -self._lnlikelihood(theta, resample)
        return nll if np.isfinite(nll) else 1e15

    def _negative_lnlikelihood(self, theta, resample=False):
        """Negative log-likelihood function for optimization."""
        nll = -self._lnlikelihood(theta, resample)
        return nll if np.isfinite(nll) else 1e15

    @staticmethod
    def _build_kernel(x, y, theta):
        """Build the GP kernel. Returns None if gp.compute(x) fails."""
        noise, lnsigma, lnrho = theta[1:]
        k_noise = celerite.terms.JitterTerm(log_sigma=np.log(noise))
        k_line = celerite.terms.Matern32Term(log_sigma=lnsigma, log_rho=lnrho)
        gp = celerite.GP(k_noise + k_line, mean=np.nanmean(y), fit_mean=True)
        try:
            gp.compute(x)
        except Exception:
            return None
        return gp

    def _get_masked_spectra(self, theta, resample=True):
        """Return the masked spectra for fitting."""
        vrot = theta[0]
        x, y = self.deprojected_spectrum(vrot, resample=resample)
        mask = np.logical_and(x >= self.velax_mask[0], x <= self.velax_mask[1])
        return x[mask], y[mask]

    @staticmethod
    def _lnprior(theta, vref):
        """Uninformative log-prior function for MCMC."""
        vrot, noise, lnsigma, lnrho = theta
        if abs(vrot - vref) / vref > 0.4:
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

        # Deproject the data and resample if requested.
        x, y = self._get_masked_spectra(theta, resample=resample)

        # Build the GP model and calculate the log-likelihood.
        gp = annulus._build_kernel(x, y, theta)
        if gp is None:
            return -np.inf
        ll = gp.log_likelihood(y, quiet=True)
        return ll if np.isfinite(ll) else -np.inf

    def _lnprobability(self, theta, vref, resample=False):
        """Log-probability function for the MCMC."""
        if ~np.isfinite(annulus._lnprior(theta, vref)):
            return -np.inf
        return self._lnlikelihood(theta, resample)

    # -- Rotation Velocity by Maximizing SNR -- #

    def get_vrot_SNR(self, vref=None, resample=False, signal='int',
                     weight_SNR=True):
        """Infer the rotation velocity by finding the rotation velocity which,
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
            vrot (float): Rotation velocity which maximies the width.
        """
        if signal not in ['max', 'int']:
            raise ValueError("'signal' must be either 'max' or 'int'.")
        vref = self.guess_parameters(fit=True)[0] if vref is None else vref
        bounds = np.array([0.7, 1.3]) * vref
        res = minimize_scalar(self.get_deprojected_nSNR, method='bounded',
                              args=(resample, signal, weight_SNR),
                              bounds=bounds)
        return res.x if res.success else np.nan

    def get_deprojected_nSNR(self, vrot, resample=False, signal='int',
                             weighted_SNR=True):
        """Return the negative SNR of the deprojected spectrum."""
        x, y = self.deprojected_spectrum(vrot, resample=resample)
        x0, dx, A = annulus._fit_gaussian(x, y)
        noise = np.std(x[(x - x0) / dx > 3.0])  # Check: noise will vary.
        if signal == 'max':
            SNR = A / noise
        else:
            if weighted_SNR:
                w = annulus._gaussian(x, x0, dx, (np.sqrt(np.pi) * dx)**-1)
            else:
                w = np.ones(x.size)
            mask = (x - x0) / dx <= 1.0
            SNR = np.trapz((y * w)[mask], x=x[mask])
        return -SNR

    # -- Rotation Velocity by Minimizing Linewidth -- #

    def get_vrot_dV(self, vref=None, resample=False):
        """Infer the rotation velocity by finding the rotation velocity which,
        after shifting all spectra to a common velocity, results in the
        narrowest stacked profile.

        Args:
            vref (Optional[float]): Predicted rotation velocity, typically the
                Keplerian velocity at that radius. Will be used as the starting
                position for the minimization.
            resample (Optional[bool]): Resample the shifted spectra by this
                factor. For example, resample = 2 will shift and bin the
                spectrum down to sampling rate twice that of the original data.

        Returns:
            vrot (float): Rotation velocity which minimizes the width.

        """
        vref = self.guess_parameters(fit=True)[0] if vref is None else vref
        bounds = np.array([0.7, 1.3]) * vref
        res = minimize_scalar(self.get_deprojected_width, method='bounded',
                              bounds=bounds, args=(resample))
        return res.x if res.success else np.nan

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
            return popt, cvar
        return popt

    @staticmethod
    def _get_gaussian_width(x, y, fill_value=1e50):
        """Return the absolute width of a Gaussian fit to the spectrum."""
        dV = annulus._fit_gaussian(x, y)[1]
        if np.isfinite(dV):
            return abs(dV)
        return fill_value

    @staticmethod
    def _get_gaussian_center(x, y):
        """Return the line center from a Gaussian fit to the spectrum."""
        x0 = annulus._fit_gaussian(x, y)[0]
        if np.isfinite(x0):
            return abs(x0)
        return x[np.argmax(y)]

    def get_deprojected_width(self, vrot, resample=True):
        """Return the spectrum from a Gaussian fit."""
        x, y = self.deprojected_spectrum(vrot, resample=resample)
        mask = np.logical_and(x >= self.velax[0], x <= self.velax[-1])
        return annulus._get_gaussian_width(x[mask], y[mask])

    # -- Line Profile Functions -- #

    @staticmethod
    def _gaussian(x, x0, dV, Tb):
        """Gaussian function."""
        return Tb * np.exp(-np.power((x - x0) / dV, 2.0))

    @staticmethod
    def _thickline(x, x0, dV, Tex, tau):
        """Optically thick line profile."""
        if tau <= 0.0:
            raise ValueError("Must have positive tau.")
        return Tex * (1. - np.exp(-annulus._gaussian(x, x0, dV, tau)))

    @staticmethod
    def _SHO(x, A, y0):
        """Simple harmonic oscillator."""
        return A * np.cos(x) + y0

    @staticmethod
    def _SHOb(x, A, y0, dx):
        """Simple harmonic oscillator with offset."""
        return A * np.cos(x + dx) + y0

    # -- Deprojection Functions -- #

    def deprojected_spectra(self, vrot):
        """Returns (x, y) of all deprojected points as an ensemble."""
        spectra = []
        for theta, spectrum in zip(self.theta, self.spectra):
            shifted = interp1d(self.velax - vrot * np.cos(theta), spectrum,
                               bounds_error=False, fill_value=np.nan)
            spectra += [shifted(self.velax)]
        return np.squeeze(spectra)

    def deprojected_spectrum(self, vrot, resample=False, scatter=False):
        """Returns (x, y) of collapsed deprojected spectrum."""
        vpnts = self.velax[None, :] - vrot * np.cos(self.theta)[:, None]
        vpnts, spnts = self._order_spectra(vpnts=vpnts.flatten())
        return self._resample_spectra(vpnts, spnts, resample=resample,
                                      scatter=scatter)

    def deprojected_spectrum_maximum(self, resample=False, method='quadratic',
                                     scatter=False):
        """Deprojects data such that their max values are aligned."""
        vmax = self.line_centroids(method=method)
        vpnts = np.array([self.velax - dv for dv in vmax - np.median(vmax)])
        vpnts, spnts = self._order_spectra(vpnts=vpnts.flatten())
        return self._resample_spectra(vpnts, spnts, resample=resample,
                                      scatter=scatter)

    def line_centroids(self, method='max', spectra=None, velax=None):
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
        """Resample the spectra."""
        if not resample:
            return vpnts, spnts
        bins = self.velax.size * resample + 1
        bins = np.linspace(self.velax[0], self.velax[-1], int(bins))
        y, x, _ = binned_statistic(vpnts, spnts, statistic='mean', bins=bins)
        x = np.average([x[1:], x[:-1]], axis=0)
        if not scatter:
            return x, y
        dy, _, _ = binned_statistic(vpnts, spnts, statistic='std', bins=bins)
        return x, y, dy

    def guess_parameters(self, fit=True, fix_theta=True, method='quadratic'):
        """Guess vrot and vlsr from the spectra.."""
        vpeaks = self.line_centroids(method=method)
        vrot = 0.5 * (np.max(vpeaks) - np.min(vpeaks))
        vlsr = np.mean(vpeaks)
        if not fit:
            return vrot, vlsr
        try:
            if fix_theta:
                return curve_fit(annulus._SHO, self.theta, vpeaks,
                                 p0=[vrot, vlsr], maxfev=10000)[0]
            else:
                return curve_fit(annulus._SHOb, self.theta, vpeaks,
                                 p0=[vrot, vlsr, 0.0], maxfev=10000)[0]
        except Exception:
            return vrot, vlsr

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

    def plot_river(self, ax=None, vrot=None, tgrid=None, vgrid=None,
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
            ax: The fugure axes instance.
            cb: The colorbar instance
        """
        if vrot is None:
            toplot = self.spectra
        else:
            toplot = self.deprojected_spectra(vrot)
        vgrid, tgrid, sgrid = self._grid_river(spnts=toplot,
                                               tgrid=tgrid,
                                               vgrid=vgrid)
        if normalize:
            sgrid /= np.nanmax(sgrid, axis=1)[:, None]

        if residual:
            scatter = np.nanstd(sgrid, axis=0)
            sgrid -= np.nanmean(sgrid, axis=0)[None, :]

        ax = annulus._make_axes(ax)

        if imshow:
            im = ax.imshow(sgrid, origin='lower', interpolation='nearest',
                           extent=[vgrid[0], vgrid[-1], tgrid[0], tgrid[-1]],
                           aspect='auto', **kwargs)
        else:
            im = ax.contourf(vgrid, tgrid, sgrid, 50, **kwargs)
        cb = plt.colorbar(im, pad=0.02)

        if residual and significance:
            ax.contour(vgrid, tgrid, abs(sgrid) / scatter[None, :],
                       np.arange(3, 300, 3), colors='k', linewidths=1.0)

        if plot_max:
            vmax = self.line_centroids(method=method,
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
        return ax, cb

    # -- Plotting Functions -- #

    def plot_spectra(self, ax=None):
        """Plot all the spectra."""
        ax = annulus._make_axes(ax)
        for spectrum in self.spectra:
            ax.step(self.velax, spectrum, where='mid', color='k')
        ax.set_xlabel('Velocity')
        ax.set_ylabel('Intensity')
        ax.set_xlim(self.velax[0], self.velax[-1])
        return ax

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
        labels = [r'${\rm v_{rot}}$', r'${\rm \sigma_{rms}}$',
                  r'${\rm ln(\sigma)}$', r'${\rm ln(\rho)}$']
        corner.corner(samples, labels=labels, quantiles=[0.16, 0.5, 0.84],
                      show_titles=True)

    @staticmethod
    def _plot_walkers(sampler, nburnin):
        """Plot the walkers from the MCMC."""
        labels = [r'${\rm v_{rot}}$', r'${\rm \sigma_{rms}}$',
                  r'${\rm ln(\sigma)}$', r'${\rm ln(\rho)}$']
        for s, sample in enumerate(sampler.chain.T):
            _, ax = plt.subplots()
            for walker in sample.T:
                ax.plot(walker, alpha=0.1, color='k')
            ax.set_xlabel('Steps')
            ax.set_ylabel(labels[s])
            ax.axvline(nburnin, ls=':', color='k')
