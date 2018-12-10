"""Simple class to deproject spectra and measure the rotation velocity."""

import celerite
import numpy as np
from scipy.stats import binned_statistic
from scipy.optimize import curve_fit
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d


class ensemble(object):

    def __init__(self, spectra, theta, velax, suppress_warnings=True,
                 remove_empty=True, sort_spectra=True):
        """Initialize the class."""

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
            idxs = np.sum(spectra, axis=-1) > 0.0
            self.theta = self.theta[idxs]
            self.spectra = self.spectra[idxs]

        # Easier to use variables.
        self.theta_deg = np.degrees(theta) % 360.
        self.spectra_flat = spectra.flatten()

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
            resample (Optional[bool]): Resample the shifted spectra down to the
                original velocity resolution. Not recommended.
            optimize (Optional[bool]): Optimize the starting positions before
                the MCMC runs.
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

        # Import emcee for the MCMC.
        import emcee
        if resample:
            print("WARNING: Resampling with the GP method is not advised.")

        # Initialize the starting positions.
        if vref is not None and p0 is not None:
            print("WARNING: Initial value of p0 (%.2f) " % (p0[0]) +
                  "used in place of vref (%.2f)." % (vref))
        if not isinstance(vref, (int, float)):
            vref = vref[0]
        p0 = self._guess_parameters_GP(vref) if p0 is None else p0
        if len(p0) != 4:
            raise ValueError('Incorrect length of p0.')
        vref = p0[0]

        # Optimize if necessary.
        if optimize:
            p0 = self._optimize_p0(p0, resample=resample, **kwargs)
        p0 = ensemble._randomize_p0(p0, nwalkers, scatter)
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
            ensemble._plot_walkers(sampler, nburnin)
        if plot_corner:
            ensemble._plot_corner(samples)

        # Return the perncetiles.
        percentiles = np.percentile(samples, [16, 50, 84], axis=0)
        if return_all:
            return percentiles
        return percentiles[:, 0]

    def _optimize_p0(self, p0, resample=True, **kwargs):
        """Optimize the starting positions."""
        from scipy.optimize import minimize
        kwargs['method'] = kwargs.get('method', 'L-BFGS-B')
        res = minimize(self._negative_lnlikelihood, x0=p0,
                       args=(resample), **kwargs)
        if not res.success:
            kwargs['method'] = kwargs.get('method', 'Nelder-Mead')
            kwargs['options'] = {'maxiter': 100000}
            res = minimize(self._negative_lnlikelihood, x0=p0,
                           args=(resample), **kwargs)
            if not res.success:
                print("WARNING: scipy.optimize.minimze did not converge.")
            elif np.isfinite(ensemble._lnprior(res.x, p0[0])):
                p0 = res.x
            else:
                print("WARNING: optimized parameters inconsistent with prior.")
        elif np.isfinite(ensemble._lnprior(res.x, p0[0])):
            p0 = res.x
        else:
            print("WARNING: optimized parameters inconsistent with prior.")
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
        if abs(vrot - vref) / vref > 0.3:
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
        gp = ensemble._build_kernel(x, y, theta)
        if gp is None:
            return -np.inf
        ll = gp.log_likelihood(y, quiet=True)
        return ll if np.isfinite(ll) else -np.inf

    def _lnprobability(self, theta, vref, resample=False):
        """Log-probability function for the MCMC."""
        if ~np.isfinite(ensemble._lnprior(theta, vref)):
            return -np.inf
        return self._lnlikelihood(theta, resample)

    # -- Rotation Velocity by Minimizing Linewidth -- #

    def get_vrot_dV(self, vref=None, resample=False):
        """Infer the rotation velocity by finding the rotation velocity which,
        after shifting all spectra to a common velocity, results in the
        narrowest stacked profile.

        Args:
            vref (Optional[float]): Predicted rotation velocity, typically the
                Keplerian velocity at that radius. Will be used as the starting
                position for the minimization.
            resample (Optional[bool]): Resample the shifted spectra down to the
                original velocity resolution. Not recommended.

        Returns:
            vrot (float): Rotation velocity which minimizes the width.

        """
        vref = self.guess_parameters(fit=True)[0] if vref is None else vref
        bounds = np.array([0.7, 1.3]) * vref
        res = minimize_scalar(self.get_deprojected_width, method='bounded',
                              bounds=bounds, args=(resample))
        return res.x if res.success else np.nan

    @staticmethod
    def _get_p0_dV(x, y):
        """Estimate (x0, dV, Tb) for the spectrum."""
        if x.size != y.size:
            raise ValueError("Mismatch in array shapes.")
        Tb = np.max(x)
        x0 = x[y.argmax()]
        dV = np.trapz(y, x) / Tb / np.sqrt(2. * np.pi)
        return x0, dV, Tb

    @staticmethod
    def _get_gaussian_width(spectrum, velax, fill_value=1e50):
        """Return the absolute width of a Gaussian fit to the spectrum."""
        try:
            dV = curve_fit(ensemble._gaussian, velax, spectrum,
                           p0=ensemble._get_p0_dV(velax, spectrum),
                           maxfev=100000)[0][1]
            return abs(dV)
        except Exception:
            return fill_value

    def get_deprojected_width(self, vrot, resample=True):
        """Return the spectrum from a Gaussian fit."""
        x, y = self.deprojected_spectrum(vrot, resample=resample)
        if resample:
            mask = np.logical_and(x >= self.velax[0], x <= self.velax[-1])
            x, y = x[mask], y[mask]
        return ensemble._get_gaussian_width(y, x)

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
        return Tex * (1. - np.exp(-ensemble._gaussian(x, x0, dV, tau)))

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

    def deprojected_spectrum(self, vrot, resample=True):
        """Returns (x, y) of collapsed deprojected spectrum."""
        vpnts = self.velax[None, :] - vrot * np.cos(self.theta)[:, None]
        vpnts = vpnts.flatten()
        idxs = np.argsort(vpnts)
        vpnts, spnts = vpnts[idxs], self.spectra_flat[idxs]
        if not resample:
            return vpnts, spnts
        spectrum = binned_statistic(vpnts, spnts,
                                    statistic='mean',
                                    bins=self.velax.size,
                                    range=self.velax_range)[0]
        return self.velax, spectrum

    def guess_parameters(self, fit=True, fix_theta=True):
        """Guess vrot and vlsr from the spectra.."""
        vpeaks = np.take(self.velax, np.argmax(self.spectra, axis=1))
        vrot = 0.5 * (np.max(vpeaks) - np.min(vpeaks))
        vlsr = np.mean(vpeaks)
        if not fit:
            return vrot, vlsr
        try:
            if fix_theta:
                return curve_fit(ensemble._SHO, self.theta, vpeaks,
                                 p0=[vrot, vlsr], maxfev=10000)[0]
            else:
                return curve_fit(ensemble._SHOb, self.theta, vpeaks,
                                 p0=[vrot, vlsr, 0.0], maxfev=10000)[0]
        except Exception:
            return vrot, vlsr

    # -- Plotting Functions -- #

    def plot_spectra(self, ax=None):
        """Plot all the spectra."""
        ax = ensemble._make_axes(ax)
        for spectrum in self.spectra:
            ax.step(self.velax, spectrum, where='mid', color='k')
        ax.set_xlabel('Velocity')
        ax.set_ylabel('Intensity')
        ax.set_xlim(self.velax[0], self.velax[-1])
        return ax

    def plot_river(self, vrot=None, ax=None, xlims=None, ylims=None, max=True):
        """Plot a river plot."""
        if vrot is None:
            toplot = self.spectra
        else:
            toplot = self.deprojected_spectra(vrot)
        toplot /= np.nanmax(toplot, axis=1)[:, None]
        toplot = np.where(np.isnan(toplot), 0.0, toplot)
        ax = ensemble._make_axes(ax)
        ax.imshow(toplot, origin='lower', interpolation='nearest',
                  aspect='auto', vmin=0.0, vmax=1.0,
                  extent=[self.velax.min(), self.velax.max(),
                          self.theta.min(), self.theta.max()])
        if max:
            ax.errorbar(np.take(self.velax, np.argmax(toplot, axis=1)),
                        self.theta, color='k', fmt='o', mew=0.0, ms=2)
        if xlims is not None:
            ax.set_xlim(xlims[0], xlims[1])
        if ylims is not None:
            ax.set_ylim(ylims[0], ylims[1])
        ax.set_xlabel(r'${\rm Velocity \, (m\,s^{-1})}$')
        ax.set_ylabel(r'${\rm Polar \,\, Angle \quad (rad)}$')
        return ax

    @staticmethod
    def _make_axes(ax):
        """Make an axis to plot on."""
        import matplotlib.pyplot as plt
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
        import matplotlib.pyplot as plt
        labels = [r'${\rm v_{rot}}$', r'${\rm \sigma_{rms}}$',
                  r'${\rm ln(\sigma)}$', r'${\rm ln(\rho)}$']
        for s, sample in enumerate(sampler.chain.T):
            _, ax = plt.subplots()
            for walker in sample.T:
                ax.plot(walker, alpha=0.1, color='k')
            ax.set_xlabel('Steps')
            ax.set_ylabel(labels[s])
            ax.axvline(nburnin, ls=':', color='k')
