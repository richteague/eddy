"""Simple class to deproject spectra and measure the rotation velocity."""

import celerite
import numpy as np
from scipy.stats import binned_statistic
from scipy.optimize import curve_fit
from scipy.optimize import minimize_scalar


class ensemble:

    def __init__(self, spectra, theta, velax, suppress_warnings=True):
        """Initialize the class."""

        # Suppress warnings.
        if suppress_warnings:
            import warnings
            warnings.filterwarnings("ignore")

        # Disk polar angles.
        self.theta = theta
        self.theta_deg = np.degrees(theta)

        # Intensity / brightness temperature.
        self.spectra = spectra
        self.spectra_flat = spectra.flatten()

        # Velocity axis.
        self.velax = velax
        self.channel = np.diff(velax)[0]
        self.velax_range = (self.velax[0] - 0.5 * self.channel,
                            self.velax[-1] + 0.5 * self.channel)
        self.velax_mask = np.percentile(self.velax, [30, 70])

    # -- Rotation Velocity by Gaussian Process Modelling -- #

    def _lnprior(self, theta, vref):
        """Uninformative log-prior function for MCMC."""
        vrot, noise, lnsigma, lnrho = theta
        if abs(vrot - vref) / vref > 0.3:
            return -np.inf
        if noise <= 0.0:
            return -np.inf
        if not -5.0 < lnsigma < 10.:
            return -np.inf
        if not 0.0 <= lnrho <= 10.:
            return -np.inf
        return 0.0

    def _lnlikelihood(self, theta, resample=False):
        """Log-likelihood function for the MCMC."""

        # Unpack the free parameters.

        vrot, noise, lnsigma, lnrho = theta

        # Deproject the data and resample if requested.

        if resample:
            x, y = self.deprojected_spectrum(vrot)
        else:
            x, y = self.deprojected_spectra(vrot)

        # Remove pesky points. This is not necessary but speeds it up
        # and is useful to remove regions where there is a low
        # sampling of points.

        mask = np.logical_and(x > self.velax_mask[0], x < self.velax_mask[1])
        x, y = x[mask], y[mask]

        # Build the GP model.

        k_noise = celerite.terms.JitterTerm(log_sigma=np.log(noise))
        k_line = celerite.terms.Matern32Term(log_sigma=lnsigma, log_rho=lnrho)
        gp = celerite.GP(k_noise + k_line, mean=np.nanmean(y), fit_mean=True)

        # Calculate and the log-likelihood.

        try:
            gp.compute(x)
        except:
            return -np.inf
        ll = gp.log_likelihood(y, quiet=True)
        return ll if np.isfinite(ll) else -np.inf

    def _lnprobability(self, theta, vref, resample=False):
        """Log-probability function for the MCMC."""
        if ~np.isfinite(self._lnprior(theta, vref)):
            return -np.inf
        return self._lnlikelihood(theta, resample)

    def get_vrot_GP(self, vref=None, resample=False, nwalkers=16, nburnin=300,
                    nsteps=300, scatter=1e-2, plot_walkers=False,
                    plot_corner=False, return_all=False):
        """Get rotation velocity by modelling deprojected spectrum as a GP."""
        import emcee
        if resample:
            print("WARNING: Resampling with the GP method is not advised.")
        vref = self.guess_parameters(fit=True)[0] if vref is None else vref

        # Set up emcee.
        p0 = self._get_p0_GP(vref, nwalkers, scatter)
        sampler = emcee.EnsembleSampler(nwalkers, 4, self._lnprobability,
                                        args=(vref, resample))

        # Run the sampler.
        sampler.run_mcmc(p0, nburnin + nsteps)
        samples = sampler.chain[:, -nsteps:]
        samples = samples.reshape(-1, samples.shape[-1])

        # Diagnosis plots if appropriate.
        if plot_walkers:
            self._plot_walkers(sampler, nburnin)
        if plot_corner:
            self._plot_corner(samples)

        # Return the perncetiles.
        percentiles = np.percentile(samples, [16, 50, 84], axis=0)
        if return_all:
            return percentiles
        return percentiles[:, 0]

    def _get_p0_GP(self, vref, nwalkers, scatter):
        """Estimate (vrot, noise, lnp, lns) for the spectrum."""
        p0 = np.array([vref, np.std(self.spectra[:, :10]),
                       np.log(np.std(self.spectra)), np.log(150.)])
        dp0 = np.random.randn(nwalkers * len(p0)).reshape(nwalkers, len(p0))
        dp0 = np.where(p0 == 0.0, 1.0, p0)[None, :] * (1.0 + scatter * dp0)
        return np.where(p0[None, :] == 0.0, dp0 - 1.0, dp0)

    # -- Rotation Velocity by Minimizing Linewidth -- #

    def _get_p0_dV(self, velax, spectrum):
        """Estimate (x0, dV, Tb) for the spectrum."""
        Tb = np.max(spectrum)
        x0 = velax[spectrum.argmax()]
        dV = np.trapz(spectrum, velax) / Tb / np.sqrt(2. * np.pi)
        return x0, dV, Tb

    def _get_gaussian_width(self, spectrum, fill_value=1e50):
        """Return the absolute width of a Gaussian fit to the spectrum."""
        try:
            dV = curve_fit(self.gaussian, self.velax, spectrum,
                           p0=self._get_p0_dV(self.velax, spectrum),
                           maxfev=100000)[0][1]
            return abs(dV)
        except:
            return fill_value

    def _get_deprojected_width(self, vrot, resample=True):
        """Return the spectrum from a Gaussian fit."""
        if resample:
            x, y = self.deprojected_spectrum(vrot)
        else:
            x, y = self.deprojected_spectra(vrot, sort=True)
            y = y[np.logical_and(x >= self.velax[0], x <= self.velax[-1])]
        return self._get_gaussian_width(y)

    def get_vrot_dV(self, guess=None, resample=True):
        """Get the rotation velocity by minimizing the linewidth."""
        guess = self.guess_parameters(fit=True)[0] if guess is None else guess
        bounds = np.array([0.7, 1.3]) * guess
        return minimize_scalar(self._get_deprojected_width, method='bounded',
                               bounds=bounds, args=(resample)).x

    # -- Line Profile Functions -- #

    def _gaussian(self, x, x0, dV, Tb):
        """Gaussian function."""
        return Tb * np.exp(-np.power((x - x0) / dV, 2.0))

    def _thickline(self, x, x0, dV, Tex, tau):
        """Optically thick line profile."""
        if tau <= 0.0:
            raise ValueError("Must have positive tau.")
        return Tex * (1. - np.exp(-self.gaussian(x, x0, dV, tau)))

    def _SHO(self, x, A, y0):
        """Simple harmonic oscillator."""
        return A * np.cos(x) + y0

    # -- Deprojection Functions -- #

    def deprojected_spectra(self, vrot, sort=True):
        """Returns (x, y) of all deprojected points."""
        vpnts = self.velax[None, :] - vrot * np.cos(self.theta)[:, None]
        vpnts = vpnts.flatten()
        if not sort:
            return vpnts, self.spectra_flat
        return np.squeeze(zip(*sorted(zip(vpnts, self.spectra_flat))))

    def deprojected_spectrum(self, vrot):
        """Returns (x, y) of deprojected and binned spectrum."""
        vpnts, spnts = self.deprojected_spectra(vrot, sort=True)
        spectrum = binned_statistic(vpnts, spnts, statistic='mean',
                                    bins=self.velax.size,
                                    range=self.velax_range)[0]
        return self.velax, spectrum

    def guess_parameters(self, fit=True):
        """Guess vrot and vlsr from the spectra.."""
        vpeaks = np.take(self.velax, np.argmax(self.spectra, axis=1))
        vrot = 0.5 * (np.max(vpeaks) - np.min(vpeaks))
        vlsr = np.mean(vpeaks)
        if not fit:
            return vrot, vlsr
        try:
            return curve_fit(self.SHO, self.theta, vpeaks,
                             p0=[vrot, vlsr], maxfev=10000)[0]
        except:
            return vrot, vlsr

    # - Plotting Functions - #

    def plot_spectra(self, ax=None):
        """Plot all the spectra."""
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots()
        for spectrum in self.spectra:
            ax.step(self.velax, spectrum, where='mid', color='k')
        ax.set_xlabel('Velocity')
        ax.set_ylabel('Intensity')
        ax.set_xlim(self.velax[0], self.velax[-1])
        return ax

    def _plot_corner(self, samples):
        """Plot the corner plot for the MCMC."""
        import corner
        labels = [r'${\rm v_{rot}}$', r'${\rm \sigma_{rms}}$',
                  r'${\rm ln(\sigma)}$', r'${\rm ln(\rho)}$']
        corner.corner(samples, labels=labels, quantiles=[0.34, 0.5, 0.86],
                      show_titles=True)

    def _plot_walkers(self, sampler, nburnin):
        """Plot the walkers from the MCMC."""
        import matplotlib.pyplot as plt
        labels = [r'${\rm v_{rot}}$', r'${\rm \sigma_{rms}}$',
                  r'${\rm ln(\sigma)}$', r'${\rm ln(\rho)}$']
        for s, sample in enumerate(sampler.chain.T):
            fig, ax = plt.subplots()
            for walker in sample.T:
                ax.plot(walker, alpha=0.1)
            ax.set_xlabel('Steps')
            ax.set_ylabel(labels[s])
            ax.axvline(nburnin, ls=':', color='k')
