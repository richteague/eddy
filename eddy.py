"""
Class to derive a rotation velocity profile by minimizing the width of a line.
This implicitly assumes a Gaussian line profile which may not be correct for
optically thick lines or those with significant structure due to the velocity
profile.
"""

import matplotlib.pyplot as plt
from prettyplots.prettyplots import running_mean
from imgcube.imagecube import imagecube
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import flaring.cube as flaring
import numpy as np
import emcee
import corner
import celerite


class linecube:

    param_names = {}
    param_names['SHO'] = [r'$v_{\rm rot}$', r'$\theta$',
                          r'$\sigma_{\rm rms}$', r'$\log\,S_0$',
                          r'$\log\,Q$', r'$\log\,\omega_0$']
    param_names['M32'] = [r'$v_{\rm rot}$', r'$\theta$',
                          r'$\sigma_{\rm rms}$', r'$\log\,\sigma$',
                          r'$\log\,\rho$']

    def __init__(self, path, inc=49., dist=122., x0=0.0, y0=0.0,
                 orientation='east', nearest='top', rmin=30., rmax=270.,
                 nbins=30., verbose=True):
        """
        Initial instance of a line cube based on `imagecube`. The cube must be
        rotated such that the major axis is aligned with the x-axis.

        - Input Variables -

        path:           Relative path to the fits cube.
        inc:            Inclination of the disk in [degrees].
        dist:           Distance to disk in [pc].
        x0, y0:         Offset of star in [arcseconds].
        orientation:    Direction of the blue shifted half of the disk. This is
                        used to calculate the position angles of the pixels.
                        Note that 'east' is to the left.
        nearest:        Which of the sides of the disk is closest to the
                        observer. If 'top' then the centre of the ellipses will
                        shift downwards. If None, assume no flaring.
        rmin:           Minimum radius for the surface profile in [au].
        rmax:           Maximum radius for the surface profile in [au].
        nbins:          Number of radial points between rmin and rmax.
        verbose:        Boolean describing if output message should be used.
        """

        self.path = path
        self.file = path.split('/')[-1]
        self.cube = imagecube(path)
        self.velax = self.cube.velax
        self.verbose = verbose
        self.dist = dist
        self.inc = inc

        self.xaxis, self.yaxis = self.cube.xaxis, self.cube.yaxis
        self.data = self.cube.data

        # Orientation of the disk. Note that this is not the true position
        # angle of the disk, rather just used interally.

        self.x0, self.y0 = x0, y0
        if orientation.lower() not in ['east', 'west']:
            raise ValueError("Orientation must be 'east' or 'west'.")
        self.orientation = orientation.lower()
        if nearest not in ['top', 'bottom', None]:
            raise ValueError("Nearest must be 'top', 'bottom' or 'None'.")
        self.nearest = nearest
        self.pa = 0.0 if self.orientation == 'east' else 180.
        self.rvals, self.tvals = self.cube._deproject(x0=self.x0,
                                                      y0=self.y0,
                                                      inc=self.inc,
                                                      pa=self.pa)

        # Define the radial sampling in [au].
        self.rmin, self.rmax, self.nbins = rmin, rmax, int(nbins)
        self.rpnts = np.linspace(self.rmin, self.rmax, self.nbins)

        # By default the disk is assumed to be geometrically thin.
        self.surface = interp1d(self.rpnts, np.zeros(self.nbins),
                                fill_value='extrapolate')

        return

    def get_rotation_profile(self, rpnts=None, width=None, nwalkers=200,
                             nburnin=50, nsteps=150, sampling=None,
                             kern='M32', plot=False):
        """
        Calculate the rotation profile using Gaussian Processes to model
        non-parametric spectra. The best model should be the smoothest model.

        - Input -

        rpnts:      Array of radial points in [au] to calculate the rotation
                    profile at. If not specified, will default to the sampling
                    used to derive the emission surface.
        width:      Width of the annuli to use. If not specified, will use the
                    full width between rpnts.
        nwalkers:   Number of walkers used for the MCMC.
        nburnin:    Number of samples used for the burn-in of the MCMC.
        nsteps:     Number of samples used to calculate the posterior
                    distributions.
        sampling:   Spatial sampling to be applied for selecting pixels. If
                    None, will use try to use the beamsize of the attached
                    cube. If all pixels are to be considered, use 0.0,
                    otherwise any value will do.
        kern:       Kernel to use for the correlation. Either the simple
                    harmonic oscillator, 'SHO', or the Mattern32 kernel, 'M32'.
                    More information can be found at the celerite website,
                    http://celerite.readthedocs.io/en/stable/python/kernel/.
        plot:       Plot the samples and corner plot for each radial point.

        - Output -

        rpnts:      Radial points used for the sampling.
        vrot:       Percentiles of the posterior distribution of vrot [m/s].
        posang:     Percentiles of the posterior distribution for the position
                    angle [degrees].
        """

        # Define the radial sampling points.
        if rpnts is None:
            rpnts = self.surface.x
        if width is None:
            width = np.mean(np.diff(rpnts))
        if sampling is None:
            sampling = self.cube.bmaj

        # Check the kernel selection.
        if kern not in ['SHO', 'M32']:
            raise ValueError("Must specified either 'SHO' or 'M32'.")
        elif kern == 'SHO':
            log_probability = self._log_probability_SHO
        else:
            log_probability = self._log_probability_M32

        # Cycle through each radial point performing the fitting.
        pcnts = []
        ndim = 6
        for r, radius in enumerate(rpnts):

            # Find the annulus.
            spectra, angles = self._get_annulus(radius, width, sampling)

            # Run the MCMC with **kwargs from the user.
            p0, ndim, vrot = self._get_p0(spectra, kern)
            sampler = emcee.EnsembleSampler(nwalkers, ndim,
                                            log_probability,
                                            args=(spectra, angles, vrot))
            sampler.run_mcmc(p0, nburnin + nsteps)

            # Plot the sampling.
            if plot:
                self._plot_walkers(sampler, nburnin, kern)

            # Save the percentiles of the posterior.
            samples = sampler.chain[:, -nsteps:]
            samples = samples.reshape(-1, samples.shape[-1])
            pcnts.append(np.percentile(samples, [16, 50, 84], axis=0).T)

            # Plot the corner plot.
            if plot:
                corner.corner(samples, quantiles=[0.16, 0.5, 0.84],
                              show_titles=True, labels=self.param_names[kern])

            if self.verbose:
                print("Completed %d out of %d." % (r + 1, len(rpnts)))

        # Parse the variables and return.
        pcnts = np.rollaxis(np.squeeze(pcnts), 0, 3)
        return rpnts, pcnts[0], pcnts[1], pcnts[2:]

    def _get_p0(self, spectra, kern, nwalkers):
        """
        Return the starting positions for the MCMC. The default starting
        hyperparameters were found by testing multiple runs. With a sufficient
        burn-in period it should still yield reasonable results.

        - Input -

        spectra:    Array of the spectra used for the fitting.
        kern:       Kernel used for the fitting.
        nwalkers:   Number of walkers used in the fitting.

        - Returns -

        p0:         Starting positions for the walkers.
        ndim:       Number of dimensions.
        vrot:       Estimated rotation velocity [m/s].
        """

        # Estimate values from the spectra.
        vrot = self._estimate_vrot(spectra)
        noise = np.nanmean([self._estimate_noise(s) for s in spectra])**2

        # Best guesses for parameters.
        p0 = [vrot, 1.0, noise]
        if kern == 'SHO':
            p0 += [3.0, -1.0, 5.0]
        elif kern == 'M32':
            p0 += [-2.0, 7.0]
        else:
            raise ValueError("Kern must be 'SHO' or 'M32'.")
        ndim = len(p0)

        # Include scatter and rescale the PA guess.
        dp0 = np.random.randn(nwalkers * ndim).reshape((nwalkers, ndim))
        p0 = p0[None, :] * (1.0 + 3e-2 * dp0)
        p0[:, 1] -= 1.0
        return p0, ndim, vrot

    def set_emission_surface_analytical(self, psi=None, z_0=None, z_q=1.0,
                                        r_0=None):
        """
        Use an analytical prescription for the emission surface instead. Two
        options are available: a conical shape following Rosenfeld et al.
        (2013) where the surface is described by an opening angle between the
        surface and the midplane (~15 degrees for 12CO). Alternatively use a
        power-law description of the form:

            z(r) = z_0 * (r / r_0)^(z_q)

        which allows for a more flared surface.

        Note that both of these options are monotonically increasing or
        decreasing with radius and thus may be a poor fit in the outer disk
        where the emission surface decreases,

        - Input -

        psi:        Opening angle [degrees] of the conical surface.
        z_0:        Height of the emission surface at r_0 in [au].
        z_q:        Exponent for the power-law surface.
        r_0:        Characteristic radius for the power-law surface. By default
                    this is self.rmin.
        """

        if psi is not None and z_0 is not None:
            raise ValueError("Specify either 'psi' or 'z_0', not both.")

        # Flat surface.
        if psi is not None:
            z = self.rpnts * np.tan(np.radians(psi))
            self.surface = interp1d(self.rpnts, z, fill_value='extrapolate')
            return

        # Power-law surface.
        if r_0 is None:
            r_0 = self.rmin
        z = z_0 * np.power(self.rpnts / r_0, z_q)
        self.surface = interp1d(self.rpnts, z, fill_value='extrapolate')
        return

    def set_emission_surface_data(self, nsigma=3, downsample=1, smooth=1):
        """
        Derive the emission surface profile following the method of Pinte et
        al. (2018) (https://ui.adsabs.harvard.edu/#abs/2017arXiv171006450P).

        - Input -

        nsigma:     The sigma clipping value.
        downsample: Downsample the cube by this factor to improve signal to
                    noise. Note that this shouldn't increase the channel size
                    above the linewidth.
        smooth:     Number of radial points to smooth over.
        """
        if self.nearest is None:
            r = np.linspace(self.rmin, self.rmax, self.nbins)
            z = np.zeros(self.nbins)
        else:
            cube = flaring.linecube(self.path, inc=self.inc, dist=self.dist,
                                    nsigma=nsigma, downsample=downsample)
            r, z, _ = cube.emission_surface(rmin=self.rmin, rmax=self.rmax,
                                            nbins=self.nbins)
            z = z[1]
        if smooth > 1:
            z = running_mean(z, smooth)
        return interp1d(r, z, fill_value='extrapolate')

    def _log_probability_SHO(self, theta, spectra, angles, vkep):
        """
        Log-probability function for the Gaussian Processes approach. This uses
        celerite to calculate the Gaussian processes. The kernel is the sum of
        a white noise term (JitterTerm) and the line (SHOTerm).

        - Input -

        theta:      Free parameters for the fit.
        spectra:    Array of the lines to deproject.
        angles:     Relative position angles of the spectra in [radians].
        vkep:       Keplerian velocity for the given radius in [m/s]. Not used
                    as a full prior but rather to set the bounds for vrot.

        - Output -

        loglike:    Log-likelihood calculated with geroge.
        """

        # Unpack the free parameters.
        vrot, posang, noise, lnS, lnQ, lnw0 = theta

        # Uninformative priors. The bounds for lnw0 are somewhat arbitrary and
        # testing has shown that it makes little difference to the result. It
        # is also highly correlated with lnQ which has a much broader prior.
        if not 0.8 <= vrot / vkep <= 1.2:
            return -np.inf
        if abs(posang) > 0.2:
            return -np.inf
        if noise <= 0.0:
            return -np.inf
        if not np.isfinite(lnS):
            return -np.inf
        if not -16. < lnQ < 0.:
            return -np.inf
        if not 4.0 <= lnw0 <= 6.0:
            return - np.inf

        # Deproject the model and make sure arrays are sorted.
        x = self.velax[None, :] + vrot * np.cos(angles)[:, None]
        if x.shape == spectra.shape:
            x = x.flatten()
        else:
            x = x.T.flatten()
        y = spectra.flatten()
        idxs = np.argsort(x)
        x, y = x[idxs], y[idxs]

        # Generate the Gaussian process.
        k_noise = celerite.terms.JitterTerm(log_sigma=np.log(noise))
        k_line = celerite.terms.SHOTerm(log_S0=lnS, log_Q=lnQ, log_omega0=lnw0)
        kernel = k_noise + k_line
        gp = celerite.GP(kernel, mean=np.nanmean(y), fit_mean=True)
        gp.compute(x)

        # Return the log-likelihood.
        ll = gp.log_likelihood(y, quiet=True)
        return ll if np.isfinite(ll) else -np.inf

    def _log_probability_M32(self, theta, spectra, angles, vkep):
        """
        Log-probability function for the Gaussian Processes approach. This uses
        celerite to calculate the Gaussian processes. The kernel is the sum of
        a white noise term (JitterTerm) and the line (Matern32Term).

        - Input -

        theta:      Free parameters for the fit.
        spectra:    Array of the lines to deproject.
        angles:     Relative position angles of the spectra in [radians].
        vkep:       Keplerian velocity for the given radius in [m/s]. Not used
                    as a full prior but rather to set the bounds for vrot.

        - Output -

        loglike:    Log-likelihood calculated with geroge.
        """

        # Unpack the free parameters.
        vrot, posang, noise, lnsigma, lnrho = theta

        # Uninformative priors. The bounds for lnw0 are somewhat arbitrary and
        # testing has shown that it makes little difference to the result. It
        # is also highly correlated with lnQ which has a much broader prior.
        if not 0.8 <= vrot / vkep <= 1.2:
            return -np.inf
        if abs(posang) > 0.2:
            return -np.inf
        if noise <= 0.0:
            return -np.inf
        if not np.isfinite(lnsigma):
            return -np.inf
        if not np.isfinite(lnrho):
            return -np.inf

        # Deproject the model and make sure arrays are sorted.
        x = self.velax[None, :] + vrot * np.cos(angles)[:, None]
        if x.shape == spectra.shape:
            x = x.flatten()
        else:
            x = x.T.flatten()
        y = spectra.flatten()
        idxs = np.argsort(x)
        x, y = x[idxs], y[idxs]

        # Generate the Gaussian process.
        k_noise = celerite.terms.JitterTerm(log_sigma=np.log(noise))
        k_line = celerite.terms.Matern32Term(log_sigma=lnsigma, log_rho=lnrho)
        kernel = k_noise + k_line
        gp = celerite.GP(kernel, mean=np.nanmean(y), fit_mean=True)
        try:
            gp.compute(x)
        except:
            return -np.inf

        # Return the log-likelihood.
        ll = gp.log_likelihood(y, quiet=True)
        return ll if np.isfinite(ll) else -np.inf

    def _get_annulus(self, radius, width, sampling=0.0):
        """
        Return all pixels which fall within an annulus of the given radius and
        width. The deprojection will take into account any flaring through
        self.surface(). If sampling is provided, will return a pixel roughly
        every `sampling` arcseconds.

        - Input -

        radius:     Radius of the annulus in [au].
        width:      Width of the annulus in [au].
        sampling:   The spatial sampling rate in [arcseconds]. Typically the
                    beam major axis.

        - Returns -

        spectra:    An array of all the pixels in the annulus.
        angles:     Relative position angles for all the spectra.
        """

        # Calculate the deprojected axes.
        xaxis = self.xaxis
        yaxis = self.surface(radius) * np.sin(self.inc) / self.dist
        if self.nearest == 'bottom':
            yaxis *= -1.
        yaxis = (self.yaxis - yaxis) / np.cos(self.inc)

        # Flat arrays of the pixel coordinates.
        rpnts = np.hypot(yaxis[:, None], xaxis[None, :]).flatten() * self.dist
        tpnts = np.arctan2(yaxis[:, None], xaxis[None, :]).flatten()
        xpnts, ypnts = np.meshgrid(self.xaxis, self.yaxis)
        xpnts, ypnts = xpnts.flatten(), ypnts.flatten()
        dflat = self.data.reshape(self.data.shape[0], -1).T

        # Define the radial mask.
        rmask = np.where(abs(rpnts - radius) < 0.5 * width, True, False)

        if sampling > abs(np.mean([np.diff(self.xaxis), np.diff(self.yaxis)])):

            # Create the bins used to sample the pixels further.
            nx = int(abs(self.xaxis[0] - self.xaxis[-1]) / sampling)
            ny = int(abs(self.yaxis[0] - self.yaxis[-1]) / sampling)
            xbins = np.linspace(self.xaxis[0], self.xaxis[-1], nx)
            ybins = np.linspace(self.yaxis[0], self.yaxis[-1], ny)
            dx = 0.5 * np.mean([abs(np.diff(xbins)), abs(np.diff(ybins))])
            rbins = np.hypot(ybins[:, None], xbins[None, :]) * self.dist
            rbins = np.where(abs(rbins - radius) < 4 * width, True, False)

            pixels, angles = [], []

            # Cycle through each bin, selecting all pixels which are possible.
            # Then, for the selection of pixels, randomly select one.
            for xidx, xbin in enumerate(xbins):
                for yidx, ybin in enumerate(ybins):

                    if not rbins[yidx, xidx]:
                        continue

                    in_x = abs(xpnts - xbin) < dx
                    in_y = abs(ypnts - ybin) < dx
                    mask = rmask * in_x * in_y

                    if np.sum(mask) == 0.0:
                        continue

                    temp_pixels = dflat[mask]
                    temp_angles = tpnts[mask]
                    npix = temp_pixels.shape[0]
                    if npix == 1:
                        idx = 0
                    else:
                        idx = np.random.randint(0, npix - 1)
                    pixels += [temp_pixels[idx]]
                    angles += [temp_angles[idx]]

            return np.squeeze(pixels), np.squeeze(angles)
        else:
            return dflat[rmask], tpnts[rmask]

        mask = np.hypot(xaxis[None, :], yaxis[:, None]) - radius / self.dist
        mask = np.where(abs(mask) <= 0.5 * width / self.dist, True, False)
        mask = mask.flatten()

        angles = np.arctan2(yaxis[:, None], xaxis[None, :]).flatten()
        spectra = self.cube.data.reshape((self.cube.data.shape[0], -1)).T
        return spectra[mask], angles[mask]

    def _estimate_vrot(self, spectra):
        """Estimate the rotation velocity from line peaks."""
        centers = np.take(self.velax, np.argmax(spectra, axis=1))
        vmin, vmax = centers.min(), centers.max()
        vlsr = np.average([vmin, vmax])
        return np.average([vlsr - vmin, vmax - vlsr])

    def _estimate_noise(self, spectrum):
        """Estimate the noise in the spectrum's end channels."""
        Tb = spectrum.max()
        x0 = self.velax[spectrum.argmax()]
        dV = np.trapz(spectrum, self.velax) / Tb / np.sqrt(2. * np.pi)
        try:
            x0, dV, _ = curve_fit(gaussian, self.velax, spectrum,
                                  maxfev=10000, p0=[x0, dV, Tb])[0]
        except:
            pass
        noise = np.nanstd(spectrum[abs(self.velax - x0) > 3. * dV])
        return noise if np.isfinite(noise) else np.nanstd(spectrum)

    def _estimate_width(self, spectra):
        """Estimate the Doppler width of the lines."""
        if spectra.ndim == 2:
            if spectra.shape[1] != self.velax.size:
                spectra = spectra.T
        elif spectra.ndim == 1:
            spectra = spectra[None, :]
        dV = np.trapz(spectra, self.velax, axis=1)
        dV /= np.nanmax(spectra, axis=1) * np.sqrt(np.pi)
        return np.nanmean(dV)

    def plot_emission_surface(self, ax=None):
        """Plot the emission surface."""
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.surface.x, self.surface.y)
        ax.set_xlabel('Radius (au)')
        ax.set_ylabel('Height (au)')
        return ax

    def _plot_walkers(self, sampler, nburnin, kern):
        """
        Plot the samples used in the MCMC.

        - Input -

        sampler:    The emcee sampler object.
        nburnin:    The number of steps used for the burn-in.
        kern:       Name of the kernel.
        """
        nwalkers, nsamples, ndim = sampler.chain.shape
        fig, axs = plt.subplots(ncols=ndim)
        for a, ax in enumerate(axs):
            for w in range(nwalkers):
                ax.plot(sampler.chain[w, :, a], alpha=0.1)
            ax.axvline(nburnin, ls='--', color='k')
            ax.set_ylabel(self.param_names[kern][a])
        plt.tight_layout()
        return

    def _fit_width(self, spectrum, failed=1e20):
        """Fit the spectrum with a Gaussian function."""
        Tb = spectrum.max()
        x0 = self.velax[spectrum.argmax()]
        dV = np.trapz(spectrum, self.velax) / Tb / np.sqrt(2. * np.pi)
        try:
            popt, _ = curve_fit(gaussian, self.velax, spectrum,
                                maxfev=10000, p0=[x0, dV, Tb])
            return popt[1]
        except:
            return failed


def gaussian(x, x0, dx, A):
    """Gaussian function with Doppler width."""
    return A * np.exp(-np.power((x-x0) / dx, 2))
