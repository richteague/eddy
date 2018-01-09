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
from scipy.optimize import minimize
from scipy.stats import binned_statistic_2d
import flaring.cube as flaring
import numpy as np
import emcee
import george


class linecube:

    def __init__(self, path, inc=0.0, dist=122., x0=0.0, y0=0.0,
                 orientation='east', nearest='top', nsigma=3,
                 downsample=1, rmin=30., rmax=350., nbins=30., smooth=False,
                 verbose=True):
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
        nsigma:         Clip all voxels below nsigma * rms when calculating the
                        emission surface.
        downsample:     Number of channels to average over when calculating the
                        emission surface.
        rmin:           Minimum radius for the surface profile in [au].
        rmax:           Maximum radius for the surface profile in [au].
        nbins:          Number of radial points between rmin and rmax.
        smooth:         Number of points used in a running mean to smooth data.
        verbose:        Boolean describing if output message should be used.
        """

        self.path = path
        self.file = path.split('/')[-1]
        self.cube = imagecube(path)
        self.velax = self.cube.velax
        self.verbose = verbose
        self.dist = dist
        self.inc = inc

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

        # Use the flaring module to calculate the emission surface.
        # We only need z(r) so everything else is forgotten.

        self.nsigma = nsigma
        self.downsample = downsample
        self.rmin, self.rmax, self.nbins = rmin, rmax, nbins
        self.smooth = smooth

        if self.verbose:
            print("Calculating emission surface...")
        self.surface = self.get_emission_surface()
        if self.verbose:
            print("Done.")

        return

    def gaussian_processes(self, rpnts=None, width=None, nwalkers=50,
                           nsteps=50, nburnin=50):
        """
        Calculate the rotation profile using Gaussian Processes to model
        non-parametric spectra. The best model should be the smoothest model.
        This does not require a model beforehand which should speed up the
        fitting.

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

        - Output -

        rpnts:      Radial points used for the sampling.
        vrot:       Percentiles of the posterior distribution of vrot [m/s].
        posang:     Percentiles of the posterior distribution for the position
                    angle [degrees].
        hyper:      Posterior distributions of the hyper-parameters used in the
                    fitting.
        """

        # Define the radial sampling points.
        if rpnts is None:
            rpnts = self.surface.x
        if width is None:
            width = np.mean(np.diff(rpnts))

        # Cycle through each radial point performing the fitting.
        pcnts = []
        for radius in rpnts:

            # Find the annulus.
            spectra, angles = self._get_annulus(radius, width)

            # Estimate the starting positions.
            p0 = np.array([self._estimate_vrot(spectra),
                           0.0,
                           0.5 * np.nanvar(spectra),
                           self._estimate_width(spectra)])
            dp = 1e-2 * np.random.randn(nwalkers * 2).reshape((nwalkers, 4))
            p0 *= 1.0 + dp

            # Run the MCMC with **kwargs from the user.
            sampler = emcee.EnsembleSampler(nwalkers, 4,
                                            self._log_probability_gp,
                                            args=(spectra, angles))
            sampler.run_mcmc(np.squeeze(p0).T, nburnin + nsteps)

            # Save the percentiles of the posterior.
            samples = sampler.chain[:, -nsteps:]
            samples = samples.reshape(-1, samples.shape[-1])
            pcnts.append(np.percentile(samples, [16, 50, 84], axis=0).T)

        # Parse the variables and return.
        pcnts = np.rollaxis(np.squeeze(pcnts), 0, 3)
        return rpnts, pcnts[0], pcnts[1], pcnts[2:]

    def get_rotation_profile(self, rpnts=None, width=None, errors=False,
                             nwalkers=50, nburnin=50, nsteps=50):
        """
        Derive the rotation profile by minimizing the width of the average of
        the deprojected spectra. The inital value is found by using scipy's
        minimize function which is relatively fast.

        If errors are wanted, thess values are used as the models for emcee
        runs which estimate the posterior on vrot. This is relatively time
        consuming so make sure that the simple minimize approach works well
        first.

        - Input Variables -

        rpnts:      Array of radial points to calculate the rotation profile.
                    If None then will use the radial sampling of the surface
                    profile.
        width:      Width of the annuli in [arcsec]. If None is specified, will
                    use the spacing of rpnts.
        errors:     If True, use an MCMC approach to estimate uncertainties on
                    vrot. This will take much longer.
        nwalkers:   Number of walkers used for the MCMC runs.
        nburnin:    Number of steps used for the burn-in of the MCMC run.
        nsteps:     Number of steps to use for the production MCMC runs.

        - Output -

        rpnts:      Radial sampling points in [au].
        vrot:       Rotation profile in [m/s] at the sampled points. If errors
                    is True, this is a [3 x len(rpnts)] array with the
                    [16, 50, 84] percentiles for each radial point, otherwise
                    it is just a len(rpnts) array.
        angle:      Relative position angle of the annulus in [degrees].
        """

        # Define the radial sampling points.
        if rpnts is None:
            rpnts = self.surface.x
        if width is None:
            width = np.mean(np.diff(rpnts))

        # Use scipy.optimize.minimize to find the rortation velocity at each
        # radial point. Allows for the position angle to be free too.
        if self.verbose:
            print("Running minimization for %d points..." % self.nbins)

        p0 = []
        models = []
        for radius in rpnts:
            spectra, angles = self._get_annulus(radius, width)

            if spectra.size < 3:
                p0.append([np.nan, np.nan])
                continue

            vrot = self._estimate_vrot(spectra)
            res = minimize(self._to_minimize, (vrot, 0.0),
                           args=(spectra, angles), method='Nelder-Mead')
            p0.append(res.x)
            models.append(self._spectral_deproject(res.x[0], spectra,
                                                   angles, res.x[1]))
        p0 = np.squeeze(p0).T
        models = np.squeeze(models)
        if self.verbose:
            print("Done.")

        # Smooth the data if appropriate.
        if self.smooth > 1:
            p0 = np.squeeze([running_mean(p, self.smooth) for p in p0])

        # If no errors are required, return the values.
        if not errors:
            return rpnts, p0[0], np.degrees(p0[1])

        # TODO: Probably a hugely more efficient way to do this.
        pcnts = []
        if self.verbose:
            print("Using models to run MCMC...")

        for radius, model in zip(rpnts, models):
            vrot = self._estimate_vrot(spectra)
            noise = self._estimate_noise(model)
            spectra, angles = self._get_annulus(radius, width)

            if spectra.shape[0] < 3:
                pcnts.append(np.zeros((2, 3)))
                continue

            p0 = [vrot + np.random.randn(nwalkers), np.random.randn(nwalkers)]
            args = (spectra, angles, model, noise)
            sampler = emcee.EnsembleSampler(nwalkers, 2,
                                            self._log_probability,
                                            args=args)
            sampler.run_mcmc(np.squeeze(p0).T, nsteps+nburnin)
            samples = sampler.chain[:, -nsteps:]
            samples = samples.reshape((-1, samples.shape[-1]))
            pcnts.append(np.percentile(samples, [16, 50, 84], axis=0).T)
        if self.verbose:
            print("Done.")

        pcnts = np.rollaxis(np.squeeze(pcnts), 0, 3)
        return rpnts, pcnts[0], pcnts[1]

    def get_emission_surface(self):
        """
        Derive the emission surface profile. See flaring.cube.linecube for
        more help.

        - Returns -

        surface:    Interpolation function for z(r), with both r and z in [au].
                    The values are linearly interpolated and extrapolated
                    beyond the bounds.
        """
        if self.nearest is None:
            r = np.linspace(self.rmin, self.rmax, self.nbins)
            z = np.zeros(self.nbins)
        else:
            cube = flaring.linecube(self.path, inc=self.inc, dist=self.dist,
                                    nsigma=self.nsigma,
                                    downsample=self.downsample)
            r, z, _ = cube.emission_surface(rmin=self.rmin, rmax=self.rmax,
                                            nbins=self.nbins)
            z = z[1]
        if self.smooth > 1:
            z = running_mean(z, self.smooth)
        return interp1d(r, z, fill_value='extrapolate')

    def _log_probability_gp(self, theta, spectra, angles):
        """
        Log-probability function for the Gaussian Processes approach. We keep
        nearly all functionality within this function which is probably not a
        good idea.

        - Input -

        theta:      Free parameters for the fit: (vrot, posang, sigma, dV).
                    These have units of [m/s], [radians], [K] and [m/s]
                    respectively.
        spectra:    Array of the lines to deproject.
        angles:     Relative position angles of the spectra in [radians].

        - Output -

        loglike:    Log-likelihood calculated with geroge.

        """

        # Unpack the free parameters.
        vrot, posang, sigma, dV = theta

        # Enforce flat priors.
        if vrot <= 0.0:
            return -np.inf
        if abs(posang) > 0.2:
            return -np.inf
        if sigma > np.nanmax(spectra):
            return -np.inf
        if dV <= 0.0:
            return -np.inf

        # Deproject the model.
        x = self.velax[None, :] + vrot * np.cos(angles)[:, None]
        if x.shape == spectra.shape:
            x = x.flatten()
        else:
            x = x.T.flatten()
        y = spectra.flatten()

        # Generate the Gaussian process.
        white_noise = np.log(np.nanvar(y[abs(x) > 3. * dV]))
        if not np.isfinite(white_noise):
            white_noise = np.log(np.nanvar(y))
        kernel = sigma**2 * george.kernels.ExpSquaredKernel(dV**2)
        gp = george.GP(kernel, mean=np.nanmean(y), fit_mean=True,
                       white_noise=white_noise, fit_white_noise=True)
        gp.compute(x)

        # Return the log-likelihood.
        ll = gp.log_likelihood(y, quiet=True)
        return ll if np.isfinite(ll) else -np.inf

    def _log_probability(self, theta, spectra, angles, model, noise):
        """
        Log-probability function for the MCMC modelling. Wrapper for _log_prior
        and _log_likelihood functions.
        """
        lp = self._log_prior(theta)
        if np.isfinite(lp):
            return self._log_likelihood(theta, spectra, angles, model, noise)
        return -np.inf

    def _log_likelihood(self, theta, spectra, angles, model, noise):
        """
        Log-likelihood function for the MCMC modelling.

        theta:      Free parameters for the fit: vrot [m/s] and pa [degrees].
        spectra:    Spectra to deproject and average over.
        angles:     Relative position angles of the pixels [radians].
        model:      Model spectra from the scipy.optimize.minimize fit.
        noise:      RMS noise of the model calculate with _estimate_rms.

        - Returns -

        lnx2:       Log-chi-squared value.
        """
        attempt = self._spectral_deproject(theta[0], spectra, angles,
                                           np.radians(theta[1]))
        lnx2 = np.nansum(np.power((attempt - model) / noise, 2))
        return -0.5 * (lnx2 + np.log(noise**2 * np.sqrt(2. * np.pi)))

    def _log_prior(self, theta):
        """
        Log-prior function for the MCMC modelling.

        - Input -

        theta:      Free parameters for the fit: vrot [m/s] and pa [degrees].

        - Output -

        lp:         Log prior value, 0.0 if allowed, -np.inf otherwise.
        """
        if theta[0] <= 0.0:
            return -np.inf
        if abs(theta[1]) > 10.:
            return - np.inf
        return 0.0

    def _spectral_deproject(self, vrot, spectra, angles, pa=0.0):
        """
        Deperoject the spectra to a common center. It is assumed that all share
        the same velocity axis: self.cube.velax.

        - Input Variables -

        vrot:       Rotation velocity used for the deprojection in [m/s].
        spectra:    Array of spectra to deproject.
        angles:     Position angle of the pixel in [radians], measured East
                    from the blue-shifted major axis.
        pa:         Optional position angle value.

        - Returns -

        deproj:     Average of all deprojected spectra.
        """
        deproj = [np.interp(self.velax, self.velax + vrot * np.cos(t + pa), s)
                  for s, t in zip(spectra, angles)]
        return np.average(deproj, axis=0)

    def _get_annulus(self, radius, width):
        """
        Return an annulus of pixels at a given radius and width. Will take into
        account the offset from a flared emission surface.

        - Input Variables -

        radius:     Radius of the annulus in [au].
        width:      Width of the annulus in [au].

        - Returns -

        spectra:    An array of all the pixels in the annulus.
        angles:     Relative position angles for all the spectra.
        """

        xaxis = self.cube.xaxis
        yaxis = self.surface(radius) * np.sin(self.inc) / self.dist
        if self.nearest == 'bottom':
            yaxis *= -1.
        yaxis = (self.cube.yaxis - yaxis) / np.cos(self.inc)

        mask = np.hypot(xaxis[None, :], yaxis[:, None]) - radius / self.dist
        mask = np.where(abs(mask) <= 0.5 * width / self.dist, True, False)
        mask = mask.flatten()

        angles = np.arctan2(yaxis[:, None], xaxis[None, :]).flatten()
        spectra = self.cube.data.reshape((self.cube.data.shape[0], -1)).T
        return spectra[mask], angles[mask]

    def _bin_annulus(self, size, xaxis, yaxis, mask):
        """
        Bin the annulus into near beamsize areas and only select one value from
        each of these. This helps reduce the number of pixels we are working
        with, speeding up the calculation, and removes possible correlations.

        - Input -

        size:       Size of the bin size in the same units as xaxis and yaxis.
        xaxis:      X-axis of the points.
        yaxis:      Y-axis of the points.
        mask:       Flattened boolean mask of the annulus.

        - Returns -

        mask:       Mask sampled such that only one sample per bin.
        """

        # Define the bin edges.
        dx, dy = 0.5 * np.mean(np.diff(xaxis)), 0.5 * np.mean(np.diff(yaxis))
        xedge = np.linspace(xaxis[0] - dx, xaxis[-1] + dx,
                            int(np.ceil(abs(xaxis[-1] - xaxis[0]) / size)))
        yedge = np.linspace(yaxis[0] - dy, yaxis[-1] + dy,
                            int(np.ceil(abs(yaxis[-1] - yaxis[0]) / size)))

        # Calculate the (x, y) coordinate of each of the masked values.
        xpnts, ypnts = np.meshgrid(xaxis, yaxis)
        xpnts = xpnts.flatten()[mask]
        ypnts = ypnts.flatten()[mask]
        dummy = np.arange(ypnts.size)
        _, _, _, bins = binned_statistic_2d(xpnts, ypnts, dummy,
                                            bins=[xedge, yedge])

        return mask

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

    def _to_minimize(self, theta, spectra, angles):
        """Return width of the average of deprojected spectra."""
        vrot, pa = theta
        if abs(pa) > 0.4:
            return 1e20
        deprojected = self._spectral_deproject(vrot, spectra, angles, pa)
        return self._fit_width(deprojected)


def gaussian(x, x0, dx, A):
    """Gaussian function with Doppler width."""
    return A * np.exp(-np.power((x-x0) / dx, 2))
