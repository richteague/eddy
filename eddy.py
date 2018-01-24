"""
Class to derive a rotation velocity profile by minimizing the width of a line.
This implicitly assumes a Gaussian line profile which may not be correct for
optically thick lines or those with significant structure due to the velocity
profile.
"""

import matplotlib.pyplot as plt
import scipy.constants as sc
from prettyplots.prettyplots import percentiles_to_errors
from prettyplots.prettyplots import running_mean
from imgcube.imagecube import imagecube
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.optimize import curve_fit
import flaring.cube as flaring
import numpy as np
import emcee
import corner
import celerite


class linecube:

    param_names = {}
    param_names['SHO'] = [r'$v_{\rm rot}$', r'$\theta$',
                          r'$\sigma_{\rm rms}^2$', r'$\log\,S_0$',
                          r'$\log\,Q$', r'$\log\,\omega_0$']
    param_names['M32'] = [r'$v_{\rm rot}$', r'$\theta$',
                          r'$\sigma_{\rm rms}^2$', r'$\log\,\sigma$',
                          r'$\log\,\rho$']

    def __init__(self, path, inc=49., dist=122., x0=0.0, y0=0.0, mstar=2.33,
                 orientation='east', brightest='top', rmin=30., rmax=270.,
                 nbins=30., verbose=True, mask=None, downsample=1):
        """
        Initial instance of a line cube based on `imagecube`. The cube must be
        rotated such that the major axis is aligned with the x-axis.

        - Input Variables -

        path:           Relative path to the fits cube.
        inc:            Inclination of the disk in [degrees].
        dist:           Distance to disk in [pc].
        x0, y0:         Offset of star in [arcseconds].
        mstar:          Mass of the star in [Msun].
        orientation:    Direction of the blue shifted half of the disk. This is
                        used to calculate the position angles of the pixels.
                        Note that 'east' is to the left.
        brightest:      Which of the sides of the disk is closest to the
                        observer. If 'top' then the centre of the ellipses will
                        shift downwards. If None, assume no flaring.
        rmin:           Minimum radius for the surface profile in [au].
        rmax:           Maximum radius for the surface profile in [au].
        nbins:          Number of radial points between rmin and rmax.
        verbose:        Boolean describing if output message should be used.
        mask:           Fits file containing the mask to use.
        """

        self.path = path
        self.file = path.split('/')[-1]
        self.cube = imagecube(path)
        if downsample > 1:
            self.cube.downsample_cube(downsample)
        self.velax = self.cube.velax
        self.chan = np.nanmean(abs(np.diff(self.velax)))
        self.verbose = verbose
        self.dist = dist
        self.inc = inc

        self.mstar = mstar
        self.xaxis, self.yaxis = self.cube.xaxis, self.cube.yaxis
        self.dpix = np.average([abs(np.diff(self.xaxis)),
                                abs(np.diff(self.yaxis))])
        self.data = self.cube.data

        # Mask the data if appropriate.
        if mask is not None:
            self.mask_path = mask
            self.mask = imagecube(self.mask_path, brightness=False).data
        else:
            self.mask_path = None
            self.mask = np.ones(self.data.shape)
        self.data *= self.mask

        # Orientation of the disk. Note that this is not the true position
        # angle of the disk, rather just used interally.

        self.x0, self.y0 = x0, y0
        if orientation.lower() not in ['east', 'west']:
            raise ValueError("Orientation must be 'east' or 'west'.")
        self.orientation = orientation.lower()
        if brightest not in ['top', 'bottom', None]:
            raise ValueError("Brightest must be 'top', 'bottom' or 'None'.")
        self.brightest = brightest
        self.pa = 0.0 if self.orientation == 'east' else 180.
        coords = self.cube._deproject_polar(x0=self.x0, y0=self.y0,
                                            inc=self.inc, PA=self.pa)
        self.rvals, self.tvals = coords

        # Define the radial sampling in [au] and calculate Keplerian rotation.
        self.rmin, self.rmax, self.nbins = rmin, rmax, int(nbins)
        self.rpnts = np.linspace(self.rmin, self.rmax, self.nbins)
        self.width = np.nanmean(np.diff(self.rpnts))

        # By default the disk is assumed to be geometrically thin.
        self.surface = np.zeros((3, self.nbins))
        self.interp_surface = interp1d(self.rpnts, self.surface[0],
                                       fill_value=(0.0, self.surface[0][-1]),
                                       bounds_error=False)
        return

    @property
    def vkep(self):
        """Keplerian velocity in [m/s]."""
        vkep = np.sqrt(sc.G * self.mstar * 1.988e30 / self.rpnts / sc.au)
        return vkep * np.sin(np.radians(self.inc))

    @property
    def vref(self):
        """Reference velocity in [m/s] including alititude component."""
        top = sc.G * self.mstar * 1.988e30 * np.power(self.rpnts * sc.au, 2)
        bot = np.hypot(self.rpnts * sc.au, self.surface[0] * sc.au)
        return np.sqrt(top / np.power(bot, 3)) * np.sin(np.radians(self.inc))

    def get_rotation_profile_dV(self, sampling=0.0):
        """
        Calculate the rotation profile uby minimizing the width of a Gaussian
        fit to the deprojected data.

        - Input -

        sampling:   Downsample the number of pixels by roughly this factor.

        - Returns -

        fits:       The rotation velocity in [m/s] at each radial point and the
                    best-fist position angle.
        """

        vrot = []
        for r, radius in enumerate(self.rpnts):
            spectra, angles = self._get_annulus(radius, sampling)
            vrot += [minimize(self._deprojected_dV,
                              (self._estimate_vrot(spectra, angles), 0.0),
                              args=(spectra, angles),
                              method='Nelder-Mead').x]
        return np.squeeze(vrot).T

    def _deprojected_dV(self, theta, spectra, angles):
        """Return the width of the deprojected spectra."""

        vrot, posang = theta

        # Deproject the model and make sure arrays are sorted.
        x = self.velax[None, :] - vrot * np.cos(angles + posang)[:, None]
        if x.shape == spectra.shape:
            x = x.flatten()
        else:
            x = x.T.flatten()
        y = spectra.flatten()
        idxs = np.argsort(x)
        x, y = x[idxs], y[idxs]

        # Bin the data down to the true resolution.
        bins = np.linspace(self.velax[0] - 0.5 * self.chan,
                           self.velax[-1] + 0.5 * self.chan,
                           self.velax.size + 1)
        idxs = np.digitize(x, bins)
        avgy = [np.nanmean(y[idxs == i]) for i in range(1, bins.size)]
        avgy = np.array(avgy)
        stdy = [np.nanstd(y[idxs == i]) for i in range(1, bins.size)]
        stdy = np.array(stdy)

        # Fit a Gaussian profile to this data.
        Tb = np.nanmax(avgy)
        dV = self._estimate_width(spectra)
        x0 = self.velax[avgy.argmax()]
        try:
            popt, _ = curve_fit(gaussian, self.velax, avgy, sigma=stdy,
                                p0=[x0, dV, Tb], absolute_sigma=True)
        except:
            return 1e20
        return popt[1]

    def get_rotation_profile(self, rpnts=None, width=None, nwalkers=100,
                             nburnin=150, nsteps=50, sampling=None,
                             kern='M32', plot=False, fit_theta=True):
        """
        Calculate the rotation profile using Gaussian Processes to model
        non-parametric spectra. The best model should be the smoothest model.
        Note that within the burn-in, sometimes the hyperparameters will not
        have sufficiently burnt in, although (vrot, pa) will have.

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
        sampling:   Spatial sampling to be applied for selecting pixels.
        kern:       Kernel to use for the correlation. Either the simple
                    harmonic oscillator, 'SHO', or the Mattern32 kernel, 'M32'.
                    More information can be found at the celerite website,
                    http://celerite.readthedocs.io/en/stable/python/kernel/.
        plot:       Plot the samples and corner plot for each radial point.
        fit_theta:  Include the relative position angle as a free parameter.

        - Output -

        rpnts:      Radial points used for the sampling.
        vrot:       Percentiles of the posterior distribution of vrot [m/s].
        posang:     Percentiles of the posterior distribution for the position
                    angle [degrees].
        """

        # Define the radial sampling points.
        if rpnts is None:
            rpnts = self.rpnts
        if width is None:
            width = np.mean(np.diff(rpnts))
        if sampling is None:
            sampling = 0.0

        # Check the kernel selection.
        if kern not in ['SHO', 'M32']:
            raise ValueError("Must specified either 'SHO' or 'M32'.")
        elif kern == 'SHO':
            log_probability = self._log_probability_SHO
        else:
            log_probability = self._log_probability_M32

        # Cycle through each radial point performing the fitting.
        pcnts = []
        for r, radius in enumerate(rpnts):

            # Find the annulus and normalise the spectra.
            spectra, angles = self._get_annulus(radius, sampling)

            # Run the MCMC with **kwargs from the user.
            vkep = np.sqrt(sc.G * self.mstar * 1.988e30 / radius / sc.au)
            vkep *= np.sin(np.radians(self.inc))
            vrot = self._estimate_vrot(spectra, angles)
            p0, ndim = self._get_p0(spectra, kern, nwalkers, vrot, fit_theta)

            # Skip any empty annuli.
            if len(spectra) == 0.0:
                pcnts.append(np.zeros((ndim, 3)))
                continue

            # TODO: Is it better to have more walkers or steps?
            sampler = emcee.EnsembleSampler(nwalkers, ndim,
                                            log_probability,
                                            args=(spectra, angles, vkep))
            sampler.run_mcmc(p0, nburnin + nsteps)

            # Correct the label terms.
            labels = list(self.param_names[kern])
            if not fit_theta:
                labels.pop(1)
            if len(labels) != ndim:
                raise ValueError("Wrong number of parameters.")

            # Plot the sampling.
            if plot:
                self._plot_walkers(sampler, nburnin, labels)

            # Save the percentiles of the posterior.
            samples = sampler.chain[:, -nsteps:]
            samples = samples.reshape(-1, samples.shape[-1])
            pcnts.append(np.percentile(samples, [16, 50, 84], axis=0).T)

            # Plot the corner plot.
            if plot:
                corner.corner(samples, quantiles=[0.16, 0.5, 0.84],
                              show_titles=True, labels=labels)

            if self.verbose:
                print("Completed %d out of %d." % (r + 1, len(rpnts)))

        # Parse the variables and return.
        return np.rollaxis(np.squeeze(pcnts), 0, 3)

    def _get_p0(self, spectra, kern, nwalkers, vkep, fit_theta):
        """
        Return the starting positions for the MCMC. The default starting
        hyperparameters were found by testing multiple runs. With a sufficient
        burn-in period it should still yield reasonable results.

        - Input -

        spectra:    Array of the spectra used for the fitting.
        kern:       Kernel used for the fitting.
        nwalkers:   Number of walkers used in the fitting.
        vkep:       The keplerian velocity the center of the annulus.
        fit_theta:  Inlcude a free parameter for the relative position angle.

        - Returns -

        p0:         Starting positions for the walkers.
        ndim:       Number of dimensions.
        """

        # Best guesses for parameters.
        noise = np.nanmean([self._estimate_noise(s) for s in spectra])**2
        if fit_theta:
            p0 = [vkep, 1.0, noise]
        else:
            p0 = [vkep, noise]
        if kern == 'SHO':
            p0 += [12.0, -12.0, 5.0]
        elif kern == 'M32':
            p0 += [2.0, 7.0]
        else:
            raise ValueError("Kern must be 'SHO' or 'M32'.")
        p0 = np.squeeze(p0)
        ndim = int(p0.size)

        # Include scatter and rescale the PA guess.
        dp0 = np.random.randn(int(nwalkers * ndim))
        dp0 = dp0.reshape((int(nwalkers), int(ndim)))
        p0 = p0[None, :] * (1.0 + 3e-2 * dp0)
        if fit_theta:
            p0[:, 1] -= 1.0
        return p0, ndim

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

        if psi is not None:
            z = self.rpnts * np.tan(np.radians(psi))
        else:
            if r_0 is None:
                r_0 = self.rmin
            z = z_0 * np.power(self.rpnts / r_0, z_q)
        dz = np.zeros(z.size)
        self.surface = np.vstack([z, dz, dz])
        self.interp_surface = interp1d(self.rpnts, self.surface[0],
                                       fill_value=(0.0, self.surface[0][-1]),
                                       bounds_error=False)
        return self.surface

    def set_emission_surface_data(self, nsigma=3, downsample=1, smooth=1,
                                  minTb=0.0):
        """
        Derive the emission surface profile following the method of Pinte et
        al. (2018) (https://ui.adsabs.harvard.edu/#abs/2017arXiv171006450P).
        TODO: Find someway to include uncertainties on this.

        - Input -

        nsigma:     The sigma clipping value.
        downsample: Downsample the cube by this factor to improve signal to
                    noise. Note that this shouldn't increase the channel size
                    above the linewidth.
        smooth:     Number of radial points to smooth over.
        """
        if self.brightest is None:
            r = np.linspace(self.rmin, self.rmax, self.nbins)
            z = np.zeros(self.nbins)
        else:
            cube = flaring.linecube(self.path, inc=self.inc, dist=self.dist,
                                    nsigma=nsigma, downsample=downsample,
                                    mask=self.mask_path)
            r, z, _ = cube.emission_surface(rmin=self.rmin, rmax=self.rmax,
                                            nbins=self.nbins, mph=minTb)
        if smooth > 1:
            z = np.squeeze([running_mean(zz, smooth) for zz in z])
        self.surface = percentiles_to_errors(z)
        self.interp_surface = interp1d(self.rpnts, self.surface[0],
                                       fill_value=(0.0, self.surface[0][-1]),
                                       bounds_error=False)
        return self.surface

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
        try:
            vrot, posang, noise, lnS, lnQ, lnw0 = theta
        except ValueError:
            vrot, noise, lnS, lnQ, lnw0 = theta
            posang = 0.0

        # Uninformative priors. The bounds for lnw0 are somewhat arbitrary and
        # testing has shown that it makes little difference to the result. It
        # is also highly correlated with lnQ which has a much broader prior.
        if abs(vrot - vkep) / vkep > 0.3:
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
            return -np.inf

        # Deproject the model and make sure arrays are sorted.
        x = self.velax[None, :] - vrot * np.cos(angles + posang)[:, None]
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
        try:
            vrot, posang, noise, lnsigma, lnrho = theta
        except ValueError:
            vrot, noise, lnsigma, lnrho = theta
            posang = 0.0

        # Uninformative priors. The bounds for lnw0 are somewhat arbitrary and
        # testing has shown that it makes little difference to the result. It
        # is also highly correlated with lnQ which has a much broader prior.
        if abs(vrot - vkep) / vkep > 0.3:
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
        x = self.velax[None, :] - vrot * np.cos(angles + posang)[:, None]
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

    def get_coordinates(self, niter=5):
        """
        For each pixel return the disk polar coordaintes r, theta in [au, rad],
        respectively. This takes into account the emission surface so returns
        an array for the near and far cone.

        - Input -

        niter:      Number of iterations to include. More is more accurate but
                    slower. Not tested which is the best.

        - Returns -

        near:       Deprojected radial positions [au] and position angles [red]
                    for the near side cone.
        far:        Deprojected radial positions [au] and position angles [rad]
                    for the far side cone.
        """
        r_near, r_far = None, None
        for _ in range(niter):
            r_near, t_near = self._get_coords_single(r_near, 'near')
            r_far, t_far = self._get_coords_single(r_far, 'far')
        return [r_near, t_near], [r_far, t_far]

    def _get_annulus(self, radius, sampling=0, which='near'):
        """
        Return all pixels which fall within an annulus of the given radius and
        width. The deprojection will take into account any flaring through
        self.surface(). If sampling is provided, will return a pixel roughly
        every `sampling` arcseconds.

        - Input -

        radius:     Radius of the annulus in [au].
        which:      Which side of the disk, near or far.
        sampling:   Return a sample roughly a factor `sample` fewer.

        - Returns -

        spectra:    An array of all the pixels in the annulus.
        angles:     Relative position angles for all the spectra.
        """

        # Calculate the deprojected axes.
        near, far = self.get_coordinates()
        near_r, near_t = near[0].flatten(), near[1].flatten()
        far_r, far_t = far[0].flatten(), far[1].flatten()
        dflat = self.data.reshape(self.data.shape[0], -1).T

        # Select the near-side points using both sides of the disk.
        dr = 0.5 * self.width
        mask_n = np.where(abs(near_r - radius) <= dr, True, False)
        mask_f = np.where(abs(far_r - radius) <= dr, True, False)
        mask_n = mask_n.flatten()
        mask_f = mask_f.flatten()

        # Mask the arrays.
        if which == 'both':
            spectra = np.concatenate([dflat[mask_n], dflat[mask_f]])
            angles = np.concatenate([near_t[mask_n], far_t[mask_f]])
        elif which == 'near':
            spectra, angles = dflat[mask_n], near_t[mask_n]
        elif which == 'far':
            spectra, angles = dflat[mask_f], far_t[mask_f]
        else:
            raise ValueError("'which' should be 'near', 'far' or 'both'.")

        # Apply the sampling.
        if sampling > 0:
            mask = np.random.randint(0, sampling, size=angles.size)
            mask = np.where(mask == 0, True, False)
            if np.nansum(mask) > 0:
                spectra, angles = spectra[mask], angles[mask]

        return spectra, angles

    def _estimate_vrot_old(self, spectra):
        """Estimate the rotation velocity from line peaks."""
        centers = np.take(self.velax, np.argmax(spectra, axis=1))
        vmin, vmax = centers.min(), centers.max()
        vlsr = np.average([vmin, vmax])
        return np.average([vlsr - vmin, vmax - vlsr])

    def _estimate_noise(self, spectra, nchan=15):
        """Estimate the noise in the spectrum's end channels."""
        if spectra.shape[0] != self.velax.shape[0]:
            spectra = spectra.T
        return np.nanstd([spectra[:nchan], spectra[-nchan:]])

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

    def plot_rotation_velocities(self, ax=None):
        """Plot the rotation velocities."""
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.rpnts, self.vkep, label='Keplerian')
        ax.plot(self.rpnts, self.vref, label='Reference')
        ax.legend()
        ax.set_xlabel('Radius (au)')
        ax.set_ylabel('Rotation (m/s)')
        return ax

    def plot_emission_surface(self, ax=None):
        """Plot the emission surface."""
        if ax is None:
            fig, ax = plt.subplots()
        ax.errorbar(self.rpnts, self.surface[0], self.surface[1:], capsize=0.0)
        ax.set_xlabel('Radius (au)')
        ax.set_ylabel('Height (au)')
        return ax

    def _get_coords_single(self, rpnts=None, side='near'):
        """Return the radius and position angle of each pixel."""
        xsky, ysky = np.meshgrid(self.xaxis*self.dist, self.yaxis*self.dist)
        if rpnts is None:
            rpnts = np.zeros(xsky.shape)
        else:
            if rpnts.shape != xsky.shape:
                raise ValueError("'rpnts' must have 'xsky' shape.")
        zpnts = self.interp_surface(rpnts)
        if side == 'near':
            zpnts *= -1.
        elif side != 'far':
            raise ValueError("'side' must be 'near' or 'far'.")
        xdisk = xsky
        ydisk = ysky / np.cos(np.radians(self.inc))
        ydisk += zpnts * np.tan(np.radians(self.inc))
        return np.hypot(ydisk, xdisk), np.arctan2(ydisk, xdisk)

    def _plot_walkers(self, sampler, nburnin, labels):
        """Plot the samples used in the MCMC."""
        nwalkers, nsamples, ndim = sampler.chain.shape
        fig, axs = plt.subplots(ncols=ndim, figsize=(ndim*3.0, 2.0))
        for a, ax in enumerate(axs):
            for w in range(nwalkers):
                ax.plot(sampler.chain[w, :, a], alpha=0.1)
            ax.axvline(nburnin, ls='--', color='k')
            ax.set_ylabel(labels[a])
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

    def _estimate_vrot(self, spectra, angles):
        """Estimate the rotation velocity from line peaks."""
        centers = np.take(self.velax, np.argmax(spectra, axis=1))
        vlsr = np.median([centers.min(), centers.max()])
        vrot = centers.max() - vlsr
        try:
            vrot, _ = curve_fit(offsetSHO, angles, centers, p0=[vrot, vlsr])
            return vrot[0]
        except:
            return self._estimate_vrot_old(spectra)


def gaussian(x, x0, dx, A):
    """Gaussian function with Doppler width."""
    return A * np.exp(-np.power((x-x0) / dx, 2))


def offsetSHO(theta, A, y0):
    """Simple harmonic oscillator with an offset."""
    return A * np.cos(theta) + y0
