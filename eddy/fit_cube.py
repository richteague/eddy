"""
Class to load up a velocity map and fit a Keplerian profile to it. The main
functions of interest are:

    disk_coords - To be filled in.
    keplerian   - To be filled in.
"""

import numpy as np
from astropy.io import fits
import scipy.constants as sc


class rotationmap:

    def __init__(self, path, error=None, clip=None, downsample=None):
        """Initialize."""

        # Read in the data and position axes.
        self.data = fits.getdata(path)
        self.header = fits.getheader(path)
        try:
            self.error = fits.getdata(error)
        except:
            print("No uncertainties found, assuming uncertainties of 10%.")
            self.error = 0.1 * self.data
        self.xaxis = self._read_position_axis(a=1)
        self.yaxis = self._read_position_axis(a=2)
        self.dpix = abs(np.diff(self.xaxis)).mean()

        # Clip and downsample the cube to speed things up.
        if clip is not None:
            self._clip_cube(clip)
        if downsample is not None:
            self._downsample_cube(downsample)

        # Estimate the systemic velocity.
        self.vlsr = np.nanmedian(self.data)

        # Beam parameters. TODO: Make sure CASA beam tables work.
        try:
            self.bmaj = self.header['bmaj'] * 3600.
            self.bmin = self.header['bmin'] * 3600.
            self.bpa = self.header['bpa']
        except KeyError:
            self.bmaj = self.dpix
            self.bmin = self.dpix
            self.bpa = 0.0

    # -- Fitting functions. -- #

    def fit_keplerian(self, p0, params, r_min=None, r_max=None, optimize=False,
                      nwalkers=None, nburnin=300, nsteps=100, scatter=1e-3,
                      plot_walkers=True, plot_corner=True):
        """
        Fit a Keplerian rotation profile to the data.

        Args:
            p0:
            params:
            r_min:
            r_max:
            optimize:
            nwalkers:
            scatter:

        Returns:
            what.
        """

        # Load up emcee.
        try:
            import emcee
        except ImportError:
            raise ImportError("Cannot find emcee.")

        # Check the dictionary. May need some more work.
        params = rotationmap._verify_dictionary(params)

        # Calculate the inverse variance mask.
        r_min = r_min if r_min is not None else 0.0
        r_max = r_max if r_max is not None else 1e5
        self.ivar = self._calculate_ivar(x0=params['x0'], y0=params['y0'],
                                         inc=params['inc'], PA=params['PA'],
                                         z0=params['z0'], psi=params['psi'],
                                         tilt=params['tilt'], r_min=r_min,
                                         r_max=r_max)

        # Run an initial optimization using scipy.minimize.
        if optimize:
            raise NotImplementedError("Not working yet.")

        # Make sure all starting positions are valid.
        # COMING SOON.

        labels = rotationmap._get_labels(params)
        if len(labels) != len(p0):
            raise ValueError("Mismatch in labels and p0. Check for integers.")

        # Set up the MCMC.
        ndim = len(p0)
        nwalkers = 2 * ndim if nwalkers is None else nwalkers
        p0 = rotationmap._random_p0(p0, scatter, nwalkers)

        sampler = emcee.EnsembleSampler(nwalkers, ndim, self._ln_probability,
                                        args=(params))
        sampler.run_mcmc(p0, nburnin + nsteps)
        samples = sampler.chain[:, -int(nsteps):]
        samples = samples.reshape(-1, samples.shape[-1])

        # Diagnostic plots.
        if plot_walkers:
            rotationmap.plot_walkers(sampler.chain.T, nburnin, labels)
        if plot_corner:
            functions.plot_corner(samples, labels)

        return

    @staticmethod
    def _random_p0(p0, scatter, nwalkers):
        """Get the starting positions."""
        p0 = np.squeeze(p0)
        dp0 = np.random.randn(nwalkers * len(p0)).reshape(nwalkers, len(p0))
        dp0 = np.where(p0 == 0.0, 1.0, p0)[None, :] * (1.0 + scatter * dp0)
        return np.where(p0[None, :] == 0.0, dp0 - 1.0, dp0)

    def _ln_likelihood(self, params):
        """Log-likelihood function. Simple chi-squared likelihood."""
        lnx2 = np.power((self.data - self._make_model(params)), 2)
        lnx2 = -0.5 * np.sum(lnx2 * self.ivar)
        return lnx2 if np.isfinite(lnx2) else -np.inf

    def _ln_probability(self, theta, params):
        """Log-probablility function."""
        params = rotationmap._populate_dictionary(theta, params)
        if np.isfinite(self._ln_priors(params)):
            return self._ln_likelihood(params)
        return -np.inf

    def _ln_prior(self, params):
        """Log-priors. Uniform and uninformative."""
        if abs(params['x0']) > 0.5:
            return -np.inf
        if abs(params['y0']) > 0.5:
            return -np.inf
        if not 0. < params['inc'] < 90.:
            return -np.inf
        if not -360. < params['PA'] < 360.:
            return -np.inf
        if not 0.0 < params['mstar'] < 5.0:
            return -np.inf
        if abs(self.vlsr - params['vlsr'] * 1e3) > 1e3:
            return -np.inf
        if not 0. <= params['z0'] < 1.0:
            return -np.inf
        if not 0. < params['psi'] < 2.0:
            return -np.inf
        if not -1.0 < params['tilt'] < 1.0:
            return -np.inf
        return 0.0

    def _calc_ivar(self, x0=0.0, y0=0.0, inc=None, PA=0.0, z0=0.0, psi=0.0,
                   tilt=0.0, r_min=0.0, r_max=1e5):
        """Calculate the inverse variance including radius mask."""
        rvals = self.disk_coords(x0=x0, y0=y0, inc=inc, PA=PA,
                                 z0=z0, psi=psi, tilt=tilt)[0]
        mask = np.logical_and(rvals >= r_min, rvals <= r_max)
        mask = np.logical_and(mask, np.isfinite(self.data))
        return np.where(mask, np.power(self.error, -2.0), 0.0)

    @staticmethod
    def _get_labels(params):
        """Return the labels of the parameters to fit."""
        idxs, labs = [], []
        for k in params.keys():
            if type(params[k]) is int:
                idxs.append(params[k])
                labs.append(k)
        return np.array(labs)[np.argsort(idxs)]

    @staticmethod
    def _populate_dictionary(theta, params):
        """Populate the dictionary of free parameters."""
        for k, key in enumerate(params.keys()):
            if type(params[key]) is int:
                params[key] = theta[params[key]]
                return params

    @staticmethod
    def _verify_dictionary(params):
        """Check there are the the correct keys."""
        if params.get('x0') is None:
            params['x0'] = 0.0
        if params.get('y0') is None:
            params['y0'] = 0.0
        if params.get('z0') is None:
            params['z0'] = 0.0
        if params.get('psi') is None:
            params['psi'] = 0.0
        if params.get('dist') is None:
            params['dist'] = 100.
        if params.get('beam') is None:
            params['beam'] = False
        return params

    # -- Deprojection functions. -- #

    def disk_coords(self, x0=0.0, y0=0.0, inc=0.0, PA=0.0, z0=0.0, psi=0.0,
                    tilt=0.0, frame='polar'):
        """
        Get the disk coordinates given certain geometrical parameters and an
        emission surface. The emission surface is parameterized as a powerlaw
        profile: z(r) = z0 * (r / 1")^psi. For a razor thin disk, z0 = 0.0,
        while for a conical disk, as described in Rosenfeld et al. (2013),
        psi = 1.0.

        Args:
            x0 (Optional[float]): -
            y0 (Optional[float]): -
            inc (Optional[float]): -
            PA (Optional[float]): -
            frame (Optional[str]): -
            z0 (Optional[float]): -
            psi (Optional[float]): -
            nearest (Optional[str]): -
            frame (Optional[str]): -

        Returns:
            c1 (ndarryy): -
            c2 (ndarray): -
            c3 (ndarray): -
        """

        # Check the input variables.

        frame = frame.lower()
        if frame not in ['cartesian', 'polar']:
            raise ValueError("frame must be 'cartesian' or 'polar'.")

        # Define the emission surface function. This approach should leave
        # some flexibility for more complex emission surface parameterizations.

        def func(r):
            return z0 * np.power(r, psi)

        # Calculate the pixel values.

        if frame == 'cartesian':
            c1, c2 = self._get_flared_cart_coords(x0, y0, inc, PA, func, tilt)
            c3 = func(np.hypot(c1, c2))
        else:
            c1, c2 = self._get_flared_polar_coords(x0, y0, inc, PA, func, tilt)
            c3 = func(c1)
        return c1, c2, c3

    def _rotate_coords(self, x, y, PA):
        """Rotate (x, y) by PA [deg]."""
        x_rot = x * np.cos(np.radians(PA)) - y * np.sin(np.radians(PA))
        y_rot = y * np.cos(np.radians(PA)) + x * np.sin(np.radians(PA))
        return x_rot, y_rot

    def _deproject_coords(self, x, y, inc):
        """Deproject (x, y) by inc [deg]."""
        return x, y / np.cos(np.radians(inc))

    def _get_cart_sky_coords(self, x0, y0):
        """Return caresian sky coordinates in [arcsec, arcsec]."""
        return np.meshgrid(self.xaxis - x0, self.yaxis - y0)

    def _get_polar_sky_coords(self, x0, y0):
        """Return polar sky coordinates in [arcsec, radians]."""
        x_sky, y_sky = self._get_cart_sky_coords(x0, y0)
        return np.hypot(y_sky, x_sky), np.arctan2(x_sky, y_sky)

    def _get_midplane_cart_coords(self, x0, y0, inc, PA):
        """Return cartesian coordaintes of midplane in [arcsec, arcsec]."""
        x_sky, y_sky = self._get_cart_sky_coords(x0, y0)
        x_rot, y_rot = self._rotate_coords(y_sky, x_sky, -PA)
        return self._deproject_coords(x_rot, y_rot, inc)

    def _get_midplane_polar_coords(self, x0, y0, inc, PA):
        """Return the polar coordinates of midplane in [arcsec, radians]."""
        x_mid, y_mid = self._get_midplane_cart_coords(x0, y0, inc, PA)
        return np.hypot(y_mid, x_mid), np.arctan2(y_mid, x_mid)

    def _get_flared_polar_coords(self, x0, y0, inc, PA, func, tilt):
        """Return polar coordinates of surface in [arcsec, radians]."""
        x_mid, y_mid = self._get_midplane_cart_coords(x0, y0, inc, PA)
        r_mid, t_mid = self._get_midplane_polar_coords(x0, y0, inc, PA)
        for _ in range(5):
            y_tmp = y_mid - func(r_mid) * tilt * np.tan(np.radians(inc))
            r_mid = np.hypot(y_tmp, x_mid)
            t_mid = np.arctan2(y_tmp, x_mid)
        return r_mid, t_mid

    def _get_flared_cart_coords(self, x0, y0, inc, PA, func, tilt):
        """Return cartesian coordinates of surface in [arcsec, arcsec]."""
        r_mid, t_mid = self._get_flared_polar_coords(x0, y0, inc,
                                                     PA, func, tilt)
        return r_mid * np.cos(t_mid), r_mid * np.sin(t_mid)

    # -- Functions to build Keplerian rotation profiles. -- #

    def keplerian(self, x0=0.0, y0=0.0, inc=0.0, PA=0.0, z0=0.0, psi=0.0,
                  nearest='north', mstar=1.0, dist=100., vlsr=0.0):
        """
        Return a Keplerian rotation profile (not including pressure). This
        includes the deivation due to non-zero heights above the midplane,
        see Teague et al. (2018a,c) for a thorough description.

        Args:
            x0 (Optional[float]): -
            y0 (Optional[float]): -
            inc (Optional[float]): -
            PA (Optional[float]): -
            frame (Optional[str]): -
            z0 (Optional[float]): -
            psi (Optional[float]): -
            nearest (Optional[str]): -
            mstar (Optional[float]): -
            disk (Optional[float]): -
            vlsr (Optional[floar]): -

        Returns:
            vproj (ndarray): Projected Keplerian rotation at each pixel [m/s].
        """
        coords = self.disk_coords(x0=x0, y0=y0, inc=inc, PA=PA, z0=z0, psi=psi,
                                  nearest=nearest, frame='polar')
        rvals = coords[0] * sc.au * dist
        zvals = coords[2] * sc.au * dist
        vkep = sc.G * mstar * self.msun * np.power(rvals, 2.0)
        vkep = np.sqrt(vkep * np.power(np.hypot(rvals, zvals), -3.0))
        return vkep * np.sin(np.radians(inc)) * np.cos(coords[1]) + vlsr

    # -- Helper functions for loading up the data. -- #

    def _clip_cube(self, radius):
        """Clip the cube to  clip arcseconds from the origin."""
        xa = abs(self.xaxis - radius).argmin()
        xb = abs(self.xaxis + radius).argmin()
        ya = abs(self.yaxis - radius).argmin()
        yb = abs(self.yaxis + radius).argmin()
        self.data = self.data[yb:ya, xa:xb]
        self.error = self.error[yb:ya, xa:xb]
        self.xaxis = self.xaxis[xa:xb]
        self.yaxis = self.yaxis[yb:ya]

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
        if a == 1 and self.absolute:
            a_del /= np.cos(np.radians(self.header['crval2']))
        a_pix = self.header['crpix%d' % a]
        a_ref = self.header['crval%d' % a]
        if not self.absolute:
            a_ref = 0.0
            a_pix -= 0.5
        axis = a_ref + (np.arange(a_len) - a_pix + 1.0) * a_del
        if self.absolute:
            return axis
        return 3600 * axis

    # -- Plotting functions. -- #

    @staticmethod
    def plot_walkers(samples, nburnin=None, labels=None):
        """Plot the walkers to check if they are burning in."""

        # Import matplotlib.
        import matplotlib.pyplot as plt

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

    @staticmethod
    def plot_corner(samples, labels=None, quantiles=[0.16, 0.5, 0.84]):
        """Plot the corner plot to check for covariances."""
        import corner
        corner.corner(samples, labels=labels, title_fmt='.4f',
                      quantiles=quantiles, show_titles=True)
