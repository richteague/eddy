import numpy as np
import scipy.constants as sc

class profile(object):

    # constants

    mu = 2.37
    msun = 1.98847e30

    # priors

    mstar_min = 0.0
    mstar_max = 10.0
    gamma_min = -10.0
    gamma_max = 10.0
    mdisk_min = 0.0
    mdisk_max = 0.1

    def __init__(self, rvals, v_phi, zvals=None, T_gas=None, dvphi=None,
                 maskNaN=True):
        """
        Initialize the class.

        Args:
            rvals (ndarray): Array of radial positions in [au].
            v_phi (ndarray): Array of rotational velocities in [m/s].
            dv_phi (optional[ndarray]): Uncertainties on the rotational
                velocities in [m/s].
            zvals (optional[ndarray]): Array of emission height position in
                [au]. If none is specified, assume all to be 0.
            T_gas (optional[ndarray]): Array of the gas temperatures in [K].
            maskNaN (optional[bool]): If True, mask out all NaN numbers in the
                input arrays. True by default.
        """

        # Read in the arrays.
        self.rvals = rvals
        self.zvals = zvals if zvals is not None else np.zeros(self.rvals.size)
        self.v_phi = v_phi
        self.dvphi = dvphi if dvphi is not None else 0.1 * self.v_phi.copy()
        self.T_gas = T_gas if T_gas is not None else np.ones(self.rvals.size)

        # Mask out the NaN values.
        if maskNaN:
            self._mask_NaN()

        # Check everything is in order.
        assert self.rvals.size == self.zvals.size, "Wrong number of zvals."
        assert self.rvals.size == self.v_phi.size, "Wrong number of v_phi."
        assert self.rvals.size == self.dvphi.size, "Wrong number of dvphi."
        assert self.rvals.size == self.T_gas.size, "Wrong number of T_gas."

        # Calculate some other default values to speed the fitting.
        self.cs = self._calculate_soundspeed()
        self.rvals_m = self.rvals * sc.au
        self.zvals_m = self.zvals * sc.au
        self._fit_gamma = False
        self._fit_mstar = False
        self._fit_perts = False

    # -- Main Fitting Function -- #

    def fit_vphi(self, p0, fit_gamma=False, fit_mdisk=False,
                 nwalkers=64, nburnin=1000, nsteps=1000,
                 percentiles=[16, 50, 84], niter=None,
                 minimize_kwargs=None, emcee_kwargs=None):
        """
        Find the best-fit density profile to explain the rotation profile.
        Currently there are two components: a normalized volume density, n(r),
        where the height is the height traced by the emission. This is a simple
        powerlaw,

            n(r) = (r / 100au)^(gamma)

        with the inclusion of Gaussian perturbations, each parameterised by a
        center, width and depth, (r_i, dr_i, dn_i).

        It is additionally possible to include (an approximate) self-gravity
        term. which again is described by,

            sigma(r) = sig_0 * (r / 100au)^(gamma)

        and the sig_0 term describes the disk mass contained between the inner
        and outer radial points.

        Args:
            p0 (list): Starting positions for the walkers.
            fit_gamma (Optional[bool]): Whether gamma should be included in the
                fit. If not, no pressure correction will be included.
            fit_mdisk (Optional[bool]): Whether the disk mass should be
                included in the fit. If not, no self-gravity correction will be
                included.
            nwalkers (Optional[int]): Number of walkers to use for the MCMC.
            nburnin (Optional[int]):

        Returns:
            samples: Samples of the posterior distributions.
        """

        # Import necessary packages.
        try:
            import emcee
            if int(emcee.__version__[0]) < 3:
                raise ImportError("Install v3.0 or higher of emcee.")
        except:
            raise ImportError("Install emcee (v3.0 or higher).")

        # Number of walkers.
        p0 = np.atleast_1d(p0)
        nwalkers = int(max(2*p0.size, nwalkers))

        # Set the global variables for the fitting.
        self._fit_gamma = int(fit_gamma)
        self._fit_mdisk = int(fit_mdisk)
        self._fit_perts = (len(p0) - int(fit_gamma) - int(fit_mdisk))
        self._fit_perts = int(np.floor(self._fit_perts / 3))
        ndim = 1 + self._fit_gamma + self._fit_mdisk + 3 * self._fit_perts
        assert p0.size == ndim, "Incorrect number of starting positions."

        # Cycle through this some times to converge.
        if niter is None:
            niter = self._fit_perts + 1
        niter = max(1, int(niter))

        # Sample the posteriors.
        emcee_kwargs = {} if emcee_kwargs is None else emcee_kwargs
        for _ in range(niter):
            sampler = self._run_mcmc(p0=p0, nwalkers=nwalkers, nburnin=nburnin,
                                     nsteps=nsteps, **emcee_kwargs)
            samples = sampler.chain[:, -int(nsteps):]
            samples = samples.reshape(-1, samples.shape[-1])
            p0 = np.median(samples, axis=0)

        # Diagnostic plots.
        labels = self._get_labels()
        self.plot_walkers(sampler.chain.T, nburnin=nburnin,
                          labels=labels)
        v_mod = self.calc_v_phi(theta=p0)
        self.plot_v_phi(v_mod=v_mod, return_fig=True)

        # Reset the fitting values before returning samples.
        self._fit_gamma = False
        self._fit_mdisk = False
        self._fit_perts = False
        return samples

    # -- MCMC Functions -- #

    def _run_mcmc(self, p0, nwalkers=16, nburnin=1000, nsteps=1000, **kwargs):
        """Explore the posteriors using MCMC."""
        p0 = self._random_p0(p0, kwargs.pop('scatter', 1e-3), nwalkers)
        sampler = emcee.EnsembleSampler(nwalkers, p0.shape[1],
                                        self._fit_density_ln_prob,
                                        pool=kwargs.pop('pool', None))
        sampler.run_mcmc(p0, nburnin + nsteps,
                         progress=kwargs.pop('progress', True))
        return sampler

    def _fit_density_ln_prob(self, theta):
        """Log-probability function for density profile fits."""
        lnp = self._fit_density_ln_prior(theta)
        if np.isfinite(lnp):
            return lnp + self._fit_density_ln_like(theta)
        return -np.inf

    def _fit_density_ln_like(self, theta):
        """Log-likelihood function for density profile fits."""
        v_mod = self.calc_v_phi(theta=theta)
        lnx2 = np.power((self.v_phi - v_mod), 2)
        lnx2 = -0.5 * np.sum(lnx2 * self.dvphi**-2)
        return lnx2 if np.isfinite(lnx2) else -np.inf

    def _fit_density_ln_prior(self, theta):
        """Log-prior functions for density profile fits."""
        mstar, gamma, mdisk, perts = self._unpack_theta(theta)

        # Stellar mass.
        if not profile.mstar_min < mstar < profile.mstar_max:
            return -np.inf

        # Surface density profile.
        if self._fit_gamma:
            if not profile.gamma_min < gamma < profile.gamma_max:
                return -np.inf

        # Disk mass.
        if self._fit_mdisk:
            if not profile.mdisk_min < mdisk / mstar < profile.mdisk_max:
                return -np.inf

        # Priors on the perturbations.
        for n in range(self._fit_perts):
            p0, dp, P = perts[(n)*3:(n+1)*3]
            if not -1.5 * self.rvals[0] < p0 < 1.5 * self.rvals[-1]:
                return -np.inf
            if not 0.0 < dp < self.rvals[-1]:
                return -np.inf
            if not 0.0 < P < 1.0:
                return- -np.inf

        return 0.0

    def _unpack_theta(self, theta):
        """Unpack theta."""
        theta = np.atleast_1d(theta)
        mstar, gamma, mdisk, perts = theta[0], None, None, None
        i = 1
        if self._fit_gamma:
            gamma = theta[i]
            i += 1
        if self._fit_mdisk:
            mdisk = theta[i]
            i += 1
        if self._fit_perts:
            perts = theta[i:]
        return mstar, gamma, mdisk, perts

    @staticmethod
    def _random_p0(p0, scatter, nwalkers):
        """Get the starting positions."""
        dp0 = np.random.randn(nwalkers * len(p0)).reshape(nwalkers, len(p0))
        dp0 = np.where(p0 == 0.0, 1.0, p0)[None, :] * (1.0 + scatter * dp0)
        return np.where(p0[None, :] == 0.0, dp0 - 1.0, dp0)

    # -- Velocity Functions -- #

    def calc_v_phi(self, theta):
        """Calculate the rotation velocity."""
        mstar, gamma, mdisk, perts = self._unpack_theta(theta)
        v_kep2 = self._calc_v_kep2(mstar)
        v_prs2 = 0.0 if gamma is None else self._calc_v_prs2(gamma, perts)
        v_grv2 = 0.0 if mdisk is None else self._calc_v_grv2(mdisk, gamma)
        v_phi2 = v_kep2 + v_prs2 + v_grv2
        return np.sqrt(np.where(v_phi2 < 0.0, 0.0, v_phi2))

    def _calc_v_prs2(self, gamma, perts):
        """Calculate the pressure correction term."""
        n_mol = self._calc_n_mol(gamma, perts)
        return self.rvals_m * self._dPdR(n_mol) / self._n2rho(n_mol)

    def _calc_v_kep2(self, mstar):
        """Calculate the Keplerian rotation term."""
        v_kep2 = sc.G * mstar * self.msun * self.rvals_m**2
        v_kep2 *= (self.rvals_m**2 + self.zvals_m**2)**(-3./2.)
        return v_kep2

    def _calc_v_grv2(self, gamma, mdisk):
        """Calcaulte the disk self-gravity correction term."""
        mdisk_cum = self._calc_enclosed_mass(gamma=gamma, mdisk=mdisk)
        return sc.G * mdisk_cum / self.rvals_m

    @staticmethod
    def _dphidR(sigma):
        """Gravitational potential gradient for a given sigma in [g/cm^2]."""
        return 2. * np.pi * sc.G * sigma * 1e3

    def _dPdR(self, n_mol):
        """Pressure gradient for a given n_mol in [/cm^3]."""
        P = n_mol * sc.k * self.T_gas * 1e6
        return np.gradient(P, self.rvals_m)

    # -- Density Functions -- #

    def _n2rho(self, n):
        """Convert n in [/cm^3] to rho in [kg/m^3]."""
        return n * self.mu * sc.m_p * 1e6

    def _rho2n(self, rho):
        """Convert rho in [kg/m^3] to n in [/cm^3]."""
        return rho / self.mu / sc.m_p / 1e6

    def _calc_sig0(self, gamma, mdisk):
        """Return the normalization constant in [kg/m^2]."""
        rmin, rmax = self.rvals_m[0], self.rvals_m[-1]
        sig0 = mdisk * self.msun * (2+gamma) * (100. * sc.au)**gamma
        sig0 /= 2.0 * np.pi * (rmax**(2+gamma) - rmin**(2+gamma))
        return sig0

    def _calc_sigma(self, gamma, mdisk):
        """Return sigma in [kg/m^2] for a given disk mass in [Msun]."""
        sig0 = self._calc_sig0(gamma=gamma, mdisk=mdisk)
        return sig0 * np.power(self.rvals / 100., gamma)

    def _calc_enclosed_mass(self, gamma, mdisk):
        """Return the enclosed mass [kg] as a function of radius."""
        sig0 = self._calc_sig0(gamma=gamma, mdisk=mdisk)
        mass = self.rvals_m**(2. + gamma) - self.rvals_m[0]**(2. + gamma)
        mass *= 2. * np.pi * sig0 * (100. * sc.au)**gamma
        return mass / (2. + gamma)

    # -- Perturbation Functions -- #

    def _calc_n_mol(self, gamma, perts):
        """Build the gas volume density profile."""
        n_mol =  1e6 * np.power(self.rvals / 100., gamma)
        dn_mol = self.perturbation_multi(perts)
        return n_mol * dn_mol

    @staticmethod
    def gaussian(x, x0, dx, A):
        """Gaussian function."""
        return A * np.exp(-0.5 * np.power((x-x0)/dx, 2))

    def perturbation(self, x0, dx, A):
        """Perturbation function."""
        return 1.0 - profile.gaussian(self.rvals, x0, dx, A)

    def perturbation_multi(self, perts):
        """Multiple perturbations."""
        dp = np.ones(self.rvals.size)
        if perts is not None:
            for i in range(self._fit_perts):
                dp *= self.perturbation(perts[3*i],
                                        perts[3*i+1],
                                        perts[3*i+2])
        return dp

    # -- Miscellaneous Functions -- #

    def _fit_mstar_optimize(self, **kwargs):
        """Optimize the parameters for the MCMC."""
        from scipy.optimize import minimize

        def nlnL(theta):
            return -self._fit_mstar_ln_prob(theta)

        x0 = kwargs.pop('x0', [1.0])
        method = kwargs.pop('method', 'TNC')
        options = kwargs.pop('options', {})
        options['maxiter'] = kwargs.pop('maxiter', 10000)
        res = minimize(nlnL, x0=x0, method=method, options=options)

        if res.success:
            print("Optimized value: {:.2f} Msun.".format(res.x[0]))
        else:
            print("Optimization failed.")
            res.x = x0
        time.sleep(0.3)
        return res.x

    def _calculate_soundspeed(self, T_gas=None, avg_mu=2.37):
        """Calculate the soundspeed for the gas in [m/s]."""
        T_gas = self.T_gas if T_gas is None else T_gas
        return np.sqrt(sc.k * T_gas / avg_mu / sc.m_p)

    def _mask_NaN(self):
        """Mask all the NaN values from all arrays."""
        mask_A = np.isfinite(self.rvals) & np.isfinite(self.zvals)
        mask_B = np.isfinite(self.v_phi) & np.isfinite(self.T_gas)
        mask = mask_A & mask_B
        self.rvals = self.rvals[mask]
        self.zvals = self.zvals[mask]
        self.v_phi = self.v_phi[mask]
        self.T_gas = self.T_gas[mask]

    # -- Plotting Functions -- #

    def _get_labels(self):
        """Return the labels of the fitting."""
        labels = [r'${\rm M_{star} \,\, (M_{sun})}$']
        if self._fit_gamma:
            labels += [r'${\rm \gamma}$']
        if self._fit_mdisk:
            labels += [r'${\rm M_{disk} \,\, (M_{sun})}$']
        for i in range(self._fit_perts):
            labels += [r'${\rm r_{%d} \,\, (au)}$' % i,
                       r'${\rm \Delta r_{%d} \,\, (au)}$' % i,
                       r'${\rm \delta n_{%d}}$' % i]
        return labels

    def plot_v_phi(self, v_mod=None, plot_residual=True, return_fig=False):
        """
        Plot the rotation profile.

        Args:
            v_mod (Optional[ndarray]): Model v_phi profile to compared to the
                data.
            plot_residual (Optional[bool]): If a model is provided, plot the
                residual at the bottom of the axis.
            return_fig (Optional[bool]): Return the figure.
        """

        # Imports.
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

        fig, ax = plt.subplots(figsize=(5.5, 3.0))

        ax.errorbar(self.rvals, self.v_phi, self.dvphi, color='k', fmt=' ')
        ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1])
        if v_mod is not None:
            ax.plot(self.rvals, v_mod, color='orangered')
        ax.set_ylabel(r'${\rm V_{\phi} \,\, (m\,s^{-1})}$')
        ax.set_xlim(self.rvals[0], self.rvals[-1])

        # Include a residual box.
        if v_mod is not None and plot_residual:
            fig.set_size_inches(5.5, 5.0, forward=True)
            ax_divider = make_axes_locatable(ax)
            ax1 = ax_divider.append_axes("bottom", size="40%", pad="2%")
            ax1.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1])
            ax.set_xticks([])
            ax1.set_xlabel('Radius (au)')
            ax1.axhline(0.0, color='orangered', lw=1.0, zorder=-10)
            ax1.errorbar(self.rvals, self.v_phi - v_mod, self.dvphi,
                         color='k', fmt=' ', linewidth=1.0)
        else:
            ax.set_xlabel('Radius (au)')
        if return_fig:
            return fig

    def plot_walkers(self, samples, nburnin=None, labels=None, return_fig=False,
                     plot_kwargs=None, histogram=True):
        """
        Plot the walkers to check if they are burning in.

        Args:
            samples (ndarray): Flattened array of posterior samples.
            nburnin (Optional[int]): Number of steps used to burn in. If
                provided will annotate the burn-in region.
            labels (Optional[list]): Labels for the different parameters.
            histogram (Optional[bool]): Include a histogram of the PDF samples
                at the right-hand side of the plot.
            return_fit (Optional[bool]): If True, return the figure.
            plot_kwargs (Optional[dict]): Kwargs used for ax.plot().

        Returns:
            fig (matplotlib figure): If requested.
        """

        # Imports.
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

        # Check the length of the label list.
        if labels is not None:
            if samples.shape[0] != len(labels):
                raise ValueError("Not correct number of labels.")

        # Cycle through the plots.
        if plot_kwargs is None:
            plot_kwargs = {}
        for s, sample in enumerate(samples):
            lc = plot_kwargs.get('c', plot_kwargs.get('color', 'k'))
            la = plot_kwargs.get('alpha', 0.1)
            fig, ax = plt.subplots(figsize=(4.0, 2.0))
            for walker in sample.T:
                ax.plot(walker, color=lc, alpha=la, zorder=-1, **plot_kwargs)
            ax.set_xlabel('Steps')
            ax.set_xlim(0, len(walker))
            if labels is not None:
                ax.set_ylabel(labels[s])
            if nburnin is not None:
                ax.axvspan(0, nburnin, facecolor='w', alpha=0.75, zorder=1)
                ax.axvline(nburnin, ls=':', color='k')
                y0 =  0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])
                ax.text(nburnin / 2.0, y0 + ax.get_ylim()[1], 'burn-in',
                ha='center', va='bottom', fontsize=8)
            ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1])

            # Add the aide histogram if required.
            if histogram:
                fig.set_size_inches(6.0, 2.0, forward=True)
                ax_divider = make_axes_locatable(ax)

                # Make the histogram.
                bins = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 50)
                hist, _ = np.histogram(sample[nburnin:].flatten(), bins=bins,
                                       density=True)
                bins = np.average([bins[1:], bins[:-1]], axis=0)

                # Plot it.
                ax1 = ax_divider.append_axes("right", size="25%", pad="2%")
                ax1.fill_betweenx(bins, hist, np.zeros(bins.size), step='mid',
                                  color='darkgray', lw=0.0)
                ax1.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1])
                ax1.set_xlim(0, ax1.get_xlim()[1])

                # Gentrification
                ax1.set_yticklabels([])
                ax1.set_xticklabels([])
                ax1.tick_params(which='both', left=0, bottom=0)
                ax1.spines['right'].set_visible(False)
                ax1.spines['bottom'].set_visible(False)
                ax1.spines['top'].set_visible(False)

        # Return the figure.
        if return_fig:
            return fig
