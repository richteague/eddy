# -*- coding: utf-8 -*-

from .datacube import datacube
from .annulus import annulus
import numpy as np
import warnings

warnings.filterwarnings("ignore")


class linecube(datacube):
    """
    Read in a line cube and initialize the class.

    Args:
        path (str): Relative path to the rotation map you want to fit.
        FOV (Optional[float]): If specified, clip the data down to a
            square field of view with sides of `FOV` [arcsec].
        fill (Optional[float/None]): Value to fill any NaN values.
    """

    def __init__(self, path, FOV=None, fill=0.0):
        datacube.__init__(self, path=path, FOV=FOV, fill=fill)

    # -- ROTATION PROFILE FUNCTIONS -- #

    def get_velocity_profile(self, rbins=None, fit_method='GP', fit_vrad=False,
                             x0=0.0, y0=0.0, inc=0.0, PA=0.0, z0=None,
                             psi=None, r_cavity=None, r_taper=None,
                             q_taper=None, w_i=None, w_r=None, w_t=None,
                             z_func=None, shadowed=False, phi_min=None,
                             phi_max=None, exclude_phi=False, abs_phi=False,
                             mask_frame='disk', user_mask=None,
                             beam_spacing=True, deproject=False, niter=1,
                             get_vlos_kwargs=None, weighted_average=True,
                             return_samples=False):
        """
        Returns the rotational and, optionally, radial velocity profiles under
        the assumption that the disk is azimuthally symmetric (at least across
        the regions extracted).

        Several different inference methods can be used through the
        ``fit_method`` argument:

            - ``'GP'``:  Models the aligned and stacked spectrum as a Gaussian
                         Process to remain agnostic about the underlying true
                         line profile. This is the default.
            - ``'dV'``:  Minimize the line width of the aligned and stacked
                         spectrum. Assumes the underyling profile is a
                         Gaussian.
            - ``'SNR'``: Maximizes the SNR of the aligned and stacked spectrum.
                         Assumes the underlying profile is a Gaussian.
            - ``'SHO'``: Fits the azimuthal dependence of the line centroids
                         with a simple harmonic oscillator. Requires a choice
                         of ``centroid_method`` to determine the line centers.

        By default, these methods are applied to annuli across the whole radial
        and azimuthal range of the disk. Masks can be adopted, either limiting
        or excluding azimuthal regions, or a user-defined mask can be provided.

        Boostrapping is possible (multiple iterations using random draws) to
        estimate the uncertainties on the velocities which may be
        underestimated with a single interation.

        Args:
            rbins (Optional[arr]): Array of bin edges of the annuli.
            fit_method (Optional[str]): Method used to infer the velocities.
            fit_vrad (Optional[bool]): Whether to include radial velocities in
                the fit.
            x0 (Optional[float]): Source right ascension offset [arcsec].
            y0 (Optional[float]): Source declination offset [arcsec].
            inc (Optional[float]): Source inclination [degrees]. A positive
                inclination denotes a disk rotating clockwise on the sky, while
                a negative inclination represents a counter-clockwise rotation.
            PA (Optional[float]): Source position angle [degrees]. Measured
                between north and the red-shifted semi-major axis in an
                easterly direction.
            z0 (Optional[float]): Aspect ratio at 1" for the emission surface.
                To get the far side of the disk, make this number negative.
            psi (Optional[float]): Flaring angle for the emission surface.
            z1 (Optional[float]): Correction term for ``z0``.
            phi (Optional[float]): Flaring angle correction term for the
                emission surface.
            r_cavity (Optional[float]): Outer radius of a cavity. Within this
                region the emission surface is taken to be zero.
            w_i (Optional[float]): Warp inclination in [degrees] at the disk
                center.
            w_r (Optional[float]): Scale radius of the warp in [arcsec].
            w_t (Optional[float]): Angle of nodes of the warp in [degrees].
            z_func=None,
            shadowed=False,
            phi_min=None,
            phi_max=None,
            exclude_phi=False,
            abs_phi=False,
            mask_frame='disk',
            user_mask=None,
            beam_spacing=True,
            deproject=False,
            niter=1,
            get_vlos_kwargs=None,
            weighted_average=True,
            return_samples=False

        Returns:
            TBD
        """

        if niter == 0:
            raise ValueError("`niter` must be >= 1.")

        # Single iteration.

        if niter == 1:
            return self._velocity_profile(rbins=rbins,
                                          fit_method=fit_method,
                                          fit_vrad=fit_vrad,
                                          x0=x0,
                                          y0=y0,
                                          inc=inc,
                                          PA=PA,
                                          z0=z0,
                                          psi=psi,
                                          r_cavity=r_cavity,
                                          r_taper=r_taper,
                                          q_taper=q_taper,
                                          w_i=w_i,
                                          w_r=w_r,
                                          w_t=w_t,
                                          z_func=z_func,
                                          shadowed=shadowed,
                                          phi_min=phi_min,
                                          phi_max=phi_max,
                                          exclude_phi=exclude_phi,
                                          abs_phi=abs_phi,
                                          mask_frame=mask_frame,
                                          user_mask=user_mask,
                                          beam_spacing=beam_spacing,
                                          get_vlos_kwargs=get_vlos_kwargs)

        # Multiple iterations.

        if beam_spacing is False:
            raise ValueError("niter must equal 1 when beam_spacing=False.")

        samples = [self._velocity_profile(rbins=rbins,
                                          fit_method=fit_method,
                                          fit_vrad=fit_vrad,
                                          x0=x0,
                                          y0=y0,
                                          inc=inc,
                                          PA=PA,
                                          z0=z0,
                                          psi=psi,
                                          r_cavity=r_cavity,
                                          r_taper=r_taper,
                                          q_taper=q_taper,
                                          w_i=w_i,
                                          w_r=w_r,
                                          w_t=w_t,
                                          z_func=z_func,
                                          shadowed=shadowed,
                                          phi_min=phi_min,
                                          phi_max=phi_max,
                                          exclude_phi=exclude_phi,
                                          abs_phi=abs_phi,
                                          mask_frame=mask_frame,
                                          user_mask=user_mask,
                                          beam_spacing=beam_spacing,
                                          get_vlos_kwargs=get_vlos_kwargs)
                   for _ in range(niter)]

        # Just return the samples.

        if return_samples:
            return samples
        rpnts = samples[0][0]
        profiles = np.array([s[1] for s in samples])

        # Calculate weights.

        if weighted_average:
            weights = [1.0 / s[2] for s in samples]
            weights = np.where(np.isfinite(weights), weights, 0.0)
        else:
            weights = np.ones(profiles.shape)
        if np.all(np.sum(weights, axis=0) == 0.0):
            weights = np.ones(profiles.shape)
        M = np.sum(weights != 0.0, axis=0)

        # Weighted average.

        profile = np.average(profiles, weights=weights, axis=0)

        # Weighted standard deviation.

        uncertainty = weights * (profiles - profile[None, :, :])**2
        uncertainty = np.sum(uncertainty, axis=0)
        uncertainty /= (M - 1.0) / M * np.sum(weights, axis=0)
        uncertainty = np.sqrt(uncertainty)
        return rpnts, profile, uncertainty

    def _velocity_profile(self, rbins=None, fit_method='GP', fit_vrad=False,
                          x0=0.0, y0=0.0, inc=0.0, PA=0.0, z0=None,
                          psi=None, r_cavity=None, r_taper=None, q_taper=None,
                          w_i=None, w_r=None, w_t=None, z_func=None,
                          shadowed=False, phi_min=None, phi_max=None,
                          exclude_phi=False, abs_phi=False, mask_frame='disk',
                          user_mask=None, beam_spacing=True, deproject=False,
                          get_vlos_kwargs=None):
        """Returns the velocity (rotational and radial) profiles."""

        # Define the radial binning.

        if rbins is None:
            rbins = np.arange(0, self.xaxis.max(), 0.25 * self.bmaj)
        rpnts = np.mean([rbins[1:], rbins[:-1]], axis=0)

        # Set up the kwargs for the fitting.

        kw = {} if get_vlos_kwargs is None else get_vlos_kwargs
        kw['fit_vrad'] = kw.pop('fit_vrad', fit_vrad)
        kw['fit_method'] = fit_method

        # Cycle through the annuli.

        profiles = []
        uncertainties = []
        for r_min, r_max in zip(rbins[:-1], rbins[1:]):
            annulus = self.get_annulus(r_min=r_min,
                                       r_max=r_max,
                                       phi_min=phi_min,
                                       phi_max=phi_max,
                                       exclude_phi=exclude_phi,
                                       abs_phi=abs_phi,
                                       x0=x0,
                                       y0=y0,
                                       inc=inc,
                                       PA=PA,
                                       z0=z0,
                                       psi=psi,
                                       r_cavity=r_cavity,
                                       r_taper=r_taper,
                                       q_taper=q_taper,
                                       w_i=w_i,
                                       w_r=w_r,
                                       w_t=w_t,
                                       z_func=z_func,
                                       shadowed=shadowed,
                                       mask_frame=mask_frame,
                                       user_mask=user_mask,
                                       beam_spacing=beam_spacing)
            output = annulus.get_vlos(**kw)

            # Parse the outputs.

            par = 2 if fit_vrad else 1
            output = np.empty(par) if np.all(np.isnan(output)) else output
            if fit_method.lower() in ['dv', 'snr']:
                profiles += [output]
                uncertainties += [np.ones(par) * np.nan]
            elif fit_method.lower() == 'sho':
                profiles += [output[0]]
                uncertainties += [output[1]]
            elif fit_method.lower() == 'gp':
                profiles += [output[0]]
                uncertainties += [output[1]]

        # Make sure the returned arrays are in the (nparam, nrad) form.

        profiles = np.atleast_2d(profiles)
        uncertainties = np.atleast_2d(uncertainties)
        if profiles.shape[0] == rpnts.size:
            profiles = profiles.T
            uncertainties = uncertainties.T

        # Deproject the velocity profiles.

        if deproject:
            profiles /= np.sin(np.radians(abs(inc)))
            uncertainties /= np.sin(np.radians(abs(inc)))

        return rpnts, profiles, uncertainties

    # -- ANNULUS FUNCTIONS -- #

    def get_annulus(self, r_min, r_max, phi_min=None, phi_max=None,
                    exclude_phi=False, abs_phi=False, x0=0.0, y0=0.0, inc=0.0,
                    PA=0.0, z0=None, psi=None, r_cavity=None, r_taper=None,
                    q_taper=None, w_i=None, w_r=None, w_t=None, z_func=None,
                    shadowed=False, mask_frame='disk', user_mask=None,
                    beam_spacing=True, annulus_kwargs=None):
        """
        Returns an annulus instance.
        """

        # Calculate and flatten the mask.

        mask = self.get_mask(r_min=r_min,
                             r_max=r_max,
                             phi_min=phi_min,
                             phi_max=phi_max,
                             exclude_phi=exclude_phi,
                             abs_phi=abs_phi,
                             x0=x0,
                             y0=y0,
                             inc=inc,
                             PA=PA,
                             z0=z0,
                             psi=psi,
                             r_cavity=r_cavity,
                             r_taper=r_taper,
                             q_taper=q_taper,
                             w_i=w_i,
                             w_r=w_r,
                             w_t=w_t,
                             z_func=z_func,
                             shadowed=shadowed,
                             mask_frame=mask_frame,
                             user_mask=user_mask)
        if mask.shape != self.data[0].shape:
            raise ValueError("mask is incorrect shape: {}.".format(mask.shape))
        mask = mask.flatten()

        # Flatten the data and get the deprojected pixel coordinates.

        dvals = self.data.copy().reshape(self.data.shape[0], -1)
        rvals, pvals = self.disk_coords(x0=x0,
                                        y0=y0,
                                        inc=inc,
                                        PA=PA,
                                        z0=z0,
                                        psi=psi,
                                        r_cavity=r_cavity,
                                        r_taper=r_taper,
                                        q_taper=q_taper,
                                        w_i=w_i,
                                        w_r=w_r,
                                        w_t=w_t,
                                        z_func=z_func,
                                        shadowed=shadowed)[:2]
        rvals, pvals = rvals.flatten(), pvals.flatten()
        dvals, rvals, pvals = dvals[:, mask].T, rvals[mask], pvals[mask]

        # Thin down to spatially independent pixels.

        rvals, pvals, dvals = self._independent_samples(beam_spacing,
                                                        rvals, pvals, dvals)

        # Return the annulus instance.

        annulus_kwargs = {} if annulus_kwargs is None else annulus_kwargs
        return annulus(spectra=dvals, pvals=pvals, velax=self.velax,
                       rotation='clockwise' if inc >= 0.0 else 'anticlockwise',
                       **annulus_kwargs)

    def _independent_samples(self, beam_spacing, rvals, pvals, dvals):
        """Returns spatially independent samples."""

        if not beam_spacing:
            return rvals, pvals, dvals

        # Order pixels in increasing phi.

        idxs = np.argsort(pvals)
        dvals, pvals = dvals[idxs], pvals[idxs]

        # Calculate the sampling rate.

        sampling = float(beam_spacing) * self.bmaj
        sampling /= np.mean(rvals) * np.median(np.diff(pvals))
        sampling = np.floor(sampling).astype('int')

        # If the sampling rate is above 1, start at a random location in
        # the array and sample at this rate, otherwise don't sample. This
        # happens at small radii, for example.

        if sampling > 1:
            start = np.random.randint(0, pvals.size)
            rvals = np.concatenate([rvals[start:], rvals[:start]])
            pvals = np.concatenate([pvals[start:], pvals[:start]])
            dvals = np.vstack([dvals[start:], dvals[:start]])
            rvals = rvals[::sampling]
            pvals = pvals[::sampling]
            dvals = dvals[::sampling]
        else:
            print("Pixels appear to be close to spatially independent.")

        return rvals, pvals, dvals

    # -- PLOTTING FUNCTIONS -- #

    def plot_mask(self, ax, r_min=None, r_max=None, exclude_r=False,
                  phi_min=None, phi_max=None, exclude_phi=False, abs_phi=False,
                  mask_frame='disk', mask=None, x0=0.0, y0=0.0, inc=0.0,
                  PA=0.0, z0=None, psi=None, r_cavity=None, r_taper=None,
                  q_taper=None, w_i=None, w_r=None, w_t=None, z_func=None,
                  mask_color='k', mask_alpha=0.5, contour_kwargs=None,
                  contourf_kwargs=None, shadowed=False):
        """
        Plot the boolean mask on the provided axis to check that it makes
        sense.

        Args:
            ax (matplotib axis instance): Axis to plot the mask.
            r_min (Optional[float]): Minimum midplane radius of the annulus in
                [arcsec]. Defaults to minimum deprojected radius.
            r_max (Optional[float]): Maximum midplane radius of the annulus in
                [arcsec]. Defaults to the maximum deprojected radius.
            exclude_r (Optional[bool]): If ``True``, exclude the provided
                radial range rather than include.
            PA_min (Optional[float]): Minimum polar angle of the segment of the
                annulus in [degrees]. Note this is the polar angle, not the
                position angle.
            PA_max (Optional[float]): Maximum polar angle of the segment of the
                annulus in [degrees]. Note this is the polar angle, not the
                position angle.
            exclude_PA (Optional[bool]): If ``True``, exclude the provided
                polar angle range rather than include it.
            abs_PA (Optional[bool]): If ``True``, take the absolute value of
                the polar angle such that it runs from 0 [deg] to 180 [deg].
            x0 (Optional[float]): Source center offset along the x-axis in
                [arcsec].
            y0 (Optional[float]): Source center offset along the y-axis in
                [arcsec].
            inc (Optional[float]): Inclination of the disk in [degrees].
            PA (Optional[float]): Position angle of the disk in [degrees],
                measured east-of-north towards the redshifted major axis.
            z0 (Optional[float]): Emission height in [arcsec] at a radius of
                1".
            psi (Optional[float]): Flaring angle of the emission surface.
            z1 (Optional[float]): Correction to emission height at 1" in
                [arcsec].
            phi (Optional[float]): Flaring angle correction term.
            z_func (Optional[function]): A function which provides
                :math:`z(r)`. Note that no checking will occur to make sure
                this is a valid function.
            mask_color (Optional[str]): Color used for the mask lines.
            mask_alpha (Optional[float]): The alpha value of the filled contour
                of the masked regions. Setting ``mask_alpha=0.0`` will remove
                the filling.

            contour_kwargs (Optional[dict]): Kwargs to pass to contour for
                drawing the mask.

        Returns:
            ax : The matplotlib axis instance.
        """
        # Grab the mask.
        if mask is None:
            mask = self.get_mask(r_min=r_min, r_max=r_max, exclude_r=exclude_r,
                                 phi_min=phi_min, phi_max=phi_max,
                                 exclude_phi=exclude_phi, abs_phi=abs_phi,
                                 mask_frame=mask_frame, x0=x0, y0=y0, inc=inc,
                                 PA=PA, z0=z0, psi=psi, r_cavity=r_cavity,
                                 r_taper=r_taper, q_taper=q_taper, w_i=w_i,
                                 w_r=w_r, w_t=w_t, z_func=z_func,
                                 shadowed=shadowed)
        assert mask.shape[0] == self.yaxis.size, "Wrong y-axis shape for mask."
        assert mask.shape[1] == self.xaxis.size, "Wrong x-axis shape for mask."

        # Set the default plotting style.

        contour_kwargs = {} if contour_kwargs is None else contour_kwargs
        contour_kwargs['colors'] = contour_kwargs.pop('colors', mask_color)
        contour_kwargs['linewidths'] = contour_kwargs.pop('linewidths', 1.0)
        contour_kwargs['linestyles'] = contour_kwargs.pop('linestyles', '-')
        contourf_kwargs = {} if contourf_kwargs is None else contourf_kwargs
        contourf_kwargs['alpha'] = contourf_kwargs.pop('alpha', mask_alpha)
        contourf_kwargs['colors'] = contourf_kwargs.pop('colors', mask_color)

        # Plot the contour and return the figure.

        ax.contourf(self.xaxis, self.yaxis, mask, [-.5, .5], **contourf_kwargs)
        ax.contour(self.xaxis, self.yaxis, mask, 1, **contour_kwargs)
