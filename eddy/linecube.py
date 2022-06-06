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
                             x0=0.0, y0=0.0, inc=0.0, PA=0.0, z0=0.0,
                             psi=1.0, r_cavity=None, r_taper=None,
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
                          x0=0.0, y0=0.0, inc=0.0, PA=0.0, z0=0.0,
                          psi=1.0, r_cavity=None, r_taper=None, q_taper=None,
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
                    PA=0.0, z0=0.0, psi=1.0, r_cavity=None, r_taper=None,
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
                  PA=0.0, z0=0.0, psi=1.0, r_cavity=None, r_taper=None,
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

    def get_spectrum(self, coords, x0=0.0, y0=0.0, inc=0.0, PA=0.0,
                     z0=0.0, psi=1.0, z_func=None, frame='sky',
                     coord_type='cartesian', area=0.0, beam_weighting=False,
                     return_mask=False):
        """
        Return a spectrum at a position defined by a coordinates given either
        in sky-frame position (``frame='sky'``) or a disk-frame location
        (``frame='disk'``). The coordinates can be either in a cartesian system
        (``coord_type='cartesian'``) or cylindrical system
        (``coord_type='cylindrical'``).

        By default the returned spectrum is extracted at the pixel closest to
        the provided coordinates. If ``area`` is set to a positive value, then
        a beam-shaped area is averaged over, where ``area`` sets the size of
        this region in number of beams. For example ``area=2.0`` will result
        in an average over an area twice the size of the beam.

        If an area is averaged over, you can also weight the pixels by the beam
        response with ``beam_weighting=True``. This will reduce the weight of
        pixels that are further away from the beam center.

         Finally, to check that you're extracting what you think you are, you
         can return the mask (and weights) used for the extraction with
         ``return_mask=True``. Note that if ``beam_weighting=False`` then all
         ``weights`` will be 1.

         TODO: Check that the returned uncertainties are reasonable.

        Args:
            coords (tuple): The coordinates from where you want to extract a
                spectrum. Must be a length 2 tuple.
            x0 (Optional[float]): RA offset in [arcsec].
            y0 (Optional[float]): Dec offset in [arcsec].
            inc (Optional[float]): Inclination of source in [deg]. Only
                required for ``frame='disk'``.
            PA (Optional[float]): Position angle of source in [deg]. Only
                required for ``frame='disk'``.
            frame (Optional[str]): The frame that the ``coords`` are given.
                Either ``'disk'`` or ``'sky'``.
            coord_type (Optional[str]): The type of coordinates given, either
                ``'cartesian'`` or ``'cylindrical'``.
            area (Optional[float]): The area to average over in units of the
                beam area. Note that this take into account the beam aspect
                ratio and position angle. For a single pixel extraction use
                ``area=0.0``.
            beam_weighting (Optional[bool]): Whether to use the beam response
                function to weight the averaging of the spectrum.
            return_mask (Optional[bool]): Whether to return the mask and
                weights used to extract the spectrum.

        Retuns (if ``return_mask=False``):
            x, y, dy (arrays): The velocity axis, extracted spectrum and
            associated uncertainties.
        (if ``return_mask=True``):
            mask, weights (arrays): Arrays of the mask used to extract the
            spectrum and the weighted used for the averaging.
        """

        # TODO:
        #   1 - Check if it's three coordinates, or just two.
        #   2 - Check which frame it's in.

        # Convert the input coordinate into on-sky cartesian coordinates
        # relative to the center of the image.

        if frame.lower() == 'sky':
            if inc != 0.0 or PA != 0.0:
                message = "WARNING: You shouldn't need to specify `inc` or "
                message += "`PA` when using `frame='sky'`."
                print(message)
            c1 = np.squeeze(coords[0])
            c2 = np.squeeze(coords[1])
            if coord_type.lower() == 'cartesian':
                x, y = c1 + x0, c2 + y0
            elif coord_type.lower() == 'cylindrical':
                x = x0 + c1 * np.cos(c2 - np.radians(90.0))
                y = y0 - c1 * np.sin(c2 - np.radians(90.0))
        elif frame.lower() == 'disk':
            x, y = self.disk_to_sky(coords=coords, coord_type=coord_type,
                                    inc=inc, PA=PA, x0=x0, y0=y0)
        assert x.size == y.size == 1

        # Define the area to average over.

        if area == 0.0:
            x_pix = abs(self.xaxis - x).argmin()
            y_pix = abs(self.yaxis - y).argmin()
            mask = np.zeros(self.data[0].shape)
            weights = np.zeros(mask.shape)
            mask[y_pix, x_pix] = 1
            weights[y_pix, x_pix] = 1
        elif area > 0.0:
            mask = self._beam_mask(x, y, stretch=area)
            weights = self._beam_mask(x, y, stretch=area, response=True)
        else:
            raise ValueError("`area` must be a non-negative value.")
        weights = weights if beam_weighting else mask

        # If requested, return the mask and the weighting instead.

        if return_mask:
            return mask, weights

        # Otherwise, extract the spectrum and average it.

        y = [np.average(c * mask, weights=weights) for c in self.data]
        dy = max(1.0, mask.sum() * self.beams_per_pix)**-0.5 * self.rms
        return self.velax, np.array(y), np.array([dy for _ in y])
