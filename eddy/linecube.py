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
        return annulus(spectra=dvals, rvals=rvals, pvals=pvals,
                       velax=self.velax, **annulus_kwargs)

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

    # -- MASKING FUNCTIONS -- #

    def get_mask(self, r_min=None, r_max=None, exclude_r=False, phi_min=None,
                 phi_max=None, exclude_phi=False, abs_phi=False, x0=0.0,
                 y0=0.0, inc=0.0, PA=0.0, z0=None, psi=None, r_cavity=None,
                 r_taper=None, q_taper=None, w_i=None, w_r=None, w_t=None,
                 z_func=None, shadowed=False, mask_frame='disk',
                 user_mask=None):
        """
        Returns a 2D mask for pixels in the given region. The mask can be
        specified in either disk-centric coordinates, ``mask_frame='disk'``,
        or on the sky, ``mask_frame='sky'``. If sky-frame coordinates are
        requested, the geometrical parameters (``inc``, ``PA``, ``z0``, etc.)
        are ignored, however the source offsets, ``x0``, ``y0``, are still
        considered.

        Args:
            r_min (Optional[float]): Minimum midplane radius of the annulus in
                [arcsec]. Defaults to minimum deprojected radius.
            r_max (Optional[float]): Maximum midplane radius of the annulus in
                [arcsec]. Defaults to the maximum deprojected radius.
            exclude_r (Optional[bool]): If ``True``, exclude the provided
                radial range rather than include.
            phi_min (Optional[float]): Minimum polar angle of the segment of
                the annulus in [deg]. Note this is the polar angle, not the
                position angle.
            phi_max (Optional[float]): Maximum polar angle of the segment of
                the annulus in [deg]. Note this is the polar angle, not the
                position angle.
            exclude_phi (Optional[bool]): If ``True``, exclude the provided
                polar angle range rather than include it.
            abs_phi (Optional[bool]): If ``True``, take the absolute value of
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
            z_func (Optional[function]): A function which provides
                :math:`z(r)`. Note that no checking will occur to make sure
                this is a valid function.

        Returns:
            A 2D array mask matching the shape of a channel.
        """

        # Check the requested frame.

        mask_frame = mask_frame.lower()
        if mask_frame not in ['disk', 'sky']:
            raise ValueError("mask_frame must be 'disk' or 'sky'.")
        if mask_frame == 'sky':
            inc = 0.0
            PA = 0.0

        # Calculate the deprojected pixel coordaintes.

        rvals, pvals = self.disk_coords(x0=x0, y0=y0, inc=inc, PA=PA, z0=z0,
                                        psi=psi, r_cavity=r_cavity,
                                        r_taper=r_taper, q_taper=q_taper,
                                        w_i=w_i, w_r=w_r, w_t=w_t,
                                        z_func=z_func, frame='cylindrical',
                                        shadowed=shadowed)[:2]
        pvals = abs(pvals) if abs_phi else pvals

        # Radial mask.

        r_min = np.nanmin(rvals) if r_min is None else r_min
        r_max = np.nanmax(rvals) if r_max is None else r_max
        if r_min >= r_max:
            raise ValueError("`r_min` must be smaller than `r_max`.")
        r_mask = np.logical_and(rvals >= r_min, rvals <= r_max)
        r_mask = ~r_mask if exclude_r else r_mask

        # Azimuthal mask.

        phi_min = np.nanmin(pvals) if phi_min is None else np.radians(phi_min)
        phi_max = np.nanmax(pvals) if phi_max is None else np.radians(phi_max)
        if phi_min >= phi_max:
            raise ValueError("`PA_min` must be smaller than `PA_max`.")
        phi_mask = np.logical_and(pvals >= phi_min, pvals <= phi_max)
        phi_mask = ~phi_mask if exclude_phi else phi_mask

        # Combine and return.

        mask = r_mask * phi_mask
        if np.sum(mask) == 0:
            raise ValueError("There are zero pixels in the mask.")
        if user_mask is not None:
            mask *= user_mask
        return mask

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
