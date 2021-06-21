# -*- coding: utf-8 -*-

import os
import numpy as np
from astropy.io import fits
import scipy.constants as sc
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


class datacube(object):
    """
    A ``datacube`` instance to read in data in a ``FITS`` format. This should
    contain all the functions to interact with a spectral line dataset.

    Args:
        path (str): Path to the datacube to load.
        FOV (Optional[float]): If specified, clip the data down to a
            square field of view with sides of `FOV` [arcsec].
        velocity_range (Optional[list]): A tuple or list of the minimum and
            maximum velocities in [m/s] to clip the data cube to.
    """

    flared_niter = 5
    shadowed_extend = 1.5
    shadowed_oversample = 2.0
    shadowed_method = 'nearest'

    msun = 1.98847e30
    fwhm = 2. * np.sqrt(2 * np.log(2))

    def __init__(self, path, FOV=None, velocity_range=None, fill=None):

        # Read in the data

        self.path = path
        self._read_FITS(path=self.path, fill=fill)

        # Clip down the cube.

        if FOV is not None:
            self._clip_cube_spatial(FOV / 2.0)
        if velocity_range is not None:
            self._clip_cube_velocity(*velocity_range)

    # -- PIXEL DEPROJECTION -- #

    def disk_coords(self, x0=0.0, y0=0.0, inc=0.0, PA=0.0, z0=None, psi=None,
                    r_cavity=0.0, r_taper=None, q_taper=None, w_i=0.0, w_r=1.0,
                    w_t=0.0, outframe='cylindrical', z_func=None,
                    shadowed=False, **_):
        r"""
        Get the disk coordinates given certain geometrical parameters and an
        emission surface. The emission surface is most simply described as a
        power law profile,

        .. math::

            z(r) = z_0 \times \left(\frac{r}{1^{\prime\prime}}\right)^{\psi}

        where ``z0`` and ``psi`` can be provided by the user. With the increase
        in spatial resolution afforded by interferometers such as ALMA there
        are a couple of modifications that can be used to provide a better
        match to the data.

        An inner cavity can be included with the ``r_cavity`` argument which
        makes the transformation:

        .. math::

            \tilde{r} = {\rm max}(0, r - r_{\rm cavity})

        Note that the inclusion of a cavity will mean that other parameters,
        such as ``z0``, would need to change as the radial axis has effectively
        been shifted.

        To account for the drop in emission surface in the outer disk where the
        gas surface density decreases there are two descriptions. The preferred
        way is to include an exponential taper to the power law profile,

        .. math::

            z_{\rm tapered}(r) = z(r) \times \exp\left( -\left[
            \frac{r}{r_{\rm taper}} \right]^{q_{\rm taper}} \right)

        where both ``r_taper`` and ``q_taper`` values must be set.

        We can also include a warp which is parameterized by,

        .. math::

            z_{\rm warp}(r,\, t) = r \times \tan \left(w_i \times \exp\left(-
            \frac{r^2}{2 w_r^2} \right) \times \sin(t - w_t)\right)

        where ``w_i`` is the inclination in [radians] describing the warp at
        the disk center. The width of the warp is given by ``w_r`` [arcsec] and
        ``w_t`` in [radians] is the angle of nodes (where the warp is zero),
        relative to the position angle of the disk, measured east of north.

        .. WARNING::

            The use of warps is largely untested. Use with caution!
            If you are using a warp, increase the number of iterations
            for the inference through ``self.flared_niter`` (by default at
            5). For high inclinations, also set ``shadowed=True``.

        Args:
            x0 (optional[float]): Source right ascension offset [arcsec].
            y0 (optional[float]): Source declination offset [arcsec].
            inc (optional[float]): Source inclination [degrees]. A positive
                inclination denotes a disk rotating clockwise on the sky, while
                a negative inclination represents a counter-clockwise rotation.
            PA (optional[float]): Source position angle [degrees]. Measured
                between north and the red-shifted semi-major axis in an
                easterly direction.
            z0 (optional[float]): Aspect ratio at 1" for the emission surface.
                To get the far side of the disk, make this number negative.
            psi (optional[float]): Flaring angle for the emission surface.
            z1 (optional[float]): Correction term for ``z0``.
            phi (optional[float]): Flaring angle correction term for the
                emission surface.
            r_cavity (optional[float]): Outer radius of a cavity. Within this
                region the emission surface is taken to be zero.
            w_i (optional[float]): Warp inclination in [degrees] at the disk
                center.
            w_r (optional[float]): Scale radius of the warp in [arcsec].
            w_t (optional[float]): Angle of nodes of the warp in [degrees].
            outframe (optional[str]): Frame of reference for the returned
                coordinates. Either ``'cartesian'`` or ``'cylindrical'``.

        Returns:
            array, array, array: Three coordinate arrays with ``(r, phi, z)``,
            in units of [arcsec], [radians], [arcsec], if
            ``frame='cylindrical'`` or ``(x, y, z)``, all in units of [arcsec]
            if ``frame='cartesian'``.
        """

        # Check the input variables.

        outframe = outframe.lower()
        if outframe not in ['cylindrical', 'cartesian']:
            raise ValueError("frame must be 'cylindrical' or 'cartesian'.")

        # Apply the inclination convention to be consistent with orbits.

        inc = inc if inc < 90.0 else inc - 180.0

        # Check that the necessary pairs are provided.

        msg = "Must specify either both or neither of `{}` and `{}`."
        if (r_taper is not None) != (q_taper is not None):
            raise ValueError(msg.format('r_taper', 'q_taper'))

        msg = "Must specify all of the warp coorinates: (w_r, w_i, w_t)."
        if (w_i is not None) != (w_t is not None) != (w_r is not None):
            raise ValueError(msg)

        flared = r_taper is not None or w_i is not None or r_cavity is not None

        # Select the quickest pixel deprojection method.

        if z0 is None or z0 == 0.0 and z_func is None:

            r, t = self._get_midplane_polar_coords(x0, y0, inc, PA)
            z = np.zeros(r.shape)

        elif z0 > 0.0 and psi == 1.0 and not flared and z_func is None:

            r, t, z = self._get_conical_polar_coords(x0, y0, inc, PA, z0)

        else:

            r_cavity = 0.0 if r_cavity is None else r_cavity
            r_taper = np.inf if r_taper is None else r_taper
            q_taper = 1.0 if q_taper is None else q_taper
            w_i = 0.0 if w_i is None else w_i
            w_r = 1.0 if w_r is None else w_r
            w_t = 0.0 if w_t is None else w_t

            if z_func is None:
                def z_func(r_in):
                    r = np.clip(r_in - r_cavity, a_min=0.0, a_max=None)
                    z = r**psi * np.exp(-np.power(r / r_taper, q_taper))
                    return np.clip(z0 * z, a_min=0.0, a_max=None)

            def w_func(r_in, t):
                r = np.clip(r_in - r_cavity, a_min=0.0, a_max=None)
                warp = np.radians(w_i) * np.exp(-0.5 * (r / w_r)**2)
                return r * np.tan(warp * np.sin(t - np.radians(w_t)))

            if shadowed:
                coords = self._get_shadowed_coords(x0, y0, inc, PA,
                                                   z_func, w_func)
            else:
                coords = self._get_flared_coords(x0, y0, inc, PA,
                                                 z_func, w_func)
            r, t, z = coords

        # Return the values.

        if outframe == 'cylindrical':
            return r, t, z
        return r * np.cos(t), r * np.sin(t), z

    @staticmethod
    def _rotate_coords(x, y, PA):
        """Rotate (x, y) by PA [deg]."""
        x_rot = y * np.cos(np.radians(PA)) + x * np.sin(np.radians(PA))
        y_rot = x * np.cos(np.radians(PA)) - y * np.sin(np.radians(PA))
        return x_rot, y_rot

    @staticmethod
    def _deproject_coords(x, y, inc):
        """Deproject (x, y) by inc [deg]."""
        return x, y / np.cos(np.radians(inc))

    def _get_cart_sky_coords(self, x0, y0):
        """Return cartesian sky coordinates in [arcsec, arcsec]."""
        return np.meshgrid(self.xaxis - x0, self.yaxis - y0)

    def _get_midplane_cart_coords(self, x0, y0, inc, PA):
        """Return cartesian coordaintes of midplane in [arcsec, arcsec]."""
        x_sky, y_sky = self._get_cart_sky_coords(x0, y0)
        x_rot, y_rot = self._rotate_coords(x_sky, y_sky, PA)
        return datacube._deproject_coords(x_rot, y_rot, inc)

    def _get_midplane_polar_coords(self, x0, y0, inc, PA):
        """Return the polar coordinates of midplane in [arcsec, radians]."""
        x_mid, y_mid = self._get_midplane_cart_coords(x0, y0, inc, PA)
        return np.hypot(y_mid, x_mid), np.arctan2(y_mid, x_mid)

    def _get_conical_cart_coords(self, x0, y0, inc, PA, z0):
        """Return the cartesian coords of a conical surface."""
        inc = np.radians(inc)
        PA = np.radians(PA - 90.0)
        x_sky, y_sky = self._get_cart_sky_coords(x0, y0)
        x_rot = x_sky * np.cos(PA) - y_sky * np.sin(PA)
        y_rot = x_sky * np.sin(PA) + y_sky * np.cos(PA)
        psi = np.tan(z0)
        a = np.cos(2 * inc) + np.cos(2 * psi)
        b = -4.0 * np.sin(psi)**2 * y_rot * np.tan(inc)
        c = -2.0 * np.sin(psi)**2 * (x_rot**2 + y_rot**2 / np.cos(inc)**2)
        t = -b + np.sqrt(b**2 - 4 * a * c) / 2 / a
        x_d = x_rot
        y_d = y_rot / np.cos(inc) + t * np.sin(inc)
        z_d = z0 * np.hypot(x_d, y_d)
        return x_d, y_d, z_d

    def _get_conical_polar_coords(self, x0, y0, inc, PA, z0):
        """Return the cylindrical coords of a conical surface."""
        x_d, y_d, z_d = self._get_conical_cart_coords(x0, y0, inc, PA, z0)
        return np.hypot(y_d, x_d), np.arctan2(y_d, x_d), z_d

    def _get_flared_coords(self, x0, y0, inc, PA, z_func, w_func):
        """Return cyclindrical coords of surface in [arcsec, rad, arcsec]."""
        x_mid, y_mid = self._get_midplane_cart_coords(x0, y0, inc, PA)
        r_tmp, t_tmp = np.hypot(x_mid, y_mid), np.arctan2(y_mid, x_mid)
        for _ in range(self.flared_niter):
            z_tmp = z_func(r_tmp) + w_func(r_tmp, t_tmp)
            y_tmp = y_mid + z_tmp * np.tan(np.radians(inc))
            r_tmp = np.hypot(y_tmp, x_mid)
            t_tmp = np.arctan2(y_tmp, x_mid)
        return r_tmp, t_tmp, z_func(r_tmp)

    def _get_shadowed_coords(self, x0, y0, inc, PA, z_func, w_func):
        """Return cyclindrical coords of surface in [arcsec, rad, arcsec]."""

        # Make the disk-frame coordinates.

        diskframe_coords = self._get_diskframe_coords()
        xdisk, ydisk, rdisk, tdisk = diskframe_coords
        zdisk = z_func(rdisk) + w_func(rdisk, tdisk)

        # Incline the disk.

        inc = np.radians(inc)
        x_dep = xdisk
        y_dep = ydisk * np.cos(inc) - zdisk * np.sin(inc)

        # Remove shadowed pixels.
        # TODO: Check how this handles the bottom side of the disk.

        if inc < 0.0:
            y_dep = np.maximum.accumulate(y_dep, axis=0)
        else:
            y_dep = np.minimum.accumulate(y_dep[::-1], axis=0)[::-1]

        # Rotate and the disk.

        x_rot, y_rot = self._rotate_coords(x_dep, y_dep, PA)
        x_rot, y_rot = x_rot + x0, y_rot + y0

        # Grid the disk.

        from scipy.interpolate import griddata
        disk = (x_rot.flatten(), y_rot.flatten())
        grid = (self.xaxis[None, :], self.yaxis[:, None])
        r_obs = griddata(disk, rdisk.flatten(), grid,
                         method=self.shadowed_method)
        t_obs = griddata(disk, tdisk.flatten(), grid,
                         method=self.shadowed_method)
        return r_obs, t_obs, z_func(r_obs)

    def _get_diskframe_coords(self):
        """Disk-frame coordinates based on the cube axes."""
        x_disk = np.linspace(self.shadowed_extend * self.xaxis[0],
                             self.shadowed_extend * self.xaxis[-1],
                             int(self.nxpix * self.shadowed_oversample))[::-1]
        y_disk = np.linspace(self.shadowed_extend * self.yaxis[0],
                             self.shadowed_extend * self.yaxis[-1],
                             int(self.nypix * self.shadowed_oversample))
        x_disk, y_disk = np.meshgrid(x_disk, y_disk)
        r_disk = np.hypot(x_disk, y_disk)
        t_disk = np.arctan2(y_disk, x_disk)
        return x_disk, y_disk, r_disk, t_disk

    # -- DATA I/O -- #

    def _read_FITS(self, path, fill=None):
        """Reads the data from the FITS file."""

        # File names.

        self.path = os.path.expanduser(path)
        self.fname = self.path.split('/')[-1]

        # FITS data.

        self.header = fits.getheader(path)
        self.data = np.squeeze(fits.getdata(self.path))
        if fill is not None:
            self.data = np.where(np.isfinite(self.data), self.data, fill)

        # Position axes.

        self.xaxis = self._readpositionaxis(a=1)
        self.yaxis = self._readpositionaxis(a=2)

        # Spectral axis.

        self.nu0 = self._readrestfreq()
        try:
            self.velax = self._readvelocityaxis()
            if self.velax.size > 1:
                self.chan = np.mean(np.diff(self.velax))
            else:
                self.chan = np.nan
            self.freqax = self._readfrequencyaxis()
            if self.chan < 0.0:
                self.data = self.data[::-1]
                self.velax = self.velax[::-1]
                self.freqax = self.freqax[::-1]
                self.chan *= -1.0
        except KeyError:
            self.velax = None
            self.chan = None
            self.freqax = None

        # Check that the data is saved such that increasing indices in x are
        # decreasing in offset counter to the yaxis.

        if np.diff(self.xaxis).mean() > 0.0:
            self.xaxis = self.xaxis[::-1]
            self.data = self.data[:, ::-1]

        # Read the beam properties.

        self._read_beam()

    def _read_beam(self):
        """Reads the beam properties from the header."""
        try:
            if self.header.get('CASAMBM', False):
                beam = fits.open(self.path)[1].data
                beam = np.median([b[:3] for b in beam.view()], axis=0)
                self.bmaj, self.bmin, self.bpa = beam
            else:
                self.bmaj = self.header['bmaj'] * 3600.
                self.bmin = self.header['bmin'] * 3600.
                self.bpa = self.header['bpa']
            self.beamarea_arcsec = self._calculate_beam_area_arcsec()
            self.beamarea_str = self._calculate_beam_area_str()
        except Exception:
            print("WARNING: No beam values found. Assuming pixel as beam.")
            self.bmaj = self.dpix
            self.bmin = self.dpix
            self.bpa = 0.0
            self.beamarea_arcsec = self.dpix**2.0
            self.beamarea_str = np.radians(self.dpix / 3600.)**2.0
        self.bpa %= 180.0

    def print_beam(self):
        """Print the beam properties."""
        print('{:.2f}" x {:.2f}" at {:.1f} deg'.format(*self.beam))

    @property
    def beam(self):
        """Returns beam properties."""
        return self.bmaj, self.bmin, self.bpa

    @property
    def beams_per_pix(self):
        """Number of beams per pixel."""
        return self.dpix**2.0 / self.beamarea_arcsec

    @property
    def pix_per_beam(self):
        """Number of pixels in a beam."""
        return self.beamarea_arcsec / self.dpix**2.0

    def _clip_cube_velocity(self, v_min=None, v_max=None):
        """Clip the cube to within ``vmin`` and ``vmax``."""
        if self.data.ndim == 2:
            raise ValueError("Attaced cube has no velocity axis.")
        v_min = self.velax[0] if v_min is None else v_min
        v_max = self.velax[-1] if v_max is None else v_max
        i = abs(self.velax - v_min).argmin()
        i += 1 if self.velax[i] < v_min else 0
        j = abs(self.velax - v_max).argmin()
        j -= 1 if self.velax[j] > v_max else 0
        self.velax = self.velax[i:j+1]
        self.data = self.data[i:j+1]

    def _clip_cube_spatial(self, radius):
        """Clip the cube plus or minus clip arcseconds from the origin."""
        if radius > min(self.xaxis.max(), self.yaxis.max()):
            print('WARNING: FOV = {:.1f}" larger than '.format(radius * 2)
                  + 'FOV of cube: {:.1f}".'.format(self.xaxis.max() * 2))
        else:
            self._original_shape = self.data.shape
            xa = abs(self.xaxis - radius).argmin()
            if self.xaxis[xa] < radius:
                xa -= 1
            self._xa = xa
            xb = abs(self.xaxis + radius).argmin()
            if -self.xaxis[xb] < radius:
                xb += 1
            xb += 1
            self._xb = xb
            ya = abs(self.yaxis + radius).argmin()
            if -self.yaxis[ya] < radius:
                ya -= 1
            self._ya = ya
            yb = abs(self.yaxis - radius).argmin()
            if self.yaxis[yb] < radius:
                yb += 1
            yb += 1
            self._yb = yb
            if self.data.ndim == 3:
                self.data = self.data[:, ya:yb, xa:xb]
            else:
                self.data = self.data[ya:yb, xa:xb]
            self.xaxis = self.xaxis[xa:xb]
            self.yaxis = self.yaxis[ya:yb]

    @property
    def nxpix(self):
        return self.xaxis.size

    @property
    def nypix(self):
        return self.yaxis.size

    @property
    def dpix(self):
        return np.diff(self.yaxis).mean()

    @property
    def nchan(self):
        if self.velax is not None:
            return self.velax.size
        return 0

    def _readspectralaxis(self, a):
        """Returns the spectral axis in [Hz] or [m/s]."""
        a_len = self.header['naxis%d' % a]
        a_del = self.header['cdelt%d' % a]
        a_pix = self.header['crpix%d' % a]
        a_ref = self.header['crval%d' % a]
        return a_ref + (np.arange(a_len) - a_pix + 1.0) * a_del

    def _readpositionaxis(self, a=1):
        """Returns the position axis in [arcseconds]."""
        if a not in [1, 2]:
            raise ValueError("'a' must be in [1, 2].")
        try:
            a_len = self.header['naxis%d' % a]
            a_del = self.header['cdelt%d' % a]
            a_pix = self.header['crpix%d' % a] - 0.5
        except KeyError:
            if self._user_pixel_scale is None:
                print('WARNING: No axis information found.')
                _input = input("\t Enter pixel scale size in [arcsec]: ")
                self._user_pixel_scale = float(_input) / 3600.0
            a_len = self.data.shape[-1] if a == 1 else self.data.shape[-2]
            if a == 1:
                a_del = -1.0 * self._user_pixel_scale
            else:
                a_del = 1.0 * self._user_pixel_scale
            a_pix = a_len / 2.0 + 0.5
        axis = (np.arange(a_len) - a_pix + 1.0) * a_del
        return 3600 * axis

    def _readrestfreq(self):
        """Read the rest frequency."""
        try:
            nu = self.header['restfreq']
        except KeyError:
            try:
                nu = self.header['restfrq']
            except KeyError:
                try:
                    nu = self.header['crval3']
                except KeyError:
                    nu = np.nan
        return nu

    def _readvelocityaxis(self):
        """Wrapper for _velocityaxis and _spectralaxis."""
        a = 4 if 'stokes' in self.header['ctype3'].lower() else 3
        if 'freq' in self.header['ctype%d' % a].lower():
            specax = self._readspectralaxis(a)
            velax = (self.nu0 - specax) * sc.c
            velax /= self.nu0
        else:
            velax = self._readspectralaxis(a)
        return velax

    def _readfrequencyaxis(self):
        """Returns the frequency axis in [Hz]."""
        a = 4 if 'stokes' in self.header['ctype3'].lower() else 3
        if 'freq' in self.header['ctype3'].lower():
            return self._readspectralaxis(a)
        return self._readrestfreq() * (1.0 - self._readvelocityaxis() / sc.c)

    # -- UNIT CONVERSIONS -- #

    def jybeam_to_Tb_RJ(self, data=None, nu=None):
        """[Jy/beam] to [K] conversion using Rayleigh-Jeans approximation."""
        nu = self.nu0 if nu is None else nu
        data = self.data if data is None else data
        jy2k = 1e-26 * sc.c**2 / nu**2 / 2. / sc.k
        return jy2k * data / self._calculate_beam_area_str()

    def jybeam_to_Tb(self, data=None, nu=None):
        """[Jy/beam] to [K] conversion using the full Planck law."""
        nu = self.nu0 if nu is None else nu
        data = self.data if data is None else data
        Tb = 1e-26 * abs(data) / self._calculate_beam_area_str()
        Tb = 2.0 * sc.h * nu**3 / Tb / sc.c**2
        Tb = sc.h * nu / sc.k / np.log(Tb + 1.0)
        return np.where(data >= 0.0, Tb, -Tb)

    def Tb_to_jybeam_RJ(self, data=None, nu=None):
        """[K] to [Jy/beam] conversion using Rayleigh-Jeans approxmation."""
        nu = self.nu0 if nu is None else nu
        data = self.data if data is None else data
        jy2k = 1e-26 * sc.c**2 / nu**2 / 2. / sc.k
        return data * self._calculate_beam_area_str() / jy2k

    def Tb_to_jybeam(self, data=None, nu=None):
        """[K] to [Jy/beam] conversion using the full Planck law."""
        nu = self.nu0 if nu is None else nu
        data = self.data if data is None else data
        Fnu = 2. * sc.h * nu**3 / sc.c**2
        Fnu /= np.exp(sc.h * nu / sc.k / abs(data)) - 1.0
        Fnu *= self._calculate_beam_area_str() / 1e-26
        return np.where(data >= 0.0, Fnu, -Fnu)

    # -- BEAM FUNCTIONS -- #

    def _calculate_beam_area_arcsec(self):
        """Beam area in square arcseconds."""
        omega = self.bmin * self.bmaj
        if self.bmin == self.dpix and self.bmaj == self.dpix:
            return omega
        return np.pi * omega / 4. / np.log(2.)

    def _calculate_beam_area_str(self):
        """Beam area in steradians."""
        omega = np.radians(self.bmin / 3600.)
        omega *= np.radians(self.bmaj / 3600.)
        if self.bmin == self.dpix and self.bmaj == self.dpix:
            return omega
        return np.pi * omega / 4. / np.log(2.)

    def _beamkernel(self):
        """Returns the 2D Gaussian kernel for convolution."""
        from astropy.convolution import Gaussian2DKernel
        bmaj = self.bmaj / self.dpix / self.fwhm
        bmin = self.bmin / self.dpix / self.fwhm
        return Gaussian2DKernel(bmin, bmaj, np.radians(self.bpa))

    @staticmethod
    def _convolve_image(image, kernel, fast=True):
        """Convolve the image with the provided kernel."""
        if fast:
            from astropy.convolution import convolve_fft
            return convolve_fft(image, kernel, preserve_nan=True)
        from astropy.convolution import convolve
        return convolve(image, kernel, preserve_nan=True)

    # -- PLOTTING FUNCTIONS -- #

    @staticmethod
    def cmap():
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        c2 = plt.cm.Reds(np.linspace(0, 1, 32))
        c1 = plt.cm.Blues_r(np.linspace(0, 1, 32))
        c1 = np.vstack([c1, [1, 1, 1, 1]])
        colors = np.vstack((c1, c2))
        return mcolors.LinearSegmentedColormap.from_list('eddymap', colors)

    @property
    def extent(self):
        """Cube field of view for use with Matplotlib's ``imshow``."""
        return [self.xaxis[0], self.xaxis[-1],
                self.yaxis[0], self.yaxis[-1]]

    @property
    def FOV(self):
        """Cube field of view."""
        xFOV = self.xaxis[0] - self.xaxis[-1]
        yFOV = self.yaxis[-1] - self.yaxis[0]
        return np.mean([xFOV, yFOV])

    def extent_au(self, dist=1.0):
        """Cube field of view in [au] for use with Matplotlib's ``imshow``."""
        return dist * np.squeeze(self.extent)

    def plot_beam(self, ax, x0=0.1, y0=0.1, **kwargs):
        """Plot the sythensized beam on the provided axes."""
        from matplotlib.patches import Ellipse
        beam = Ellipse(ax.transLimits.inverted().transform((x0, y0)),
                       width=self.bmin, height=self.bmaj, angle=-self.bpa,
                       fill=kwargs.get('fill', False),
                       hatch=kwargs.get('hatch', '//////////'),
                       lw=kwargs.get('linewidth', kwargs.get('lw', 1)),
                       color=kwargs.get('color', kwargs.get('c', 'k')),
                       zorder=kwargs.get('zorder', 1000))
        ax.add_patch(beam)

    def _gentrify_plot(self, ax):
        """Gentrify the plot with a grid, label axes and a beam."""
        from matplotlib.ticker import MaxNLocator
        ax.set_aspect(1)
        ax.grid(ls='--', color='k', alpha=0.2, lw=0.5)
        ax.tick_params(which='both', right=True, top=True)
        ax.set_xlim(self.xaxis.max(), self.xaxis.min())
        ax.set_ylim(self.yaxis.min(), self.yaxis.max())
        ax.xaxis.set_major_locator(MaxNLocator(5, min_n_ticks=3))
        ax.yaxis.set_major_locator(MaxNLocator(5, min_n_ticks=3))
        ax.set_xlabel('Offset (arcsec)')
        ax.set_ylabel('Offset (arcsec)')
        if self.bmaj is not None:
            self.plot_beam(ax=ax)

    def plot_surface(self, x0=0.0, y0=0.0, inc=0.0, PA=0.0, z0=None, psi=None,
                     r_cavity=None, r_taper=None, q_taper=None, w_i=None,
                     w_r=None, w_t=None, z_func=None, w_func=None,
                     shadowed=False, r_max=None, mask=None, fill=None, ax=None,
                     contour_kwargs=None, imshow_kwargs=None, return_fig=True,
                     **_):
        """
        Overplot the emission surface onto the provided axis.

        Args:
            ax (Optional[AxesSubplot]): Axis to plot onto.
            x0 (Optional[float]): Source right ascension offset [arcsec].
            y0 (Optional[float]): Source declination offset [arcsec].
            inc (Optional[float]): Source inclination [deg].
            PA (Optional[float]): Source position angle [deg]. Measured
                between north and the red-shifted semi-major axis in an
                easterly direction.
            z0 (Optional[float]): Aspect ratio at 1" for the emission surface.
                To get the far side of the disk, make this number negative.
            psi (Optional[float]): Flaring angle for the emission surface.
            z1 (Optional[float]): Aspect ratio correction term at 1" for the
                emission surface. Should be opposite sign to z0.
            phi (Optional[float]): Flaring angle correction term for the
                emission surface.
            w_i (Optional[float]): Warp inclination in [degrees] at the disk
                center.
            w_r (Optional[float]): Scale radius of the warp in [arcsec].
            w_t (Optional[float]): Angle of nodes of the warp in [degrees].
            r_max (Optional[float]): Outer radius to plot.
            mask (Optional[array]): A 2D mask to define where the surcace is
                plotted.
            ntheta (Optional[int]): Number of theta contours to plot.
            nrad (Optional[int]): Number of radial contours to plot.
            mask (Optional[array]): Mask used to define regions where the
                surface should be plot. If not provided, will plot everywhere.
            check_mask (Optional[bool]): Mask regions which are like projection
                errors for highly flared surfaces.

        Returns:
            ax (AxesSubplot): Axis with the contours overplotted.
        """

        # Dummy axis to overplot.

        if ax is None:
            fig, ax = plt.subplots()

        rvals, tvals, zvals = self.disk_coords(x0=x0, y0=y0,
                                               inc=inc, PA=PA,
                                               z0=z0, psi=psi,
                                               r_cavity=r_cavity,
                                               r_taper=r_taper,
                                               q_taper=q_taper,
                                               w_i=w_i, w_r=w_r, w_t=w_t,
                                               z_func=z_func,
                                               w_func=w_func,
                                               shadowed=shadowed)

        # Mask the data based on r_max.

        r_max = np.nanmax(rvals) if r_max is None else r_max
        zvals = np.where(rvals <= r_max, zvals, np.nan)
        tvals = np.where(rvals <= r_max, tvals, np.nan)
        tvals = np.where(rvals >= 0.5 * self.bmaj, tvals, np.nan)
        rvals = np.where(rvals <= r_max, rvals, np.nan)

        # Mask the data based on user-defined mask.

        if mask is not None:
            rvals = np.where(mask, rvals, np.nan)
            tvals = np.where(mask, tvals, np.nan)
            zvals = np.where(mask, zvals, np.nan)

        # Fill in the background.

        if fill is not None:
            kw = {} if imshow_kwargs is None else imshow_kwargs
            kw['origin'] = 'lower'
            kw['extent'] = self.extent
            ax.imshow(eval(fill), **kw)

        # Draw the contours. The azimuthal angles are drawn on individually to
        # avoid having overlapping lines about the +\- pi boundary making a
        # particularly thick line.

        kw = {} if contour_kwargs is None else contour_kwargs
        kw['levels'] = kw.pop('levels', np.arange(0.5, 0.99 * r_max, 0.5))
        kw['levels'] = np.append(kw['levels'], 0.99 * r_max)
        kw['linewidths'] = kw.pop('linewidths', 1.0)
        kw['colors'] = kw.pop('colors', 'k')
        ax.contour(self.xaxis, self.yaxis, rvals, **kw)

        kw['levels'] = [0.0]
        for t in np.arange(-np.pi, np.pi, np.pi / 8.0):
            if t - 0.1 < -np.pi:
                a = np.where(abs(tvals - t) <= 0.4,
                             tvals - t, np.nan)
                b = np.where(abs(tvals - 2.0 * np.pi - t) <= 0.4,
                             tvals - 2.0 * np.pi - t, np.nan)
                amask = np.where(np.isfinite(a), 1, -1)
                bmask = np.where(np.isfinite(b), 1, -1)
                a = np.where(np.isfinite(a), a, 0.0)
                b = np.where(np.isfinite(b), b, 0.0)
                ttmp = np.where(amask * bmask < 1, a + b, np.nan)
            elif t + 0.1 > np.pi:
                a = np.where(abs(tvals - t) <= 0.4,
                             tvals - t, np.nan)
                b = np.where(abs(tvals + 2.0 * np.pi - t) <= 0.4,
                             tvals + 2.0 * np.pi - t, np.nan)
                amask = np.where(np.isfinite(a), 1, -1)
                bmask = np.where(np.isfinite(b), 1, -1)
                a = np.where(np.isfinite(a), a, 0.0)
                b = np.where(np.isfinite(b), b, 0.0)
                ttmp = np.where(amask * bmask < 1, a + b, np.nan)
            else:
                ttmp = np.where(abs(tvals - t) <= 0.4, tvals - t, np.nan)
            ax.contour(self.xaxis, self.yaxis, ttmp, **kw)
        ax.set_xlim(max(ax.get_xlim()), min(ax.get_xlim()))
        ax.set_aspect(1)

        if return_fig:
            return fig
