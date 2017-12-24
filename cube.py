"""
Image cube class to deproject the coordinates of the image. Inclination and
position angle are in degrees, center offsets are in arcseconds.
"""

import numpy as np
from astropy.io import fits
import scipy.constants as sc


class imagecube:

    def __init__(self, path, inc=0.0, pa=0.0, dx=0.0, dy=0.0):
        """Read in the image cube."""

        # Read in the data and header.
        self.path = path
        self.filename = self.path.split('/')[-1]
        self.data = np.squeeze(fits.getdata(self.path))
        self.header = fits.getheader(path)
        self.bunit = self.header['bunit']
        self.restfreq = self._read_rest_frequency()

        # Define the axes.
        self.xaxis = self._get_position_axis(1)
        self.yaxis = self._get_position_axis(2)
        self.nxpix = int(self.xaxis.size)
        self.nypix = int(self.yaxis.size)
        self.dpix = self._pixelscale()
        self.velax = self._get_velocity_axis()
        self.nvpix = int(self.velax.size)
        self.chan = np.mean(np.diff(self.velax))

        # Attempt to read the beamsize.
        self.bmin, self.bmaj, self.bpa = self._get_beam_params()

        # Calculate the deprojected values.
        self.inc, self.pa = inc, pa
        self.dx, self.dy = dx, dy
        self.rvals, self.tvals = self._deproject(inc, pa, dx, dy)

        return

    def _get_rest_frequency(self):
        """Read the rest frequency in [Hz]."""
        try:
            return self.header['restfreq']
        except KeyError:
            return self.header['restfrq']

    def _get_position_axis(self, a=1):
        """Returns the position axis in ["]."""
        if a not in [1, 2]:
            raise ValueError("'a' must be in [0, 1].")
        a_len = self.header['naxis%d' % a]
        a_del = self.header['cdelt%d' % a]
        a_pix = self.header['crpix%d' % a]
        return 3600. * ((np.arange(1, a_len+1) - a_pix + 0.5) * a_del)

    def _get_spectral_axis(self):
        """Returns the spectral axis in either [Hz] or [m/s]."""
        a_len = self.header['naxis3']
        a_del = self.header['cdelt3']
        a_pix = self.header['crpix3']
        a_ref = self.header['crval3']
        return (a_ref + (np.arange(a_len) - a_pix + 0.5) * a_del)

    def _get_velocity_axis(self, fn):
        """Returns the velocity axis in [m/s]."""
        specax = self._read_spectral_axis()
        if self.header['ctype3'].lower() == 'freq':
            return (self.restfreq - specax) * sc.c / self.restfreq
        return specax

    def _get_beam_params(self):
        """Return the parameters of the beam."""
        try:
            bmaj = self.header['bmaj'] * 3600.
            bmin = self.header['bmin'] * 3600.
            bpa = self.header['bpa']
        except KeyError:
            bmaj = self.dpix
            bmin = self.dpix
            bpa = 0.0
        return bmin, bmaj, bpa

    def _get_beamarea(self):
        """Beam area in steradians."""
        omega = np.radians(self.bmin / 3600.)
        omega *= np.radians(self.bmaj / 3600.)
        if self.bmin == self.dpix and self.bmaj == self.dpix:
            return omega
        return np.pi * omega / 4. / np.log(2.)

    def Jybeam_to_Tb(self):
        """Calculate the Jy/beam to K conversion."""
        if self.bunit == 'K':
            return 1.0
        omega = self._get_beamarea()
        return 1e-26 * sc.c**2 / self.restfreq**2 / 2. / sc.k / omega

    def _deproject(self, inc=0.0, pa=0.0, dx=0.0, dy=0.0):
        """Returns the deprojected polar pixel values in [arcsec, rad]."""
        x_sky = self.xaxis[None, :] * np.ones(self.nypix)[:, None] - dx
        y_sky = self.yaxis[:, None] * np.ones(self.nxpix)[None, :] - dy
        x_rot, y_rot = self._rotate(x_sky, y_sky, np.radians(pa))
        x_dep, y_dep = self._incline(x_rot, y_rot, np.radians(inc))
        return np.hypot(x_dep, y_dep), np.arctan2(y_dep, x_dep)

    def _rotate(self, x, y, t):
        '''Rotation by angle t [rad].'''
        x_rot = x * np.cos(t) + y * np.sin(t)
        y_rot = y * np.cos(t) - x * np.sin(t)
        return x_rot, y_rot

    def _incline(self, x, y, i):
        '''Incline the image by angle i [rad].'''
        return x, y / np.cos(i)
