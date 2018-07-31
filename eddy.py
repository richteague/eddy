"""Simple class to deproject spectra and measure the rotation velocity."""

import numpy as np
from scipy.stats import binned_statistic
from scipy.optimize import curve_fit
from scipy.optimize import minimize_scalar


class ensemble:

    def __init__(self, spectra, theta, velax):
        """Initialize the class."""

        # Disk polar angles.
        self.theta = theta
        self.theta_deg = np.degrees(theta)

        # Intensity / brightness temperature.
        self.spectra = spectra
        self.spectra_flat = spectra.flatten()

        # Velocity axis.
        self.velax = velax
        self.channel = np.diff(velax)[0]
        self.velax_range = (self.velax[0] - 0.5 * self.channel,
                            self.velax[-1] + 0.5 * self.channel)

    # -- Rotation Velocity by Minimizing Linewidth -- #

    def get_p0(self, velax, spectrum):
        """Estimate (x0, dV, Tb) for the spectrum."""
        Tb = np.max(spectrum)
        x0 = velax[spectrum.argmax()]
        dV = np.trapz(spectrum, velax) / Tb / np.sqrt(2. * np.pi)
        return x0, dV, Tb

    def get_gaussian_width(self, spectrum, fill_value=1e50):
        """Return the absolute width of a Gaussian fit to the spectrum."""
        try:
            dV = curve_fit(self.gaussian, self.velax, spectrum,
                           p0=self.get_p0(self.velax, spectrum),
                           maxfev=100000)[0][1]
            return abs(dV)
        except:
            return fill_value

    def get_deprojected_width(self, vrot, resample=True):
        """Return the spectrum from a Gaussian fit."""
        if resample:
            x, y = self.deprojected_spectrum(vrot)
        else:
            x, y = self.deprojected_spectra(vrot, sort=True)
            y = y[np.logical_and(x >= self.velax[0], x <= self.velax[-1])]
        return self.get_gaussian_width(y)

    def get_vrot_dV(self, guess=None, resample=True):
        """Get the rotation velocity by minimizing the linewidth."""
        guess = self.guess_parameters(fit=True)[0] if guess is None else guess
        bounds = np.array([0.7, 1.3]) * guess
        return minimize_scalar(self.get_deprojected_width, method='bounded',
                               bounds=bounds, args=(resample)).x

    # -- Line Profile Functions -- #

    def gaussian(self, x, x0, dV, Tb):
        """Gaussian function."""
        return Tb * np.exp(-np.power((x - x0) / dV, 2.0))

    def thickline(self, x, x0, dV, Tex, tau):
        """Optically thick line profile."""
        if tau <= 0.0:
            raise ValueError("Must have positive tau.")
        return Tex * (1. - np.exp(-self.gaussian(x, x0, dV, tau)))

    def SHO(self, x, A, y0):
        """Simple harmonic oscillator."""
        return A * np.cos(x) + y0

    # -- Deprojection Functions -- #

    def deprojected_spectra(self, vrot, sort=True):
        """Returns (x, y) of all deprojected points."""
        vpnts = self.velax[None, :] - vrot * np.cos(self.theta)[:, None]
        vpnts = vpnts.flatten()
        if not sort:
            return vpnts, self.spectra_flat
        return np.squeeze(zip(*sorted(zip(vpnts, self.spectra_flat))))

    def deprojected_spectrum(self, vrot):
        """Returns (x, y) of deprojected and binned spectrum."""
        vpnts, spnts = self.deprojected_spectra(vrot, sort=True)
        spectrum = binned_statistic(vpnts, spnts, statistic='mean',
                                    bins=self.velax.size,
                                    range=self.velax_range)[0]
        return self.velax, spectrum

    def guess_parameters(self, fit=True):
        """Guess vrot and vlsr from the spectra.."""
        vpeaks = np.take(self.velax, np.argmax(self.spectra, axis=1))
        vrot = 0.5 * (np.max(vpeaks) - np.min(vpeaks))
        vlsr = np.mean(vpeaks)
        if not fit:
            return vrot, vlsr
        try:
            return curve_fit(self.SHO, self.theta, vpeaks,
                             p0=[vrot, vlsr], maxfev=10000)[0]
        except:
            return vrot, vlsr
