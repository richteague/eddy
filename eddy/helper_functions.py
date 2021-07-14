from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np


# -- MCMC / OPTIMIZATION FUNCTIONS -- #

def random_p0(p0, scatter, nwalkers):
    """Introduce scatter to starting positions."""
    p0 = np.squeeze(p0)
    dp0 = np.random.randn(nwalkers * len(p0)).reshape(nwalkers, len(p0))
    dp0 = np.where(p0 == 0.0, 1.0, p0)[None, :] * (1.0 + scatter * dp0)
    return np.where(p0[None, :] == 0.0, dp0 - 1.0, dp0)


def fit_gaussian(x, y, dy=None, return_uncertainty=None):
    """Fit a gaussian to (x, y, [dy])."""
    if return_uncertainty is None:
        return_uncertainty = dy is not None
    dy = np.ones(x.size) * (1.0 if dy is None else dy)
    if np.all(np.isnan(dy)):
        dy = np.ones(x.size)
        absolute_sigma = False
    else:
        absolute_sigma = True
    p0 = get_p0_gaussian(x, y)
    try:
        popt, cvar = curve_fit(gaussian, x, y, sigma=dy, p0=p0,
                               absolute_sigma=absolute_sigma,
                               maxfev=100000)
        cvar = np.diag(cvar)**0.5
    except Exception:
        popt = [np.nan, np.nan, np.nan]
        cvar = popt.copy()
    return (popt, cvar) if return_uncertainty else popt


def fit_gaussian_thick(x, y, dy=None, return_uncertainty=None):
    """Fit an optically thick Gaussian function to (x, y, [dy])."""
    if return_uncertainty is None:
        return_uncertainty = dy is not None
    dy = np.ones(x.size) * (1.0 if dy is None else dy)
    if np.all(np.isnan(dy)):
        dy = np.ones(x.size)
        absolute_sigma = False
    else:
        absolute_sigma = True
    p0 = fit_gaussian(x=x, y=y, dy=dy,
                      return_uncertainty=False)
    p0 = np.append(p0, 0.5)
    try:
        popt, cvar = curve_fit(gaussian_thick, x, y, sigma=dy, p0=p0,
                               absolute_sigma=absolute_sigma,
                               maxfev=100000)
        cvar = np.diag(cvar)**0.5
    except Exception:
        popt = [np.nan, np.nan, np.nan, np.nan]
        cvar = popt.copy()
    return (popt, cvar) if return_uncertainty else popt


def get_gaussian_center(x, y, dy=None, return_uncertainty=None, fill=1e50):
    """Return the line center from a Gaussian fit to the spectrum."""
    if return_uncertainty is None:
        return_uncertainty = dy is not None
    popt, cvar = fit_gaussian(x, y, dy, return_uncertainty=True)
    if np.isfinite(popt[0]):
        return (popt[0], cvar[0]) if return_uncertainty else popt[0]
    return (fill, fill) if return_uncertainty else fill


def get_gaussthick_center(x, y, dy=None, return_uncertainty=None, fill=1e50):
    """Return the line center from a Gaussian fit to the spectrum."""
    if return_uncertainty is None:
        return_uncertainty = dy is not None
    popt, cvar = fit_gaussian_thick(x, y, dy, return_uncertainty=True)
    if np.isfinite(popt[0]):
        return (popt[0], cvar[0]) if return_uncertainty else popt[0]
    return (fill, fill) if return_uncertainty else fill


def get_gaussian_width(x, y, dy=None, return_uncertainty=None, fill=1e50):
    """Return the absolute width of a Gaussian fit to the spectrum."""
    if return_uncertainty is None:
        return_uncertainty = dy is not None
    popt, cvar = fit_gaussian(x, y, dy, return_uncertainty=True)
    if np.isfinite(popt[1]):
        return (popt[1], cvar[1]) if return_uncertainty else popt[1]
    return (fill, fill) if return_uncertainty else fill


def get_p0_gaussian(x, y):
    """Estimate (x0, dV, Tb) for the spectrum."""
    if x.size != y.size:
        raise ValueError("Mismatch in array shapes.")
    Tb = np.max(y)
    x0 = x[y.argmax()]
    dV = np.trapz(y, x) / Tb / np.sqrt(2. * np.pi)
    return x0, dV, Tb


# -- MODEL FUNCTIONS --#

def gaussian(x, x0, dV, Tb):
    """Gaussian function."""
    return Tb * np.exp(-np.power((x - x0) / dV, 2.0))


def gaussian_thick(x, x0, dV, Tex, tau0):
    """Optically thick Gaussian line."""
    tau = gaussian(x, x0, dV, tau0)
    return Tex * (1. - np.exp(-tau))


def SHO(x, A, y0):
    """Simple harmonic oscillator."""
    return A * np.cos(x) + y0


def SHO_offset(x, A, y0, dx):
    """Simple harmonic oscillator with offset."""
    return A * np.cos(x + dx) + y0


def SHO_double(x, A, B, y0):
    """Two orthogonal simple harmonic oscillators."""
    return A * np.cos(x) + B * np.sin(x) + y0


# -- PLOTTING FUNCTIONS -- #

def plot_walkers(samples, nburnin=None, labels=None, histogram=True):
    """
    Plot the walkers to check if they are burning in.

    Args:
        samples (ndarray):
        nburnin (Optional[int])
    """

    # Check the length of the label list.

    if labels is not None:
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

        # Include the histogram.

        if histogram:
            fig.set_size_inches(1.37 * fig.get_figwidth(),
                                fig.get_figheight(), forward=True)
            ax_divider = make_axes_locatable(ax)
            bins = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 50)
            hist, _ = np.histogram(sample[nburnin:].flatten(), bins=bins,
                                   density=True)
            bins = np.average([bins[1:], bins[:-1]], axis=0)
            ax1 = ax_divider.append_axes("right", size="35%", pad="2%")
            ax1.fill_betweenx(bins, hist, np.zeros(bins.size), step='mid',
                              color='darkgray', lw=0.0)
            ax1.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1])
            ax1.set_xlim(0, ax1.get_xlim()[1])
            ax1.set_yticklabels([])
            ax1.set_xticklabels([])
            ax1.tick_params(which='both', left=0, bottom=0, top=0, right=0)
            ax1.spines['right'].set_visible(False)
            ax1.spines['bottom'].set_visible(False)
            ax1.spines['top'].set_visible(False)


def plot_corner(samples, labels=None, quantiles=[0.16, 0.5, 0.84]):
    """
    A wrapper for DFM's corner plots.

    Args:
        samples (ndarry):
        labels (Optional[list]):
        quantiles (Optional[list]): Quantiles to use for plotting.
    """
    import corner
    corner.corner(samples, labels=labels, title_fmt='.4f', bins=30,
                  quantiles=quantiles, show_titles=True)
