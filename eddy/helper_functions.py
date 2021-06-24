from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import matplotlib.pyplot as plt
import numpy as np


# -- MCMC FUNCTIONS -- #

def random_p0(p0, scatter, nwalkers):
    """Introduce scatter to starting positions."""
    p0 = np.squeeze(p0)
    dp0 = np.random.randn(nwalkers * len(p0)).reshape(nwalkers, len(p0))
    dp0 = np.where(p0 == 0.0, 1.0, p0)[None, :] * (1.0 + scatter * dp0)
    return np.where(p0[None, :] == 0.0, dp0 - 1.0, dp0)


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
