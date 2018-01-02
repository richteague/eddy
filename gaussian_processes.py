"""
Functions to find the rotational velocity via george.
"""

import numpy as np


class annuli:

    def __init__(self, pixels, velax, tvals):
        """An annnulus of points to fit."""

        # Assign the arrays.
        self.velax = velax
        self.tvals = tvals
        self.pixels = pixels

        # Make sure the array is the correct shape.
        if pixels.shape != (self.velax.size, self.tvals.size):
            self.pixels = self.pixels.T

        return

    def estimate_vrot(self):
        """Estimate vrot from the line peaks."""
        centers = np.argmax(self.pixels, axis=0)
        centers = np.take(self.velax, centers)
        return np.nanmax(abs(centers))
