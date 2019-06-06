# -*- coding: utf-8 -*-
"""Provides `EllipsesDataset`."""
import numpy as np
from odl.discr.lp_discr import uniform_discr
from odl.phantom import ellipsoid_phantom
from dival.datasets.dataset import GroundTruthDataset


class EllipsesDataset(GroundTruthDataset):
    """Dataset with images of multiple random ellipses.

    This dataset uses the function `odl.phantom.ellipsoid_phantom` to create
    the images.
    """
    def __init__(self, min_pt=None, max_pt=None):
        """Construct the ellipses dataset.

        Parameters
        ----------
        min_pt : [int, int], optional
            Minimum values of the lp space. Default: [-64, -64].
        max_pt : [int, int], optional
            Maximum values of the lp space. Default: [64, 64].
        """
        self.shape = (128, 128)
        if min_pt is None:
            min_pt = [-self.shape[0]/2, -self.shape[1]/2]
        if max_pt is None:
            max_pt = [self.shape[0]/2, self.shape[1]/2]
        space = uniform_discr(min_pt, max_pt, self.shape, dtype=np.float32)
        self.train_len = 50000
        self.validation_len = 5000
        self.test_len = 5000
        super().__init__(space=space)

    def generator(self, part='train'):
        """Yield random ellipse phantom images using
        `odl.phantom.ellipsoid_phantom`.
        """
        seed = 42
        if part == 'validation':
            seed = 2
        elif part == 'test':
            seed = 1
        r = np.random.RandomState(seed)
        n = self.get_len(part=part)
        n_ellipse = 50
        ellipsoids = np.empty((n_ellipse, 6))
        for _ in range(n):
            v = (r.uniform(-.5, .5, (n_ellipse,)) *
                 r.exponential(.4, (n_ellipse,)))
            a1 = .2 * r.exponential(1., (n_ellipse,))
            a2 = .2 * r.exponential(1., (n_ellipse,))
            x = r.uniform(-1., 1., (n_ellipse,))
            y = r.uniform(-1., 1., (n_ellipse,))
            rot = r.uniform(0., 2*np.pi, (n_ellipse,))
            ellipsoids = np.stack((v, a1, a2, x, y, rot), axis=1)
            image = ellipsoid_phantom(self.space, ellipsoids)

            yield image
