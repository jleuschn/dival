# -*- coding: utf-8 -*-
"""Provides `EllipsesDataset`."""
from itertools import repeat
import numpy as np
from odl import uniform_discr
from odl.phantom import ellipsoid_phantom
from dival.datasets.dataset import GroundTruthDataset


class EllipsesDataset(GroundTruthDataset):
    """Dataset with images of multiple random ellipses.

    This dataset uses :meth:`odl.phantom.ellipsoid_phantom` to create
    the images.
    The images are normalized to have a value range of ``[0., 1.]`` with a
    background value of ``0.``.

    Attributes
    ----------
    space
        ``odl.uniform_discr(min_pt, max_pt, (image_size, image_size),
        dtype='float32')``, with the parameters passed to :meth:`__init__`.
    shape
        ``(image_size, image_size)``, with `image_size` parameter passed to
        :meth:`__init__`. Default ``(128, 128)``.
    train_len
        `train_len` parameter passed to :meth:`__init__`.
        Default ``32000``.
    validation_len
        `validation_len` parameter passed to :meth:`__init__`.
        Default ``3200``.
    test_len
        `test_len` parameter passed to :meth:`__init__`.
        Default ``3200``.
    random_access
        ``False``
    num_elements_per_sample
        ``1``
    """
    def __init__(self, image_size=128, min_pt=None, max_pt=None,
                 train_len=32000, validation_len=3200, test_len=3200,
                 fixed_seeds=False):
        """
        Parameters
        ----------
        image_size : int, optional
            Number of pixels per image dimension. Default: ``128``.
        min_pt : [int, int], optional
            Minimum values of the lp space.
            Default: ``[-image_size/2, -image_size/2]``.
        max_pt : [int, int], optional
            Maximum values of the lp space.
            Default: ``[image_size/2, image_size/2]``.
        train_len : int or `None`, optional
            Length of training set. Default: ``32000``.
            If `None`, infinitely many samples could be generated.
        validation_len : int, optional
            Length of training set. Default: ``3200``.
        test_len : int, optional
            Length of test set. Default: ``3200``.
        fixed_seeds : dict or bool, optional
            Seeds to use for random generation.
            The values of the keys ``'train'``, ``'validation'`` and ``'test'``
            are used. If a seed is `None` or omitted, it is choosen randomly.
            If ``True`` is passed, the seeds
            ``fixed_seeds={'train': 42, 'validation': 2, 'test': 1}`` are used.
            If ``False`` is passed (the default), all seeds are chosen
            randomly.
        """
        self.shape = (image_size, image_size)
        if min_pt is None:
            min_pt = [-self.shape[0]/2, -self.shape[1]/2]
        if max_pt is None:
            max_pt = [self.shape[0]/2, self.shape[1]/2]
        space = uniform_discr(min_pt, max_pt, self.shape, dtype=np.float32)
        self.train_len = train_len
        self.validation_len = validation_len
        self.test_len = test_len
        self.random_access = False
        if isinstance(fixed_seeds, bool):
            if fixed_seeds:
                self.fixed_seeds = {'train': 42, 'validation': 2, 'test': 1}
            else:
                self.fixed_seeds = {}
        else:
            self.fixed_seeds = fixed_seeds.copy()
        super().__init__(space=space)

    def generator(self, part='train'):
        """Yield random ellipse phantom images using
        :meth:`odl.phantom.ellipsoid_phantom`.

        Parameters
        ----------
        part : {``'train'``, ``'validation'``, ``'test'``}, optional
            The data part. Default is ``'train'``.

        Yields
        ------
        image : element of :attr:`space`
            Random ellipse phantom image with values in ``[0., 1.]``.
        """
        seed = self.fixed_seeds.get(part)
        r = np.random.RandomState(seed)
        max_n_ellipse = 70
        ellipsoids = np.empty((max_n_ellipse, 6))
        n = self.get_len(part=part)
        it = repeat(None, n) if n is not None else repeat(None)
        for _ in it:
            v = (r.uniform(-0.4, 1.0, (max_n_ellipse,)))
            a1 = .2 * r.exponential(1., (max_n_ellipse,))
            a2 = .2 * r.exponential(1., (max_n_ellipse,))
            x = r.uniform(-0.9, 0.9, (max_n_ellipse,))
            y = r.uniform(-0.9, 0.9, (max_n_ellipse,))
            rot = r.uniform(0., 2 * np.pi, (max_n_ellipse,))
            n_ellipse = min(r.poisson(40), max_n_ellipse)
            v[n_ellipse:] = 0.
            ellipsoids = np.stack((v, a1, a2, x, y, rot), axis=1)
            image = ellipsoid_phantom(self.space, ellipsoids)
            # normalize the foreground (all non-zero pixels) to [0., 1.]
            image[np.array(image) != 0.] -= np.min(image)
            image /= np.max(image)
            yield image
