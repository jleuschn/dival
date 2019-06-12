# -*- coding: utf-8 -*-
"""Provides an interface to the ODL element classes."""
from dival.util.odl_noise_random_state import (white_noise, uniform_noise,
                                               poisson_noise,
                                               salt_pepper_noise)
from odl.discr.lp_discr import uniform_discr
from odl.operator.operator import Operator
import numpy as np


def uniform_discr_element(inp, space=None):
    """Generate an element of a ODL space from an array-like.

    Parameters
    ----------
    inp : array-like
        The input data from which the element is generated.
    space : `odl.discr.DiscretizedSpace`, optional
        The space which the element will belong to. If not given, a uniform
        discretization space with cell size 1 centered around the origin is
        generated.
    """
    inp = np.asarray(inp)
    if space is None:
        space = uniform_discr(-np.array(inp.shape)/2, np.array(inp.shape)/2,
                              inp.shape)
    element = space.element(inp)
    return element


def apply_noise(x, noise_type, noise_kwargs=None, seed=None,
                random_state=None):
    """Apply noise to an odl element.
    Calls noise functions from ``odl.phantom.noise`` or their equivalents
    from ``dival.util.odl_noise_random_state``, which support a custom random
    state.

    Parameters
    ----------
    x : odl element
        The element to which the noise is applied (in-place).
    noise_type : {'white', 'uniform', 'poisson', 'salt_pepper'}
        Type of noise.
    noise_kwargs : dict, optional
        Keyword arguments to be passed to the noise function, e.g. ``'stddev'``
        for ``'white'`` noise.
        The arguments are:

            * ``'white'`` noise:
                * ``'stddev'``: float, optional
                    Standard deviation of each component of the normal
                    distribution. Default is 1.
                * ``'relative_stddev'``: bool, optional
                    Whether to multiply ``'stddev'`` with ``mean(abs(x))``.
                    Default is ``False``.
            * ``'poisson'`` noise:
                * ``'scaling_factor'``: float, optional
                    If specified, the intensity is multiplied and the samples
                    from the poisson distribution are divided by this factor:
                    ``poisson(x * scaling_factor) / scaling_factor``.
                    Default is ``None``.
    seed : int, optional
        Random seed passed to the noise function.
    random_state : `np.random.RandomState`, optional
        Random state passed to the noise function.
    """
    n_kwargs = noise_kwargs.copy()
    n_kwargs['seed'] = seed
    n_kwargs['random_state'] = random_state
    if noise_type == 'white':
        relative_stddev = n_kwargs.pop('relative_stddev', False)
        stddev = n_kwargs.pop('stddev', 1.)
        if relative_stddev:
            mean_abs = np.mean(np.abs(x))
            stddev *= mean_abs
        noise = white_noise(x.space, stddev=stddev, **n_kwargs)
        x += noise
    elif noise_type == 'uniform':
        noise = uniform_noise(x.space, **n_kwargs)
        x += noise
    elif noise_type == 'poisson':
        scaling_factor = n_kwargs.pop('scaling_factor', None)
        if scaling_factor:
            x.assign(poisson_noise(x * scaling_factor, **n_kwargs) /
                     scaling_factor)
        else:
            x.assign(poisson_noise(x, **n_kwargs))
    elif noise_type == 'salt_pepper':
        noise = salt_pepper_noise(x.domain, **n_kwargs)
        x += noise
    else:
        raise ValueError("unknown noise type '{}'".format(noise_type))


class ExpOperator(Operator):
    def _call(self, x, out):
        np.exp(x, out=out)

    @property
    def inverse(self):
        return LogOperator(self.range, self.domain)


class LogOperator(Operator):
    def _call(self, x, out):
        np.log(x, out=out)

    @property
    def inverse(self):
        return ExpOperator(self.range, self.domain)


class NoiseOperator(Operator):
    """Operator applying noise.

    Wraps `apply_noise`, which calls noise functions from ``odl.phantom.noise``
    or their equivalents from ``dival.util.odl_noise_random_state``, which
    support a custom random state.
    """
    def __init__(self, domain, noise_type, noise_kwargs=None, seed=None,
                 random_state=None):
        """Construct a noise operator.

        Parameters
        ----------
        space : odl space
            Domain and range.
        noise_type : {'white', 'uniform', 'poisson', 'salt_pepper'}
            Type of noise.
        noise_kwargs : dict, optional
            Keyword arguments to be passed to the noise function, e.g.
            ``'stddev'`` for ``'white'`` noise. Cf. docs for `apply_noise`.
        seed : int, optional
            Random seed passed to the noise function.
        random_state : `np.random.RandomState`, optional
            Random state passed to the noise function.
        """
        super().__init__(domain, domain)
        self.noise_type = noise_type or 'white'
        self.noise_kwargs = noise_kwargs or {}
        self.seed = seed
        self.random_state = random_state

    def _call(self, x, out):
        if out is not x:
            out.assign(x)
        apply_noise(out, self.noise_type, noise_kwargs=self.noise_kwargs,
                    seed=self.seed, random_state=self.random_state)
