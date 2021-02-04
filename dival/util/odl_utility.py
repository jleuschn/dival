# -*- coding: utf-8 -*-
"""Provides utilities related to ODL."""
import warnings
import copy
from dival.util.odl_noise_random_state import (white_noise, uniform_noise,
                                               poisson_noise,
                                               salt_pepper_noise)
from odl import uniform_discr
from odl.operator.operator import Operator
from odl.solvers.util.callback import Callback
from odl.util import signature_string
import numpy as np
from skimage.transform import resize


def uniform_discr_element(inp, space=None):
    """Generate an element of a ODL space from an array-like.

    Parameters
    ----------
    inp : array-like
        The input data from which the element is generated.
    space : :class:`odl.discr.DiscretizedSpace`, optional
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

    Calls noise functions from :mod:`odl.phantom.noise` or their equivalents
    from :mod:`dival.util.odl_noise_random_state`.

    Parameters
    ----------
    x : odl element
        The element to which the noise is applied (in-place).
    noise_type : {``'white'``, ``'uniform'``, ``'poisson'``, ``'salt_pepper'``}
        Type of noise.
    noise_kwargs : dict, optional
        Keyword arguments to be passed to the noise function, e.g. ``'stddev'``
        for ``'white'`` noise.
        The arguments are:

            * for ``noise_type='white'``:
                * ``'stddev'``: float, optional
                    Standard deviation of each component of the normal
                    distribution. Default is 1.
                * ``'relative_stddev'``: bool, optional
                    Whether to multiply ``'stddev'`` with ``mean(abs(x))``.
                    Default is ``False``.
            * for ``noise_type='poisson'``:
                * ``'scaling_factor'``: float, optional
                    If specified, the intensity is multiplied and the samples
                    from the poisson distribution are divided by this factor:
                    ``poisson(x * scaling_factor) / scaling_factor``.
                    Default is `None`.
    seed : int, optional
        Random seed passed to the noise function.
    random_state : :class:`np.random.RandomState`, optional
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


class NoiseOperator(Operator):
    """Operator applying noise.

    Wraps :func:`apply_noise`, which calls noise functions from
    :mod:`odl.phantom.noise` or their equivalents from
    :mod:`dival.util.odl_noise_random_state`.
    """
    def __init__(self, domain, noise_type, noise_kwargs=None, seed=None,
                 random_state=None):
        """
        Parameters
        ----------
        space : odl space
            Domain and range.
        noise_type : {``'white'``, ``'uniform'``, ``'poisson'``,\
                      ``'salt_pepper'``}
            Type of noise.
        noise_kwargs : dict, optional
            Keyword arguments to be passed to the noise function, cf. docs for
            :func:`apply_noise`.
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


class CallbackStore(Callback):
    """This is a modified copy of odl.solvers.util.callback.CallbackStore,
    Copyright held by The ODL contributors, subject to the terms of the
    Mozilla Public License, v. 2.0. If a copy of the MPL was not distributed
    with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
    This copy incorporates https://github.com/odlgroup/odl/pull/1539.

    Callback for storing all iterates of a solver.
    Can optionally apply a function, for example the norm or calculating the
    residual.
    By default, calls the ``copy()`` method on the iterates before storing.
    """

    def __init__(self, results=None, function=None, step=1):
        """Initialize a new instance.

        Parameters
        ----------
        results : list, optional
            List in which to store the iterates.
            Default: new list (``[]``)
        function : callable, optional
            Deprecated, use composition instead. See examples.
            Function to be called on all incoming results before storage.
            Default: copy
        step : int, optional
            Number of iterates between storing iterates.

        Examples
        --------
        Store results as-is:
        >>> callback = CallbackStore()
        Provide list to store iterates in:
        >>> results = []
        >>> callback = CallbackStore(results=results)
        Store the norm of the results:
        >>> norm_function = lambda x: x.norm()
        >>> callback = CallbackStore() * norm_function
        """
        self.results = [] if results is None else results
        self.function = function
        if function is not None:
            warnings.warn('`function` argument is deprecated and will be '
                          'removed in a future release. Use composition '
                          'instead. '
                          'See Examples in the documentation.',
                          DeprecationWarning)
        self.step = int(step)
        self.iter = 0

    def __call__(self, result):
        """Append result to results list."""
        if self.iter % self.step == 0:
            if self.function:
                self.results.append(self.function(result))
            else:
                self.results.append(copy.copy(result))
        self.iter += 1

    def reset(self):
        """Clear the results list."""
        self.results = []
        self.iter = 0

    def __iter__(self):
        """Allow iteration over the results."""
        return iter(self.results)

    def __getitem__(self, index):
        """Return ``self[index]``.
        Get iterates by index.
        """
        return self.results[index]

    def __len__(self):
        """Number of results stored."""
        return len(self.results)

    def __repr__(self):
        """Return ``repr(self)``."""
        optargs = [('results', self.results, []),
                   ('function', self.function, None),
                   ('step', self.step, 1)]
        inner_str = signature_string([], optargs)
        return '{}({})'.format(self.__class__.__name__, inner_str)


class CallbackStoreAfter(Callback):
    """Callback for storing after specific numbers of iterations of a solver.
    Calls the ``copy()`` method on the iterates before storing.

    The source code of this class is based on
    odl.solvers.util.callback.CallbackStore, Copyright held by The ODL
    contributors, subject to the terms of the Mozilla Public License, v. 2.0.
    If a copy of the MPL was not distributed with this file, You can obtain one
    at https://mozilla.org/MPL/2.0/.
    """

    def __init__(self, results=None, store_after_iters=None):
        """Initialize a new instance.

        Parameters
        ----------
        results : list, optional
            List in which to store the iterates.
            Default: new list (``[]``)
        store_after_iters : list of int, optional
            Numbers of iterations after which the result should be stored.
        """
        self.results = results if results is not None else []
        self.store_after_iters = (store_after_iters
                                  if store_after_iters is not None else [])
        self.iter = 0

    def __call__(self, result):
        """Append result to results list."""
        if (self.iter + 1) in self.store_after_iters:
            self.results.append(copy.copy(result))
        self.iter += 1

    def reset(self):
        """Clear the results list."""
        self.results = []
        self.iter = 0

    def __iter__(self):
        """Allow iteration over the results."""
        return iter(self.results)

    def __getitem__(self, index):
        """Return ``self[index]``.
        Get iterates by index.
        """
        return self.results[index]

    def __len__(self):
        """Number of results stored."""
        return len(self.results)

    def __repr__(self):
        """Return ``repr(self)``."""
        optargs = [('results', self.results, []),
                   ('store_after_iters', self.store_after_iters, [])]
        inner_str = signature_string([], optargs)
        return '{}({})'.format(self.__class__.__name__, inner_str)


class ResizeOperator(Operator):
    def __init__(self, reco_space, space, order=1):
        self.target_shape = space.shape
        self.order = order
        super().__init__(reco_space, space)

    def _call(self, x, out):
        out.assign(self.range.element(
            resize(x, self.target_shape, order=self.order)))


class RayBackProjection(Operator):
    """Adjoint of the discrete Ray transform between L^p spaces.

    This class is copied and modified from
    `odl <https://github.com/odlgroup/odl/blob/25ec783954a85c2294ad5b76414f8c7c3cd2785d/odl/tomo/operators/ray_trafo.py#L324>`_.

    This main-scope class definition is used by
    :func:`patch_ray_trafo_for_pickling` to make a ray transform object
    pickleable by replacing its :attr:`_adjoint` attribute with an instance of
    this class.
    """
    def __init__(self, ray_trafo, **kwargs):
        self.ray_trafo = ray_trafo
        super().__init__(**kwargs)

    def _call(self, x, out=None, **kwargs):
        """Backprojection.

        Parameters
        ----------
        x : DiscretizedSpaceElement
            A sinogram. Must be an element of
            `RayTransform.range` (domain of `RayBackProjection`).
        out : `RayBackProjection.domain` element, optional
            A volume to which the result of this evaluation is
            written.
        **kwargs
            Extra keyword arguments, passed on to the
            implementation backend.

        Returns
        -------
        DiscretizedSpaceElement
            Result of the transform in the domain
            of `RayProjection`.
        """
        return self.ray_trafo.get_impl(
            self.ray_trafo.use_cache
        ).call_backward(x, out, **kwargs)

    @property
    def geometry(self):
        return self.ray_trafo.geometry

    @property
    def adjoint(self):
        return self.ray_trafo


def patch_ray_trafo_for_pickling(ray_trafo):
    """
    Make an object of type :class:`odl.tomo.operators.RayTransform` pickleable
    by overwriting the :attr:`_adjoint` (which originally has a local class
    type) with a :class:`dival.util.torch_utility.RayBackProjection` object.
    This can be required for multiprocessing.

    Parameters
    ----------
    ray_trafo : :class:`odl.tomo.operators.RayTransform`
        The ray transform to patch for pickling.
    """
    kwargs = ray_trafo._extra_kwargs.copy()
    kwargs['domain'] = ray_trafo.range
    ray_trafo._adjoint = RayBackProjection(
        ray_trafo, range=ray_trafo.domain, linear=True, **kwargs
    )
