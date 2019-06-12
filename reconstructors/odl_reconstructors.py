# -*- coding: utf-8 -*-
"""Provides wrappers for reconstruction methods of odl."""
from odl.tomo import fbp_op
from odl.operator import Operator
from odl.discr.discretization import DiscretizedSpaceElement
from odl.solvers.iterative import iterative, statistical
from dival import Reconstructor


class FBPReconstructor(Reconstructor):
    HYPER_PARAMS = {
        'filter_type':
            {'default': 'Ram-Lak',
             'choices': ['Ram-Lak', 'Shepp-Logan', 'Cosine', 'Hamming',
                         'Hann']},
        'frequency_scaling':
            {'default': 1.,
             'range': [0, 1],
             'grid_search_options': {'num_samples': 11}}
    }

    """Reconstructor applying filtered back-projection.

    Attributes
    ----------
    fbp_op : `odl.operator.Operator`
        The operator applying filtered back-projection.
    """
    def __init__(self, ray_trafo, padding=True, hyper_params=None,
                 post_processor=None, **kwargs):
        """Construct an FBP reconstructor.

        Parameters
        ----------
        ray_trafo : `odl.tomo.operators.RayTransform`
            The forward operator. See `odl.tomo.fbp_op` for details.
        padding : bool, optional
            Whether to use padding (the default is ``True``).
            See `odl.tomo.fbp_op` for details.
        post_processor : callable, optional
            Callable that takes the filtered backprojection and returns the
            final reconstruction.
        """
        super().__init__(hyper_params=hyper_params, **kwargs)
        self.ray_trafo = ray_trafo
        self.padding = padding
        self.post_processor = post_processor

    def reconstruct(self, observation):
        self.fbp_op = fbp_op(self.ray_trafo, padding=self.padding,
                             **self.hyper_params)
        reconstruction = self.fbp_op(observation)
        if self.post_processor is not None:
            reconstruction = self.post_processor(reconstruction)
        return reconstruction


# TODO class IterativeReconstructor


class CGReconstructor(Reconstructor):
    """Reconstructor applying the conjugate gradient method.

    Attributes
    ----------
    op : `odl.operator.Operator`
        The forward operator of the inverse problem.
    x0 : ``op.domain`` element
        Initial value.
    niter : int
        Number of iterations.
    callback : odl.solvers.util.callback.Callback
        Object that is called in each iteration.
    op_is_symmetric : bool
        If ``False`` (default), the normal equations are solved. If ``True``,
        `op` is assumed to be self-adjoint and the system of equations is
        solved directly.
    """
    def __init__(self, op, x0, niter, callback=None, op_is_symmetric=False):
        """Construct a CG reconstructor.

        Calls `odl.solvers.iterative.iterative.conjugate_gradient_normal` (if
        ``op_is_symmetric == False``) or
        `odl.solvers.iterative.iterative.conjugate_gradient` (if
        ``op_is_symmetric == True``).

        Arguments
        ---------
        op : `odl.operator.Operator`
            The forward operator of the inverse problem.
            If ``op_is_symmetric == False`` (default), the call
            ``op.derivative(x).adjoint`` must be valid.
            If ``op_is_symmetric == True``, `op` must be linear and
            self-adjoint.
        x0 : ``op.domain`` element
            Initial value.
        niter : int
            Number of iterations.
        callback : odl.solvers.util.callback.Callback or callable, optional
            Object that is called in each iteration.
        op_is_symmetric : bool, optional
            If ``False`` (default), the normal equations are solved. If
            ``True``, `op` is required to be self-adjoint and the system of
            equations is solved directly.
        """
        super().__init__()
        self.op = op
        self.x0 = x0
        self.niter = niter
        self.callback = callback
        self.op_is_symmetric = op_is_symmetric

    def reconstruct(self, observation):
        x = self.x0.copy()
        if self.op_is_symmetric:
            iterative.conjugate_gradient(self.op, x, observation,
                                         self.niter, self.callback)
        else:
            iterative.conjugate_gradient_normal(self.op, x, observation,
                                                self.niter, self.callback)
        return x


class GaussNewtonReconstructor(Reconstructor):
    """Reconstructor applying the Gauss-Newton method.

    Attributes
    ----------
    op : `odl.operator.Operator`
        The forward operator of the inverse problem.
    x0 : ``op.domain`` element
        Initial value.
    niter : int
        Maximum number of iterations.
    zero_seq : iterable
        Zero sequence used for regularization.
    callback : odl.solvers.util.callback.Callback
        Object that is called in each iteration.
    """
    def __init__(self, op, x0, niter, zero_seq=None, callback=None):
        """Construct a Gauss-Newton reconstructor.

        Calls `odl.solvers.iterative.iterative.gauss_newton`.

        Arguments
        ---------
        op : `odl.operator.Operator`
            The forward operator of the inverse problem.
            The call ``op.derivative(x).adjoint`` must be valid.
        x0 : ``op.domain`` element
            Initial value.
        niter : int
            Maximum number of iterations.
        zero_seq : iterable, optional
            Zero sequence used for regularization.
        callback : odl.solvers.util.callback.Callback or callable, optional
            Object that is called in each iteration.
        """
        super().__init__()
        self.op = op
        self.x0 = x0
        self.niter = niter
        self.zero_seq = zero_seq
        self.callback = callback

    def reconstruct(self, observation):
        x = self.x0.copy()
        kwargs = {'callback': self.callback}
        if self.zero_seq is not None:
            kwargs['zero_seq'] = self.zero_seq
        iterative.gauss_newton(self.op, x, observation, self.niter,
                               **kwargs)
        return x


class KaczmarzReconstructor(Reconstructor):
    """Reconstructor applying Kaczmarz's method.

    Attributes
    ----------
    ops : sequence of `odl.Operator`
        The forward operators of the inverse problem.
    x0 : ``op.domain`` element
        Initial value.
    niter : int
        Number of iterations.
    omega : positive float or sequence of positive floats
        Relaxation parameter.
        If a single float is given it is used for all operators.
    random : bool
        Whether the order of the operators is randomized in each iteration.
    projection : callable
        Callable that can be used to modify the iterates in each iteration.
    callback : odl.solvers.util.callback.Callback
        Object that is called in each iteration.
    callback_loop : {'inner', 'outer'}
        Whether the `callback` should be called in the inner or outer loop.
    """
    def __init__(self, ops, x0, niter, omega=1, random=False, projection=None,
                 callback=None, callback_loop='outer'):
        """Construct a Kaczmarz reconstructor.

        Calls `odl.solvers.iterative.iterative.kaczmarz`.

        Arguments
        ---------
        ops : sequence of `odl.Operator`
            The forward operators of the inverse problem.
            The call ``ops[i].derivative(x).adjoint`` must be valid for all i.
        x0 : ``op.domain`` element
            Initial value.
        niter : int
            Number of iterations.
        omega : positive float or sequence of positive floats, optional
            Relaxation parameter.
            If a single float is given it is used for all operators.
        random : bool, optional
            Whether the order of the operators is randomized in each iteration.
        projection : callable, optional
            Callable that can be used to modify the iterates in each iteration.
        callback : odl.solvers.util.callback.Callback or callable, optional
            Object that is called in each iteration.
        callback_loop : {'inner', 'outer'}
            Whether the `callback` should be called in the inner or outer loop.
        """
        super().__init__()
        self.ops = ops
        self.x0 = x0
        self.niter = niter
        self.omega = omega
        self.projection = projection
        self.random = random
        self.callback = callback
        self.callback_loop = callback_loop

    def reconstruct(self, observation):
        x = self.x0.copy()
        iterative.kaczmarz(self.ops, x, observation, self.niter,
                           self.omega, self.projection, self.random,
                           self.callback, self.callback_loop)
        return x


class LandweberReconstructor(Reconstructor):
    """Reconstructor applying Landweber's method.

    Attributes
    ----------
    op : `odl.operator.Operator`
        The forward operator of the inverse problem.
    x0 : ``op.domain`` element
        Initial value.
    niter : int
        Number of iterations.
    omega : positive float
        Relaxation parameter.
    projection : callable
        Callable that can be used to modify the iterates in each iteration.
    callback : odl.solvers.util.callback.Callback
        Object that is called in each iteration.
    """
    def __init__(self, op, x0, niter, omega=None, projection=None,
                 callback=None):
        """Construct a Landweber reconstructor.

        Calls `odl.solvers.iterative.iterative.landweber`.

        Arguments
        ---------
        op : `odl.operator.Operator`
            The forward operator of the inverse problem.
            The call ``op.derivative(x).adjoint`` must be valid.
        x0 : ``op.domain`` element
            Initial value.
        niter : int
            Number of iterations.
        omega : positive float, optional
            Relaxation parameter.
        projection : callable, optional
            Callable that can be used to modify the iterates in each iteration.
            One argument is passed and expected be modified in-place.
        callback : odl.solvers.util.callback.Callback or callable, optional
            Object that is called in each iteration.
        """
        super().__init__()
        self.op = op
        self.x0 = x0
        self.niter = niter
        self.omega = omega
        self.projection = projection
        self.callback = callback

    def reconstruct(self, observation):
        x = self.x0.copy()
        iterative.landweber(self.op, x, observation, self.niter,
                            self.omega, self.projection, self.callback)
        return x


class MLEMReconstructor(Reconstructor):
    """Reconstructor applying Maximum Likelihood Expectation Maximization.

    If multiple operators are given, Ordered Subsets MLEM is applied.

    Attributes
    ----------
    op : `odl.operator.Operator` or sequence of `odl.operator.Operator`
        The forward operator(s) of the inverse problem.
    x0 : ``op.domain`` element
        Initial value.
    niter : int
        Number of iterations.
    noise : {'poisson'}, optional
        Noise model determining the variant of MLEM.
    callback : odl.solvers.util.callback.Callback
        Object that is called in each iteration.
    sensitivities : float or ``op.domain`` `element-like`
        Usable with ``noise='poisson'``. The algorithm contains an ``A^T 1``
        term, if this parameter is given, it is replaced by it.
        Default: ``op[i].adjoint(op[i].range.one())``
    """
    def __init__(self, op, x0, niter, noise='poisson', callback=None,
                 sensitivities=None):
        """Construct a (OS)MLEM reconstructor.

        Calls `odl.solvers.iterative.statistical.osmlem`.

        Arguments
        ---------
        op : `odl.operator.Operator` or sequence of `odl.operator.Operator`
            The forward operator(s) of the inverse problem.
            If an operator sequence is given, Ordered Subsets MLEM is applied.
        x0 : ``op.domain`` element
            Initial value.
        niter : int
            Number of iterations.
        noise : {'poisson'}, optional
            Noise model determining the variant of MLEM.
            For ``'poisson'``, the initial value of ``x`` should be
            non-negative.
        callback : odl.solvers.util.callback.Callback or callable, optional
            Object that is called in each iteration.
        sensitivities : float or ``op.domain`` `element-like`, optional
            Usable with ``noise='poisson'``. The algorithm contains an
            ``A^T 1`` term, if this parameter is given, it is replaced by it.
            Default: ``op[i].adjoint(op[i].range.one())``
        """
        super().__init__()
        self.op = [op] if isinstance(op, Operator) else op
        self.x0 = x0
        self.niter = niter
        self.noise = noise
        self.callback = callback
        self.sensitivities = sensitivities

    def reconstruct(self, observation):
        x = self.x0.copy()
        if isinstance(observation, DiscretizedSpaceElement):
            observation = [observation]
        statistical.osmlem(self.op, x, observation, self.niter,
                           noise=self.noise, callback=self.callback,
                           sensitivities=self.sensitivities)
        return x
