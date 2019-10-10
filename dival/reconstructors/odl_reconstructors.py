# -*- coding: utf-8 -*-
"""Provides wrappers for reconstruction methods of odl."""
from odl.tomo import fbp_op
from odl.operator import Operator
from odl.space.pspace import ProductSpace
from odl.solvers.iterative import iterative, statistical
from dival import Reconstructor, IterativeReconstructor


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
        It is computed in the constructor, and is recomputed for each
        reconstruction if ``recompute_fbp_op == True`` (since parameters could
        change).
    """
    def __init__(self, ray_trafo, padding=True, hyper_params=None,
                 pre_processor=None, post_processor=None,
                 recompute_fbp_op=True, **kwargs):
        """
        Parameters
        ----------
        ray_trafo : `odl.tomo.operators.RayTransform`
            The forward operator. See `odl.tomo.fbp_op` for details.
        padding : bool, optional
            Whether to use padding (the default is ``True``).
            See `odl.tomo.fbp_op` for details.
        pre_processor : callable, optional
            Callable that takes the observation and returns the sinogram that
            is passed to the filtered back-projection operator.
        post_processor : callable, optional
            Callable that takes the filtered back-projection and returns the
            final reconstruction.
        recompute_fbp_op : bool, optional
            Whether :attr:`fbp_op` should be recomputed on each call to
            :meth:`reconstruct`. Must be ``True`` (default) if changes to
            :attr:`ray_trafo`, :attr:`hyper_params` or :attr:`padding` are
            planned in order to use the updated values in :meth:`reconstruct`.
            If none of these attributes will change, you may specify
            ``recompute_fbp_op==False``, so :attr:`fbp_op` can be computed
            only once, improving reconstruction time efficiency.
        """
        self.ray_trafo = ray_trafo
        self.padding = padding
        self.pre_processor = pre_processor
        self.post_processor = post_processor
        super().__init__(
            reco_space=ray_trafo.domain, observation_space=ray_trafo.range,
            hyper_params=hyper_params, **kwargs)
        self.fbp_op = fbp_op(self.ray_trafo, padding=self.padding,
                             **self.hyper_params)
        self.recompute_fbp_op = recompute_fbp_op

    def _reconstruct(self, observation, out):
        if self.pre_processor is not None:
            observation = self.pre_processor(observation)
        if self.recompute_fbp_op:
            self.fbp_op = fbp_op(self.ray_trafo, padding=self.padding,
                                 **self.hyper_params)
        self.fbp_op(observation, out=out)
        if self.post_processor is not None:
            out[:] = self.post_processor(out)


class CGReconstructor(IterativeReconstructor):
    """Iterative reconstructor applying the conjugate gradient method.

    Attributes
    ----------
    op : `odl.operator.Operator`
        The forward operator of the inverse problem.
    x0 : ``op.domain`` element
        Initial value.
    niter : int
        Number of iterations.
    callback : :class:`odl.solvers.util.callback.Callback` or `None`
        Object that is called in each iteration.
    op_is_symmetric : bool
        If ``False`` (default), the normal equations are solved. If ``True``,
        `op` is assumed to be self-adjoint and the system of equations is
        solved directly.
    """
    def __init__(self, op, x0, niter, callback=None, op_is_symmetric=False,
                 **kwargs):
        """
        Calls `odl.solvers.iterative.iterative.conjugate_gradient_normal` (if
        ``op_is_symmetric == False``) or
        `odl.solvers.iterative.iterative.conjugate_gradient` (if
        ``op_is_symmetric == True``).

        Parameters
        ----------
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
        callback : :class:`odl.solvers.util.callback.Callback`, optional
            Object that is called in each iteration.
        op_is_symmetric : bool, optional
            If ``False`` (default), the normal equations are solved. If
            ``True``, `op` is required to be self-adjoint and the system of
            equations is solved directly.
        """
        self.op = op
        self.x0 = x0
        self.niter = niter
        self.callback = callback
        self.op_is_symmetric = op_is_symmetric
        super().__init__(
            reco_space=self.op.domain, observation_space=self.op.range,
            **kwargs)

    def _reconstruct(self, observation, out):
        observation = self.observation_space.element(observation)
        out[:] = self.x0
        if self.op_is_symmetric:
            iterative.conjugate_gradient(self.op, out, observation,
                                         self.niter, self.callback)
        else:
            iterative.conjugate_gradient_normal(self.op, out, observation,
                                                self.niter, self.callback)
        return out


class GaussNewtonReconstructor(IterativeReconstructor):
    """Iterative reconstructor applying the Gauss-Newton method.

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
    callback : :class:`odl.solvers.util.callback.Callback` or `None`
        Object that is called in each iteration.
    """
    def __init__(self, op, x0, niter, zero_seq=None, callback=None, **kwargs):
        """
        Calls `odl.solvers.iterative.iterative.gauss_newton`.

        Parameters
        ----------
        op : `odl.operator.Operator`
            The forward operator of the inverse problem.
            The call ``op.derivative(x).adjoint`` must be valid.
        x0 : ``op.domain`` element
            Initial value.
        niter : int
            Maximum number of iterations.
        zero_seq : iterable, optional
            Zero sequence used for regularization.
        callback : :class:`odl.solvers.util.callback.Callback`, optional
            Object that is called in each iteration.
        """
        self.op = op
        self.x0 = x0
        self.niter = niter
        self.zero_seq = zero_seq
        self.callback = callback
        super().__init__(
            reco_space=self.op.domain, observation_space=self.op.range,
            **kwargs)

    def _reconstruct(self, observation, out):
        observation = self.observation_space.element(observation)
        out[:] = self.x0
        kwargs = {'callback': self.callback}
        if self.zero_seq is not None:
            kwargs['zero_seq'] = self.zero_seq
        iterative.gauss_newton(self.op, out, observation, self.niter,
                               **kwargs)
        return out


class KaczmarzReconstructor(IterativeReconstructor):
    """Iterative reconstructor applying Kaczmarz's method.

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
    callback : :class:`odl.solvers.util.callback.Callback` or `None`
        Object that is called in each iteration.
    callback_loop : {'inner', 'outer'}
        Whether the `callback` should be called in the inner or outer loop.
    """
    def __init__(self, ops, x0, niter, omega=1, random=False, projection=None,
                 callback=None, callback_loop='outer', **kwargs):
        """
        Calls `odl.solvers.iterative.iterative.kaczmarz`.

        Parameters
        ----------
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
        callback : :class:`odl.solvers.util.callback.Callback`, optional
            Object that is called in each iteration.
        callback_loop : {'inner', 'outer'}
            Whether the `callback` should be called in the inner or outer loop.
        """
        self.ops = ops
        self.x0 = x0
        self.niter = niter
        self.omega = omega
        self.projection = projection
        self.random = random
        self.callback = callback
        self.callback_loop = callback_loop
        super().__init__(
            reco_space=self.ops[0].domain,
            observation_space=ProductSpace(*(op.range for op in self.ops)),
            **kwargs)

    def _reconstruct(self, observation, out):
        observation = self.observation_space.element(observation)
        out[:] = self.x0
        iterative.kaczmarz(self.ops, out, observation, self.niter,
                           self.omega, self.projection, self.random,
                           self.callback, self.callback_loop)
        return out


class LandweberReconstructor(IterativeReconstructor):
    """Iterative reconstructor applying Landweber's method.

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
    callback : :class:`odl.solvers.util.callback.Callback` or `None`
        Object that is called in each iteration.
    """
    def __init__(self, op, x0, niter, omega=None, projection=None,
                 callback=None, **kwargs):
        """
        Calls `odl.solvers.iterative.iterative.landweber`.

        Parameters
        ----------
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
        callback : :class:`odl.solvers.util.callback.Callback`, optional
            Object that is called in each iteration.
        """
        self.op = op
        self.x0 = x0
        self.niter = niter
        self.omega = omega
        self.projection = projection
        self.callback = callback
        super().__init__(
            reco_space=self.op.domain, observation_space=self.op.range,
            **kwargs)

    def _reconstruct(self, observation, out):
        observation = self.observation_space.element(observation)
        out[:] = self.x0
        iterative.landweber(self.op, out, observation, self.niter,
                            self.omega, self.projection, self.callback)
        return out


class MLEMReconstructor(IterativeReconstructor):
    """Iterative reconstructor applying Maximum Likelihood Expectation
    Maximization.

    If multiple operators are given, Ordered Subsets MLEM is applied.

    Attributes
    ----------
    op : `odl.operator.Operator` or sequence of `odl.operator.Operator`
        The forward operator(s) of the inverse problem.
    x0 : ``op.domain`` element
        Initial value.
    niter : int
        Number of iterations.
    noise : {'poisson'} or `None`
        Noise model determining the variant of MLEM.
    callback : :class:`odl.solvers.util.callback.Callback` or `None`
        Object that is called in each iteration.
    sensitivities : float or ``op.domain`` `element-like`
        Usable with ``noise='poisson'``. The algorithm contains an ``A^T 1``
        term, if this parameter is given, it is replaced by it.
        Default: ``op[i].adjoint(op[i].range.one())``
    """
    def __init__(self, op, x0, niter, noise='poisson', callback=None,
                 sensitivities=None, **kwargs):
        """
        Calls `odl.solvers.iterative.statistical.osmlem`.

        Parameters
        ----------
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
        callback : :class:`odl.solvers.util.callback.Callback`, optional
            Object that is called in each iteration.
        sensitivities : float or ``op.domain`` `element-like`, optional
            Usable with ``noise='poisson'``. The algorithm contains an
            ``A^T 1`` term, if this parameter is given, it is replaced by it.
            Default: ``op[i].adjoint(op[i].range.one())``
        """
        self.os_mode = not isinstance(op, Operator)
        self.op = op if self.os_mode else [op]
        self.x0 = x0
        self.niter = niter
        self.noise = noise
        self.callback = callback
        self.sensitivities = sensitivities
        observation_space = (ProductSpace(*(op.range for op in self.op)) if
                             self.os_mode else self.op[0].range)
        super().__init__(
            reco_space=self.op[0].domain, observation_space=observation_space,
            **kwargs)

    def _reconstruct(self, observation, out):
        out[:] = self.x0
        observation = self.observation_space.element(observation)
        if not self.os_mode:
            observation = [observation]
        statistical.osmlem(self.op, out, observation, self.niter,
                           noise=self.noise, callback=self.callback,
                           sensitivities=self.sensitivities)
        return out
