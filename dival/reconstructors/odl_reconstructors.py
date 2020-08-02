# -*- coding: utf-8 -*-
"""Provides wrappers for reconstruction methods of odl."""
from odl import power_method_opnorm, ScalingOperator
from odl.tomo import fbp_op
from odl.discr.diff_ops import Gradient
from odl.operator.pspace_ops import BroadcastOperator
from odl.operator import Operator
from odl.space.pspace import ProductSpace
from odl.solvers import L1Norm, L2NormSquared, ZeroFunctional, SeparableSum,\
    forward_backward_pd, GroupL1Norm, MoreauEnvelope
from odl.solvers.smooth import newton
from odl.solvers.iterative import iterative, statistical
from odl.solvers.iterative.iterative import exp_zero_seq
from odl.solvers.nonsmooth import proximal_gradient_solvers,\
    primal_dual_hybrid_gradient, douglas_rachford, admm
from odl.solvers.functional.functional import Functional
from dival.reconstructors import Reconstructor, IterativeReconstructor


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
        if out in self.reco_space:
            self.fbp_op(observation, out=out)
        else:  # out is e.g. numpy array, cannot be passed to fbp_op
            out[:] = self.fbp_op(observation)
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
        self.op_is_symmetric = op_is_symmetric
        super().__init__(
            reco_space=self.op.domain, observation_space=self.op.range,
            callback=callback, **kwargs)

    def _reconstruct(self, observation, out):
        observation = self.observation_space.element(observation)
        out_ = out
        if out not in self.reco_space:
            out_ = self.reco_space.zero()
        out_[:] = self.x0
        if self.op_is_symmetric:
            iterative.conjugate_gradient(self.op, out_, observation,
                                         self.niter, self.callback)
        else:
            iterative.conjugate_gradient_normal(self.op, out_, observation,
                                                self.niter, self.callback)
        if out not in self.reco_space:
            out[:] = out_
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
    zero_seq_gen : generator
        Zero sequence generator used for regularization.
    callback : :class:`odl.solvers.util.callback.Callback` or `None`
        Object that is called in each iteration.
    """
    def __init__(self, op, x0, niter, zero_seq_gen=None, callback=None,
                 **kwargs):
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
        zero_seq_gen : generator, optional
            Zero sequence generator used for regularization.
            Default: generator yielding 2^(-i).
        callback : :class:`odl.solvers.util.callback.Callback`, optional
            Object that is called in each iteration.
        """
        self.op = op
        self.x0 = x0
        self.niter = niter
        self.zero_seq_gen = zero_seq_gen
        super().__init__(
            reco_space=self.op.domain, observation_space=self.op.range,
            callback=callback, **kwargs)

    def _reconstruct(self, observation, out):
        observation = self.observation_space.element(observation)
        out_ = out
        if out not in self.reco_space:
            out_ = self.reco_space.zero()
        out_[:] = self.x0
        zero_seq = (self.zero_seq_gen() if self.zero_seq_gen is not None else
                    exp_zero_seq(2.0))
        iterative.gauss_newton(self.op, out_, observation, self.niter,
                               callback=self.callback, zero_seq=zero_seq)
        if out not in self.reco_space:
            out[:] = out_
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
        self.callback_loop = callback_loop
        super().__init__(
            reco_space=self.ops[0].domain,
            observation_space=ProductSpace(*(op.range for op in self.ops)),
            callback=callback, **kwargs)

    def _reconstruct(self, observation, out):
        observation = self.observation_space.element(observation)
        out_ = out
        if out not in self.reco_space:
            out_ = self.reco_space.zero()
        out_[:] = self.x0
        iterative.kaczmarz(self.ops, out_, observation, self.niter,
                           self.omega, self.projection, self.random,
                           self.callback, self.callback_loop)
        if out not in self.reco_space:
            out[:] = out_
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
        super().__init__(
            reco_space=self.op.domain, observation_space=self.op.range,
            callback=callback, **kwargs)

    def _reconstruct(self, observation, out):
        observation = self.observation_space.element(observation)
        out_ = out
        if out not in self.reco_space:
            out_ = self.reco_space.zero()
        out_[:] = self.x0
        iterative.landweber(self.op, out_, observation, self.niter,
                            self.omega, self.projection, self.callback)
        if out not in self.reco_space:
            out[:] = out_
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
        self.sensitivities = sensitivities
        observation_space = (ProductSpace(*(op.range for op in self.op)) if
                             self.os_mode else self.op[0].range)
        super().__init__(
            reco_space=self.op[0].domain, observation_space=observation_space,
            callback=callback, **kwargs)

    def _reconstruct(self, observation, out):
        out_ = out
        if out not in self.reco_space:
            out_ = self.reco_space.zero()
        out_[:] = self.x0
        observation = self.observation_space.element(observation)
        if not self.os_mode:
            observation = [observation]
        statistical.osmlem(self.op, out_, observation, self.niter,
                           noise=self.noise, callback=self.callback,
                           sensitivities=self.sensitivities)
        if out not in self.reco_space:
            out[:] = out_
        return out


class ISTAReconstructor(IterativeReconstructor):
    """Iterative reconstructor applying proximal gradient
    algorithm for convex optimization, also known as
    Iterative Soft-Thresholding Algorithm (ISTA).

    Attributes
    ----------
    op : `odl.operator.Operator`
        The forward operator of the inverse problem.
    x0 : ``op.domain`` element
        Initial value.
    niter : int
        Number of iterations.
    gamma : positive float
        Step size parameter.
    reg : `odl.solvers.functional.Functional`
        The regularization operator of the inverse problem. Needs to have
        ``f.proximal``.
    accelerated : boolean
        Indicates which algorithm to use. If `False`, then the "Iterative
        Soft-Thresholding Algorithm" (ISTA) is used. If `True`, then the
        accelerated version FISTA is used.
    lam : float or callable
        Overrelaxation step size (default ``1.0``).
        If callable, it should take an index (starting at zero) and return
        the corresponding step size.
    callback : :class:`odl.solvers.util.callback.Callback` or `None`
        Object that is called in each iteration.
    """

    def __init__(self, op, x0, niter, gamma=0.001, reg=L1Norm,
                 accelerated=True,  lam=1., callback=None, **kwargs):
        """
        Calls
        `odl.solvers.nonsmooth.proximal_gradient_solvers.proximal_gradient` or
        `odl.solvers.nonsmooth.proximal_gradient_solvers\
        .accelerated_proximal_gradient`.

        Parameters
        ----------
        op : `odl.operator.Operator`
            The forward operator of the inverse problem.
        x0 : ``op.domain`` element
            Initial value.
        niter : int
            Number of iterations.
        gamma : positive float, optional
            Step size parameter (default ``0.001``).
        reg : [type of ] `odl.solvers.functional.Functional`, optional
            The regularization functional of the inverse problem. Needs to have
            ``f.proximal``. If a type is passed instead, the functional is
            constructed by calling ``reg(op.domain)``.
            Default: :class:`odl.solvers.L1Norm`.
        accelerated : boolean, optional
            Indicates which algorithm to use. If `False`, then the "Iterative
            Soft-Thresholding Algorithm" (ISTA) is used. If `True`, then the
            accelerated version FISTA is used. Default: `True`.
        lam : float or callable, optional
            Overrelaxation step size (default ``1.0``).
            If callable, it should take an index (starting at zero) and return
            the corresponding step size.
        callback : :class:`odl.solvers.util.callback.Callback`, optional
            Object that is called in each iteration.
        """
        self.op = op
        self.x0 = x0
        self.niter = niter
        self.gamma = gamma
        if isinstance(reg, Functional):
            self.reg = reg
        elif issubclass(reg, Functional):
            self.reg = reg(self.op.domain)
        else:
            raise ValueError('`reg` must be an odl `Functional` object or an '
                             'odl `Functional` type')
        self.accelerated = accelerated
        self.lam = lam
        self.callback = callback
        super().__init__(
            reco_space=self.op.domain, observation_space=self.op.range,
            callback=callback, **kwargs)

    def _reconstruct(self, observation, out):
        # The proximal_gradient and accelerated_proximal_gradient methods
        # from ODL have as input the function, which needs to be minimized
        # (and not the forward operator itself). Therefore, we calculate the
        # discrepancy ||Ax-b||_2^2. Note that the terms `f` and `g` are
        # switched in odl compared to the usage in the FISTA paper
        # (https://epubs.siam.org/doi/abs/10.1137/080716542).
        observation = self.observation_space.element(observation)
        out_ = out
        if out not in self.reco_space:
            out_ = self.reco_space.zero()
        out_[:] = self.x0
        l2_norm_sq_trans = L2NormSquared(self.op.range).translated(observation)
        discrepancy = l2_norm_sq_trans * self.op
        if not self.accelerated:
            proximal_gradient_solvers.proximal_gradient(
                out_, self.reg, discrepancy, self.gamma, self.niter,
                callback=self.callback, lam=self.lam)
        else:
            proximal_gradient_solvers.accelerated_proximal_gradient(
                out_, self.reg, discrepancy, self.gamma, self.niter,
                callback=self.callback, lam=self.lam)
        if out not in self.reco_space:
            out[:] = out_
        return out


class PDHGReconstructor(IterativeReconstructor):
    """Primal-Dual Hybrid Gradient (PDHG) algorithm from the 2011 paper
    https://link.springer.com/article/10.1007/s10851-010-0251-1 with TV
    regularization.

    Attributes
    ----------
    op : `odl.operator.Operator`
        The forward operator of the inverse problem.
    x0 : ``op.domain`` element
        Initial value.
    niter : int
        Number of iterations.
    lam : positive float, optional
        TV-regularization rate (default ``0.01``).
    callback : :class:`odl.solvers.util.callback.Callback` or `None`
        Object that is called in each iteration.
    """

    def __init__(self, op, x0, niter, lam=0.01, callback=None, **kwargs):
        """
        Calls `odl.solvers.nonsmooth.primal_dual_hybrid_gradient.pdhg`.

        Parameters
        ----------
        op : `odl.operator.Operator`
            The forward operator of the inverse problem.
        x0 : ``op.domain`` element
            Initial value.
        niter : int
            Number of iterations.
        lam : positive float, optional
            TV-regularization rate (default ``0.01``).
        callback : :class:`odl.solvers.util.callback.Callback` or `None`
            Object that is called in each iteration.
        """
        self.op = op
        self.x0 = x0
        self.niter = niter
        self.lam = lam
        self.callback = callback
        super().__init__(
            reco_space=self.op.domain, observation_space=self.op.range,
            callback=callback, **kwargs)

    def _reconstruct(self, observation, out):
        observation = self.observation_space.element(observation)
        out_ = out
        if out not in self.reco_space:
            out_ = self.reco_space.zero()
        out_[:] = self.x0
        gradient = Gradient(self.op.domain)
        L = BroadcastOperator(self.op, gradient)
        f = ZeroFunctional(self.op.domain)
        l2_norm = L2NormSquared(self.op.range).translated(observation)
        l1_norm = self.lam * L1Norm(gradient.range)
        g = SeparableSum(l2_norm, l1_norm)
        tau, sigma = primal_dual_hybrid_gradient.pdhg_stepsize(L)
        primal_dual_hybrid_gradient.pdhg(out_, f, g, L, self.niter,
                                         tau, sigma, callback=self.callback)
        if out not in self.reco_space:
            out[:] = out_
        return out


class DouglasRachfordReconstructor(IterativeReconstructor):
    """Douglas-Rachford primal-dual splitting algorithm from the 2012 paper
    https://arxiv.org/abs/1212.0326 with TV regularization.

    Attributes
    ----------
    op : `odl.operator.Operator`
        The forward operator of the inverse problem.
    x0 : ``op.domain`` element
        Initial value.
    niter : int
        Number of iterations.
    lam : positive float, optional
        TV-regularization rate (default ``0.01``).
    callback : :class:`odl.solvers.util.callback.Callback` or `None`
        Object that is called in each iteration.
    """

    def __init__(self, op, x0, niter, lam=0.01, callback=None, **kwargs):
        """
        Calls `odl.solvers.nonsmooth.douglas_rachford.douglas_rachford_pd`.

        Parameters
        ----------
        op : `odl.operator.Operator`
            The forward operator of the inverse problem.
        x0 : ``op.domain`` element
            Initial value.
        niter : int
            Number of iterations.
        lam : positive float, optional
            TV-regularization rate (default ``0.01``).
        callback : :class:`odl.solvers.util.callback.Callback` or `None`
            Object that is called in each iteration.
        """
        self.op = op
        self.x0 = x0
        self.niter = niter
        self.lam = lam
        self.callback = callback
        super().__init__(
            reco_space=self.op.domain, observation_space=self.op.range,
            callback=callback, **kwargs)

    def _reconstruct(self, observation, out):
        observation = self.observation_space.element(observation)
        out_ = out
        if out not in self.reco_space:
            out_ = self.reco_space.zero()
        out_[:] = self.x0
        gradient = Gradient(self.op.domain)
        L = BroadcastOperator(self.op, gradient)
        f = ZeroFunctional(self.op.domain)
        l2_norm = L2NormSquared(self.op.range).translated(observation)
        l1_norm = self.lam * L1Norm(gradient.range)
        g = [l2_norm, l1_norm]
        tau, sigma = douglas_rachford.douglas_rachford_pd_stepsize(L)
        douglas_rachford.douglas_rachford_pd(out_, f, g, L, self.niter, tau,
                                             sigma, callback=self.callback)
        if out not in self.reco_space:
            out[:] = out_
        return out


class ForwardBackwardReconstructor(IterativeReconstructor):
    """ The forward-backward primal-dual splitting algorithm.

    Attributes
    ----------
    op : `odl.operator.Operator`
        The forward operator of the inverse problem.
    x0 : ``op.domain`` element
        Initial value.
    niter : int
        Number of iterations.
    lam : positive float, optional
        TV-regularization rate (default ``0.01``).
    tau : positive float, optional
        Step-size like parameter for ``f`` (default is ``0.01``).
    callback : :class:`odl.solvers.util.callback.Callback` or `None`
        Object that is called in each iteration.
    """

    def __init__(self, op, x0, niter, lam=0.01, tau=0.01, callback=None,
                 **kwargs):
        """
        Calls `odl.solvers.forward_backward_pd`.

        Parameters
        ----------
        op : `odl.operator.Operator`
            The forward operator of the inverse problem.
        x0 : ``op.domain`` element
            Initial value.
        niter : int
            Number of iterations.
        lam : positive float, optional
            TV-regularization rate (default ``0.01``).
        tau : positive float, optional
            Step-size like parameter for ``f`` (default is ``0.01``).
        callback : :class:`odl.solvers.util.callback.Callback` or `None`
            Object that is called in each iteration.
        """
        self.op = op
        self.x0 = x0
        self.niter = niter
        self.lam = lam
        self.tau = tau
        self.callback = callback
        super().__init__(
            reco_space=self.op.domain, observation_space=self.op.range,
            callback=callback, **kwargs)

    def _reconstruct(self, observation, out):
        observation = self.observation_space.element(observation)
        out_ = out
        if out not in self.reco_space:
            out_ = self.reco_space.zero()
        out_[:] = self.x0
        gradient = Gradient(self.op.domain)
        L = [self.op, gradient]
        f = ZeroFunctional(self.op.domain)
        l2_norm = 0.5 * L2NormSquared(self.op.range).translated(observation)
        l12_norm = self.lam * GroupL1Norm(gradient.range)
        g = [l2_norm, l12_norm]
        op_norm = power_method_opnorm(self.op, maxiter=20)
        gradient_norm = power_method_opnorm(gradient, maxiter=20)
        sigma_ray_trafo = 45.0 / op_norm ** 2
        sigma_gradient = 45.0 / gradient_norm ** 2
        sigma = [sigma_ray_trafo, sigma_gradient]
        h = ZeroFunctional(self.op.domain)
        forward_backward_pd(out_, f, g, L, h, self.tau, sigma,
                            self.niter, callback=self.callback)
        if out not in self.reco_space:
            out[:] = out_
        return out


class ADMMReconstructor(IterativeReconstructor):
    """ Generic linearized ADMM method for convex problems. ADMM stands for
    'Alternating Direction Method of Multipliers'.

    Attributes
    ----------
    op : `odl.operator.Operator`
        The forward operator of the inverse problem.
    x0 : ``op.domain`` element
        Initial value.
    niter : int
        Number of iterations.
    lam : positive float, optional
        TV-regularization weight (default ``0.01``).
    tau : positive float, optional
        Step-size like parameter for ``f`` (default is ``0.01``).
    callback : :class:`odl.solvers.util.callback.Callback` or `None`
        Object that is called in each iteration.
    """

    def __init__(self, op, x0, niter, lam=0.01, tau=0.01, callback=None,
                 **kwargs):
        """
        Calls `odl.solvers.nonsmooth.admm.admm_linearized`.

        Parameters
        ----------
        op : `odl.operator.Operator`
            The forward operator of the inverse problem.
        x0 : ``op.domain`` element
            Initial value.
        niter : int
            Number of iterations.
        lam : positive float, optional
            TV-regularization weight (default ``0.01``).
        tau : positive float, optional
            Step-size like parameter for ``f`` (default is ``0.01``).
        callback : :class:`odl.solvers.util.callback.Callback` or `None`
            Object that is called in each iteration.
        """
        self.op = op
        self.x0 = x0
        self.niter = niter
        self.lam = lam
        self.tau = tau
        self.callback = callback
        super().__init__(
            reco_space=self.op.domain, observation_space=self.op.range,
            callback=callback, **kwargs)

    def _reconstruct(self, observation, out):
        observation = self.observation_space.element(observation)
        out_ = out
        if out not in self.reco_space:
            out_ = self.reco_space.zero()
        out_[:] = self.x0
        gradient = Gradient(self.op.domain)
        L = BroadcastOperator(self.op, gradient)
        f = ZeroFunctional(self.op.domain)
        l2_norm = L2NormSquared(self.op.range).translated(observation)
        l1_norm = self.lam * L1Norm(gradient.range)
        g = SeparableSum(l2_norm, l1_norm)
        op_norm = 1.1 * power_method_opnorm(L, maxiter=20)
        sigma = self.tau * op_norm ** 2
        admm.admm_linearized(out_, f, g, L, self.tau, sigma,
                             self.niter, callback=self.callback)
        if out not in self.reco_space:
            out[:] = out_
        return out


class BFGSReconstructor(IterativeReconstructor):
    """ Quasi-Newton BFGS method to minimize a differentiable function. The
    TV regularization term is smoothed using the Moreau envelope.

    Attributes
    ----------
    op : `odl.operator.Operator`
        The forward operator of the inverse problem.
    x0 : ``op.domain`` element
        Initial value.
    niter : int
        Number of iterations.
    lam : positive float, optional
        TV-regularization weight (default ``0.01``).
    callback : :class:`odl.solvers.util.callback.Callback` or `None`
        Object that is called in each iteration.
    """

    def __init__(self, op, x0, niter, lam=0.01, callback=None, **kwargs):
        """
        Calls `odl.solvers.smooth.newton.bfgs_method`.

        Parameters
        ----------
        op : `odl.operator.Operator`
            The forward operator of the inverse problem.
        x0 : ``op.domain`` element
            Initial value.
        niter : int
            Number of iterations.
        lam : positive float, optional
            TV-regularization weight (default ``0.01``).
        callback : :class:`odl.solvers.util.callback.Callback` or `None`
            Object that is called in each iteration.
        """
        self.op = op
        self.x0 = x0
        self.niter = niter
        self.lam = lam
        self.callback = callback
        super().__init__(
            reco_space=self.op.domain, observation_space=self.op.range,
            callback=callback, **kwargs)

    def _reconstruct(self, observation, out):
        observation = self.observation_space.element(observation)
        out_ = out
        if out not in self.reco_space:
            out_ = self.reco_space.zero()
        out_[:] = self.x0
        l2_norm = L2NormSquared(self.op.range)
        discrepancy = l2_norm * (self.op - observation)
        gradient = Gradient(self.op.domain)
        l1_norm = GroupL1Norm(gradient.range)
        smoothed_l1 = MoreauEnvelope(l1_norm, sigma=0.03)
        regularizer = smoothed_l1 * gradient
        f = discrepancy + self.lam * regularizer
        opnorm = power_method_opnorm(self.op)
        hessinv_estimate = ScalingOperator(self.op.domain, 1 / opnorm ** 2)
        newton.bfgs_method(f, out_, maxiter=self.niter,
                           hessinv_estimate=hessinv_estimate,
                           callback=self.callback)
        if out not in self.reco_space:
            out[:] = out_
        return out
