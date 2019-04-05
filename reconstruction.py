# -*- coding: utf-8 -*-
"""Provides the abstract reconstructor base class."""
from abc import ABC, abstractmethod
from odl.solvers.iterative import iterative


class Reconstructor(ABC):
    """Abstract reconstructor base class.

    Subclasses must implement the abstract method `reconstruct`.
    """
    @abstractmethod
    def reconstruct(self, observation_data):
        """Reconstruct input data from observation data.

        This method must be implemented by subclasses.

        Parameters
        ----------
        observation_data : "observation space" element

        Returns
        -------
        "input space" element
            The reconstruction.
        """


class FunctionReconstructor(Reconstructor):
    """Reconstructor defined by a callable calculating the reconstruction."""
    def __init__(self, function, *args, **kwargs):
        """Construct a reconstructor by specifying a callable.

        Parameters
        ----------
        function : callable
            Callable that is used in `reconstruct`.
        args : list
            arguments to be passed to `function`.
        kwargs : dict
            keyword arguments to be passed to `function`.
        """
        self.function = function
        self.args = args
        self.kwargs = kwargs

    def reconstruct(self, observation_data):
        return self.function(observation_data, *self.args, **self.kwargs)


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
    callback : callable
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
        callback : callable, optional
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

    def reconstruct(self, observation_data):
        x = self.x0.copy()
        if self.op_is_symmetric:
            iterative.conjugate_gradient(self.op, x, observation_data,
                                         self.niter, self.callback)
        else:
            iterative.conjugate_gradient_normal(self.op, x, observation_data,
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
    callback : callable
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
        callback : callable, optional
            Object that is called in each iteration.
        """
        self.op = op
        self.x0 = x0
        self.niter = niter
        self.zero_seq = zero_seq
        self.callback = callback

    def reconstruct(self, observation_data):
        x = self.x0.copy()
        kwargs = {'callback': self.callback}
        if self.zero_seq is not None:
            kwargs['zero_seq'] = self.zero_seq
        iterative.gauss_newton(self.op, x, observation_data, self.niter,
                               **kwargs)
        return x


class KaczmarzReconstructor(Reconstructor):
    """Reconstructor applying Kaczmarz's method.

    Attributes
    ----------
    op : sequence of `odl.Operator`
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
    callback : callable
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
            Maximum number of iterations.
        omega : positive float or sequence of positive floats, optional
            Relaxation parameter.
            If a single float is given it is used for all operators.
        random : bool, optional
            Whether the order of the operators is randomized in each iteration.
        projection : callable, optional
            Callable that can be used to modify the iterates in each iteration.
        callback : callable, optional
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

    def reconstruct(self, observation_data):
        x = self.x0.copy()
        iterative.kaczmarz(self.ops, x, observation_data, self.niter,
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
    callback : callable
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
            Maximum number of iterations.
        omega : positive float, optional
            Relaxation parameter.
        projection : callable, optional
            Callable that can be used to modify the iterates in each iteration.
            One argument is passed and expected be modified in-place.
        callback : callable, optional
            Object that is called in each iteration.
        """
        self.op = op
        self.x0 = x0
        self.niter = niter
        self.omega = omega
        self.projection = projection
        self.callback = callback

    def reconstruct(self, observation_data):
        x = self.x0.copy()
        iterative.landweber(self.op, x, observation_data, self.niter,
                            self.omega, self.projection, self.callback)
        return x
