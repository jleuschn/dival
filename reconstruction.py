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


class OperatorReconstructor(Reconstructor):
    """Reconstructor defined by an ODL operator calculating the reconstruction.
    """
    def __init__(self, op):
        """Construct a reconstructor by specifying an ODL operator.

        Parameters
        ----------
        op : `odl.operator.Operator`
            ODL operator that gets called in `reconstruct`.
        """
        self.op = op

    def reconstruct(self, observation_data):
        return self.op(observation_data)


class CGReconstructor(Reconstructor):
    """Reconstructor applying the conjugate gradient method for self-adjoint
    operators.

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
