# -*- coding: utf-8 -*-
"""Provides the abstract reconstructor base class."""
from abc import ABC, abstractmethod
from odl.operator import Operator

class Reconstructor(ABC):
    """Abstract reconstructor base class.

    Subclasses must implement the abstract method `reconstruct`.
    """
    @abstractmethod
    def reconstruct(self, observation_data):
        """Reconstruct input data from observation data.

        Returns
        -------
        `odl.discr.discretization.DiscretizedSpaceElement`
            The reconstruction.
        """


class OperatorReconstructor(Reconstructor):
    def __init__(self, op):
        self.op = op

    def reconstruct(self, observation_data):
        return self.op(observation_data)


def reconstructor_from_op(op):
    return OperatorReconstructor(op)
