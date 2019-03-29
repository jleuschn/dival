# -*- coding: utf-8 -*-
"""Provides the abstract reconstructor base class."""
from abc import ABC, abstractmethod


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
