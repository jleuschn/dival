# -*- coding: utf-8 -*-
class DataPairs:
    """
    Bundles :attr:`observations` with :attr:`ground_truth`.

    Attributes
    ----------
    observations : list of observation space elements
        The observations, possibly distorted or low-dimensional.
    ground_truth: list of reconstruction space elements or `None`
        The ground truth data (may be replaced with good quality references).
        If not known, it may be omitted (`None`).
    """
    def __init__(self, observations, ground_truth=None, name='',
                 description=''):
        self.observations = observations
        if not isinstance(self.observations, list):
            self.observations = [self.observations]
        self.ground_truth = ground_truth
        if (self.ground_truth is not None and
                not isinstance(self.ground_truth, list)):
            self.ground_truth = [self.ground_truth]
        self.name = name
        self.description = description

    def __repr__(self):
        return ("DataPairs(observations=\n{observations}, "
                "ground_truth=\n{ground_truth}, name='{name}', "
                "description='{description}')".format(
                    observations=self.observations,
                    ground_truth=self.ground_truth,
                    name=self.name,
                    description=self.description))

    def __str__(self):
        return ("DataPairs '{name}'".format(name=self.name) if self.name
                else self.__repr__())

    def __len__(self):
        return len(self.observations)
