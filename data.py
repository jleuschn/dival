# -*- coding: utf-8 -*-
class TestData:
    """
    Bundles an `observation` with a `ground_truth`.

    Attributes
    ----------
    observation : observation space element
        The observation, possibly distorted or low-dimensional.
    ground_truth : reconstruction space element
        The ground truth. May be replaced with a good quality reference.
        Reconstructors will be evaluated by comparing their reconstructions
        with this value. May also be ``None`` if no evaluation based on
        ground truth shall be performed.
    """
    def __init__(self, observation, ground_truth=None,
                 name='', description=''):
        self.observation = observation
        self.ground_truth = ground_truth
        self.name = name
        self.description = description

    def __repr__(self):
        return ("TestData(observation=\n{observation}, "
                "ground_truth=\n{ground_truth}, name='{name}', "
                "description='{description}')".format(
                    observation=self.observation,
                    ground_truth=self.ground_truth,
                    name=self.name,
                    description=self.description))

    def __str__(self):
        return ("TestData('{name}')".format(name=self.name) if self.name
                else self.__repr__())
