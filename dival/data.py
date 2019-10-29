# -*- coding: utf-8 -*-
class DataPairs:
    """
    Bundles :attr:`observations` with :attr:`ground_truth`.
    Implements :meth:`__getitem__` and :meth:`__len__`.

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

    def __getitem__(self, idx):
        """Return data pair(s).

        Parameters
        ----------
        idx : index supported by list
            The index that is applied to :attr:`observations` and
            :attr:`ground_truth`.

        Returns
        -------
        pair : tuple of odl elements or tuple of lists of odl elements
            The pair of data.
            If `idx` is an integer, a tuple is returned.
            If `idx` selects multiple entries, a list of tuples is returned.
        """
        if isinstance(idx, int):
            return (self.observations[idx], self.ground_truth[idx])
        return list(zip(self.observations[idx], self.ground_truth[idx]))

    def __len__(self):
        """Return the length.
        """
        return len(self.observations)
