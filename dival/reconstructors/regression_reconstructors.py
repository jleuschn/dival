# -*- coding: utf-8 -*-
"""Provides reconstructors performing regression."""
import os
import json
import numpy as np
from sklearn.linear_model import Ridge
from dival.util.odl_utility import uniform_discr_element
from dival import LearnedReconstructor


class LinRegReconstructor(LearnedReconstructor):
    HYPER_PARAMS = {
        'l2_regularization':
            {'default': 0.,
             'range': [0., np.inf],
             'retrain': True}
    }

    """Reconstructor learning and applying linear regression.

    Assumes the inverse operator is linear, i.e. ``x = A_inv * y``.
    Learns the entries of ``A_inv`` by l2-regularized linear regression:
    ``A_inv = 1/2N * sum_i ||x_i - A_inv * y_i||^2 + alpha/2 * ||A_inv||_F^2``,
    where (y_i, x_i) with i=1,...,N are pairs of observations and the
    corresponding ground truth.

    Attributes
    ----------
    weights : :class:`np.ndarray`
        The weight matrix.
    """
    def __init__(self, hyper_params=None, **kwargs):
        """
        Parameters
        ----------
        hyper_params : dict, optional
            A dict with no items or an item ``'l2_regularization': float``.
            Cf. :meth:`Reconstructor.init`.
        """
        super().__init__(hyper_params=hyper_params, **kwargs)
        self.weights = None

    def _reconstruct(self, observation):
        reconstruction = np.dot(self.weights, observation)
        return uniform_discr_element(reconstruction, self.reco_space)

    def train(self, dataset):
        observation_shape, reco_shape = dataset.get_shape()
        if (self.observation_space is not None
                and self.observation_space.shape != observation_shape):
            raise ValueError('Observation shape of dataset not matching '
                             '`self.observation_space.shape`')
        if (self.observation_space is not None
                and self.observation_space.shape != observation_shape):
            raise ValueError('Observation shape of dataset not matching '
                             '`self.observation_space.shape`')
        ridge = Ridge(self.hyper_params['l2_regularization'])
        train_len = dataset.get_train_len()
        n_features = dataset.shape[0][0]
        n_targets = dataset.shape[1][0]
        x = np.empty((train_len, n_features))
        y = np.empty((train_len, n_targets))
        for i, (x_i, y_i) in enumerate(dataset.get_train_generator()):
            x[i] = x_i
            y[i] = y_i
        ridge.fit(x, y)
        self.weights = ridge.coef_

    def save_params(self, path):
        """
        Save :attr:`weights` and :attr:`hyper_params` to files.

        Parameters
        ----------
        path : str
            Folder.
        """
        if not os.path.isdir(path):
            os.mkdir(path)
        np.save(os.path.join(path, 'weights.npy'), self.weights)
        with open(os.path.join(path, 'hyper_params.json'), 'w') as file:
            json.dump(self.hyper_params, file, indent=True)

    def load_params(self, path):
        """
        Load :attr:`weights` and :attr:`hyper_params` from files.

        Parameters
        ----------
        path : str
            Folder.
        """
        self.weights = np.load(os.path.join(path, 'weights.npy'))
        with open(os.path.join(path, 'hyper_params.json'), 'r') as file:
            self.hyper_params.update(json.load(file))
