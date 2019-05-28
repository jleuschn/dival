# -*- coding: utf-8 -*-
"""Provides wrappers for reconstruction methods of odl."""
import numpy as np
from sklearn.linear_model import Ridge
from hyperopt import hp
from dival.util.odl_utility import uniform_discr_element
from dival import LearnedReconstructor


class LinRegReconstructor(LearnedReconstructor):
    HYPER_PARAMS = {
        'l2_regularization':
            {'default': 0.,
             'range': [0., float('inf')],
             'retrain': True,
             'method': 'hyperopt',
             'hyperopt_options': {
                 'space': hp.loguniform('l2_regularization', 0., np.log(1e9))}}
    }

    """Reconstructor learning and applying linear regression.

    Assumes the inverse operator is linear, i.e. ``x = A_inv * y``.
    Learns the entries of ``A_inv`` by l2-regularized linear regression:
    ``A_inv = 1/2N * sum_i ||x_i - A_inv * y_i||^2 + alpha/2 * ||A_inv||_F^2``,
    where (y_i, x_i) with i=1,...,N are pairs of observations and the
    corresponding ground truth.

    Attributes
    ----------
    weights : `np.ndarray`
        The weight matrix.
    """
    def __init__(self, hyper_params=None, **kwargs):
        """Construct a LinReg reconstructor.
        """
        super().__init__(hyper_params=hyper_params, **kwargs)
        self.weights = None

    def reconstruct(self, observation_data):
        reconstruction = np.dot(self.weights, observation_data)
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
