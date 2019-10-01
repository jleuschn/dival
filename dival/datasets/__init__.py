# -*- coding: utf-8 -*-
"""Implements datasets for training and evaluating learned reconstructors.

.. autosummary::
    get_standard_dataset
    Dataset
    GroundTruthDataset
    ObservationGroundTruthPairDataset
    EllipsesDataset
    LoDoPaBDataset

The function :func:`.get_standard_dataset` returns fixed "standard" datasets
with pairs of observation and ground truth samples.
Currently the standard datasets are ``'ellipses'`` and ``'lodopab'``.

The class :class:`.ObservationGroundTruthPairDataset` can be used, either
directly or via :meth:`.GroundTruthDataset.create_pair_dataset`, to create a
custom dataset of pairs given a ground truth dataset and a forward operator.
For example:

    * define a :class:`.GroundTruthDataset` object (e.g. \
      :class:`.EllipsesDataset`)
    * define a forward operator
    * call :meth:`~.GroundTruthDataset.create_pair_dataset` of the dataset and
      pass the forward operator as well as some noise specification if wanted
"""
from warnings import warn

__all__ = ['get_standard_dataset', 'Dataset', 'GroundTruthDataset',
           'ObservationGroundTruthPairDataset', 'EllipsesDataset',
           'LoDoPaBDataset']

from .standard import get_standard_dataset
from .dataset import (Dataset, GroundTruthDataset,
                      ObservationGroundTruthPairDataset)
from .ellipses_dataset import EllipsesDataset
try:
    from .lodopab_dataset import LoDoPaBDataset
except FileNotFoundError as e:
    __all__.remove('LoDoPaBDataset')
    warn('could not import `LoDoPaBDataset` because of the following '
         'error:\n\n{}\n'.format(e))
