# -*- coding: utf-8 -*-
"""Implements datasets for training learned reconstructors.

The function `get_standard_dataset` returns fixed "standard" datasets with
pairs of ground truth and observation samples.
Currently the standard datasets are ``'ellipses'`` and ``'lidc_idri_dival'``.

The recommended way of defining custom datasets is as follows:

    * define a `GroundTruthDataset` object (e.g. of type `EllipsesDataset`,
      `LIDCIDRIDivalDataset` or a custom subclass)
    * define a forward operator
    * call the `create_pair_dataset` method of the dataset and pass the forward
      operator as well as some noise specification if wanted
"""
from .standard import get_standard_dataset
from .dataset import (Dataset, GroundTruthDataset,
                      ObservationGroundTruthPairDataset)
from .ellipses.ellipses_dataset import EllipsesDataset
from .lidc_idri_dival.lidc_idri_dival_dataset import LIDCIDRIDivalDataset

__all__ = ('get_standard_dataset', 'Dataset', 'GroundTruthDataset',
           'ObservationGroundTruthPairDataset', 'EllipsesDataset',
           'LIDCIDRIDivalDataset')
