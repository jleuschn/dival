# -*- coding: utf-8 -*-
"""Provides standard datasets for benchmarking.
"""
import odl
from odl.operator.default_ops import ScalingOperator
from odl.operator.operator import OperatorComp
from dival.util.odl_utility import ExpOperator
from dival.datasets.ellipses.ellipses_dataset import EllipsesDataset
from dival.datasets.lidc_idri_dival.lidc_idri_dival_dataset import (
    LIDCIDRIDivalDataset)


def get_standard_dataset(name):
    """
    Return a standard dataset by name.

    The datasets are:

        * ``'ellipses'``
            A typical synthetical CT dataset with ellipse phantoms.
            `EllipsesDataset` is used as ground truth dataset, a ray
            transform with parallel beam geometry using 30 angles is applied,
            and white gaussian noise with a standard deviation of 5% (i.e.
            ``0.05 * mean(abs(observation))``) is added.
            The ray transform is stored in the attribute `ray_trafo` of the
            dataset.
        * ``'lidc_idri_dival'``
            A dataset based on real CT reconstructions from the LIDC-IDRI
            dataset.
            `LIDCIDRIDivalDataset` is used as ground truth dataset and a ray
            transform with parallel beam geometry using 30 angles is applied,
            followed by ``exp(-mu * observation)`` due to Beer-Lamberts law,
            where ``mu = 0.02 cm^2/g`` (mass attenuation of water).
            The noise is chosen in the way presented in [1]_ for the human
            dataset: poisson noise corresponding to 10^4 incident photons per
            pixel before attenuation is used.
            The ray transform is stored in the attribute `ray_trafo` of the
            dataset. The deterministic pre- and postprocessing operators
            applied before and after the ray transform are stored in the
            attributes `preprocessing_op` and `postprocessing_op` of the
            dataset, which also implement the respective inverses.

    References
    ----------
    .. [1] Adler, J., & Öktem, O. (2018). Learned Primal-Dual Reconstruction.
        IEEE Transactions on Medical Imaging, 37, 1322-1332.

    Parameters
    ----------
    name : str
        Name of the dataset.

    Returns
    -------
    dataset : `ObservationGroundTruthDataset`
        The dataset.
    """
    if name == 'ellipses':
        ellipses_dataset = EllipsesDataset()
        space = ellipses_dataset.space

        geometry = odl.tomo.parallel_beam_geometry(space, num_angles=30)
        ray_trafo = odl.tomo.RayTransform(space, geometry)

        # Ensure operator has fixed operator norm for scale invariance
        opnorm = odl.power_method_opnorm(ray_trafo)
        operator = (1 / opnorm) * ray_trafo

        dataset = ellipses_dataset.create_pair_dataset(
            forward_op=operator, noise_type='white',
            noise_kwargs={'relative_stddev': True,
                          'stddev': 0.05},
            noise_seed=1)

        dataset.ray_trafo = ray_trafo

        return dataset
    elif name == 'lidc_idri_dival':
        lidc_idri_dival_dataset = LIDCIDRIDivalDataset(normalize=False)
        space = lidc_idri_dival_dataset.space

        geometry = odl.tomo.parallel_beam_geometry(space, num_angles=30)
        ray_trafo = odl.tomo.RayTransform(space, geometry)

        # Ensure operator has fixed operator norm for scale invariance
        opnorm = odl.power_method_opnorm(ray_trafo)
#        operator = (1 / opnorm) * ray_trafo

#        noise model taken from Adler, J. and Öktem, O., 2018, Learned
#        Primal-dual Reconstruction, human dataset
        mu_water = 0.02
        photons_per_pixel = 10000
        factor = 1000.
        preprocessing_op = ScalingOperator(space, 1/factor)
        postprocessing_op = (ExpOperator(ray_trafo.range, ray_trafo.range) *
                             (-mu_water))
        operator = OperatorComp(ray_trafo, preprocessing_op)
        operator = OperatorComp(postprocessing_op, operator)

        dataset = lidc_idri_dival_dataset.create_pair_dataset(
            forward_op=operator, noise_type='poisson',
            noise_kwargs={'scaling_factor': photons_per_pixel},
            noise_seed=1)

        dataset.forward_op = operator
        dataset.ray_trafo = ray_trafo
        dataset.preprocessing_op = preprocessing_op
        dataset.postprocessing_op = postprocessing_op

        return dataset
    else:
        raise ValueError("unknown dataset '{}'".format(name))
