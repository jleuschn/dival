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
