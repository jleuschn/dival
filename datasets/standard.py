# -*- coding: utf-8 -*-
"""Provides standard datasets for benchmarking.
"""
import odl
from datasets import EllipsesDataset


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
    else:
        raise ValueError("unknown dataset '{}'".format(name))  # TODO
