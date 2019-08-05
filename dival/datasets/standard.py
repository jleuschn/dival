# -*- coding: utf-8 -*-
"""Provides standard datasets for benchmarking.
"""
from warnings import warn
import numpy as np
from skimage.transform import resize
import odl
from dival.datasets.ellipses.ellipses_dataset import EllipsesDataset
from dival.util.constants import MU_MAX
try:
    from dival.datasets.lidc_idri_dival.lidc_idri_dival_dataset import (
        LIDCIDRIDivalDataset)
except FileNotFoundError as e:
    warn('could not import LIDCIDRIDivalDataset because of the following '
         'error:\n\n{}\n'.format(e))


def get_standard_dataset(name):
    """
    Return a standard dataset by name.

    The datasets are:

        * ``'ellipses'``
            A typical synthetical CT dataset with ellipse phantoms.

            `EllipsesDataset` is used as ground truth dataset, a ray
            transform with parallel beam geometry using 30 angles is applied,
            and white gaussian noise with a standard deviation of 5% (i.e.
            ``0.05 * mean(abs(observation))``) is added. The ray transform is
            normalized by its spectral norm.

            A normalized ray transform that corresponds to the noiseless
            forward operator is stored in the attribute `ray_trafo` of the
            dataset.

            In order to avoid the inverse crime, the ground truth images of
            shape (128, 128) are upscaled by bilinear interpolation to a
            resolution of (400, 400) before the ray transform is applied (whose
            discretization is different from the one of `ray_trafo`).

        * ``'lidc_idri_dival'``
            A dataset based on real CT reconstructions from the `LIDC-IDRI
            <https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI>`_
            dataset.

            `LIDCIDRIDivalDataset` is used as ground truth dataset and a ray
            transform with parallel beam geometry using 1000 angles is applied.
            The noise is chosen in the way presented in [1]_ for the human
            dataset: poisson noise corresponding to 10^4 incident photons per
            pixel before attenuation is used.

            A ray transform that corresponds to the noiseless forward operator
            is stored in the attribute `ray_trafo` of the dataset.

            In order to avoid the inverse crime, the ground truth images of
            shape (362, 362) are upscaled by bilinear interpolation to a
            resolution of (1000, 1000) before the ray transform is applied
            (whose discretization is different from the one of `ray_trafo`).

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
        MIN_PT = [-1., -1.]
        MAX_PT = [1., 1.]

        ellipses_dataset = EllipsesDataset(min_pt=MIN_PT, max_pt=MAX_PT)

        NUM_ANGLES = 30
        # image shape for simulation
        IM_SHAPE = (400, 400)  # images will be scaled up from (128, 128)

        reco_space = ellipses_dataset.space
        space = odl.uniform_discr(min_pt=MIN_PT, max_pt=MAX_PT, shape=IM_SHAPE,
                                  dtype=np.float64)

        reco_geometry = odl.tomo.parallel_beam_geometry(
            reco_space, num_angles=NUM_ANGLES)
        geometry = odl.tomo.parallel_beam_geometry(
            space, num_angles=NUM_ANGLES,
            det_shape=reco_geometry.detector.shape)

        IMPL = 'astra_cuda'
        reco_ray_trafo = odl.tomo.RayTransform(reco_space, reco_geometry,
                                               impl=IMPL)
        ray_trafo = odl.tomo.RayTransform(space, geometry, impl=IMPL)

        class _ResizeOperator(odl.Operator):
            def __init__(self):
                super().__init__(reco_space, space)

            def _call(self, x, out):
                out.assign(space.element(resize(x, IM_SHAPE, order=1)))

        # Ensure operator has fixed operator norm for scale invariance
        opnorm = odl.power_method_opnorm(ray_trafo)
        resize_op = _ResizeOperator()
        forward_op = (1 / opnorm) * ray_trafo * resize_op

        dataset = ellipses_dataset.create_pair_dataset(
            forward_op=forward_op, noise_type='white',
            noise_kwargs={'relative_stddev': True,
                          'stddev': 0.05},
            noise_seeds={'train': 1, 'validation': 2, 'test': 3})

        dataset.ray_trafo = reco_ray_trafo

        return dataset
    elif name == 'lidc_idri_dival':
        # ~26cm x 26cm images
        MIN_PT = [-0.13, -0.13]
        MAX_PT = [0.13, 0.13]

        lidc_idri_dival_dataset = LIDCIDRIDivalDataset(min_pt=MIN_PT,
                                                       max_pt=MAX_PT,
                                                       return_val='mu_normed')

        NUM_ANGLES = 1000
        # image shape for simulation
        IM_SHAPE = (1000, 1000)  # images will be scaled up from (362, 362)

        reco_space = lidc_idri_dival_dataset.space
        space = odl.uniform_discr(min_pt=MIN_PT, max_pt=MAX_PT, shape=IM_SHAPE,
                                  dtype=np.float64)

        reco_geometry = odl.tomo.parallel_beam_geometry(
            reco_space, num_angles=NUM_ANGLES)
        geometry = odl.tomo.parallel_beam_geometry(
            space, num_angles=NUM_ANGLES,
            det_shape=reco_geometry.detector.shape)

        IMPL = 'astra_cuda'
        reco_ray_trafo = odl.tomo.RayTransform(reco_space, reco_geometry,
                                               impl=IMPL)
        ray_trafo = odl.tomo.RayTransform(space, geometry, impl=IMPL)

        PHOTONS_PER_PIXEL = 10000

        class _LIDCIDRIDivalForwardOp(odl.Operator):
            def __init__(self):
                super().__init__(reco_space, ray_trafo.range)

            def _call(self, x, out):
                im = x * MU_MAX
                im_resized = resize(im, IM_SHAPE, order=1)

                # apply forward operator
                data = ray_trafo(im_resized).asarray().astype(np.float64)

                # apply poisson noise
                data *= (-1)
                np.exp(data, out=data)
                out.assign(ray_trafo.range.element(data))

        class _LIDCIDRIDivalPostProcessor(odl.Operator):
            def __init__(self):
                super().__init__(ray_trafo.range, ray_trafo.range)

            def _call(self, x, out):
                data = np.maximum(1 / PHOTONS_PER_PIXEL, x)  # assume at least
                # one photon per pixel to avoid log(0)
                np.log(data, out=data)
                data /= (-MU_MAX)
                out.assign(ray_trafo.range.element(data))

        forward_op = _LIDCIDRIDivalForwardOp()
        post_processor = _LIDCIDRIDivalPostProcessor()

        dataset = lidc_idri_dival_dataset.create_pair_dataset(
            forward_op=forward_op, post_processor=post_processor,
            noise_type='poisson',
            noise_kwargs={'scaling_factor': PHOTONS_PER_PIXEL},
            noise_seeds={'train': 1, 'validation': 2, 'test': 3})

        dataset.ray_trafo = reco_ray_trafo

        return dataset
    else:
        raise ValueError("unknown dataset '{}'".format(name))
