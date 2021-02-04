# -*- coding: utf-8 -*-
"""Provides standard datasets for benchmarking.
"""
from warnings import warn
from functools import partial
import numpy as np
import odl
from dival.datasets.ellipses_dataset import EllipsesDataset
from dival.datasets.lodopab_dataset import LoDoPaBDataset
from dival.datasets.angle_subset_dataset import get_angle_subset_dataset
from dival.util.odl_utility import ResizeOperator


STANDARD_DATASET_NAMES = ['ellipses', 'lodopab']


def get_standard_dataset(name, **kwargs):
    """
    Return a standard dataset by name.

    The standard datasets are (currently):

        ``'ellipses'``
            A typical synthetical CT dataset with ellipse phantoms.

            `EllipsesDataset` is used as ground truth dataset, a ray
            transform with parallel beam geometry using 30 angles is applied,
            and white gaussian noise with a standard deviation of 2.5% (i.e.
            ``0.025 * mean(abs(observation))``) is added.

            In order to avoid the inverse crime, the ground truth images of
            shape (128, 128) are upscaled by bilinear interpolation to a
            resolution of (400, 400) before the ray transform is applied (whose
            discretization is different from the one of :attr:`ray_trafo`).
            
            Attributes of the returned dataset:
                `ray_trafo` : :class:`odl.tomo.RayTransform`
                    Ray transform corresponding to the noiseless forward
                    operator.
                ``get_ray_trafo(**kwargs)`` : function
                    Function that returns a ray transform corresponding to the
                    noiseless forward operator. Keyword arguments (e.g. `impl`)
                    are forwarded to the :class:`RayTransform` constructor.

        ``'lodopab'``
            The LoDoPaB-CT dataset, which is documented in the
            preprint `<https://arxiv.org/abs/1910.01113>`_ hosted on
            `<https://zenodo.org/record/3384092>`_.
            It is a simulated low dose CT dataset based on real reconstructions
            from the `LIDC-IDRI
            <https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI>`_
            dataset.

            The dataset contains 42895 pairs of images and projection data.
            For simulation, a ray transform with parallel beam geometry using
            1000 angles and 513 detector pixels is used. Poisson noise
            corresponding to 4096 incident photons per pixel before attenuation
            is applied to the projection data.
            
            Attributes of the returned dataset:
                `ray_trafo` : :class:`odl.tomo.RayTransform`
                    Ray transform corresponding to the noiseless forward
                    operator.
            Methods of the returned dataset:
                ``get_ray_trafo(**kwargs)``
                    Function that returns a ray transform corresponding to the
                    noiseless forward operator. Keyword arguments (e.g. `impl`)
                    are forwarded to the :class:`RayTransform` constructor.

    Parameters
    ----------
    name : str
        Name of the dataset.
    kwargs : dict
        Keyword arguments.
        Supported parameters for the datasets are:

            ``'ellipses'``
                impl : {``'skimage'``, ``'astra_cpu'``, ``'astra_cuda'``},\
                        optional
                    Implementation passed to :class:`odl.tomo.RayTransform`
                    Default: ``'astra_cuda'``.
                fixed_seeds : dict or bool, optional
                    Seeds to use for random ellipse generation, passed to
                    :meth:`.EllipsesDataset.__init__`.
                    Default: ``False``.
                fixed_noise_seeds : dict or bool, optional
                    Seeds to use for noise generation, passed as `noise_seeds`
                    to :meth:`.GroundTruthDataset.create_pair_dataset`.
                    If ``True`` is passed (the default), the seeds
                    ``{'train': 1, 'validation': 2, 'test': 3}`` are used.
            ``'lodopab'``
                num_angles : int, optional
                    Number of angles to use from the full 1000 angles.
                    Must be a divisor of 1000.
                observation_model : {``'post-log'``, ``'pre-log'``}, optional
                    The observation model to use. Default is ``'post-log'``.
                min_photon_count : float, optional
                    Replacement value for a simulated photon count of zero.
                    If ``observation_model == 'post-log'``, a value greater
                    than zero is required in order to avoid undefined values.
                    The default is 0.1, both for ``'post-log'`` and
                    ``'pre-log'`` model.
                sorted_by_patient : bool, optional
                    Whether to sort the samples by patient id.
                    Useful to resplit the dataset.
                    Default: ``False``.
                impl : {``'skimage'``, ``'astra_cpu'``, ``'astra_cuda'``},\
                        optional
                    Implementation passed to :class:`odl.tomo.RayTransform`
                    Default: ``'astra_cuda'``.

    Returns
    -------
    dataset : :class:`.Dataset`
        The standard dataset.
        It has an attribute `standard_dataset_name` that stores its name.
    """
    name = name.lower()
    if name == 'ellipses':
        fixed_seeds = kwargs.pop('fixed_seeds', False)
        ellipses_dataset = EllipsesDataset(image_size=128,
                                           fixed_seeds=fixed_seeds)

        NUM_ANGLES = 30
        # image shape for simulation
        IM_SHAPE = (400, 400)  # images will be scaled up from (128, 128)

        reco_space = ellipses_dataset.space
        space = odl.uniform_discr(min_pt=reco_space.min_pt,
                                  max_pt=reco_space.max_pt,
                                  shape=IM_SHAPE, dtype=np.float32)

        reco_geometry = odl.tomo.parallel_beam_geometry(
            reco_space, num_angles=NUM_ANGLES)
        geometry = odl.tomo.parallel_beam_geometry(
            space, num_angles=NUM_ANGLES,
            det_shape=reco_geometry.detector.shape)

        impl = kwargs.pop('impl', 'astra_cuda')
        ray_trafo = odl.tomo.RayTransform(space, geometry, impl=impl)

        reco_ray_trafo = odl.tomo.RayTransform(reco_space, reco_geometry,
                                               impl=impl)

        # forward operator
        resize_op = ResizeOperator(reco_space, space)
        forward_op = ray_trafo * resize_op

        noise_seeds = kwargs.pop('fixed_noise_seeds', True)
        if isinstance(noise_seeds, bool):
            noise_seeds = ({'train': 1, 'validation': 2, 'test': 3}
                           if noise_seeds else None)

        dataset = ellipses_dataset.create_pair_dataset(
            forward_op=forward_op, noise_type='white',
            noise_kwargs={'relative_stddev': True,
                          'stddev': 0.025},
            noise_seeds=noise_seeds)

        dataset.get_ray_trafo = partial(odl.tomo.RayTransform,
                                        reco_space, reco_geometry)
        dataset.ray_trafo = reco_ray_trafo

    elif name == 'lodopab':

        num_angles = kwargs.pop('num_angles', None)
        lodopab_kwargs = {}
        for k in ['observation_model', 'min_photon_count', 'sorted_by_patient',
                  'impl']:
            if k in kwargs:
                lodopab_kwargs[k] = kwargs.pop(k)

        dataset = LoDoPaBDataset(**lodopab_kwargs)

        if num_angles is not None:
            dataset = get_angle_subset_dataset(
                dataset, num_angles, impl=kwargs.get('impl', 'astra_cuda'))

    else:
        raise ValueError("unknown dataset '{}'. Known standard datasets are {}"
                         .format(name, STANDARD_DATASET_NAMES))

    if kwargs:
        warn('unused keyword arguments: {}'
             .format(', '.join(kwargs.keys())))

    dataset.standard_dataset_name = name
    return dataset
