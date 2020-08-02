import numpy as np
from odl import nonuniform_partition
from odl.tomo import Parallel2dGeometry, RayTransform

from dival.datasets.dataset import Dataset


def get_angle_subset_dataset(dataset, num_angles, **kwargs):
    """
    Return a :class:`AngleSubsetDataset` with a reduced number of angles.
    The angles in the subset are equidistant if the original angles are.

    Parameters
    ----------
    dataset : `Dataset`
        Basis CT dataset.
        The number of angles must be divisible by `num_angles`.
    num_angles : int
        Number of angles in the subset.

    Keyword arguments are passed to ``AngleSubsetDataset.__init__``.

    Raises
    ------
    ValueError
        If the number of angles of `dataset` is not divisible by
        `num_angles`.

    Returns
    -------
    angle_subset_dataset : :class:`AngleSubsetDataset`
        The dataset with the reduced number of angles.
    """
    num_angles_total = dataset.get_shape()[0][0]
    angle_indices = range(0, num_angles_total, num_angles_total // num_angles)
    if len(angle_indices) != num_angles:
        raise ValueError(
            'Number of angles {:d} is not divisible by requested number of '
            'angles {:d}'.format(num_angles_total, num_angles))
    return AngleSubsetDataset(dataset, angle_indices, **kwargs)


class AngleSubsetDataset(Dataset):
    """
    CT dataset that selects a subset of the angles of a basis CT dataset.
    """

    def __init__(self, dataset, angle_indices, impl=None):
        """
        Parameters
        ----------
        dataset : `Dataset`
            Basis CT dataset.
            Requirements:

                - sample elements are ``(observation, ground_truth)``
                - :meth:`get_ray_trafo` gives corresponding ray transform.

        angle_indices : array-like or slice
            Indices of the angles to use from the observations.
        impl : {``'skimage'``, ``'astra_cpu'``, ``'astra_cuda'``},\
                optional
            Implementation passed to :class:`odl.tomo.RayTransform` to
            construct :attr:`ray_trafo`.
        """
        self.dataset = dataset
        self.angle_indices = (angle_indices if isinstance(angle_indices, slice)
                              else np.asarray(angle_indices))
        self.train_len = self.dataset.get_len('train')
        self.validation_len = self.dataset.get_len('validation')
        self.test_len = self.dataset.get_len('test')
        self.random_access = self.dataset.supports_random_access()
        self.num_elements_per_sample = (
            self.dataset.get_num_elements_per_sample())
        orig_geometry = self.dataset.get_ray_trafo(impl=impl).geometry
        apart = nonuniform_partition(
            orig_geometry.angles[self.angle_indices])
        self.geometry = Parallel2dGeometry(
            apart=apart, dpart=orig_geometry.det_partition)
        orig_shape = self.dataset.get_shape()
        self.shape = ((apart.shape[0], orig_shape[0][1]), orig_shape[1])
        self.space = (None, self.dataset.space[1])  # preliminary, needed for
        # call to get_ray_trafo
        self.ray_trafo = self.get_ray_trafo(impl=impl)
        super().__init__(space=(self.ray_trafo.range, self.dataset.space[1]))

    def get_ray_trafo(self, **kwargs):
        """
        Return the ray transform that matches the subset of angles specified to
        the constructor via `angle_indices`.
        """
        return RayTransform(self.space[1], self.geometry, **kwargs)

    def generator(self, part='train'):
        for (obs, gt) in self.dataset.generator(part=part):
            yield (self.space[0].element(obs[self.angle_indices]), gt)

    def get_sample(self, index, part='train', out=None):
        if out is None:
            out = (True, True)
        (out_obs, out_gt) = out
        out_basis = (out_obs is not False, out_gt)
        obs_basis, gt = self.dataset.get_sample(index, part=part,
                                                out=out_basis)
        if isinstance(out_obs, bool):
            obs = (self.space[0].element(obs_basis[self.angle_indices])
                   if out_obs else None)
        else:
            out_obs[:] = obs_basis[self.angle_indices]
            obs = out_obs
        return (obs, gt)

    def get_samples(self, key, part='train', out=None):
        if out is None:
            out = (True, True)
        (out_obs, out_gt) = out
        out_basis = (out_obs is not False, out_gt)
        obs_arr_basis, gt_arr = self.dataset.get_samples(key, part=part,
                                                         out=out_basis)
        if isinstance(out_obs, bool):
            obs_arr = obs_arr_basis[:, self.angle_indices] if out_obs else None
        else:
            out_obs[:] = obs_arr_basis[:, self.angle_indices]
            obs_arr = out_obs
        return (obs_arr, gt_arr)
