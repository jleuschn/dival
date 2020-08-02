import numpy as np
from odl.tomo import fbp_op

from dival.datasets import Dataset, CachedDataset, generate_cache_files


def generate_fbp_cache_files(dataset, ray_trafo, cache_files, size=None,
                             filter_type='Hann', frequency_scaling=1.0):
    """
    Generate cache files for a CT dataset, whereby FBPs are precomputed
    from the observations.

    The cache files can be accessed by a :class:`CachedDataset`, which can
    be obtained by :func:`get_cached_fbp_dataset`.

    Parameters
    ----------
    dataset : :class:`.Dataset`
        CT dataset with observation and ground truth pairs.
        The FBPs are computed from the observations.
    ray_trafo : :class:`odl.tomo.RayTransform`
        Ray transform from which the FBP operator is constructed.
    cache_files : dict of 2-tuple of (str or `None`)
        See :func:`cached_dataset.generate_cache_files`.

        As an example, to cache the FBPs (but not the ground truths) for parts
        ``'train'`` and ``'validation'``:

        .. code-block::

            {'train':      ('cache_train_fbp.npy',      None),
             'validation': ('cache_validation_fbp.npy', None)}

    size : dict of int, optional
        Numbers of samples to cache for each dataset part.
        If a field is omitted or has value `None`, all samples are cached.
        Default: ``{}``.
    filter_type : str, optional
        Filter type accepted by :func:`odl.tomo.fbp_op`.
        Default: ``'Hann'``.
    frequency_scaling : float, optional
        Relative cutoff frequency passed to :func:`odl.tomo.fbp_op`.
        Default: ``1.0``.
    """
    fbp_dataset = FBPDataset(dataset, ray_trafo, filter_type=filter_type,
                             frequency_scaling=frequency_scaling)
    generate_cache_files(fbp_dataset, cache_files, size=size)


def get_cached_fbp_dataset(dataset, ray_trafo, cache_files, size=None,
                           filter_type='Hann', frequency_scaling=1.0):
    """
    Return :class:`CachedDataset` with FBP and ground truth pairs
    corresponding to the passed CT dataset.

    See :func:`generate_fbp_cache_files` for generating the cache files.

    If for a dataset part no FBP cache is specified, these FBPs are
    computed from the observations on the fly.

    Parameters
    ----------
    dataset : :class:`.Dataset`
        CT dataset with observation and ground truth pairs.
        For all parts and components, for which caches are specified,
        the samples of this dataset are ignored.
    ray_trafo : :class:`odl.tomo.RayTransform`
        Ray transform from which the FBP operator is constructed that is called
        if an FBP cache is missing.
    cache_files : dict of 2-tuple of (str or `None`)
        See :func:`cached_dataset.CachedDataset`.

        As an example, to use caches for the FBPs (but not the ground truths)
        for parts ``'train'`` and ``'validation'``:

        .. code-block::

            {'train':      ('cache_train_fbp.npy',      None),
             'validation': ('cache_validation_fbp.npy', None)}

    size : dict of int, optional
        Numbers of samples for each part.
        If a field is omitted or has value `None`, all available samples
        are used, which may be less than the number of samples in the
        original dataset if the cache contains fewer samples.
        Default: ``{}``.
    filter_type : str, optional
        Filter type accepted by :func:`odl.tomo.fbp_op`.
        Default: ``'Hann'``.
    frequency_scaling : float, optional
        Relative cutoff frequency passed to :func:`odl.tomo.fbp_op`.
        Default: ``1.0``.

    Returns
    -------
    cached_fbp_dataset : :class:`CachedDataset`
        Dataset with FBP and ground truth pairs that uses the specified cache
        files.
    """
    fbp_dataset = FBPDataset(dataset, ray_trafo, filter_type=filter_type,
                             frequency_scaling=frequency_scaling)
    cached_fbp_dataset = CachedDataset(
        fbp_dataset, fbp_dataset.space, cache_files, size=size)
    return cached_fbp_dataset


class FBPDataset(Dataset):
    """
    Dataset computing filtered back-projections for a CT dataset on the fly.

    Each sample is a pair of a FBP and a ground truth image.
    """
    def __init__(self, dataset, ray_trafo, filter_type='Hann',
                 frequency_scaling=1.0):
        """
        Parameters
        ----------
        dataset : :class:`.Dataset`
            CT dataset. FBPs are computed from the observations, the ground
            truth is taken directly from the dataset.
        ray_trafo : :class:`odl.tomo.RayTransform`
            Ray transform from which the FBP operator is constructed.
        filter_type : str, optional
            Filter type accepted by :func:`odl.tomo.fbp_op`.
            Default: ``'Hann'``.
        frequency_scaling : float, optional
            Relative cutoff frequency passed to :func:`odl.tomo.fbp_op`.
            Default: ``1.0``.
        """
        self.dataset = dataset
        self.ray_trafo = ray_trafo
        self.fbp_op = fbp_op(self.ray_trafo,
                             filter_type=filter_type,
                             frequency_scaling=frequency_scaling)
        self.train_len = self.dataset.get_len('train')
        self.validation_len = self.dataset.get_len('validation')
        self.test_len = self.dataset.get_len('test')
        self.shape = (self.dataset.shape[1], self.dataset.shape[1])
        self.num_elements_per_sample = 2
        self.random_access = dataset.supports_random_access()
        super().__init__(space=(self.dataset.space[1], self.dataset.space[1]))

    def generator(self, part='train'):
        gen = self.dataset.generator(part=part)
        for (obs, gt) in gen:
            fbp = self.fbp_op(obs)
            yield (fbp, gt)

    def get_sample(self, index, part='train', out=None):
        if out is None:
            out = (True, True)
        out_fbp = not (isinstance(out[0], bool) and not out[0])
        (obs, gt) = self.dataset.get_sample(index, part=part,
                                            out=(out_fbp, out[1]))
        if isinstance(out[0], bool):
            fbp = self.fbp_op(obs) if out[0] else None
        else:
            if out[0] in self.fbp_op.range:
                self.fbp_op(obs, out=out[0])
            else:
                out[0][:] = self.fbp_op(obs)
            fbp = out[0]
        return (fbp, gt)

    def get_samples(self, key, part='train', out=None):
        if out is None:
            out = (True, True)
        out_fbp = not (isinstance(out[0], bool) and not out[0])
        (obs_arr, gt_arr) = self.dataset.get_samples(key, part=part,
                                                     out=(out_fbp, out[1]))
        if isinstance(out[0], bool) and out[0]:
            fbp_arr = np.empty((len(obs_arr),) + self.dataset.shape[1],
                               dtype=self.dataset.space[1].dtype)
        elif isinstance(out[0], bool) and not out[0]:
            fbp_arr = None
        else:
            fbp_arr = out[0]
        if out_fbp:
            tmp_fbp = self.fbp_op.range.element()
            for i in range(len(obs_arr)):
                self.fbp_op(obs_arr[i], out=tmp_fbp)
                fbp_arr[i][:] = tmp_fbp
        return (fbp_arr, gt_arr)
