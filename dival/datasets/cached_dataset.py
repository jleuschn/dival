from itertools import islice
import numpy as np
from numpy.lib.format import open_memmap
from tqdm import tqdm

from dival.datasets import Dataset


def generate_cache_files(dataset, cache_files, size=None, flush_interval=1000):
    """
    Generate cache files for :class:`CachedDataset`.

    Parameters
    ----------
    dataset : :class:`.Dataset`
        Dataset from which to cache samples.
    cache_files : dict of [tuple of ] (str or `None`)
        Filenames of the cache files for each part and for each component to
        be cached.
        The part (``'train'``, ...) is the key to the dict. For each part,
        a tuple of filenames should be provided, each of which can be `None`,
        meaning that this component should not be cached.
        If the dataset only provides one element per sample, the filename does
        not have to be packed inside a tuple.
        If a key is omitted, the part is not cached.

        As an example, for a CT dataset with cached FBPs instead of
        observations for parts ``'train'`` and ``'validation'``:

        .. code-block::

            {'train':      ('cache_train_fbp.npy',      None),
             'validation': ('cache_validation_fbp.npy', None)}

    size : dict of int, optional
        Numbers of samples to cache for each dataset part.
        If a field is omitted or has value `None`, all samples are cached.
        Default: ``{}``.
    flush_interval : int, optional
        Number of samples to retrieve before flushing to file (using memmap).
        This amount of samples should fit into the systems main memory (RAM).
        If ``-1``, each file content is only flushed once at the end.
    """
    if size is None:
        size = {}
    for part in ['train', 'validation', 'test']:
        if part in cache_files:
            num_samples = min(dataset.get_len(part), size.get(part, np.inf))
            files = cache_files[part]
            if (dataset.get_num_elements_per_sample() == 1
                    and not isinstance(files, tuple)):
                files = (files,)
            memmaps = []
            for k in range(dataset.get_num_elements_per_sample()):
                space = (dataset.space[k]
                         if dataset.get_num_elements_per_sample() > 1 else
                         dataset.space)
                memmaps.append((None if files[k] is None else
                                open_memmap(
                                    files[k], mode='w+', dtype=space.dtype,
                                    shape=(num_samples,) + space.shape)))
            for i, sample in enumerate(tqdm(islice(dataset.generator(part),
                                                   num_samples),
                                            desc=('generating cache for part '
                                                  '\'{}\''.format(part)),
                                            total=num_samples)):
                for s, m in zip(sample, memmaps):
                    if m is not None:
                        m[i] = s
                if (i + 1) % flush_interval == 0:
                    for m in memmaps:
                        if m is not None:
                            m.flush()
            for m in memmaps:
                if m is not None:
                    del m  # flush completed file



class CachedDataset(Dataset):
    """Dataset that allows to replace elements of a dataset with cached data
    from .npy files.

    The arrays in the .npy files must have shape
    ``(self.get_len(part),) + self.space[i].shape`` for the i-th component.
    """

    def __init__(self, dataset, space, cache_files, size=None):
        """
        Parameters
        ----------
        dataset : :class:`.Dataset`
            Original dataset from which non-cached elements are used.
            Must support random access if any elements are not cached.
        space : [tuple of ] :class:`odl.space.base_tensors.TensorSpace`,\
                optional
            The space(s) of the elements of samples as a tuple.
            This may be different from :attr:`space`, e.g. for precomputing
            domain-changing operations on the elements.
        cache_files : dict of [tuple of ] (str or `None`)
            Filenames of the cache files for each part and for each component.
            The part (``'train'``, ...) is the key to the dict. For each part,
            a tuple of filenames should be provided, each of which can be
            `None`, meaning that this component should be fetched from the
            original dataset. If the dataset only provides one element per
            sample, the filename does not have to be packed inside a tuple.
            If a key is omitted, the part is fetched from the original dataset.

            As an example, for a CT dataset with cached FBPs instead of
            observations for parts ``'train'`` and ``'validation'``:

            .. code-block::

                {'train':      ('cache_train_fbp.npy',      None),
                 'validation': ('cache_validation_fbp.npy', None)}

        size : dict of int, optional
            Numbers of samples for each part.
            If a field is omitted or has value `None`, all available samples
            are used, which may be less than the number of samples in the
            original dataset if the cache contains fewer samples.
            Default: ``{}``.
        """
        super().__init__(space=space)

        self.dataset = dataset
        self.cache_files = cache_files
        self.size = size if size is not None else {}
        self.num_elements_per_sample = (
            self.dataset.get_num_elements_per_sample())
        self.data = {}
        cache_size = {}

        for part in ['train', 'validation', 'test']:
            if part in self.cache_files:
                self.data[part] = []
                cache_size[part] = self.dataset.get_len(part)
                files = self.cache_files[part]
                if (self.num_elements_per_sample == 1 and
                        not isinstance(files, tuple)):
                    files = (files,)
                for k in range(self.num_elements_per_sample):
                    data = None
                    if files[k]:
                        try:
                            data = np.load(files[k], mmap_mode='r')
                        except FileNotFoundError:
                            raise FileNotFoundError(
                                "Did not find cache file '{}'".format(
                                    files[k]))
                    self.data[part].append(data)
            else:
                self.data[part] = [None] * self.num_elements_per_sample

        cache_size = {}
        for part in ['train', 'validation', 'test']:
            cache_size[part] = self.dataset.get_len(part)
            for data in self.data[part]:
                if data is not None:
                    cache_size[part] = min(data.shape[0], cache_size[part])
        self.train_len = self.size.get(
            'train', cache_size['train'])
        self.validation_len = self.size.get(
            'validation', cache_size['validation'])
        self.test_len = self.size.get(
            'test', cache_size['test'])

        self.random_access = (self.dataset.supports_random_access() or
                              all((all((d is not None for d in data))
                                   for data in self.data.values())))

    def generator(self, part='train'):
        if self.num_elements_per_sample == 1:
            if self.data[part][0] is None:
                yield from self.dataset.generator(part=part)
            else:
                for i in range(self.get_len(part)):
                    yield self.space.element(np.copy(self.data[part][0][i]))
        elif all((d is not None for d in self.data[part])):  # caches only
            for i in range(self.get_len(part)):
                yield tuple((space.element(np.copy(cache[i]))
                             for cache, space in
                             zip(self.data[part], self.space)))
        else:  # some components from original dataset
            gen = self.dataset.generator(part=part)
            for i, from_dataset in zip(range(self.get_len(part)), gen):
                yield tuple(((from_d if cache is None else
                              space.element(np.copy(cache[i])))
                             for from_d, cache, space in
                             zip(from_dataset, self.data[part], self.space)))

    def get_sample(self, index, part='train', out=None):
        if index >= self.get_len(part):
            raise IndexError(
                "index {:d} out of bounds for dataset part '{}' (len: {:d})"
                .format(index, part, self.get_len(part)))
        if self.num_elements_per_sample == 1:
            if self.data[part][0] is None:
                sample = self.dataset.get_sample(index, part=part, out=out)
            elif out is None:
                sample = self.space.element(np.copy(self.data[part][0][index]))
            else:
                out[:] = self.data[part][0][index]
                sample = out
        else:
            if out is None:
                out = (True,) * self.num_elements_per_sample
            out_dataset = tuple(
                (out_orig if cache is None else False
                 for out_orig, cache in zip(out, self.data[part])))
            from_dataset = (
                self.dataset.get_sample(index, part=part, out=out_dataset)
                if any(o_d is not False for o_d in out_dataset) else
                (None,) * self.num_elements_per_sample)  # avoids
                                 # NotImplementedError if all values are cached
            sample = []
            for from_d, cache, out_, space in zip(
                    from_dataset, self.data[part], out, self.space):
                if cache is None:
                    sample.append(from_d)
                elif isinstance(out_, bool):
                    sample.append(space.element(np.copy(cache[index]))
                                  if out_ else None)
                else:
                    out_[:] = cache[index]
                    sample.append(out_)
            sample = tuple(sample)
        return sample

    def get_samples(self, key, part='train', out=None):
        len_part = self.get_len(part)
        if isinstance(key, range):
            if key[-1] >= len_part or key[0] >= len_part:
                raise IndexError(
                    "key {} out of bounds for dataset part '{}' (len: {:d})"
                    .format(key, part, len_part))
            slice_ = slice(key.start, key.stop, key.step)
        if self.num_elements_per_sample == 1:
            if self.data[part][0] is None:
                samples = self.dataset.get_samples(key, part=part, out=out)
            elif out is None:
                samples = np.copy(self.data[part][0][slice_])
            else:
                out[:] = self.data[part][0][slice_]
                samples = out
        else:
            if out is None:
                out = (True,) * self.num_elements_per_sample
            out_dataset = tuple(
                (out_orig if cache is None else False
                 for out_orig, cache in zip(out, self.data[part])))
            from_dataset = (
                self.dataset.get_samples(key, part=part, out=out_dataset)
                if any(o_d is not False for o_d in out_dataset) else
                (None,) * self.num_elements_per_sample)  # avoids
                                 # NotImplementedError if all values are cached
            samples = []
            for from_d, cache, out_ in zip(from_dataset, self.data[part], out):
                if cache is None:
                    samples.append(from_d)
                elif isinstance(out_, bool):
                    samples.append(np.copy(cache[slice_]) if out_ else None)
                else:
                    out_[:] = cache[slice_]
                    samples.append(out_)
            samples = tuple(samples)
        return samples
