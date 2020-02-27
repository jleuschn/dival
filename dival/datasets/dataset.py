# -*- coding: utf-8 -*-
"""Provides the dataset base classes.
"""
from itertools import islice
from math import ceil
import numpy as np
from dival.data import DataPairs
from dival.util.odl_utility import NoiseOperator


class Dataset():
    """Dataset base class.

    Subclasses must either implement :meth:`generator` or provide random access
    by implementing :meth:`get_sample` and :meth:`get_samples` (which then
    should be indicated by setting the attribute ``random_access = True``).

    Attributes
    ----------
    space : [tuple of ] :class:`odl.space.base_tensors.TensorSpace` or `None`
        The spaces of the elements of samples as a tuple.
        If only one element per sample is provided, this attribute is the space
        of the element (i.e., no tuple).
        It is strongly recommended to set this attribute in subclasses, as some
        functionality may depend on it.
    shape : [tuple of ] tuple of int, optional
        The shapes of the elements of samples as a tuple of tuple of int.
        If only one element per sample is provided, this attribute is the shape
        of the element (i.e., not a tuple of tuple of int, but a tuple of int).
    train_len : int, optional
        Number of training samples.
    validation_len : int, optional
        Number of validation samples.
    test_len : int, optional
        Number of test samples.
    random_access : bool, optional
        Whether the dataset supports random access via ``self.get_sample`` and
        ``self.get_samples``.
        Setting this attribute is the preferred way for subclasses to indicate
        whether they support random access.
    num_elements_per_sample : int, optional
        Number of elements per sample.
        E.g. 1 for a ground truth dataset or 2 for a dataset of pairs of
        observation and ground truth.
    standard_dataset_name : str, optional
        Datasets returned by `get_standard_dataset` have this attribute giving
        its name.
    """
    def __init__(self, space=None):
        """
        The attributes that potentially should be set by the subclass are:
        :attr:`space` (can also be set by argument), :attr:`shape`,
        :attr:`train_len`, :attr:`validation_len`, :attr:`test_len`,
        :attr:`random_access` and :attr:`num_elements_per_sample`.

        Parameters
        ----------
        space : [tuple of ] :class:`odl.space.base_tensors.TensorSpace`,\
                optional
            The spaces of the elements of samples as a tuple.
            If only one element per sample is provided, this attribute is the
            space of the element (i.e., no tuple).
            It is strongly recommended to set `space` in subclasses, as some
            functionality may depend on it.
        """
        self.space = space

    def generator(self, part='train'):
        """Yield data.

        The default implementation calls :meth:`get_sample` if the dataset
        implements it (i.e., supports random access).

        Parameters
        ----------
        part : {``'train'``, ``'validation'``, ``'test'``}, optional
            Whether to yield train, validation or test data.
            Default is ``'train'``.

        Yields
        ------
        data : odl element or tuple of odl elements
            Sample of the dataset.
        """
        if self.supports_random_access():
            for i in range(self.get_len(part)):
                sample = self.get_sample(i, part=part)
                if self.get_num_elements_per_sample() == 1:
                    sample = self.space.element(sample)
                else:
                    sample = tuple((space.element(s) for space, s in zip(
                                        self.space, sample)))
                yield sample
        else:
            raise NotImplementedError

    def get_train_generator(self):
        return self.generator(part='train')

    def get_validation_generator(self):
        return self.generator(part='validation')

    def get_test_generator(self):
        return self.generator(part='test')

    def get_len(self, part='train'):
        """Return the number of elements the generator will yield.

        Parameters
        ----------
        part : {``'train'``, ``'validation'``, ``'test'``}, optional
            Whether to return the number of train, validation or test elements.
            Default is ``'train'``.
        """
        if part == 'train':
            return self.get_train_len()
        elif part == 'validation':
            return self.get_validation_len()
        elif part == 'test':
            return self.get_test_len()
        raise ValueError("dataset part must be 'train', "
                         "'validation' or 'test', not '{}'".format(part))

    def get_train_len(self):
        """Return the number of samples the train generator will yield."""
        try:
            return self.train_len
        except AttributeError:
            raise NotImplementedError

    def get_validation_len(self):
        """Return the number of samples the validation generator will yield.
        """
        try:
            return self.validation_len
        except AttributeError:
            raise NotImplementedError

    def get_test_len(self):
        """Return the number of samples the test generator will yield."""
        try:
            return self.test_len
        except AttributeError:
            raise NotImplementedError

    def get_shape(self):
        """Return the shape of each element.

        Returns :attr:`shape` if it is set.
        Otherwise, it is inferred from :attr:`space` (which is strongly
        recommended to be set in every subclass).
        If also :attr:`space` is not set, a :class:`NotImplementedError` is
        raised.

        Returns
        -------
        shape : [tuple of ] tuple"""
        try:
            return self.shape
        except AttributeError:
            if self.space is not None:
                if self.get_num_elements_per_sample() == 1:
                    return self.space.shape
                else:
                    return tuple(s.shape for s in self.space)
            raise NotImplementedError

    def get_num_elements_per_sample(self):
        """Return number of elements per sample.

        Returns :attr:`num_elements_per_sample` if it is set.
        Otherwise, it is inferred from :attr:`space` (which is strongly
        recommended to be set in every subclass).
        If also :attr:`space` is not set, a :class:`NotImplementedError` is
        raised.

        Returns
        -------
        num_elements_per_sample : int
        """
        try:
            return self.num_elements_per_sample
        except AttributeError:
            if self.space is not None:
                return len(self.space) if isinstance(self.space, tuple) else 1
            raise NotImplementedError

    def get_data_pairs(self, part='train', n=None):
        """
        Return first samples from data part as :class:`.DataPairs` object.

        Only supports datasets with two elements per sample.``

        Parameters
        ----------
        part : {``'train'``, ``'validation'``, ``'test'``}, optional
            The data part. Default is ``'train'``.
        n : int, optional
            Number of pairs (from beginning). If `None`, all available data
            is used (the default).
        """
        if self.get_num_elements_per_sample() != 2:
            raise ValueError('`get_data_pairs` only supports datasets with'
                             '2 elements per sample, this dataset has {:d}'
                             .format(self.get_num_elements_per_sample()))
        gen = self.generator(part=part)
        observations, ground_truth = [], []
        for obs, gt in islice(gen, n):
            observations.append(obs)
            ground_truth.append(gt)
        name = '{} part{}'.format(part,
                                  ' 0:{:d}'.format(n) if n is not None else '')
        data_pairs = DataPairs(observations, ground_truth, name=name)
        return data_pairs

    def create_torch_dataset(self, part='train', reshape=None):
        """
        Create a torch dataset wrapper for one part of this dataset.

        If :meth:`supports_random_access` returns ``False``, samples are
        fetched from :meth:`generator`. The index passed to
        :meth:`~torch.utils.data.dataset.Dataset.__getitem__` of the
        returned dataset will be ignored, and parallel data loading (with
        multiple workers) is not applicable.

        If :meth:`supports_random_access` returns `True`, samples are looked
        up using :meth:`get_sample`. For datasets that support parallel calls
        to :meth:`get_sample`, the returned torch dataset can be used by
        multiple workers.

        Parameters
        ----------
        part : {``'train'``, ``'validation'``, ``'test'``}, optional
            The data part. Default is ``'train'``.
        reshape : tuple of (tuple or `None`), optional
            Shapes to which the elements of each sample will be reshaped.
            If `None` is passed for an element, no reshape is applied.
        """
        from torch.utils.data import Dataset as TorchDataset
        import torch

        if self.supports_random_access():
            class RandomAccessTorchDataset(TorchDataset):
                def __init__(self, dataset, part, reshape=None):
                    self.dataset = dataset
                    self.part = part
                    self.reshape = reshape or (
                        (None,) * self.dataset.get_num_elements_per_sample())

                def __len__(self):
                    return self.dataset.get_len(self.part)

                def __getitem__(self, idx):
                    arrays = self.dataset.get_sample(idx, part=self.part)
                    mult_elem = isinstance(arrays, tuple)
                    if not mult_elem:
                        arrays = (arrays,)
                    tensors = []
                    for arr, s in zip(arrays, self.reshape):
                        t = torch.from_numpy(np.asarray(arr))
                        if s is not None:
                            t = t.view(*s)
                        tensors.append(t)
                    return tuple(tensors) if mult_elem else tensors[0]

            dataset = RandomAccessTorchDataset(self, part, reshape=reshape)
        else:
            class GeneratorTorchDataset(TorchDataset):
                def __init__(self, dataset, part, reshape=None):
                    self.part = part
                    self.dataset = dataset
                    self.generator = self.dataset.generator(self.part)
                    self.length = self.dataset.get_len(self.part)
                    self.reshape = reshape or (
                        (None,) * dataset.get_num_elements_per_sample())

                def __len__(self):
                    return self.length

                def __getitem__(self, idx):
                    try:
                        arrays = next(self.generator)
                    except StopIteration:
                        self.generator = self.dataset.generator(self.part)
                        arrays = next(self.generator)
                    mult_elem = isinstance(arrays, tuple)
                    if not mult_elem:
                        arrays = (arrays,)
                    tensors = []
                    for arr, s in zip(arrays, self.reshape):
                        t = torch.from_numpy(np.asarray(arr))
                        if s is not None:
                            t = t.view(*s)
                        tensors.append(t)
                    return tuple(tensors) if mult_elem else tensors[0]

            dataset = GeneratorTorchDataset(self, part, reshape=reshape)

        return dataset

    def create_keras_generator(self, part='train', batch_size=1, shuffle=True,
                               reshape=None):
        """
        Create a keras data generator wrapper for one part of this dataset.

        If :meth:`supports_random_access` returns ``False``, a generator
        wrapping :meth:`generator` is returned. In this case no shuffling is
        performed regardless of the passed `shuffle` parameter. Also, parallel
        data loading (with multiple workers) is not applicable.

        If :meth:`supports_random_access` returns `True`, a
        :class:`tf.keras.utils.Sequence` is returned, which is implemented
        using :meth:`get_sample`. For datasets that support parallel calls to
        :meth:`get_sample`, the returned data generator (sequence) can be used
        by multiple workers.

        Parameters
        ----------
        part : {``'train'``, ``'validation'``, ``'test'``}, optional
            The data part. Default is ``'train'``.
        batch_size : int, optional
            Batch size. Default is 1.
        shuffle : bool, optional
            Whether to shuffle samples each epoch.
            This option has no effect if :meth:`supports_random_access` returns
            ``False``, since in that case samples are fetched directly from
            :meth:`generator`.
            The default is `True`.
        reshape : tuple of (tuple or `None`), optional
            Shapes to which the elements of each sample will be reshaped.
            If `None` is passed for an element, no reshape is applied.
        """
        from tensorflow.keras.utils import Sequence

        if self.supports_random_access():
            class KerasGenerator(Sequence):
                def __init__(self, dataset, part, batch_size, shuffle,
                             reshape=None):
                    self.dataset = dataset
                    self.part = part
                    self.batch_size = batch_size
                    self.shuffle = shuffle
                    self.reshape = reshape or (
                        (None,) * self.dataset.get_num_elements_per_sample())
                    self.data_shape = self.dataset.get_shape()
                    self.on_epoch_end()

                def __len__(self):
                    return ceil(self.dataset.get_len(self.part) /
                                self.batch_size)

                def __getitem__(self, idx):
                    indexes = self.indexes[idx*self.batch_size:
                                           (idx+1)*self.batch_size]
                    # for last batch, indexes has len <= batch_size
                    n_elem = self.dataset.get_num_elements_per_sample()
                    arrays = []
                    for i in range(n_elem):
                        array = np.empty(
                            (len(indexes),) + self.data_shape[i],
                            dtype=self.dataset.space[i].dtype)
                        arrays.append(array)
                    for j, ind in enumerate(indexes):
                        out = tuple([array[j] for array in arrays])
                        self.dataset.get_sample(ind, part=self.part, out=out)
                    for i in range(n_elem):
                        if self.reshape[i] is not None:
                            arrays[i] = arrays[i].reshape(
                                (len(indexes),) + self.reshape[i])
                    return tuple(arrays) if n_elem > 1 else arrays[0]

                def on_epoch_end(self):
                    self.indexes = np.arange(self.dataset.get_len(self.part))
                    if self.shuffle:
                        np.random.shuffle(self.indexes)

            generator = KerasGenerator(self, part, batch_size=batch_size,
                                       shuffle=shuffle, reshape=reshape)

        else:
            def keras_generator(dataset, part, batch_size, reshape=None):
                generator = dataset.generator(part)
                n_elem = dataset.get_num_elements_per_sample()
                num_steps_per_epoch = ceil(dataset.get_len(part) / batch_size)
                if reshape is None:
                    reshape = (None,) * n_elem
                data_shape = dataset.get_shape()
                while True:
                    for k in range(num_steps_per_epoch):
                        batch_size_ = (batch_size if k < num_steps_per_epoch-1
                                       else dataset.get_len(part) % batch_size)
                        arrays = []
                        for i in range(n_elem):
                            array = np.empty(
                                (batch_size_,) + data_shape[i],
                                dtype=dataset.space[i].dtype)
                            arrays.append(array)
                        for j in range(batch_size_):
                            sample = next(generator)
                            if n_elem == 1:
                                sample = (sample,)
                            for i, array in enumerate(arrays):
                                array[j, :] = sample[i]
                        for i in range(n_elem):
                            if reshape[i] is not None:
                                arrays[i] = arrays[i].reshape(
                                    (batch_size_,) + reshape[i])
                        yield tuple(arrays) if n_elem > 1 else arrays[0]

            generator = keras_generator(self, part, batch_size=batch_size,
                                        reshape=reshape)

        return generator

    def get_sample(self, index, part='train', out=None):
        """Get single sample by index.

        Parameters
        ----------
        index : int
            Index of the sample.
        part : {``'train'``, ``'validation'``, ``'test'``}, optional
            The data part. Default is ``'train'``.
        out : array-like or tuple of (array-like or bool) or `None`
            Array(s) (or e.g. odl element(s)) to which the sample is written.
            A tuple should be passed, if the dataset returns two or more arrays
            per sample (i.e. pairs, ...).
            If a tuple element is a bool, it has the following meaning:

                ``True``
                    Create a new array and return it.
                ``False``
                    Do not return this array, i.e. `None` is returned.

        Returns
        -------
        sample : [tuple of ] (array-like or `None`)
            E.g. for a pair dataset: ``(array, None)`` if
            ``out=(True, False)``.
        """
        raise NotImplementedError

    def get_samples(self, key, part='train', out=None):
        """Get samples by slice or range.

        The default implementation calls :meth:`get_sample` if the dataset
        implements it.

        Parameters
        ----------
        key : slice or range
            Indexes of the samples.
        part : {``'train'``, ``'validation'``, ``'test'``}, optional
            The data part. Default is ``'train'``.
        out : array-like or tuple of (array-like or bool) or `None`
            Array(s) (or e.g. odl element(s)) to which the sample is written.
            The first dimension must match the number of samples requested.
            A tuple should be passed, if the dataset returns two or more arrays
            per sample (i.e. pairs, ...).
            If a tuple element is a bool, it has the following meaning:

                ``True``
                    Create a new array and return it.
                ``False``
                    Do not return this array, i.e. `None` is returned.

        Returns
        -------
        samples : [tuple of ] (array-like or `None`)
            If the dataset has multiple arrays per sample, a tuple holding
            arrays is returned.
            E.g. for a pair dataset: ``(array, None)`` if
            ``out=(True, False)``.
            The samples are stacked in the first
            (additional) dimension of each array.
        """
        if self.supports_random_access():
            if isinstance(key, slice):
                key = range(*key.indices(self.get_len(part)))
            if self.get_num_elements_per_sample() == 1:
                if out is None:
                    out = True
                if isinstance(out, bool):
                    samples = np.empty((len(key),) + self.space.shape,
                                       dtype=self.space.dtype) if out else None
                else:
                    samples = out
                if samples is not None:
                    for i, index in enumerate(key):
                        self.get_sample(index, part=part, out=samples[i])
            else:
                if out is None:
                    out = (True,) * self.get_num_elements_per_sample()
                samples = ()
                for out_val, space in zip(out, self.space):
                    if isinstance(out_val, bool):
                        s = np.empty((len(key),) + space.shape,
                                     dtype=space.dtype) if out_val else None
                    else:
                        s = out_val
                    samples = samples + (s,)
                for i, index in enumerate(key):
                    self.get_sample(index, part=part, out=tuple((
                        s[i] if s is not None else None for s in samples)))
            return samples
        raise NotImplementedError

    def supports_random_access(self):
        """Whether random access seems to be supported.

        If the object has the attribute `self.random_access`, its value is
        returned (this is the preferred way for subclasses to indicate whether
        they support random access). Otherwise, a simple duck-type check is
        performed which tries to get the first sample by random access.

        Returns
        -------
        supports : bool
            ``True`` if the dataset supports random access, otherwise
            ``False``.
        """
        try:
            return self.random_access
        except AttributeError:
            try:
                self.get_sample(0)
            except NotImplementedError:
                return False
            return True


class ObservationGroundTruthPairDataset(Dataset):
    """
    Dataset of pairs generated from a ground truth generator by applying a
    forward operator and noise.
    """
    def __init__(self, ground_truth_gen, forward_op, post_processor=None,
                 train_len=None, validation_len=None, test_len=None,
                 domain=None, noise_type=None, noise_kwargs=None,
                 noise_seeds=None):
        """
        Parameters
        ----------
        ground_truth_gen : generator function
            Function returning a generator providing ground truth.
            Must accept a `part` parameter like :meth:`Dataset.generator`.
        forward_op : odl operator
            Forward operator to apply on the ground truth.
        post_processor : odl operator, optional
            Post-processor to apply on the result of the forward operator.
        train_len : int, optional
            Number of training samples.
        validation_len : int, optional
            Number of validation samples.
        test_len : int, optional
            Number of test samples.
        domain : odl space, optional
            Ground truth domain.
            If not specified, it is inferred from `forward_op`.
        noise_type : str, optional
            Noise type. See :class:`~dival.util.odl_utility.NoiseOperator` for
            the list of supported noise types.
        noise_kwargs : dict, optional
            Keyword arguments passed to
            :class:`~dival.util.odl_utility.NoiseOperator`.
        noise_seeds : dict of int, optional
            Seeds to use for random noise generation.
            The part (``'train'``, ...) is the key to the dict.
            If a key is omitted or a value is `None`, no fixed seed is used
            for that part. By default, no fixed seeds are used.
        """
        self.ground_truth_gen = ground_truth_gen
        self.forward_op = forward_op
        self.post_processor = post_processor
        if train_len is not None:
            self.train_len = train_len
        if validation_len is not None:
            self.validation_len = validation_len
        if test_len is not None:
            self.test_len = test_len
        if domain is None:
            domain = self.forward_op.domain
        self.noise_type = noise_type
        self.noise_kwargs = noise_kwargs
        self.noise_seeds = noise_seeds or {}
        range_ = (self.post_processor.range if self.post_processor is not None
                  else self.forward_op.range)
        super().__init__(space=(range_, domain))
        self.shape = (self.space[0].shape, self.space[1].shape)
        self.num_elements_per_sample = 2

    def generator(self, part='train'):
        gt_gen_instance = self.ground_truth_gen(part=part)
        if self.noise_type is not None:
            random_state = np.random.RandomState(self.noise_seeds.get(part))
            noise_op = NoiseOperator(self.forward_op.range, self.noise_type,
                                     noise_kwargs=self.noise_kwargs,
                                     random_state=random_state)
            full_op = noise_op * self.forward_op
        else:
            full_op = self.forward_op
        if self.post_processor is not None:
            full_op = self.post_processor * full_op
        for ground_truth in gt_gen_instance:
            yield (full_op(ground_truth), ground_truth)


class GroundTruthDataset(Dataset):
    """
    Ground truth dataset base class.
    """
    def __init__(self, space=None):
        """
        Parameters
        ----------
        space : :class:`odl.space.base_tensors.TensorSpace`, optional
            The space of the samples.
            It is strongly recommended to set `space` in subclasses, as some
            functionality may depend on it.
        """
        self.num_elements_per_sample = 1
        super().__init__(space=space)

    def create_pair_dataset(self, forward_op, post_processor=None,
                            noise_type=None, noise_kwargs=None,
                            noise_seeds=None):
        """
        The parameters are a subset of those of
        :meth:`ObservationGroundTruthPairDataset.__init__`.
        """
        try:
            train_len = self.get_train_len()
        except NotImplementedError:
            train_len = None
        try:
            validation_len = self.get_validation_len()
        except NotImplementedError:
            validation_len = None
        try:
            test_len = self.get_test_len()
        except NotImplementedError:
            test_len = None
        dataset = ObservationGroundTruthPairDataset(
            self.generator, forward_op, post_processor=post_processor,
            train_len=train_len, validation_len=validation_len,
            test_len=test_len, noise_type=noise_type,
            noise_kwargs=noise_kwargs, noise_seeds=noise_seeds)
        return dataset
