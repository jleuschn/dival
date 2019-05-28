# -*- coding: utf-8 -*-
"""Provides the dataset base classes.
"""
import numpy as np
from dival.util.odl_utility import NoiseOperator


class Dataset():
    def generator(self, test=False):
        """Yield data.

        Parameters
        ----------
        test : bool, optional
            Whether to yield train or test data.
            Default is ``False``, i.e. train data.

        Yields
        ------
        data : odl element or sequence of odl elements
            Sample of the dataset.
        """
        raise NotImplementedError

    def get_train_generator(self):
        return self.generator()

    def get_test_generator(self):
        return self.generator(test=True)

    def get_train_len(self):
        """Return the number of elements the train generator will yield."""
        try:
            return self.train_len
        except AttributeError:
            raise NotImplementedError

    def get_test_len(self):
        """Return the number of elements the test generator will yield."""
        try:
            return self.test_len
        except AttributeError:
            raise NotImplementedError

    def get_shape(self):
        """Return the shape of each element."""
        try:
            return self.shape
        except AttributeError:
            raise NotImplementedError

    def create_torch_dataset(self, test=False):
        import torch

        class _GeneratorTorchDataset(torch.Dataset):
            def __init__(self, generator, length):
                self.generator = generator
                self.length = length

            def __len__(self):
                return self.length

            def __getitem__(self):
                return next(self.generator)

        gen = self.get_test_generator() if test else self.get_train_generator()
        length = self.get_test_len() if test else self.get_train_len()
        dataset = _GeneratorTorchDataset(gen, length)
        return dataset

    def create_tensorflow_dataset(self, test=False):
        import tensorflow as tf

        gen = self.get_test_generator() if test else self.get_train_generator()
        dataset = tf.data.Dataset.from_generator(
            gen, tf.float32, tf.TensorShape(self.get_shape()))
        return dataset


class ObservationGroundTruthPairDataset(Dataset):
    def __init__(self, ground_truth_gen, forward_op, train_len=None,
                 test_len=None, shape=None, noise_type=None,
                 noise_kwargs=None, noise_seed=None):
        self.ground_truth_gen = ground_truth_gen
        self.forward_op = forward_op
        if train_len is not None:
            self.train_len = train_len
        if test_len is not None:
            self.test_len = test_len
        if shape is not None:
            self.shape = shape
        if noise_type is not None:
            noise_random_state = np.random.RandomState(noise_seed)
            noise_op = NoiseOperator(self.forward_op.range, noise_type,
                                     noise_kwargs=noise_kwargs,
                                     random_state=noise_random_state)
            self.forward_op = noise_op * self.forward_op

    def generator(self, test=False):
        gt_gen_instance = self.ground_truth_gen(test=test)
        while True:
            try:
                ground_truth = next(gt_gen_instance)
            except StopIteration:
                break
            yield (self.forward_op(ground_truth), ground_truth)


class GroundTruthDataset(Dataset):
    def create_pair_dataset(self, forward_op, noise_type=None,
                            noise_kwargs=None, noise_seed=None):
        try:
            train_len = self.get_train_len()
        except NotImplementedError:
            train_len = None
        try:
            test_len = self.get_test_len()
        except NotImplementedError:
            test_len = None
        try:
            shape = (forward_op.range.shape, self.get_shape())
        except NotImplementedError:
            shape = None
        dataset = ObservationGroundTruthPairDataset(
            self.generator, forward_op, train_len=train_len, test_len=test_len,
            shape=shape, noise_type=noise_type, noise_kwargs=noise_kwargs,
            noise_seed=noise_seed)
        return dataset
