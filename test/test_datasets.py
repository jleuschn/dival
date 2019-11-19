# -*- coding: utf-8 -*-
import unittest
import numpy as np
import odl
from dival.datasets.dataset import Dataset


class TestDataset(unittest.TestCase):
    def test(self):
        TRAIN_LEN = 10
        VALIDATION_LEN = 1
        TEST_LEN = 1

        class DummyDataset(Dataset):
            def __init__(self):
                self.space = odl.uniform_discr([0, 0], [1, 1], (1, 1))
                self.train_len = TRAIN_LEN
                self.validation_len = VALIDATION_LEN
                self.test_len = TEST_LEN

            def generator(self, part='train'):
                for i in range(self.get_len(part)):
                    yield self.space.element(i)

        d = DummyDataset()
        self.assertEqual(d.get_len('train'), TRAIN_LEN)
        self.assertEqual(d.get_len('validation'), VALIDATION_LEN)
        self.assertEqual(d.get_len('test'), TEST_LEN)
        for part in ['train', 'validation', 'test']:
            for i, s in zip(range(d.get_len(part)), d.generator(part)):
                self.assertEqual(s, d.space.element(i))

    def test_generator(self):
        TRAIN_LEN = 10
        VALIDATION_LEN = 1
        TEST_LEN = 1

        class DummyDataset(Dataset):
            def __init__(self):
                self.space = odl.uniform_discr([0, 0], [1, 1], (1, 1))
                self.train_len = TRAIN_LEN
                self.validation_len = VALIDATION_LEN
                self.test_len = TEST_LEN

            def get_sample(self, index, part='train', out=None):
                if out is None:
                    out = True
                if isinstance(out, bool):
                    out = self.space.zero() if out else None
                if out is not None:
                    out[:] = index
                return out

        d = DummyDataset()
        self.assertEqual(d.get_len('train'), TRAIN_LEN)
        self.assertEqual(d.get_len('validation'), VALIDATION_LEN)
        self.assertEqual(d.get_len('test'), TEST_LEN)
        for part in ['train', 'validation', 'test']:
            for i, s in zip(range(d.get_len(part)), d.generator(part)):
                self.assertEqual(s, d.space.element(i))

    def test_get_samples(self):
        class DummyDataset(Dataset):
            def __init__(self):
                self.space = odl.uniform_discr([0, 0], [1, 1], (1, 1))
                self.train_len = 20
                self.validation_len = 2
                self.test_len = 2

            def get_sample(self, index, part='train', out=None):
                if index >= self.get_len(part):
                    raise ValueError('index out of bound')
                if out is None:
                    out = True
                if isinstance(out, bool):
                    out = self.space.zero() if out else None
                if out is not None:
                    out[:] = index
                return out

        d = DummyDataset()
        for part in ['train', 'validation', 'test']:
            samples = d.get_samples(range(d.get_len(part)), part=part)
            self.assertEqual(type(samples), np.ndarray)
            self.assertEqual(samples[0].shape, d.space.shape)
            self.assertEqual(samples.dtype, d.space.dtype)
            self.assertEqual(len(samples), d.get_len(part))
            for i, s in enumerate(samples):
                self.assertEqual(s, d.space.element(i))

        class DummyDataset2(Dataset):
            def __init__(self):
                self.space = (odl.uniform_discr([0, 0], [1, 1], (4, 4)),
                              odl.uniform_discr([0, 0], [1, 1], (1, 1)))
                self.train_len = 20
                self.validation_len = 2
                self.test_len = 2

            def get_sample(self, index, part='train', out=None):
                if index >= self.get_len(part):
                    raise ValueError('index out of bound')
                if out is None:
                    out = (True, True)
                out0, out1 = out
                if isinstance(out0, bool):
                    out0 = self.space[0].zero() if out0 else None
                if isinstance(out[1], bool):
                    out1 = self.space[1].zero() if out1 else None
                if out0 is not None:
                    out0[:] = self.space[0].one() * index
                if out1 is not None:
                    out1[:] = self.space[1].one() * index
                return (out0, out1)

        d2 = DummyDataset2()
        for part in ['train', 'validation', 'test']:
            STEP = 2
            samples = d2.get_samples(slice(None, d2.get_len(part), STEP),
                                     part=part)
            self.assertEqual(type(samples), tuple)
            for s, space in zip(samples, d2.space):
                self.assertEqual(type(s), np.ndarray)
                self.assertEqual(s[0].shape, space.shape)
                self.assertEqual(s.dtype, space.dtype)
                self.assertEqual(len(s), d2.get_len(part) // STEP)
                for i, s_ in zip(range(0, d2.get_len(part), STEP), s):
                    self.assertTrue(np.all(np.equal(s_, space.one() * i)))

    def test_getters(self):
        class DummyDataset(Dataset):
            def __init__(self, TRAIN_LEN=4, VALIDATION_LEN=2,
                         TEST_LEN=2, SHAPE=(1, 1)):
                self.space = odl.uniform_discr([0, 0], [1, 1], SHAPE)
                self.train_len = TRAIN_LEN
                self.validation_len = VALIDATION_LEN
                self.test_len = TEST_LEN

            def generator(self, part='train'):
                for i in range(self.get_len(part)):
                    yield self.space.element(i)

        class DummyDataset2(Dataset):
            def __init__(self, TRAIN_LEN=4, VALIDATION_LEN=2,
                         TEST_LEN=2, SHAPE=((1, 1), (1, 1))):
                self.space = (odl.uniform_discr([0, 0], [1, 1], SHAPE[0]),
                              odl.uniform_discr([0, 0], [1, 1], SHAPE[1]))
                self.train_len = TRAIN_LEN
                self.validation_len = VALIDATION_LEN
                self.test_len = TEST_LEN

            def generator(self, part='train'):
                for i in range(self.get_len(part)):
                    yield (self.space[0].element(i), self.space[1].element(i))

        class DummyDataset3(Dataset):
            def __init__(self, NUM_ELEMENTS_PER_SAMPLE=2):
                assert NUM_ELEMENTS_PER_SAMPLE >= 2
                self.num_elements_per_sample = NUM_ELEMENTS_PER_SAMPLE

            def generator(self, part='train'):
                space = odl.uniform_discr([0, 0], [1, 1], SHAPE)
                for i in range(self.get_len(part)):
                    yield (space.element(i),) * self.num_elements_per_sample

        TRAIN_LEN = 10
        VALIDATION_LEN = 2
        TEST_LEN = 1
        SHAPE = (7, 3)
        d = DummyDataset(TRAIN_LEN=TRAIN_LEN, VALIDATION_LEN=VALIDATION_LEN,
                         TEST_LEN=TEST_LEN, SHAPE=SHAPE)
        self.assertEqual(d.get_train_len(), TRAIN_LEN)
        self.assertEqual(d.get_validation_len(), VALIDATION_LEN)
        self.assertEqual(d.get_test_len(), TEST_LEN)
        self.assertEqual(d.get_num_elements_per_sample(), 1)
        self.assertEqual(d.get_shape(), SHAPE)

        TRAIN_LEN = 10
        VALIDATION_LEN = 2
        TEST_LEN = 1
        SHAPE = ((7, 7), (3, 3))
        d2 = DummyDataset2(TRAIN_LEN=TRAIN_LEN, VALIDATION_LEN=VALIDATION_LEN,
                           TEST_LEN=TEST_LEN, SHAPE=SHAPE)
        self.assertEqual(d2.get_train_len(), TRAIN_LEN)
        self.assertEqual(d2.get_validation_len(), VALIDATION_LEN)
        self.assertEqual(d2.get_test_len(), TEST_LEN)
        self.assertEqual(d2.get_num_elements_per_sample(), 2)
        self.assertEqual(d2.get_shape(), SHAPE)

        NUM_ELEMENTS_PER_SAMPLE = 3
        d3 = DummyDataset3(NUM_ELEMENTS_PER_SAMPLE=NUM_ELEMENTS_PER_SAMPLE)
        self.assertEqual(d3.get_num_elements_per_sample(),
                         NUM_ELEMENTS_PER_SAMPLE)

    def test_supports_random_access(self):
        class DummyDataset(Dataset):
            def generator(self, part='train'):
                space = odl.uniform_discr([0, 0], [1, 1], (1, 1))
                for i in range(self.get_len(part)):
                    yield (space.element(i),) * 2

        d = DummyDataset()
        self.assertEqual(d.supports_random_access(), False)

        class DummyDataset2(Dataset):
            def get_sample(self, index, part='train', out=None):
                space = odl.uniform_discr([0, 0], [1, 1], (1, 1))
                if out is None:
                    out = True
                if isinstance(out, bool):
                    out = space.zero() if out else None
                if out is not None:
                    out[:] = index
                return out

        d2 = DummyDataset2()
        self.assertEqual(d2.supports_random_access(), True)


if __name__ == '__main__':
    unittest.main()
