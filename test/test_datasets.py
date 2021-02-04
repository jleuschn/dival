# -*- coding: utf-8 -*-
import unittest
import os
from itertools import islice
import numpy as np
import odl
from dival import get_standard_dataset
from dival.datasets.dataset import Dataset
from dival.datasets.lodopab_dataset import LoDoPaBDataset
from dival.datasets.cached_dataset import CachedDataset, generate_cache_files
from dival.datasets.angle_subset_dataset import AngleSubsetDataset


class TestDataset(unittest.TestCase):
    def setUp(self):
        self.TRAIN_LEN = 20
        self.VALIDATION_LEN = 2
        self.TEST_LEN = 2

        class SequenceGeneratorDataset(Dataset):
            def __init__(self, train_len=self.TRAIN_LEN,
                         validation_len=self.VALIDATION_LEN,
                         test_len=self.TEST_LEN):
                self.space = odl.uniform_discr([0, 0], [1, 1], (1, 1))
                self.train_len = train_len
                self.validation_len = validation_len
                self.test_len = test_len

            def generator(self, part='train'):
                for i in range(self.get_len(part)):
                    yield self.space.element(i)

        class SequenceGeneratorDataset2(Dataset):
            def __init__(self, train_len=self.TRAIN_LEN,
                         validation_len=self.VALIDATION_LEN,
                         test_len=self.TEST_LEN):
                self.space = (odl.uniform_discr([0, 0], [1, 1], (3, 4)),
                              odl.uniform_discr([0, 0], [1, 1], (2, 1)))
                self.train_len = train_len
                self.validation_len = validation_len
                self.test_len = test_len

            def generator(self, part='train'):
                for i in range(self.get_len(part)):
                    yield (self.space[0].one() * i,
                           self.space[1].one() * i)

        class SequenceRandomAccessDataset(Dataset):
            def __init__(self, train_len=self.TRAIN_LEN,
                         validation_len=self.VALIDATION_LEN,
                         test_len=self.TEST_LEN):
                self.space = odl.uniform_discr([0, 0], [1, 1], (1, 1))
                self.train_len = train_len
                self.validation_len = validation_len
                self.test_len = test_len

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

        class SequenceRandomAccessDataset2(Dataset):
            def __init__(self, train_len=self.TRAIN_LEN,
                         validation_len=self.VALIDATION_LEN,
                         test_len=self.TEST_LEN):
                self.space = (odl.uniform_discr([0, 0], [1, 1], (4, 4)),
                              odl.uniform_discr([0, 0], [1, 1], (1, 1)))
                self.train_len = train_len
                self.validation_len = validation_len
                self.test_len = test_len

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

        self.dseqgen1 = SequenceGeneratorDataset()
        self.dseqgen2 = SequenceGeneratorDataset2()
        self.dseq1 = SequenceRandomAccessDataset()
        self.dseq2 = SequenceRandomAccessDataset2()

    def test(self):
        self.assertEqual(
            self.dseqgen1.get_len('train'), self.TRAIN_LEN)
        self.assertEqual(
            self.dseqgen1.get_len('validation'), self.VALIDATION_LEN)
        self.assertEqual(
            self.dseqgen1.get_len('test'), self.TEST_LEN)
        for part in ['train', 'validation', 'test']:
            for i, s in enumerate(self.dseqgen1.generator(part)):
                self.assertEqual(s, self.dseqgen1.space.element(i))

    def test_generator(self):
        for part in ['train', 'validation', 'test']:
            for i, s in enumerate(self.dseq1.generator(part)):
                self.assertEqual(s, self.dseq1.space.element(i))
                self.assertEqual(s, self.dseq1.get_sample(i))

    def test_get_samples(self):
        for part in ['train', 'validation', 'test']:
            samples = self.dseq1.get_samples(
                range(self.dseq1.get_len(part)), part=part)
            self.assertEqual(type(samples), np.ndarray)
            self.assertEqual(samples[0].shape, self.dseq1.space.shape)
            self.assertEqual(samples.dtype, self.dseq1.space.dtype)
            self.assertEqual(len(samples), self.dseq1.get_len(part))
            for i, s in enumerate(samples):
                self.assertEqual(s, self.dseq1.space.element(i))
                self.assertEqual(s, self.dseq1.get_sample(i))

        for part in ['train', 'validation', 'test']:
            STEP = 2
            samples = self.dseq2.get_samples(
                slice(None, self.dseq2.get_len(part), STEP), part=part)
            self.assertEqual(type(samples), tuple)
            for s, space in zip(samples, self.dseq2.space):
                self.assertEqual(type(s), np.ndarray)
                self.assertEqual(s[0].shape, space.shape)
                self.assertEqual(s.dtype, space.dtype)
                self.assertEqual(len(s), self.dseq2.get_len(part) // STEP)
                for i, s_ in zip(range(0, self.dseq2.get_len(part), STEP), s):
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

    def test_get_data_pairs(self):
        PART = 'train'
        N = 10
        data_pairs = self.dseqgen2.get_data_pairs(PART, n=N)
        gen = self.dseqgen2.generator(part=PART)
        for (obs2, gt2), (obs, gt) in zip(data_pairs, gen):
            self.assertEqual(obs2, obs)
            self.assertEqual(gt2, gt)

        data_pairs2 = self.dseqgen2.get_data_pairs_per_index(
            PART, index=list(range(N)))
        gen = self.dseqgen2.generator(part=PART)
        for (obs2, gt2), (obs, gt) in zip(data_pairs2, gen):
            self.assertEqual(obs2, obs)
            self.assertEqual(gt2, gt)

        INDEX = [4, 3, 3, 5]
        data_pairs2 = self.dseqgen2.get_data_pairs_per_index(PART, index=INDEX)
        data_pairs_all = self.dseqgen2.get_data_pairs(PART, n=max(INDEX)+1)
        for j, i in enumerate(INDEX):
            obs2, gt2 = data_pairs2[j]
            obs, gt = data_pairs_all[i]
            self.assertEqual(obs2, obs)
            self.assertEqual(gt2, gt)

        PART = 'train'
        N = 10
        data_pairs = self.dseq2.get_data_pairs(PART, n=N)
        gen = self.dseq2.generator(part=PART)
        for (obs2, gt2), (obs, gt) in zip(data_pairs, gen):
            self.assertEqual(obs2, obs)
            self.assertEqual(gt2, gt)

        data_pairs2 = self.dseq2.get_data_pairs_per_index(
            PART, index=list(range(N)))
        gen = self.dseq2.generator(part=PART)
        for (obs2, gt2), (obs, gt) in zip(data_pairs2, gen):
            self.assertEqual(obs2, obs)
            self.assertEqual(gt2, gt)

        INDEX = [4, 3, 3, 5]
        data_pairs2 = self.dseq2.get_data_pairs_per_index(PART, index=INDEX)
        data_pairs_all = self.dseq2.get_data_pairs(PART, n=max(INDEX)+1)
        for j, i in enumerate(INDEX):
            obs2, gt2 = data_pairs2[j]
            obs, gt = data_pairs_all[i]
            self.assertEqual(obs2, obs)
            self.assertEqual(gt2, gt)


class TestLoDoPaBDataset(unittest.TestCase):
    def test_patient_ids(self):
        if not LoDoPaBDataset.check_for_lodopab():
            return
        d = LoDoPaBDataset(impl='skimage')
        if d.rel_patient_ids is not None:
            for part in ['train', 'validation', 'test']:
                self.assertEqual(len(d.rel_patient_ids[part]), d.get_len(part))
                self.assertTrue(np.all(np.unique(d.rel_patient_ids[part]) ==
                                       range(d.get_num_patients(part))))
                self.assertTrue(np.all(np.diff(d.rel_patient_ids[part][
                    LoDoPaBDataset.get_idx_sorted_by_patient()[part]]) >= 0))
            d2 = LoDoPaBDataset(sorted_by_patient=True, impl='skimage')
            REL_PATIENT_ID = 42
            ifp = d.get_indices_for_patient(REL_PATIENT_ID, part)
            ifp2 = d2.get_indices_for_patient(REL_PATIENT_ID, part)
            self.assertGreater(len(ifp), 0)
            self.assertEqual(len(ifp), len(ifp2))
            for i, i2 in zip(ifp[:3], ifp2[:3]):
                self.assertEqual(d.get_sample(i, part),
                                 d2.get_sample(i2, part))

    def test_get_samples(self):
        if not LoDoPaBDataset.check_for_lodopab():
            return
        KEY = range(420, 423)
        d = LoDoPaBDataset(impl='skimage')
        for part in ['train', 'validation', 'test']:
            samples = [d.get_sample(i, part) for i in KEY]
            samples2 = d.get_samples(KEY, part)
            for (s_obs, s_gt), s2_obs, s2_gt in zip(samples, samples2[0],
                                                    samples2[1]):
                self.assertTrue(np.all(np.asarray(s_obs) == s2_obs))
                self.assertTrue(np.all(np.asarray(s_gt) == s2_gt))
        if d.rel_patient_ids is not None:
            d2 = LoDoPaBDataset(sorted_by_patient=True, impl='skimage')
            for part in ['train', 'validation', 'test']:
                samples = [d2.get_sample(i, part) for i in KEY]
                samples2 = d2.get_samples(KEY, part)
                for (s_obs, s_gt), s2_obs, s2_gt in zip(samples, samples2[0],
                                                        samples2[1]):
                    self.assertTrue(np.all(np.asarray(s_obs) == s2_obs))
                    self.assertTrue(np.all(np.asarray(s_gt) == s2_gt))

    def test_generator(self):
        if not LoDoPaBDataset.check_for_lodopab():
            return
        NUM_SAMPLES = 3
        d = LoDoPaBDataset(impl='skimage')
        for part in ['train', 'validation', 'test']:
            samples = [d.get_sample(i, part) for i in range(NUM_SAMPLES)]
            samples2 = [s for s in islice(d.generator(part), NUM_SAMPLES)]
            for (s_obs, s_gt), (s2_obs, s2_gt) in zip(samples, samples2):
                self.assertTrue(np.all(np.asarray(s_obs) == s2_obs))
                self.assertTrue(np.all(np.asarray(s_gt) == s2_gt))
        if d.rel_patient_ids is not None:
            d2 = LoDoPaBDataset(sorted_by_patient=True, impl='skimage')
            for part in ['train', 'validation', 'test']:
                samples = [d2.get_sample(i, part) for i in range(NUM_SAMPLES)]
                samples2 = [s for s in islice(d2.generator(part), NUM_SAMPLES)]
                for (s_obs, s_gt), (s2_obs, s2_gt) in zip(samples, samples2):
                    self.assertTrue(np.all(np.asarray(s_obs) == s2_obs))
                    self.assertTrue(np.all(np.asarray(s_gt) == s2_gt))


class TestCachedDataset(unittest.TestCase):
    def setUp(self):
        class DummyGeneratorDataset(Dataset):
            def __init__(self):
                self.space = odl.uniform_discr([0, 0], [1, 1], (1, 1))
                self.train_len = 20
                self.validation_len = 2
                self.test_len = 2

            def generator(self, part='train'):
                for i in range(self.get_len(part)):
                    yield self.space.element(i)

        class DummyGeneratorDataset2(Dataset):
            def __init__(self):
                self.space = (odl.uniform_discr([0, 0], [1, 1], (4, 4)),
                              odl.uniform_discr([0, 0], [1, 1], (1, 1)))
                self.train_len = 20
                self.validation_len = 2
                self.test_len = 2

            def generator(self, part='train'):
                for i in range(self.get_len(part)):
                    yield (self.space[0].one() * i,
                           self.space[1].one() * i)

        class DummyRandomAccessDataset(Dataset):
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

        class DummyRandomAccessDataset2(Dataset):
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

        self.dgen1 = DummyGeneratorDataset()
        self.dgen2 = DummyGeneratorDataset2()
        self.d1 = DummyRandomAccessDataset()
        self.d2 = DummyRandomAccessDataset2()

    def test_generator(self):
        cache_files = {'train': 'train.npy',
                       'validation': 'validation.npy'}
        size = {'train': 10, 'validation': 1}
        generate_cache_files(self.dgen1, cache_files, size=size)
        cd = CachedDataset(self.dgen1, self.dgen1.space, cache_files,
                           size=size)
        self.assertEqual(cd.get_len('train'), size['train'])
        self.assertEqual(cd.get_len('validation'), size['validation'])
        self.assertEqual(cd.get_len('test'), self.dgen1.get_len('test'))
        self.assertEqual(cd.get_num_elements_per_sample(),
                         self.dgen1.get_num_elements_per_sample())
        for part in ['train', 'validation', 'test']:
            len_counter = 0
            for s, cs in zip(self.dgen1.generator(part), cd.generator(part)):
                self.assertTrue(np.all(np.asarray(cs) == np.asarray(s)))
                len_counter += 1
            self.assertEqual(len_counter, cd.get_len(part))
        for f in cache_files.values():
            os.remove(f)

        cache_files = {
            'train': ('train_obs.npy', 'train_gt.npy'),
            'validation': ('validation_obs.npy', 'validation_gt.npy')}
        size = {'train': 10, 'validation': 1}
        generate_cache_files(self.dgen2, cache_files, size=size)
        cd = CachedDataset(self.dgen2, self.dgen2.space, cache_files,
                           size=size)
        self.assertEqual(cd.get_len('train'), size['train'])
        self.assertEqual(cd.get_len('validation'), size['validation'])
        self.assertEqual(cd.get_len('test'), self.dgen2.get_len('test'))
        self.assertEqual(cd.get_num_elements_per_sample(),
                         self.dgen2.get_num_elements_per_sample())
        for part in ['train', 'validation', 'test']:
            len_counter = 0
            for s, cs in zip(self.dgen2.generator(part), cd.generator(part)):
                self.assertEqual(len(cs), len(s))
                for s_, cs_ in zip(s, cs):
                    self.assertTrue(np.all(np.asarray(cs_) == np.asarray(s_)))
                len_counter += 1
            self.assertEqual(len_counter, cd.get_len(part))
        for files in cache_files.values():
            for f in files:
                os.remove(f)

        # previous test was using caches for all elements, now test `None`
        cache_files = {
            'train': ('train_obs.npy', None),
            'validation': ('validation_obs.npy', None)}
        size = {'train': 10, 'validation': 1}
        generate_cache_files(self.dgen2, cache_files, size=size)
        cd = CachedDataset(self.dgen2, self.dgen2.space, cache_files,
                           size=size)
        self.assertEqual(cd.get_len('train'), size['train'])
        self.assertEqual(cd.get_len('validation'), size['validation'])
        self.assertEqual(cd.get_len('test'), self.dgen2.get_len('test'))
        self.assertEqual(cd.get_num_elements_per_sample(),
                         self.dgen2.get_num_elements_per_sample())
        for part in ['train', 'validation', 'test']:
            len_counter = 0
            for s, cs in zip(self.dgen2.generator(part), cd.generator(part)):
                self.assertEqual(len(cs), len(s))
                for s_, cs_ in zip(s, cs):
                    self.assertTrue(np.all(np.asarray(cs_) == np.asarray(s_)))
                len_counter += 1
            self.assertEqual(len_counter, cd.get_len(part))
        for files in cache_files.values():
            for f in files:
                if f is not None:
                    os.remove(f)

    def test_get_sample(self):
        cache_files = {'train': 'train.npy',
                       'validation': 'validation.npy'}
        size = {'train': 10, 'validation': 1}
        generate_cache_files(self.d1, cache_files, size=size)
        cd = CachedDataset(self.d1, self.d1.space, cache_files, size=size)
        self.assertEqual(cd.get_len('train'), size['train'])
        self.assertEqual(cd.get_len('validation'), size['validation'])
        self.assertEqual(cd.get_len('test'), self.d1.get_len('test'))
        self.assertEqual(cd.get_num_elements_per_sample(),
                         self.d1.get_num_elements_per_sample())
        for part in ['train', 'validation', 'test']:
            for i in range(size.get(part, self.d1.get_len(part))):
                s = self.d1.get_sample(i, part=part)
                cs = cd.get_sample(i, part=part)
                self.assertTrue(np.all(np.asarray(cs) == np.asarray(s)))
        for f in cache_files.values():
            os.remove(f)

        cache_files = {
            'train': ('train_obs.npy', 'train_gt.npy'),
            'validation': ('validation_obs.npy', 'validation_gt.npy')}
        size = {'train': 10, 'validation': 1}
        generate_cache_files(self.d2, cache_files, size=size)
        cd = CachedDataset(self.d2, self.d2.space, cache_files, size=size)
        self.assertEqual(cd.get_len('train'), size['train'])
        self.assertEqual(cd.get_len('validation'), size['validation'])
        self.assertEqual(cd.get_len('test'), self.d2.get_len('test'))
        self.assertEqual(cd.get_num_elements_per_sample(),
                         self.d2.get_num_elements_per_sample())
        for part in ['train', 'validation', 'test']:
            for i in range(size.get(part, self.d2.get_len(part))):
                s = self.d2.get_sample(i, part=part)
                cs = cd.get_sample(i, part=part)
                self.assertEqual(len(cs), len(s))
                for s_, cs_ in zip(s, cs):
                    self.assertTrue(np.all(np.asarray(cs_) == np.asarray(s_)))
        for files in cache_files.values():
            for f in files:
                os.remove(f)

    def test_get_samples(self):
        cache_files = {'train': 'train.npy',
                       'validation': 'validation.npy'}
        size = {'train': 10, 'validation': 1}
        generate_cache_files(self.d1, cache_files, size=size)
        cd = CachedDataset(self.d1, self.d1.space, cache_files, size=size)
        self.assertEqual(cd.get_len('train'), size['train'])
        self.assertEqual(cd.get_len('validation'), size['validation'])
        self.assertEqual(cd.get_len('test'), self.d1.get_len('test'))
        self.assertEqual(cd.get_num_elements_per_sample(),
                         self.d1.get_num_elements_per_sample())
        for part in ['train', 'validation', 'test']:
            range_ = range(size.get(part, self.d1.get_len(part)))
            s = self.d1.get_samples(range_, part=part)
            cs = cd.get_samples(range_, part=part)
            self.assertEqual(cs.shape, s.shape)
            self.assertTrue(np.all(cs == s))
        for f in cache_files.values():
            os.remove(f)

        cache_files = {
            'train': ('train_obs.npy', 'train_gt.npy'),
            'validation': ('validation_obs.npy', 'validation_gt.npy')}
        size = {'train': 10, 'validation': 1}
        generate_cache_files(self.d2, cache_files, size=size)
        cd = CachedDataset(self.d2, self.d2.space, cache_files, size=size)
        self.assertEqual(cd.get_len('train'), size['train'])
        self.assertEqual(cd.get_len('validation'), size['validation'])
        self.assertEqual(cd.get_len('test'), self.d2.get_len('test'))
        self.assertEqual(cd.get_num_elements_per_sample(),
                         self.d2.get_num_elements_per_sample())
        for part in ['train', 'validation', 'test']:
            range_ = range(size.get(part, self.d2.get_len(part)))
            s = self.d2.get_samples(range_, part=part)
            cs = cd.get_samples(range_, part=part)
            self.assertEqual(len(cs), len(s))
            for s_, cs_ in zip(s, cs):
                self.assertEqual(cs_.shape, s_.shape)
                self.assertTrue(np.all(cs_ == s_))
        for files in cache_files.values():
            for f in files:
                os.remove(f)


class TestAngleSubsetDataset(unittest.TestCase):
    def test_get_ray_trafo(self):
        d = get_standard_dataset('ellipses', fixed_seeds=True, impl='skimage')
        for angle_indices in (
                range(0, d.shape[0][0], 2),
                range(0, d.shape[0][0] // 2),
                range(d.shape[0][0] // 2, d.shape[0][0]),
                np.concatenate(
                    [np.arange(0, int(d.shape[0][0] * 1/4)),
                     np.arange(int(d.shape[0][0] * 3/4), d.shape[0][0])])):
            asd = AngleSubsetDataset(d, angle_indices)
            ray_trafo = asd.get_ray_trafo(impl='skimage')
            self.assertEqual(ray_trafo.range.shape[0], len(angle_indices))
            angles_subset = d.get_ray_trafo(impl='skimage').geometry.angles[
                np.asarray(angle_indices)]
            self.assertEqual(ray_trafo.geometry.angles.shape, angles_subset.shape)
            self.assertTrue(np.all(ray_trafo.geometry.angles == angles_subset))

    def test_generator(self):
        d = get_standard_dataset('ellipses', fixed_seeds=True, impl='skimage')
        angle_indices = range(0, d.shape[0][0], 2)
        asd = AngleSubsetDataset(d, angle_indices)
        test_data_asd = asd.get_data_pairs('train', 3)
        test_data = d.get_data_pairs('train', 3)
        for (obs_asd, gt_asd), (obs, gt) in zip(test_data_asd, test_data):
            obs_subset = np.asarray(obs)[np.asarray(angle_indices), :]
            self.assertEqual(obs_asd.shape, obs_subset.shape)
            self.assertEqual(gt_asd.shape, gt.shape)
            self.assertTrue(np.all(np.asarray(obs_asd) == obs_subset))
            self.assertTrue(np.all(np.asarray(gt_asd) == np.asarray(gt)))

    def test_get_sample(self):
        # TODO: use dataset that is always available instead of lodopab
        if not LoDoPaBDataset.check_for_lodopab():
            return
        d = LoDoPaBDataset(impl='skimage')
        angle_indices = range(0, d.shape[0][0], 2)
        asd = AngleSubsetDataset(d, angle_indices)
        for i in range(3):
            obs_asd, gt_asd = asd.get_sample(i)
            obs, gt = d.get_sample(i)
            obs_subset = np.asarray(obs)[np.asarray(angle_indices), :]
            self.assertEqual(obs_asd.shape, obs_subset.shape)
            self.assertEqual(gt_asd.shape, gt.shape)
            self.assertTrue(np.all(np.asarray(obs_asd) == obs_subset))
            self.assertTrue(np.all(np.asarray(gt_asd) == np.asarray(gt)))

    def test_get_samples(self):
        # TODO: use dataset that is always available instead of lodopab
        if not LoDoPaBDataset.check_for_lodopab():
            return
        d = LoDoPaBDataset(impl='skimage')
        angle_indices = range(0, d.shape[0][0], 2)
        asd = AngleSubsetDataset(d, angle_indices)
        obs_arr_asd, gt_arr_asd = asd.get_samples(range(3))
        obs_arr, gt_arr = d.get_samples(range(3))
        obs_arr_subset = np.asarray(obs_arr)[:, np.asarray(angle_indices), :]
        self.assertEqual(obs_arr_asd.shape, obs_arr_subset.shape)
        self.assertEqual(gt_arr_asd.shape, gt_arr.shape)
        self.assertTrue(np.all(np.asarray(obs_arr_asd) == obs_arr_subset))
        self.assertTrue(np.all(np.asarray(gt_arr_asd) == np.asarray(gt_arr)))


if __name__ == '__main__':
    unittest.main()
