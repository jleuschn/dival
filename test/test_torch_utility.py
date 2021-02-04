# -*- coding: utf-8 -*-
import unittest
try:
    import torch
except ImportError:
    TORCH_AVAILABLE = False
else:
    TORCH_AVAILABLE = True
    from dival.util.torch_utility import (
        RandomAccessTorchDataset, GeneratorTorchDataset,
        load_state_dict_convert_data_parallel, TOMOSIPO_AVAILABLE,
        TorchRayTrafoParallel2DModule, TorchRayTrafoParallel2DAdjointModule)
import numpy as np
import odl
from dival import get_standard_dataset
from dival.datasets import Dataset
try:
    import astra
except ImportError:
    ASTRA_CUDA_AVAILABLE = False
else:
    ASTRA_CUDA_AVAILABLE = astra.use_cuda()


@unittest.skipUnless(TORCH_AVAILABLE, 'PyTorch not available')
class TestRandomAccessTorchDataset(unittest.TestCase):
    def setUp(self):
        self.TRAIN_LEN = 20
        self.VALIDATION_LEN = 2
        self.TEST_LEN = 2

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

        self.dseq2 = SequenceRandomAccessDataset2()

    def test(self):
        for part in ['train', 'validation', 'test']:
            torch_dataset = self.dseq2.create_torch_dataset(
                part=part, reshape=((1,) + self.dseq2.space[0].shape,
                                    (1,) + self.dseq2.space[1].shape))
            assert len(torch_dataset) == self.dseq2.get_len(part)
            for (obs, gt), (obs_torch, gt_torch) in zip(
                    self.dseq2.generator(part=part), torch_dataset):
                assert obs_torch.shape == (1,) + self.dseq2.space[0].shape
                assert gt_torch.shape == (1,) + self.dseq2.space[1].shape
                assert np.all(obs_torch[0].numpy() == np.asarray(obs))
                assert np.all(gt_torch[0].numpy() == np.asarray(gt))
        transform = lambda x: (torch.nn.functional.pad(x[0], (2, 2)),
                               torch.nn.functional.pad(x[1], (1, 1)))
        for part in ['train', 'validation', 'test']:
            torch_dataset = self.dseq2.create_torch_dataset(
                part=part, reshape=((1,) + self.dseq2.space[0].shape,
                                    (1,) + self.dseq2.space[1].shape),
                transform=transform)
            assert len(torch_dataset) == self.dseq2.get_len(part)
            for (obs, gt), (obs_torch, gt_torch) in zip(
                    self.dseq2.generator(part=part), torch_dataset):
                assert obs_torch.shape == (
                    1,
                    self.dseq2.space[0].shape[0],
                    self.dseq2.space[0].shape[1] + 2*2)
                assert gt_torch.shape == (
                    1,
                    self.dseq2.space[1].shape[0],
                    self.dseq2.space[1].shape[1] + 2*1)
                assert np.all(obs_torch[0, :, 2:-2].numpy() == np.asarray(obs))
                assert np.all(gt_torch[0, :, 1:-1].numpy() == np.asarray(gt))

@unittest.skipUnless(TORCH_AVAILABLE, 'PyTorch not available')
class TestGeneratorTorchDataset(unittest.TestCase):
    def setUp(self):
        self.TRAIN_LEN = 20
        self.VALIDATION_LEN = 2
        self.TEST_LEN = 2

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

        self.dseqgen2 = SequenceGeneratorDataset2()

    def test(self):
        for part in ['train', 'validation', 'test']:
            torch_dataset = self.dseqgen2.create_torch_dataset(
                part=part, reshape=((1,) + self.dseqgen2.space[0].shape,
                                    (1,) + self.dseqgen2.space[1].shape))
            assert len(torch_dataset) == self.dseqgen2.get_len(part)
            for (obs, gt), (obs_torch, gt_torch) in zip(
                    self.dseqgen2.generator(part=part), torch_dataset):
                assert obs_torch.shape == (1,) + self.dseqgen2.space[0].shape
                assert gt_torch.shape == (1,) + self.dseqgen2.space[1].shape
                assert np.all(obs_torch[0].numpy() == np.asarray(obs))
                assert np.all(gt_torch[0].numpy() == np.asarray(gt))
        transform = lambda x: (torch.nn.functional.pad(x[0], (2, 2)),
                               torch.nn.functional.pad(x[1], (1, 1)))
        for part in ['train', 'validation', 'test']:
            torch_dataset = self.dseqgen2.create_torch_dataset(
                part=part, reshape=((1,) + self.dseqgen2.space[0].shape,
                                    (1,) + self.dseqgen2.space[1].shape),
                transform=transform)
            assert len(torch_dataset) == self.dseqgen2.get_len(part)
            for (obs, gt), (obs_torch, gt_torch) in zip(
                    self.dseqgen2.generator(part=part), torch_dataset):
                assert obs_torch.shape == (
                    1,
                    self.dseqgen2.space[0].shape[0],
                    self.dseqgen2.space[0].shape[1] + 2*2)
                assert gt_torch.shape == (
                    1,
                    self.dseqgen2.space[1].shape[0],
                    self.dseqgen2.space[1].shape[1] + 2*1)
                assert np.all(obs_torch[0, :, 2:-2].numpy() == np.asarray(obs))
                assert np.all(gt_torch[0, :, 1:-1].numpy() == np.asarray(gt))

@unittest.skipUnless(
    TORCH_AVAILABLE and TOMOSIPO_AVAILABLE and ASTRA_CUDA_AVAILABLE,
    'PyTorch or tomosipo or ASTRA+CUDA not available')
class TestTorchRayTrafoParallel2DModule(unittest.TestCase):
    def test(self):
        dataset = get_standard_dataset('ellipses', fixed_seeds=True,
                                       impl='astra_cuda')
        ray_trafo = dataset.get_ray_trafo(impl='astra_cuda')
        module = TorchRayTrafoParallel2DModule(ray_trafo)
        for batch_size, channels in [(1, 3),
                                     (5, 1),
                                     (2, 3)]:
            test_data = dataset.get_data_pairs(part='train',
                                               n=batch_size * channels)
            torch_in = (torch.from_numpy(np.asarray(test_data.ground_truth))
                        .view(batch_size, channels, *dataset.shape[1]))
            torch_out = module(torch_in).view(-1, *dataset.shape[0])
            for i, odl_in in enumerate(test_data.ground_truth):
                odl_out = ray_trafo(odl_in)
                self.assertTrue(np.allclose(
                    torch_out[i].detach().cpu().numpy(), odl_out, rtol=1.e-2))

    def testGradient(self):
        dataset = get_standard_dataset('ellipses', fixed_seeds=True,
                                       impl='astra_cuda')
        ray_trafo = dataset.get_ray_trafo(impl='astra_cuda')
        module = TorchRayTrafoParallel2DModule(ray_trafo)
        batch_size, channels = 2, 3
        torch_in_ = torch.ones(
            (batch_size * channels,) + dataset.shape[1],
            requires_grad=True)
        torch_in = torch_in_.view(batch_size, channels, *dataset.shape[1])
        torch_out = module(torch_in).view(-1, *dataset.shape[0])
        for i in range(batch_size * channels):
            for j in range(0, dataset.shape[0][0], 3):  # angle
                for k in range(0, dataset.shape[0][1], 12):  # detector pos
                    odl_value_in = np.zeros(dataset.shape[0])
                    odl_value_in[j, k] = 1.
                    odl_grad = ray_trafo.adjoint(odl_value_in)
                    torch_in_.grad = None
                    torch_out[i, j, k].backward(retain_graph=True)
                    torch_grad_np = (
                        torch_in_.grad[i].detach().cpu().numpy())
                    # very rough check for maximum error
                    self.assertTrue(np.allclose(torch_grad_np, odl_grad,
                        rtol=1.))
                    non_zero = np.nonzero(np.asarray(odl_grad))
                    if np.any(odl_grad):  # there seem to be cases where
                                          # a pixel has no influence on the
                                          # gradient
                        # tighter check for mean error
                        self.assertLess(np.mean(np.abs(
                            torch_grad_np[non_zero] - odl_grad[non_zero])
                            / np.abs(odl_grad[non_zero])), 1e-3)

@unittest.skipUnless(
    TORCH_AVAILABLE and TOMOSIPO_AVAILABLE and ASTRA_CUDA_AVAILABLE,
    'PyTorch or tomosipo or ASTRA+CUDA not available')
class TestTorchRayTrafoParallel2DAdjointModule(unittest.TestCase):
    def test(self):
        dataset = get_standard_dataset('ellipses', fixed_seeds=True,
                                       impl='astra_cuda')
        ray_trafo = dataset.get_ray_trafo(impl='astra_cuda')
        module = TorchRayTrafoParallel2DAdjointModule(ray_trafo)
        for batch_size, channels in [(1, 3),
                                     (5, 1),
                                     (2, 3)]:
            test_data = dataset.get_data_pairs(part='train',
                                               n=batch_size * channels)
            torch_in = (torch.from_numpy(np.asarray(test_data.observations))
                        .view(batch_size, channels, *dataset.shape[0]))
            torch_out = module(torch_in).view(-1, *dataset.shape[1])
            for i, odl_in in enumerate(test_data.observations):
                odl_out = ray_trafo.adjoint(odl_in)
                self.assertTrue(np.allclose(
                    torch_out[i].detach().cpu().numpy(), odl_out, rtol=1.e-4))

    def testGradient(self):
        dataset = get_standard_dataset('ellipses', fixed_seeds=True,
                                       impl='astra_cuda')
        ray_trafo = dataset.get_ray_trafo(impl='astra_cuda')
        module = TorchRayTrafoParallel2DAdjointModule(ray_trafo)
        batch_size, channels = 2, 2
        torch_in_ = torch.ones(
            (batch_size * channels,) + dataset.shape[0],
            requires_grad=True)
        torch_in = torch_in_.view(batch_size, channels, *dataset.shape[0])
        torch_out = module(torch_in).view(-1, *dataset.shape[1])
        for i in range(batch_size * channels):
            for j in range(0, dataset.shape[1][0], 9):  # x
                for k in range(0, dataset.shape[1][1], 9):  # y
                    odl_value_in = np.zeros(dataset.shape[1])
                    odl_value_in[j, k] = 1.
                    odl_grad = ray_trafo.adjoint.adjoint(odl_value_in)
                    torch_in_.grad = None
                    torch_out[i, j, k].backward(retain_graph=True)
                    torch_grad_np = (
                        torch_in_.grad[i].detach().cpu().numpy())
                    # very rough check for maximum error
                    self.assertTrue(np.allclose(torch_grad_np, odl_grad,
                        rtol=1.))
                    non_zero = np.nonzero(np.asarray(odl_grad))
                    if np.any(odl_grad):  # there seem to be cases where
                                          # a pixel has no influence on the
                                          # gradient
                        # tighter check for mean error
                        self.assertLess(np.mean(np.abs(
                            torch_grad_np[non_zero] - odl_grad[non_zero])
                            / np.abs(odl_grad[non_zero])), 1e-2)

@unittest.skipUnless(
    TORCH_AVAILABLE and TOMOSIPO_AVAILABLE and ASTRA_CUDA_AVAILABLE,
    'PyTorch or tomosipo or ASTRA+CUDA not available')
class TestGetTorchRayTrafoParallel2d(unittest.TestCase):
    def test(self):
        pass

@unittest.skipUnless(
    TORCH_AVAILABLE and TOMOSIPO_AVAILABLE and ASTRA_CUDA_AVAILABLE,
    'PyTorch or tomosipo or ASTRA+CUDA not available')
class TestGetTorchRayTrafoParallel2dAdjoint(unittest.TestCase):
    def test(self):
        pass

@unittest.skipUnless(TORCH_AVAILABLE, 'PyTorch not available')
class TestLoadStateDictConvertDataParallel(unittest.TestCase):
    def setUp(self):
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 5, 3),
            torch.nn.Conv2d(5, 1, 3))
        self.model_parallel = torch.nn.DataParallel(self.model)

    def test(self):
        state_dict = self.model.state_dict()
        state_dict_parallel = self.model_parallel.state_dict()
        try:
            load_state_dict_convert_data_parallel(self.model,
                                                  state_dict)
        except RuntimeError:
            self.fail()
        try:
            load_state_dict_convert_data_parallel(self.model,
                                                  state_dict_parallel)
        except RuntimeError:
            self.fail()
        try:
            load_state_dict_convert_data_parallel(self.model_parallel,
                                                  state_dict)
        except RuntimeError:
            self.fail()
        try:
            load_state_dict_convert_data_parallel(self.model_parallel,
                                                  state_dict_parallel)
        except RuntimeError:
            self.fail()
        state_dict_missing = {
            k: v for i, (k, v) in enumerate(state_dict.items()) if i >= 1}
        state_dict_parallel_missing = {
            k: v for i, (k, v) in enumerate(state_dict_parallel.items())
            if i >= 1}
        with self.assertRaises(RuntimeError):
            load_state_dict_convert_data_parallel(
                self.model, state_dict_missing)
        with self.assertRaises(RuntimeError):
            load_state_dict_convert_data_parallel(
                self.model, state_dict_parallel_missing)
        with self.assertRaises(RuntimeError):
            load_state_dict_convert_data_parallel(
                self.model_parallel, state_dict_missing)
        with self.assertRaises(RuntimeError):
            load_state_dict_convert_data_parallel(
                self.model_parallel, state_dict_parallel_missing)
        state_dict_unexpected = {
            '{}_unexpected'.format(k): v for k, v in state_dict.items()}
        state_dict_parallel_unexpected = {
            '{}_unexpected'.format(k): v for k, v in
            state_dict_parallel.items()}
        with self.assertRaises(RuntimeError):
            load_state_dict_convert_data_parallel(
                self.model, state_dict_unexpected)
        with self.assertRaises(RuntimeError):
            load_state_dict_convert_data_parallel(
                self.model, state_dict_parallel_unexpected)
        with self.assertRaises(RuntimeError):
            load_state_dict_convert_data_parallel(
                self.model_parallel, state_dict_unexpected)
        with self.assertRaises(RuntimeError):
            load_state_dict_convert_data_parallel(
                self.model_parallel, state_dict_parallel_unexpected)

if __name__ == '__main__':
    unittest.main()
