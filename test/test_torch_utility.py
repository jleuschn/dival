# -*- coding: utf-8 -*-
import unittest
try:
    import torch
except ImportError:
    TORCH_AVAILABLE = False
else:
    TORCH_AVAILABLE = True
    from dival.util.torch_utility import (
        load_state_dict_convert_data_parallel, TOMOSIPO_AVAILABLE,
        TorchRayTrafoParallel2DModule, TorchRayTrafoParallel2DAdjointModule)
import numpy as np
from dival import get_standard_dataset
try:
    import astra
except ImportError:
    ASTRA_CUDA_AVAILABLE = False
else:
    ASTRA_CUDA_AVAILABLE = astra.use_cuda()

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
