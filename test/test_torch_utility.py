# -*- coding: utf-8 -*-
import unittest
try:
    import torch
    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    TORCH_AVAILABLE = False
else:
    from dival.util.torch_utility import load_state_dict_convert_data_parallel

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
