# -*- coding: utf-8 -*-
import unittest
from io import StringIO
from unittest.mock import patch
import json
import numpy as np
import odl
from dival import get_standard_dataset
from dival.data import DataPairs
from dival.evaluation import TaskTable
from dival.measure import PSNR, SSIM
from dival.reconstructors.odl_reconstructors import FBPReconstructor

np.random.seed(1)


class TestTaskTable(unittest.TestCase):
    def test(self):
        reco_space = odl.uniform_discr(
            min_pt=[-64, -64], max_pt=[64, 64], shape=[128, 128])
        phantom = odl.phantom.shepp_logan(reco_space, modified=True)
        geometry = odl.tomo.parallel_beam_geometry(reco_space, 30)
        ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl='skimage')
        proj_data = ray_trafo(phantom)
        observation = np.asarray(
            proj_data +
            np.random.normal(loc=0., scale=2., size=proj_data.shape))
        test_data = DataPairs(observation, phantom)
        tt = TaskTable()
        fbp_reconstructor = FBPReconstructor(ray_trafo, hyper_params={
            'filter_type': 'Hann',
            'frequency_scaling': 0.8})
        tt.append(fbp_reconstructor, test_data, measures=[PSNR, SSIM])
        tt.run()
        self.assertGreater(
            tt.results.results['measure_values'][0, 0]['psnr'][0], 15.)

    def test_option_save_best_reconstructor(self):
        dataset = get_standard_dataset('ellipses', impl='skimage')
        test_data = dataset.get_data_pairs('validation', 1)
        tt = TaskTable()
        fbp_reconstructor = FBPReconstructor(dataset.ray_trafo)
        hyper_param_choices = {'filter_type': ['Ram-Lak', 'Hann'],
                               'frequency_scaling': [0.1, 0.5, 1.0]}
        known_best_choice = {'filter_type': 'Hann',
                             'frequency_scaling': 0.5}
        path = 'dummypath'
        options = {'save_best_reconstructor': {'path': path,
                                               'measure': PSNR}}
        tt.append(fbp_reconstructor, test_data, measures=[PSNR],
                  hyper_param_choices=hyper_param_choices,
                  options=options)

        class ExtStringIO(StringIO):
            def __init__(self, ext, f, *args, **kwargs):
                self.ext = ext
                self.f = f
                super().__init__(*args, **kwargs)
                self.ext[self.f] = self.getvalue()

            def close(self):
                self.ext[self.f] = self.getvalue()
                super().close()
        ext = {}
        with patch('dival.reconstructors.reconstructor.open',
                   lambda f, *a, **kw: ExtStringIO(ext, f)):
            tt.run()
        self.assertIn(path + '_hyper_params.json', ext)
        self.assertDictEqual(json.loads(ext[path + '_hyper_params.json']),
                             known_best_choice)


if __name__ == '__main__':
    unittest.main()
