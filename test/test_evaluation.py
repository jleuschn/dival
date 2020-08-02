# -*- coding: utf-8 -*-
import unittest
from io import StringIO
from unittest.mock import patch
import json
import numpy as np
import odl
from odl.solvers.util.callback import CallbackStore
from dival import get_standard_dataset
from dival.data import DataPairs
from dival.evaluation import TaskTable
from dival.measure import PSNR, SSIM
from dival.reconstructors.reconstructor import StandardIterativeReconstructor
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
        dataset = get_standard_dataset('ellipses', fixed_seeds=True,
                                       impl='skimage')
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

    def test_option_save_best_reconstructor_reuse_iterates(self):
        # test 'save_best_reconstructor' option together with 'iterations' in
        # hyper_param_choices, because `run` has performance optimization for
        # it (with the default argument ``reuse_iterates=True``)
        domain = odl.uniform_discr([0, 0], [1, 1], (2, 2))
        ground_truth = domain.element([[1, 0], [0, 1]])
        observation = domain.element([[0, 0], [0, 0]])

        # Reconstruct [[1, 0], [0, 1]], iterates are
        # [[0, 0], [0, 0]], [[.1, .1], [.1, .1]], [[.2, .2], [.2, .2]], ...
        # Best will be [[.5, .5], [.5, .5]].

        class DummyReconstructor(StandardIterativeReconstructor):
            def _setup(self, observation):
                self.setup_var = 'dummy_val'

            def _compute_iterate(self, observation, reco_previous, out):
                out[:] = reco_previous + 0.1

        test_data = DataPairs([observation], [ground_truth])
        tt = TaskTable()
        r = DummyReconstructor(reco_space=domain)
        hyper_param_choices = {'iterations': list(range(10))}
        known_best_choice = {'iterations': 5}
        path = 'dummypath'
        options = {'save_best_reconstructor': {'path': path,
                                               'measure': PSNR}}
        tt.append(r, test_data, measures=[PSNR],
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

    def test_option_save_iterates(self):
        domain = odl.uniform_discr([0, 0], [1, 1], (1, 1))
        ground_truth = domain.one()
        observation = domain.one()

        # reconstruct 1., iterates 0., 0.5, 0.75, 0.875, ...

        class DummyReconstructor(StandardIterativeReconstructor):
            def _setup(self, observation):
                self.setup_var = 'dummy_val'

            def _compute_iterate(self, observation, reco_previous, out):
                out[:] = 0.5 * (observation + reco_previous)

        test_data = DataPairs([observation], [ground_truth])
        tt = TaskTable()
        r = DummyReconstructor(reco_space=domain)
        hyper_param_choices = {'iterations': [10]}
        options = {'save_iterates': True}
        tt.append(r, test_data, hyper_param_choices=hyper_param_choices,
                  options=options)

        results = tt.run()
        self.assertAlmostEqual(
            1., results.results['misc'][0, 0]['iterates'][0][2][0, 0],
            delta=0.2)
        self.assertNotAlmostEqual(
            1., results.results['misc'][0, 0]['iterates'][0][1][0, 0],
            delta=0.2)

    def test_iterations_hyper_param_choices(self):
        # test 'iterations' in hyper_param_choices, because `run` has
        # performance optimization for it
        domain = odl.uniform_discr([0, 0], [1, 1], (1, 1))
        ground_truth = domain.one()
        observation = domain.one()

        # reconstruct 1., iterates 0., 0.5, 0.75, 0.875, ...

        class DummyReconstructor(StandardIterativeReconstructor):
            def _setup(self, observation):
                self.setup_var = 'dummy_val'

            def _compute_iterate(self, observation, reco_previous, out):
                out[:] = 0.5 * (observation + reco_previous)

        test_data = DataPairs([observation], [ground_truth])
        tt = TaskTable()
        r = DummyReconstructor(reco_space=domain)
        hyper_param_choices = {'iterations': [2, 3, 10]}
        tt.append(r, test_data, hyper_param_choices=hyper_param_choices)

        iters = []
        r.callback = CallbackStore(iters)
        results = tt.run(reuse_iterates=True)
        self.assertAlmostEqual(
            1., results.results['reconstructions'][0, 1][0][0, 0],
            delta=0.2)
        self.assertNotAlmostEqual(
            1., results.results['reconstructions'][0, 0][0][0, 0],
            delta=0.2)
        self.assertEqual(len(iters), max(hyper_param_choices['iterations']))
        print(results.results['misc'])

        iters2 = []
        r.callback = CallbackStore(iters2)
        results2 = tt.run(reuse_iterates=False)
        self.assertAlmostEqual(
            1., results2.results['reconstructions'][0, 1][0][0, 0],
            delta=0.2)
        self.assertNotAlmostEqual(
            1., results2.results['reconstructions'][0, 0][0][0, 0],
            delta=0.2)
        self.assertEqual(len(iters2), sum(hyper_param_choices['iterations']))


if __name__ == '__main__':
    unittest.main()
