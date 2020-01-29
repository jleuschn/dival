# -*- coding: utf-8 -*-
import unittest
from io import StringIO
from unittest.mock import patch
import json
import odl
from odl.solvers.util.callback import CallbackApply, CallbackStore
from odl.operator.default_ops import ScalingOperator
from dival.reconstructors.reconstructor import (Reconstructor,
                                                StandardIterativeReconstructor,
                                                FunctionReconstructor)
from dival.reconstructors.odl_reconstructors import LandweberReconstructor


class TestReconstructor(unittest.TestCase):
    def test_constructor(self):
        class DummyReconstructor(Reconstructor):
            HYPER_PARAMS = {'hp1': {'default': 1.},
                            'hp2': {'default': 2.}}
        reco_space = odl.uniform_discr([0, 0], [1, 1], (1, 1))
        observation_space = odl.uniform_discr([0, 0], [1, 1], (1, 1))
        name = 'dummy'
        hyper_params = {'hp1': 0.5}
        r = DummyReconstructor(reco_space=reco_space,
                               observation_space=observation_space,
                               name=name, hyper_params=hyper_params)
        self.assertEqual(r.reco_space, reco_space)
        self.assertEqual(r.observation_space, observation_space)
        self.assertEqual(r.name, name)
        self.assertEqual(r.hyper_params,
                         {k: hyper_params.get(k, v['default']) for k, v in
                          DummyReconstructor.HYPER_PARAMS.items()})

    def test_reconstruct(self):
        domain = odl.uniform_discr([0, 0], [1, 1], (1, 1))
        # test all signatures of `_reconstruct`

        class DummyReconstructor(Reconstructor):
            def _reconstruct(self, observation, out):
                out[:] = observation
        r = DummyReconstructor(reco_space=domain)
        observation = domain.one()
        out = r.reconstruct(observation)
        self.assertEqual(out, observation)
        out = domain.zero()
        r.reconstruct(observation, out)
        self.assertEqual(out, observation)

        class DummyReconstructor2(Reconstructor):
            def _reconstruct(self, observation):
                return observation.copy()
        r2 = DummyReconstructor2(reco_space=domain)
        observation = domain.one()
        out = r2.reconstruct(observation)
        self.assertEqual(out, observation)
        out = domain.zero()
        r2.reconstruct(observation, out)
        self.assertEqual(out, observation)

        class DummyReconstructor3(Reconstructor):
            def _reconstruct(self, observation, out=None):
                if out is None:
                    return observation.copy()
                else:
                    out[:] = observation
        r3 = DummyReconstructor3(reco_space=domain)
        observation = domain.one()
        out = r3.reconstruct(observation)
        self.assertEqual(out, observation)
        out = domain.zero()
        r3.reconstruct(observation, out)
        self.assertEqual(out, observation)

    def test_hyper_params_properties(self):
        class DummyReconstructor(Reconstructor):
            HYPER_PARAMS = {'hp1': {'default': 1.},
                            'hp2': {'default': 2.}}
        r = DummyReconstructor()
        self.assertTrue(hasattr(r, 'hp1'))
        self.assertTrue(hasattr(r, 'hp2'))
        self.assertEqual(r.hp1, 1.)
        r.hp1 = 3.
        self.assertEqual(r.hyper_params['hp1'], 3.)
        r.hyper_params['hp2'] = 4.
        self.assertEqual(r.hp2, 4.)

    def test_save_hyper_params(self):
        class DummyReconstructor(Reconstructor):
            HYPER_PARAMS = {'hp1': {'default': 1.},
                            'hp2': {'default': 2.}}
        r = DummyReconstructor()

        class ExtStringIO(StringIO):
            def __init__(self, ext, *args, **kwargs):
                self.ext = ext
                super().__init__(*args, **kwargs)
                self.ext['str'] = self.getvalue()

            def close(self):
                self.ext['str'] = self.getvalue()
                super().close()
        ext = {'str': ''}
        with patch('dival.reconstructors.reconstructor.open',
                   lambda *a, **kw: ExtStringIO(ext, ext['str'])):
            r.save_hyper_params('dummyfilename')
        self.assertDictEqual(r.hyper_params, json.loads(ext['str']))
        r.hyper_params['hp1'] = 3.
        ext = {'str': ''}
        with patch('dival.reconstructors.reconstructor.open',
                   lambda *a, **kw: ExtStringIO(ext, ext['str'])):
            r.save_hyper_params('dummyfilename')
        self.assertDictEqual(r.hyper_params, json.loads(ext['str']))


class TestStandardIterativeReconstructor(unittest.TestCase):
    def test_reconstruct(self):
        domain = odl.uniform_discr([0, 0], [1, 1], (1, 1))
        observation = domain.one()

        # reconstruct 1., iterates 0., 0.5, 0.75, 0.875, ...

        class DummyReconstructor(StandardIterativeReconstructor):
            def _setup(self, observation):
                self.setup_var = 'dummy_val'

            def _compute_iterate(self, observation, reco_previous, out):
                out[:] = 0.5 * (observation + reco_previous)

        r = DummyReconstructor(reco_space=domain,
                               hyper_params={'iterations': 100})
        reco = r.reconstruct(observation)
        self.assertAlmostEqual(reco.asarray(), observation.asarray(),
                               delta=1e-7)

        r2 = DummyReconstructor(reco_space=domain,
                                hyper_params={'iterations': 2})
        reco2 = r2.reconstruct(observation)
        self.assertNotAlmostEqual(reco2.asarray(), observation.asarray(),
                                  delta=0.2)

        # test init values

        r3 = DummyReconstructor(reco_space=domain,
                                hyper_params={'iterations': 2},
                                x0=domain.element(0.5))
        reco3 = r3.reconstruct(observation)
        self.assertAlmostEqual(reco3.asarray(), observation.asarray(),
                               delta=0.2)

        r4 = DummyReconstructor(reco_space=domain,
                                hyper_params={'iterations': 2},
                                x0=domain.zero())
        reco4 = r4.reconstruct(observation, x0=domain.element(0.5))
        self.assertAlmostEqual(reco4.asarray(), observation.asarray(),
                               delta=0.2)

        r5 = DummyReconstructor(reco_space=domain,
                                hyper_params={'iterations': 2})
        reco5 = domain.element(0.5)
        r5.reconstruct(observation, out=reco5)
        self.assertAlmostEqual(reco5.asarray(), observation.asarray(),
                               delta=0.2)

        r6 = DummyReconstructor(reco_space=domain,
                                hyper_params={'iterations': 2},
                                x0=domain.zero())
        reco6 = domain.element(0.5)
        r6.reconstruct(observation, out=reco6)
        self.assertNotAlmostEqual(reco6.asarray(), observation.asarray(),
                                  delta=0.2)

    def test_callback_assignment(self):
        domain = odl.uniform_discr([0, 0], [1, 1], (1, 1))
        op = ScalingOperator(domain, 0.5)
        callback = CallbackApply(lambda x: None)
        r = LandweberReconstructor(op, domain.one(), 7, callback=callback)
        self.assertEqual(r.callback, callback)
        r.reconstruct(domain.one())
        self.assertEqual(r.callback, callback)
        callback2 = CallbackApply(lambda x: 1.)
        r.reconstruct(domain.one(), callback=callback2)
        self.assertEqual(r.callback, callback)

    def test_callback(self):
        domain = odl.uniform_discr([0, 0], [1, 1], (1, 1))
        op = ScalingOperator(domain, 0.5)
        niter = 7
        iters = []
        callback = CallbackStore(iters)
        r = LandweberReconstructor(op, domain.one(), niter, callback=callback)
        r.reconstruct(domain.one())
        self.assertEqual(len(iters), niter)


class TestFunctionReconstructor(unittest.TestCase):
    def test(self):
        def fun(y, k, return_zero=True):
            if return_zero:
                k = 0
            x = k*y
            return x
        fun_args = [0.5]
        fun_kwargs = {'return_zero': False}
        r = FunctionReconstructor(fun, name='function',
                                  fun_args=fun_args, fun_kwargs=fun_kwargs)
        domain = odl.uniform_discr([0, 0], [1, 1], (1, 1))
        observation = domain.one()
        out = r.reconstruct(observation)
        self.assertEqual(out, fun(observation, *fun_args, **fun_kwargs))
        out = domain.zero()
        r.reconstruct(observation, out)
        self.assertEqual(out, fun(observation, *fun_args, **fun_kwargs))


if __name__ == '__main__':
    unittest.main()
