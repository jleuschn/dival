# -*- coding: utf-8 -*-
import unittest
import odl
from odl.solvers.util.callback import CallbackApply, CallbackStore
from odl.operator.default_ops import ScalingOperator
from dival.reconstructors.reconstructor import (Reconstructor,
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


class TestIterativeReconstructor(unittest.TestCase):
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
        self.assertEqual(r.callback, callback2)

    def test_callback(self):
        domain = odl.uniform_discr([0, 0], [1, 1], (1, 1))
        op = ScalingOperator(domain, 0.5)
        niter = 7
        result = []
        callback = CallbackStore(result)
        r = LandweberReconstructor(op, domain.one(), niter, callback=callback)
        r.reconstruct(domain.one())
        self.assertEqual(len(result), niter)


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
