# -*- coding: utf-8 -*-
import unittest
import odl
from odl.solvers.util.callback import CallbackApply, CallbackStore
from odl.operator.default_ops import ScalingOperator
from dival.reconstructors.odl_reconstructors import LandweberReconstructor


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


if __name__ == '__main__':
    unittest.main()
