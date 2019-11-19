# -*- coding: utf-8 -*-
import unittest
import odl
from dival.data import DataPairs


class TestDataPairs(unittest.TestCase):
    def test(self):
        reco_space = odl.uniform_discr([0, 0], [1, 1], (2, 2))
        observation_space = odl.uniform_discr([0, 0], [1, 1], (1, 1))
        ground_truth = [reco_space.one(), reco_space.zero()]
        observations = [observation_space.one(), observation_space.zero()]
        data_pairs = DataPairs(observations, ground_truth)
        for obs_true, gt_true, (obs, gt) in zip(observations,
                                                ground_truth,
                                                data_pairs):
            self.assertEqual(obs, obs_true)
            self.assertEqual(gt, gt_true)
        self.assertEqual(len(data_pairs), len(observations))
        for obs_true, gt_true, (obs, gt) in zip(observations[::-1],
                                                ground_truth[::-1],
                                                data_pairs[::-1]):
            self.assertEqual(obs, obs_true)
            self.assertEqual(gt, gt_true)


if __name__ == '__main__':
    unittest.main()
