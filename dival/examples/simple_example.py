from dival.util.odl_utility import uniform_discr_element
from dival.data import DataPairs
from dival.evaluation import TaskTable
from dival.measure import L2
from dival import Reconstructor
import numpy as np

np.random.seed(1)

ground_truth = uniform_discr_element([0, 1, 2, 3, 4, 5, 6])
observation = ground_truth + 1
observation += np.random.normal(size=observation.shape)
test_data = DataPairs(observation, ground_truth, name='x + 1 + normal')
eval_tt = TaskTable()


class MinusOneReconstructor(Reconstructor):
    def reconstruct(self, observation):
        return observation - 1


reconstructor = MinusOneReconstructor(name='y-1')
eval_tt.append(reconstructor=reconstructor, test_data=test_data,
               measures=[L2])
results = eval_tt.run()
results.plot_reconstruction(0)
print(results)
