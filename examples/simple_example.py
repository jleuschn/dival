from data import uniform_discr_element
from evaluation import EvaluationTaskTable, TestData, L2Measure
from reconstruction import Reconstructor

ground_truth = uniform_discr_element([0, 1, 2, 3, 4, 5, 6])
observation = uniform_discr_element([1, 2, 3, 4, 5, 6, 7])
test_data = TestData(observation, ground_truth)
eval_tt = EvaluationTaskTable()


class MinusOneReconstructor(Reconstructor):
    def reconstruct(self, observation_data):
        return observation_data - 1


reconstructor = MinusOneReconstructor()
eval_tt.append(test_data, reconstructor, [L2Measure()])
results = eval_tt.run()
results.plot_reconstruction(0)
print(results.measure_values)
