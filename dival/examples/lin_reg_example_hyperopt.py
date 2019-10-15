import numpy as np
from hyperopt import hp
from odl.discr import uniform_discr
from dival.datasets.dataset import Dataset
from dival.evaluation import TaskTable
from dival.measure import L2, PSNR
from dival.reconstructors.regression_reconstructors import LinRegReconstructor
from dival.hyper_param_search import optimize_hyper_params

# %% data
observation_space = uniform_discr(-0.5, 6.5, 7)
reco_space = uniform_discr(-0.5, 11.5, 12)

np.random.seed(0)


class LinearDataset(Dataset):
    def __init__(self, observation_space, reco_space, train_len=10000,
                 validation_len=1000, test_len=1000):
        self.train_len = train_len
        self.validation_len = validation_len
        self.test_len = test_len
        self.observation_space = observation_space
        self.reco_space = reco_space
        self.space = (self.observation_space, self.reco_space)
        self.shape = (self.observation_space.shape, self.reco_space.shape)
        self.forward_matrix = np.random.rand(self.shape[0][0],
                                             self.shape[1][0])

    def generator(self, part='train'):
        seed = 42
        if part == 'validation':
            seed = 2
        elif part == 'test':
            seed = 1
        rs = np.random.RandomState(seed)
        for _ in range(self.get_len(part=part)):
            x = rs.rand(self.shape[1][0])
            y = (np.dot(self.forward_matrix, x) +
                 0.4 * rs.normal(scale=.1, size=self.shape[0]))
            yield (self.observation_space.element(y),
                   self.reco_space.element(x))


dataset = LinearDataset(observation_space, reco_space)
validation_data = dataset.get_data_pairs('validation', 10)
test_data = dataset.get_data_pairs('test', 10)

# %% task table and reconstructors
eval_tt = TaskTable()

reconstructor = LinRegReconstructor(observation_space=observation_space,
                                    reco_space=reco_space)
optimize_hyper_params(reconstructor,
                      validation_data=validation_data,
                      measure=L2,
                      dataset=dataset,
                      hyperopt_max_evals_retrain=100,
                      HYPER_PARAMS_override={
                          'l2_regularization': {
                              'method': 'hyperopt',
                              'hyperopt_options': {
                                  'space': hp.loguniform('l2_regularization',
                                                         0., np.log(1e9))
                              }
                          }})
print('optimized l2 reg. coeff.: {}'.format(
    reconstructor.hyper_params['l2_regularization']))

eval_tt.append(reconstructor=reconstructor, test_data=test_data,
               dataset=dataset, measures=[L2, PSNR])

# %% run task table
results = eval_tt.run()
print(results)

# %% plot reconstructions
fig = results.plot_all_reconstructions(test_index=range(3),
                                       fig_size=(9, 4), vrange='individual')
