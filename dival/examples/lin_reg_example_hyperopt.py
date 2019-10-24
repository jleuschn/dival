import numpy as np
from hyperopt import hp
from odl.discr import uniform_discr
from dival.datasets.dataset import Dataset
from dival.evaluation import TaskTable
from dival.measure import L2, PSNR
from dival.reconstructors.regression_reconstructors import LinRegReconstructor
from dival.hyper_param_search import optimize_hyper_params

# %% data
observation_space = uniform_discr(-0.5, 69.5, 70)
reco_space = uniform_discr(-0.5, 79.5, 71)

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
                 rs.normal(scale=0.01, size=self.shape[0]))
            yield (self.observation_space.element(y),
                   self.reco_space.element(x))


dataset = LinearDataset(observation_space, reco_space)
validation_data = dataset.get_data_pairs('validation', 10)
test_data = dataset.get_data_pairs('test', 10)

# %% reconstructor and hyper parameter search
reconstructor = LinRegReconstructor(observation_space=observation_space,
                                    reco_space=reco_space)

optimize_hyper_params(reconstructor,
                      validation_data=validation_data,
                      measure=L2,
                      dataset=dataset,
                      hyperopt_max_evals_retrain=10,
                      HYPER_PARAMS_override={
                          'l2_regularization': {
                              'method': 'hyperopt',
                              'hyperopt_options': {
                                  'space': hp.loguniform(
                                      'l2_regularization',
                                      np.log(.001), np.log(2.))
                              }
                          }})

# =============================================================================
# # alternative method: 'grid_search'
# optimize_hyper_params(reconstructor,
#                       validation_data=validation_data,
#                       measure=L2,
#                       dataset=dataset,
#                       HYPER_PARAMS_override={
#                           'l2_regularization': {
#                               'method': 'grid_search',
#                               'range': [0., 2.],
#                               'grid_search_options': {
#                                   'num_samples': 10
#                               }
#                           }})
# =============================================================================

# =============================================================================
# # alternative method: logarithmic 'grid_search'
# optimize_hyper_params(reconstructor,
#                       validation_data=validation_data,
#                       measure=L2,
#                       dataset=dataset,
#                       HYPER_PARAMS_override={
#                           'l2_regularization': {
#                               'method': 'grid_search',
#                               'range': [0., 2.],
#                               'grid_search_options': {
#                                   'type': 'logarithmic',
#                                   'num_samples': 10
#                               }
#                           }})
# =============================================================================

print('optimized l2 reg. coeff.: {}'.format(
    reconstructor.hyper_params['l2_regularization']))


# %% task table
eval_tt = TaskTable()
eval_tt.append(reconstructor=reconstructor, test_data=test_data,
               dataset=dataset, measures=[L2, PSNR])

results = eval_tt.run()
print(results)

# %% plot reconstructions
fig = results.plot_reconstruction(0, test_ind=range(3),
                                  fig_size=(9, 4), vrange='individual')
