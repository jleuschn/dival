import numpy as np
from dival.datasets.dataset import Dataset
from dival.evaluation import TaskTable
from dival.measure import L2
from odl.discr import uniform_discr
from dival.reconstructors.regression_reconstructors import LinRegReconstructor

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
test_data = dataset.get_data_pairs('test')

# %% task table and reconstructors
eval_tt = TaskTable()

reconstructor = LinRegReconstructor(observation_space=observation_space,
                                    reco_space=reco_space)

rs = np.random.RandomState(0)
eval_tt.append(
    reconstructor=reconstructor, test_data=test_data, dataset=dataset,
    options={'hyper_param_search': {
                'measure': L2,
                'hyperopt_max_evals_retrain': 10,
                'hyperopt_rstate': rs}})

# %% run task table
results = eval_tt.run()
print(results.to_string(formatters={'reconstructor': lambda r: r.name}))

# %% plot reconstructions
fig = results.plot_all_reconstructions(test_index=range(3),
                                       fig_size=(9, 4), vrange='individual')
