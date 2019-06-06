import numpy as np
from dival.evaluation import TaskTable
from dival.measure import L2
from dival.reconstructors.odl_reconstructors import FBPReconstructor
from dival.datasets.standard import get_standard_dataset

np.random.seed(0)

# %% data
dataset = get_standard_dataset('ellipses')
validation_data = dataset.get_data_pairs('validation', 10)
test_data = dataset.get_data_pairs('test', 10)

# %% task table and reconstructors
eval_tt = TaskTable()

reconstructor = FBPReconstructor(dataset.ray_trafo)
options = {'hyper_param_search': {'measure': L2,
                                  'validation_data': validation_data}}

eval_tt.append(reconstructor=reconstructor, test_data=test_data,
               options=options)

# %% run task table
results = eval_tt.run()
print(results.to_string(formatters={'reconstructor': lambda r: r.name}))

# %% plot reconstructions
fig = results.plot_all_reconstructions(fig_size=(9, 4), vrange='individual')
