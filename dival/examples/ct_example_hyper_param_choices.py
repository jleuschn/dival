import numpy as np
from dival.evaluation import TaskTable
from dival.measure import PSNR, SSIM
from dival.reconstructors.odl_reconstructors import FBPReconstructor
from dival.datasets.standard import get_standard_dataset

np.random.seed(0)

# %% data
dataset = get_standard_dataset('ellipses', impl='astra_cpu')
test_data = dataset.get_data_pairs('test', 10)

# %% task table and reconstructors
eval_tt = TaskTable()

reconstructor = FBPReconstructor(dataset.ray_trafo)

eval_tt.append(reconstructor=reconstructor, measures=[PSNR, SSIM],
               test_data=test_data,
               hyper_param_choices={'filter_type': ['Ram-Lak', 'Hann'],
                                    'frequency_scaling': [0.8, 0.9, 1.]})

# %% run task table
results = eval_tt.run()
print(results.to_string(show_columns=['misc']))

# %% plot reconstructions
fig = results.plot_all_reconstructions(test_ind=range(1),
                                       fig_size=(9, 4), vrange='individual')
