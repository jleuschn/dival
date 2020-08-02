import os
import numpy as np
from dival import get_standard_dataset
from dival.evaluation import TaskTable
from dival.measure import PSNR, SSIM
from dival.reference_reconstructors import get_reference_reconstructor

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

IMPL = 'astra_cuda'

np.random.seed(0)

# %% data
dataset = get_standard_dataset('lodopab', impl=IMPL)
ray_trafo = dataset.get_ray_trafo(impl=IMPL)
reco_space = ray_trafo.domain
test_data = dataset.get_data_pairs('test', 100)

# %% task table and reconstructors
eval_tt = TaskTable()

fbpunet_reconstructor = get_reference_reconstructor('fbpunet', 'lodopab')
iradonmap_reconstructor = get_reference_reconstructor('iradonmap', 'lodopab')
learnedgd_reconstructor = get_reference_reconstructor('learnedgd', 'lodopab')
learnedpd_reconstructor = get_reference_reconstructor('learnedpd', 'lodopab')

reconstructors = [fbpunet_reconstructor,  iradonmap_reconstructor,
                  learnedgd_reconstructor, learnedpd_reconstructor]

options = {'skip_training': True}
eval_tt.append_all_combinations(reconstructors=reconstructors,
                                test_data=[test_data], options=options)

# %% run task table
results = eval_tt.run()
results.apply_measures([PSNR, SSIM])
print(results)

# %% plots
results.plot_all_reconstructions(fig_size=(9, 4), test_ind=range(3))
ax = results.plot_performance(PSNR)
ax.set_ylim([28, 38])
