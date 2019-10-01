from odl.ufunc_ops.ufunc_ops import log_op
from dival.evaluation import TaskTable
from dival.measure import PSNR, SSIM
from dival.reconstructors.odl_reconstructors import FBPReconstructor
from dival.datasets.standard import get_standard_dataset
from dival.util.constants import MU_MAX

# %% data
dataset = get_standard_dataset('lodopab', observation_model='pre-log')
ray_trafo = dataset.get_ray_trafo(impl='astra_cpu')
reco_space = ray_trafo.domain
test_data = dataset.get_data_pairs('test', 10)

# %% task table and reconstructors
eval_tt = TaskTable()

fbp_reconstructor = FBPReconstructor(
    ray_trafo, hyper_params={
        'filter_type': 'Hann',
        'frequency_scaling': 0.8},
    pre_processor=(-1/MU_MAX) * log_op(ray_trafo.range))

reconstructors = [fbp_reconstructor]

eval_tt.append_all_combinations(reconstructors=reconstructors,
                                test_data=[test_data])

# %% run task table
results = eval_tt.run()
results.apply_measures([PSNR, SSIM])
print(results)

# %% plot reconstructions
fig = results.plot_all_reconstructions(test_ind=range(3),
                                       fig_size=(9, 4), vrange='individual')
