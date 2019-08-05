from dival.evaluation import TaskTable
from dival.measure import L2, PSNR, SSIM
from dival.reconstructors.odl_reconstructors import FBPReconstructor
from dival.datasets.standard import get_standard_dataset

# %% data
dataset = get_standard_dataset('lidc_idri_dival')
ray_trafo = dataset.ray_trafo
reco_space = ray_trafo.domain
test_data = dataset.get_data_pairs('test', 2)

# %% task table and reconstructors
eval_tt = TaskTable()

fbp_reconstructor = FBPReconstructor(
    ray_trafo, hyper_params={
        'filter_type': 'Hann',
        'frequency_scaling': 0.8})

reconstructors = [fbp_reconstructor]
options = {'save_iterates': True}

eval_tt.append_all_combinations(reconstructors=reconstructors,
                                test_data=[test_data], options=options)

# %% run task table
results = eval_tt.run()
results.apply_measures([L2, PSNR, SSIM])
print(results.to_string(formatters={'reconstructor': lambda r: r.name}))

# %% plot reconstructions
fig = results.plot_all_reconstructions(fig_size=(9, 4), vrange='individual')
