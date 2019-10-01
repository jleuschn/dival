import os
import requests
from dival.evaluation import TaskTable
from dival.measure import PSNR, SSIM
from dival.reconstructors.odl_reconstructors import FBPReconstructor
from dival.reconstructors.fbp_unet_reconstructor import (FBPUNetReconstructor,
                                                         CachedFBPDataset)
from dival.datasets.standard import get_standard_dataset

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # TODO adjust

# %% data
dataset = get_standard_dataset('lodopab')
dataset.fbp_dataset = CachedFBPDataset(dataset, {
    'train': '/localdata/jleuschn/lodopab/reco_fbps_train.npy',
    'validation': '/localdata/jleuschn/lodopab/reco_fbps_validation.npy',
    'test': '/localdata/jleuschn/lodopab/reco_fbps_test.npy'})
ray_trafo = dataset.get_ray_trafo(impl='astra_cpu')
reco_space = ray_trafo.domain
test_data = dataset.get_data_pairs('test', 7)

# %% task table and reconstructors
eval_tt = TaskTable()

fbp_reconstructor = FBPReconstructor(
    ray_trafo, hyper_params={
        'filter_type': 'Hann',
        'frequency_scaling': 0.8})

fbp_unet_reconstructor = FBPUNetReconstructor(ray_trafo,
                                              batch_size=64, use_cuda=True)
state_filename = 'fbp_unet_reconstructor_lodopab_baseline_state.pt'
with open(state_filename, 'wb') as file:
    r = requests.get('https://github.com/jleuschn/supp.dival/raw/master/'
                     'examples/'
                     'fbp_unet_reconstructor_lodopab_baseline_state.pt')
    file.write(r.content)
fbp_unet_reconstructor.load_params(state_filename)

reconstructors = [fbp_reconstructor, fbp_unet_reconstructor]

eval_tt.append_all_combinations(reconstructors=reconstructors,
                                test_data=[test_data],
                                datasets=[dataset],
                                options={'skip_training': True}
                                )

# %% run task table
results = eval_tt.run()
results.apply_measures([PSNR, SSIM])
print(results)

# %% plot reconstructions
fig = results.plot_all_reconstructions(fig_size=(9, 4), vrange='individual')
