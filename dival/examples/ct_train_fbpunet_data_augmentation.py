"""
Train FBPUNetReconstructor with data augmentation on 'lodopab'.
"""
import numpy as np
from dival import get_standard_dataset
from dival.measure import PSNR
from dival.reconstructors.fbpunet_reconstructor import FBPUNetReconstructor
from dival.datasets.fbp_dataset import (
    generate_fbp_cache_files, get_cached_fbp_dataset)
from dival.reference_reconstructors import (
    check_for_params, download_params, get_hyper_params_path)
from dival.util.plot import plot_images
import torch

IMPL = 'astra_cuda'

LOG_DIR = './logs/lodopab_fbpunet_data_augmentation'
SAVE_BEST_LEARNED_PARAMS_PATH = './params/lodopab_fbpunet_data_augmentation'

CACHE_FILES = {
    'train':
        ('/localdata/dival_dataset_caches/cache_train_lodopab_fbp.npy', None),
    'validation':
        ('/localdata/dival_dataset_caches/cache_validation_lodopab_fbp.npy', None)}

dataset = get_standard_dataset('lodopab', impl=IMPL)
ray_trafo = dataset.get_ray_trafo(impl=IMPL)
test_data = dataset.get_data_pairs('test', 100)


def random_flip_rotate_transform(sample):
    fbp, gt = sample
    choice = torch.randint(8, (1,))[0]
    if choice % 4 == 1:
        fbp = torch.flip(fbp, (1,))
        gt = torch.flip(gt, (1,))
    elif choice % 4 == 2:
        fbp = torch.flip(fbp, (2,))
        gt = torch.flip(gt, (2,))
    elif choice % 4 == 3:
        fbp = torch.flip(fbp, (1, 2))
        gt = torch.flip(gt, (1, 2))
    if choice // 4 == 1:
        fbp = torch.transpose(fbp, 1, 2)
        gt = torch.transpose(gt, 1, 2)
    return fbp, gt


class FBPUNetDataAugmentationReconstructor(FBPUNetReconstructor):
    def init_transform(self, dataset):
        self._transform = random_flip_rotate_transform


reconstructor = FBPUNetDataAugmentationReconstructor(
    ray_trafo, log_dir=LOG_DIR,
    save_best_learned_params_path=SAVE_BEST_LEARNED_PARAMS_PATH)

#%% obtain reference hyper parameters
if not check_for_params('fbpunet', 'lodopab', include_learned=False):
    download_params('fbpunet', 'lodopab', include_learned=False)
hyper_params_path = get_hyper_params_path('fbpunet', 'lodopab')
reconstructor.load_hyper_params(hyper_params_path)

#%% expose FBP cache to reconstructor by assigning `fbp_dataset` attribute
# uncomment the next line to generate the cache files (~20 GB)
# generate_fbp_cache_files(dataset, ray_trafo, CACHE_FILES)
cached_fbp_dataset = get_cached_fbp_dataset(dataset, ray_trafo, CACHE_FILES)
dataset.fbp_dataset = cached_fbp_dataset

#%% train
# reduce the batch size here if the model does not fit into GPU memory
# reconstructor.batch_size = 16
reconstructor.train(dataset)

#%% evaluate
recos = []
psnrs = []
for obs, gt in test_data:
    reco = reconstructor.reconstruct(obs)
    recos.append(reco)
    psnrs.append(PSNR(reco, gt))

print('mean psnr: {:f}'.format(np.mean(psnrs)))

for i in range(3):
    _, ax = plot_images([recos[i], test_data.ground_truth[i]],
                        fig_size=(10, 4))
    ax[0].set_xlabel('PSNR: {:.2f}'.format(psnrs[i]))
    ax[0].set_title('FBPUNetReconstructor')
    ax[1].set_title('ground truth')
    ax[0].figure.suptitle('test sample {:d}'.format(i))
