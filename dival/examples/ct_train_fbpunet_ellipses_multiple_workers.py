"""
Train FBPUNetReconstructor on ``'ellipses'``.
"""
from odl.operator import OperatorComp
from torch.multiprocessing import set_start_method, get_start_method
import torch
import odl
import numpy as np
from dival import get_standard_dataset
from dival.measure import PSNR
from dival.reconstructors.fbpunet_reconstructor import FBPUNetReconstructor
from dival.reference_reconstructors import (
    check_for_params, download_params, get_hyper_params_path)
from dival.util.plot import plot_images
from dival.util.torch_utility import patch_ray_trafo_for_pickling

IMPL = 'astra_cuda'

LOG_DIR = './logs/ellipses_fbpunet'
SAVE_BEST_LEARNED_PARAMS_PATH = './params/ellipses_fbpunet'


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    
    # recreate forward_op
    # (to avoid astra.data2d.py ValueError: Data object not found)
    ray_trafo = dataset.dataset.dataset.forward_op.left
    dataset.dataset.dataset.forward_op = OperatorComp(
        odl.tomo.RayTransform(
            ray_trafo.domain, ray_trafo.geometry, impl=ray_trafo.impl),
        dataset.dataset.dataset.forward_op.right)


if __name__ == '__main__':
    try:
        set_start_method('spawn')
    except RuntimeError:
        if get_start_method() != 'spawn':
            raise RuntimeError(
                'Could not set multiprocessing start method to \'spawn\'')

    dataset = get_standard_dataset('ellipses',
                                    fixed_seeds=False,
                                    fixed_noise_seeds=False,
                                    impl=IMPL)
    ray_trafo = dataset.get_ray_trafo(impl=IMPL)
    patch_ray_trafo_for_pickling(ray_trafo)
    test_data = dataset.get_data_pairs('test', 100)
    
    reconstructor = FBPUNetReconstructor(
        ray_trafo, log_dir=LOG_DIR,
        save_best_learned_params_path=SAVE_BEST_LEARNED_PARAMS_PATH,
        allow_multiple_workers_without_random_access=True,
        num_data_loader_workers=2,  # more workers lead to increased VRAM usage
        worker_init_fn=worker_init_fn,  # for recreating forward_op
        )
    
    #%% obtain reference hyper parameters
    if not check_for_params('fbpunet', 'ellipses', include_learned=False):
        download_params('fbpunet', 'ellipses', include_learned=False)
    hyper_params_path = get_hyper_params_path('fbpunet', 'ellipses')
    reconstructor.load_hyper_params(hyper_params_path)
    
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
