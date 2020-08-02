from dival import get_standard_dataset
from dival.measure import PSNR
from dival.reconstructors.dip_ct_reconstructor import (
    DeepImagePriorCTReconstructor)
from dival.reference_reconstructors import (
    check_for_params, download_params, get_params_path)
from dival.util.plot import plot_images
import matplotlib.pyplot as plt

IMPL = 'astra_cuda'

dataset = get_standard_dataset('lodopab', impl=IMPL)
TEST_SAMPLE = 0
obs, gt = dataset.get_sample(TEST_SAMPLE, 'test')

def callback_func(iteration, reconstruction, loss):
    _, ax = plot_images([reconstruction, gt],
                        fig_size=(10, 4))
    ax[0].set_xlabel('loss: {:f}'.format(loss))
    ax[0].set_title('DIP iteration {:d}'.format(iteration))
    ax[1].set_title('ground truth')
    ax[0].figure.suptitle('test sample {:d}'.format(TEST_SAMPLE))
    plt.show()

reconstructor = DeepImagePriorCTReconstructor(
    dataset.get_ray_trafo(impl=IMPL),
    callback_func=callback_func, callback_func_interval=100)

#%% obtain reference hyper parameters
if not check_for_params('diptv', 'lodopab'):
    download_params('diptv', 'lodopab')
params_path = get_params_path('diptv', 'lodopab')
reconstructor.load_params(params_path)

#%% evaluate
reco = reconstructor.reconstruct(obs)
psnr = PSNR(reco, gt)

print('psnr: {:f}'.format(psnr))
_, ax = plot_images([reco, gt],
                    fig_size=(10, 4))
ax[0].set_xlabel('PSNR: {:.2f}'.format(psnr))
ax[0].set_title('DeepImagePriorCTReconstructor')
ax[1].set_title('ground truth')
ax[0].figure.suptitle('test sample {:d}'.format(TEST_SAMPLE))
