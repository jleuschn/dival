"""
Reproduce the metrics from Figure 5 in https://doi.org/10.1038/s41597-021-00893-z
"""
import numpy as np
from dival import get_standard_dataset
from dival.measure import PSNR, SSIM
from dival.reconstructors.odl_reconstructors import FBPReconstructor
from dival.reference_reconstructors import get_reference_reconstructor

IMPL = 'astra_cuda'  # change this to 'astra_cpu' if CUDA is not available

dataset = get_standard_dataset('lodopab', impl=IMPL)

fbp_reconstructor = FBPReconstructor(dataset.ray_trafo,
                                     hyper_params={'filter_type': 'Hann',
                                                   'frequency_scaling': 0.641})
# fbp_reconstructor = get_reference_reconstructor('fbp', 'lodopab', impl=IMPL)

fbpunet_reconstructor = get_reference_reconstructor('fbpunet', 'lodopab', impl=IMPL)

reconstructors = [fbp_reconstructor, fbpunet_reconstructor]

num_samples = 4  # dataset.get_len('test')  # number of images

psnrs = {}
ssims = {}
for reconstructor in reconstructors:
    psnr_list = []
    ssim_list = []
    for i, (obs, gt) in zip(range(num_samples), dataset.generator(part='test')):
        reco = np.asarray(reconstructor.reconstruct(obs))
        psnr_list.append(PSNR(reco, gt))
        ssim_list.append(SSIM(reco, gt))
        print('{} on sample {:d}: PSNR={:.1f}, SSIM={:.2f}'.format(reconstructor.name, i, psnr_list[-1], ssim_list[-1]))
        psnrs[reconstructor.name] = psnr_list
        ssims[reconstructor.name] = ssim_list

print('Statistics using the first {:d} test samples:'.format(num_samples))
for reconstructor in reconstructors:
    print(reconstructor.name, 'mean PSNR', np.mean(psnrs[reconstructor.name]))
    print(reconstructor.name, 'mean SSIM', np.mean(ssims[reconstructor.name]))
    print(reconstructor.name, 'std PSNR', np.std(psnrs[reconstructor.name]))
    print(reconstructor.name, 'std SSIM', np.std(ssims[reconstructor.name]))
