from warnings import warn
from copy import deepcopy

import torch
import numpy as np
import torch.nn as nn
from odl.tomo import fbp_op

from dival.reconstructors import StandardLearnedReconstructor
from dival.reconstructors.networks.unet import UNet
from dival.datasets.fbp_dataset import FBPDataset


class FBPUNetReconstructor(StandardLearnedReconstructor):
    """
    CT reconstructor applying filtered back-projection followed by a
    postprocessing U-Net (e.g. [1]_).

    References
    ----------
    .. [1] K. H. Jin, M. T. McCann, E. Froustey, et al., 2017,
           "Deep Convolutional Neural Network for Inverse Problems in Imaging".
           IEEE Transactions on Image Processing.
           `doi:10.1109/TIP.2017.2713099
           <https://doi.org/10.1109/TIP.2017.2713099>`_
    """

    HYPER_PARAMS = deepcopy(StandardLearnedReconstructor.HYPER_PARAMS)
    HYPER_PARAMS.update({
        'scales': {
            'default': 5,
            'retrain': True
        },
        'skip_channels': {
            'default': 4,
            'retrain': True
        },
        'channels': {
            'default': (32, 32, 64, 64, 128, 128),
            'retrain': True
        },
        'filter_type': {
            'default': 'Hann',
            'retrain': True
        },
        'frequency_scaling': {
            'default': 1.0,
            'retrain': True
        },
        'use_sigmoid': {
            'default': False,
            'retrain': True
        },
        'init_bias_zero': {
            'default': True,
            'retrain': True
        },
        'lr': {
            'default': 0.001,
            'retrain': True
        },
        'scheduler': {
            'default': 'cosine',
            'choices': ['base', 'cosine'],  # 'base': inherit
            'retrain': True
        },
        'lr_min': {  # only used if 'cosine' scheduler is selected
            'default': 1e-4,
            'retrain': True
        }
    })

    def __init__(self, ray_trafo, **kwargs):
        """
        Parameters
        ----------
        ray_trafo : :class:`odl.tomo.RayTransform`
            Ray transform (the forward operator).

        Further keyword arguments are passed to ``super().__init__()``.
        """
        super().__init__(ray_trafo, **kwargs)

    def train(self, dataset):
        try:
            fbp_dataset = dataset.fbp_dataset
        except AttributeError:
            warn('Training FBPUNetReconstructor with no cached FBP dataset.'
                 'Will compute the FBPs on the fly. For faster training, '
                 'consider precomputing the FBPs with '
                 '`generate_fbp_cache_files(...)` and pass them to `train()` '
                 'by setting the attribute '
                 '``dataset.fbp_dataset = get_cached_fbp_dataset(...)``.')
            fbp_dataset = FBPDataset(
                dataset, self.op, filter_type=self.filter_type,
                frequency_scaling=self.frequency_scaling)

        if not fbp_dataset.supports_random_access():
            warn('Dataset does not support random access. Shuffling will not '
                 'work, and only 1 worker will be used for data loading.')
            self.num_data_loader_workers = 1

        super().train(fbp_dataset)

    def init_model(self):
        self.fbp_op = fbp_op(self.op, filter_type=self.filter_type,
                             frequency_scaling=self.frequency_scaling)
        self.model = UNet(in_ch=1, out_ch=1,
                          channels=self.channels[:self.scales],
                          skip_channels=[self.skip_channels] * (self.scales),
                          use_sigmoid=self.use_sigmoid)
        if self.init_bias_zero:
            def weights_init(m):
                if isinstance(m, torch.nn.Conv2d):
                    m.bias.data.fill_(0.0)
            self.model.apply(weights_init)

        if self.use_cuda:
            self.model = nn.DataParallel(self.model).to(self.device)

    def init_scheduler(self, dataset_train):
        if self.scheduler.lower() == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs,
                eta_min=self.lr_min)
        else:
            super().init_scheduler(dataset_train)

    def _reconstruct(self, observation):
        self.model.eval()
        fbp = self.fbp_op(observation)
        fbp_tensor = torch.from_numpy(
            np.asarray(fbp)[None, None]).to(self.device)
        reco_tensor = self.model(fbp_tensor)
        reconstruction = reco_tensor.cpu().detach().numpy()[0, 0]
        return self.reco_space.element(reconstruction)
