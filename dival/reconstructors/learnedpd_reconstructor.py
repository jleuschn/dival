from copy import deepcopy

import torch
from odl.contrib.torch import OperatorModule
from odl.tomo import fbp_op
from odl.operator.operator import OperatorRightScalarMult

from dival.reconstructors.standard_learned_reconstructor import (
    StandardLearnedReconstructor)
from dival.reconstructors.networks.iterative import PrimalDualNet


class LearnedPDReconstructor(StandardLearnedReconstructor):
    """
    CT reconstructor applying a learned primal dual iterative scheme ([1]_).

    References
    ----------
    .. [1] Jonas Adler & Ozan Öktem (2018). Learned Primal-Dual Reconstruction.
        IEEE Transactions on Medical Imaging, 37(6), 1322-1332.
    """

    HYPER_PARAMS = deepcopy(StandardLearnedReconstructor.HYPER_PARAMS)
    HYPER_PARAMS.update({
        'epochs': {
            'default': 20,
            'retrain': True
        },
        'batch_size': {
            'default': 5,
            'retrain': True
        },
        'lr': {
            'default': 0.001,
            'retrain': True
        },
        'lr_min': {
            'default': 0.0,
            'retrain': True
        },
        'normalize_by_opnorm': {
            'default': True,
            'retrain': True
        },
        'niter': {
            'default': 10,
            'retrain': True
        },
        'init_fbp': {
            'default': False,
            'retrain': True
        },
        'init_filter_type': {
            'default': 'Hann',
            'retrain': True
        },
        'init_frequency_scaling': {
            'default': 0.4,
            'retrain': True
        },
        'nprimal': {
            'default': 5,
            'retrain': True
        },
        'ndual': {
            'default': 5,
            'retrain': True
        },
        'use_sigmoid': {
            'default': False,
            'retrain': True
        },
        'nlayer': {
            'default': 3,
            'retrain': True
        },
        'internal_ch': {
            'default': 32,
            'retrain': True
        },
        'kernel_size': {
            'default': 3,
            'retrain': True
        },
        'batch_norm': {
            'default': False,
            'retrain': True
        },
        'prelu': {
            'default': True,
            'retrain': True
        },
        'lrelu_coeff': {
            'default': 0.2,
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

    def init_model(self):
        self.op_mod = OperatorModule(self.op)
        self.op_adj_mod = OperatorModule(self.op.adjoint)
        if self.hyper_params['init_fbp']:
            fbp = fbp_op(
                self.non_normed_op,
                filter_type=self.hyper_params['init_filter_type'],
                frequency_scaling=self.hyper_params['init_frequency_scaling'])
            if self.normalize_by_opnorm:
                fbp = OperatorRightScalarMult(fbp, self.opnorm)
            self.init_mod = OperatorModule(fbp)
        else:
            self.init_mod = None
        self.model = PrimalDualNet(
            n_iter=self.niter,
            op=self.op_mod,
            op_adj=self.op_adj_mod,
            op_init=self.init_mod,
            n_primal=self.hyper_params['nprimal'],
            n_dual=self.hyper_params['ndual'],
            use_sigmoid=self.hyper_params['use_sigmoid'],
            n_layer=self.hyper_params['nlayer'],
            internal_ch=self.hyper_params['internal_ch'],
            kernel_size=self.hyper_params['kernel_size'],
            batch_norm=self.hyper_params['batch_norm'],
            prelu=self.hyper_params['prelu'],
            lrelu_coeff=self.hyper_params['lrelu_coeff'])

        def weights_init(m):
            if isinstance(m, torch.nn.Conv2d):
                m.bias.data.fill_(0.0)
                torch.nn.init.xavier_uniform_(m.weight)
        self.model.apply(weights_init)

        if self.use_cuda:
            # WARNING: using data-parallel here doesn't work, probably
            # astra_cuda is not thread-safe
            self.model = self.model.to(self.device)

    def init_optimizer(self, dataset_train):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                                          betas=(0.9, 0.99))

    def init_scheduler(self, dataset_train):
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs,
            eta_min=self.hyper_params['lr_min'])
