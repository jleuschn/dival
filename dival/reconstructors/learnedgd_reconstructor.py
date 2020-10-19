from copy import deepcopy

import odl
import torch
from odl.contrib.torch import OperatorModule
from odl.tomo import fbp_op
from odl.operator.operator import OperatorRightScalarMult

from dival.reconstructors.standard_learned_reconstructor import (
    StandardLearnedReconstructor)
from dival.reconstructors.networks.iterative import IterativeNet


class LearnedGDReconstructor(StandardLearnedReconstructor):
    """
    CT reconstructor applying a learned gradient descent iterative scheme.

    Note that the weights are not shared across the blocks, like presented in
    the original paper [1]_.
    This implementation rather follows
    https://github.com/adler-j/learned_primal_dual/blob/master/ellipses/learned_primal.py.

    References
    ----------
    .. [1] Jonas Adler & Ozan Öktem (2017). Solving ill-posed inverse problems
        using iterative deep neural networks. Inverse Problems, 33(12), 124007.
    """

    HYPER_PARAMS = deepcopy(StandardLearnedReconstructor.HYPER_PARAMS)
    HYPER_PARAMS.update({
        'epochs': {
            'default': 20,
            'retrain': True
        },
        'batch_size': {
            'default': 32,
            'retrain': True
        },
        'lr': {
            'default': 0.01,
            'retrain': True
        },
        'normalize_by_opnorm': {
            'default': True,
            'retrain': True
        },
        'niter': {
            'default': 5,
            'retrain': True
        },
        'init_fbp': {
            'default': True,
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
            'default': False,
            'retrain': True
        },
        'lrelu_coeff': {
            'default': 0.2,
            'retrain': True
        },
        'lr_time_decay_rate': {
            'default': 3.2,
            'retrain': True
        },
        'init_weight_xavier_normal': {
            'default': False,
            'retrain': True
        },
        'init_weight_gain': {
            'default': 1.0,
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
        partial0 = odl.PartialDerivative(self.op.domain, axis=0)
        partial1 = odl.PartialDerivative(self.op.domain, axis=1)
        self.reg_mod = OperatorModule(partial0.adjoint * partial0 +
                                      partial1.adjoint * partial1)
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
        self.model = IterativeNet(
            n_iter=self.niter,
            n_memory=5,
            op=self.op_mod,
            op_adj=self.op_adj_mod,
            op_init=self.init_mod,
            op_reg=self.reg_mod,
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
                if self.hyper_params['init_weight_xavier_normal']:
                    torch.nn.init.xavier_normal_(
                        m.weight, gain=self.hyper_params['init_weight_gain'])
        self.model.apply(weights_init)

        if self.use_cuda:
            # WARNING: using data-parallel here doesn't work, probably
            # astra_cuda is not thread-safe
            self.model = self.model.to(self.device)

    # def init_optimizer(self, dataset_train):
    #     self.optimizer = torch.optim.RMSprop(self.model.parameters(),
    #                                          lr=self.lr, alpha=0.9)

    # def init_scheduler(self, dataset_train):
    #     self.scheduler = None
