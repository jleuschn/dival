# -*- coding: utf-8 -*-
import torch.nn as nn
from copy import deepcopy

from dival.reconstructors.standard_learned_reconstructor import (
    StandardLearnedReconstructor)
from dival.reconstructors.networks.iradonmap import get_iradonmap_model


class IRadonMapReconstructor(StandardLearnedReconstructor):
    """
    CT reconstructor that learns a fully connected layer for filtering along
    the axis of the detector pixels s, followed by the backprojection
    (segment 1). After that, a residual CNN acts as a post-processing net
    (segment 2). We use the U-Net from the FBPUnet model.

    In the original paper [1]_, a learned version of the back-
    projection layer (sinusoidal layer) is used. This layer introduces a lot
    more parameters. Therefore, we added an option to directly use the operator
    in our implementation. Additionally, we drop the tanh activation after
    the first fully connected layer, due to bad performance.

    In any configuration, the iRadonMap has less parameters than an
    Automap network [2]_.

    References
    ----------
    .. [1] J. He and J. Ma, 2018,
           "Radon Inversion via Deep Learning".
           arXiv preprint.
           `arXiv:1808.03015v1
           <https://arxiv.org/abs/1808.03015>`_
    .. [2] B. Zhu, J. Z. Liu, S. F. Cauly et al., 2018,
           "Image Reconstruction by Domain-Transform Manifold Learning".
           Nature 555, 487--492.
           `doi:10.1038/nature25988
           <https://doi.org/10.1038/nature25988>`_
    """

    HYPER_PARAMS = deepcopy(StandardLearnedReconstructor.HYPER_PARAMS)
    HYPER_PARAMS.update({
        'scales': {
            'default': 5,
            'retrain': True
        },
        'epochs': {
            'default': 20,
            'retrain': True
        },
        'lr': {
            'default': 0.01,
            'retrain': True
        },
        'skip_channels': {
            'default': 4,
            'retrain': True
        },
        'batch_size': {
            'default': 64,
            'retrain': True
        },
        'fully_learned': {
            'default': False,
            'retrain': True
        },
        'use_sigmoid': {
            'default': False,
            'retrain': True
        },
    })

    def __init__(self, ray_trafo, coord_mat=None, **kwargs):
        """
        Parameters
        ----------
        ray_trafo : :class:`odl.tomo.RayTransform`
            Ray transform (the forward operator).
        coord_mat : array, optional
            Precomputed coordinate matrix for the `LearnedBackprojection`.
            This option is provided for performance optimization.
            If `None` is passed, the matrix is computed in :meth:`init_model`.

        Further keyword arguments are passed to ``super().__init__()``.
        """
        super().__init__(ray_trafo, **kwargs)

        self.coord_mat = coord_mat

    def init_model(self):
        self.model = get_iradonmap_model(
                ray_trafo=self.op, fully_learned=self.fully_learned,
                scales=self.scales, skip=self.skip_channels,
                use_sigmoid=self.use_sigmoid, coord_mat=self.coord_mat)
        if self.use_cuda:
            self.model = nn.DataParallel(self.model).to(self.device)

    # def init_optimizer(self, dataset_train):
    #     self.optimizer = torch.optim.RMSprop(self.model.parameters(),
    #                                          lr=self.lr, momentum=0.9)
