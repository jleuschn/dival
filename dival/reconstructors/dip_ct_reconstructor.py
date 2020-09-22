from warnings import warn
from functools import partial
from tqdm import tqdm
import torch
import numpy as np

from torch.optim import Adam
from torch.nn import MSELoss

from odl.contrib.torch import OperatorModule

from dival.reconstructors import IterativeReconstructor
from dival.reconstructors.networks.unet import UNet
from dival.util.torch_losses import poisson_loss, tv_loss
from dival.util.constants import MU_MAX

MIN = -1000
MAX = 1000


class DeepImagePriorCTReconstructor(IterativeReconstructor):
    """
    CT reconstructor applying DIP with TV regularization (see [2]_).
    The DIP was introduced in [1]_.

    References
    ----------
    .. [1] V. Lempitsky, A. Vedaldi, and D. Ulyanov, 2018, "Deep Image Prior".
           IEEE/CVF Conference on Computer Vision and Pattern Recognition.
           https://doi.org/10.1109/CVPR.2018.00984
    .. [2] D. Otero Baguer, J. Leuschner, M. Schmidt, 2020, "Computed
           Tomography Reconstruction Using Deep Image Prior and Learned
           Reconstruction Methods". Inverse Problems.
           https://doi.org/10.1088/1361-6420/aba415
    """

    HYPER_PARAMS = {
        'lr':
            {'default': 1e-3,
             'range': [1e-5, 1e-1]},
        'gamma':
            {'default': 1e-4,
             'range': [1e-7, 1e-0],
             'grid_search_options': {'num_samples': 20}},
        'scales':
            {'default': 4,
             'choices': [3, 4, 5, 6, 7]},
        'channels':
            {'default': [128] * 5},
        'skip_channels':
            {'default': [4] * 5},
        'iterations':
            {'default': 5000,
             'range': [1, 50000]},
        'loss_function':
            {'default': 'mse',
             'choices': ['mse', 'poisson']},
        'photons_per_pixel':  # used by 'poisson' loss function
            {'default': 4096,
             'range': [1000, 10000]},
        'mu_max':  # used by 'poisson' loss function
            {'default': MU_MAX,
             'range': [1., 10000.]}
    }

    def __init__(self, ray_trafo, callback_func=None,
                 callback_func_interval=100, show_pbar=True,
                 torch_manual_seed=10, **kwargs):
        """
        Parameters
        ----------
        ray_trafo : `odl.tomo.operators.RayTransform`
            The forward operator
        callback_func : callable, optional
            Callable with signature
            ``callback_func(iteration, reconstruction, loss)`` that is called
            after every `callback_func_interval` iterations, starting
            after the first iteration. It is additionally called after the
            last iteration.
            Note that it differs from the inherited
            `IterativeReconstructor.callback` (which is also supported) in that
            the latter is of type :class:`odl.solvers.util.callback.Callback`,
            which only receives the reconstruction, such that the loss would
            have to be recomputed.
        callback_func_interval : int, optional
            Number of iterations between calls to `callback_func`.
            Default: `100`.
        show_pbar : bool, optional
            Whether to show a tqdm progress bar during reconstruction.
        torch_manual_seed : int, optional
            Fixed seed to set by ``torch.manual_seed`` before reconstruction.
            The default is `10`. It can be set to `None` or `False` to disable
            the manual seed.
        """

        super().__init__(
            reco_space=ray_trafo.domain, observation_space=ray_trafo.range,
            **kwargs)

        self.callback_func = callback_func
        self.ray_trafo = ray_trafo
        self.ray_trafo_module = OperatorModule(self.ray_trafo)
        self.callback_func = callback_func
        self.callback_func_interval = callback_func_interval
        self.show_pbar = show_pbar
        self.torch_manual_seed = torch_manual_seed

    def get_activation(self, layer_index):
        return self.model.layer_output(self.net_input, layer_index)

    def _reconstruct(self, observation, *args, **kwargs):
        if self.torch_manual_seed:
            torch.random.manual_seed(self.torch_manual_seed)

        output_depth = 1
        input_depth = 1
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.net_input = 0.1 * \
            torch.randn(input_depth, *self.reco_space.shape)[None].to(device)
        self.model = UNet(
            input_depth,
            output_depth,
            channels=self.channels[:self.scales],
            skip_channels=self.skip_channels[:self.scales],
            use_sigmoid=True,
            use_norm=True).to(device)

        self.optimizer = Adam(self.model.parameters(), lr=self.lr)

        y_delta = torch.tensor(np.asarray(observation), dtype=torch.float32)
        y_delta = y_delta.view(1, 1, *y_delta.shape)
        y_delta = y_delta.to(device)

        if self.loss_function == 'mse':
            criterion = MSELoss()
        elif self.loss_function == 'poisson':
            criterion = partial(poisson_loss,
                                photons_per_pixel=self.photons_per_pixel,
                                mu_max=self.mu_max)
        else:
            warn('Unknown loss function, falling back to MSE')
            criterion = MSELoss()

        best_loss = np.inf
        best_output = self.model(self.net_input).detach()

        for i in tqdm(range(self.iterations),
                      desc='DIP', disable=not self.show_pbar):
            self.optimizer.zero_grad()
            output = self.model(self.net_input)
            loss = criterion(self.ray_trafo_module(output),
                             y_delta) + self.gamma * tv_loss(output)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
            self.optimizer.step()

            for p in self.model.parameters():
                p.data.clamp_(MIN, MAX)

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_output = output.detach()

            if (self.callback_func is not None and
                    (i % self.callback_func_interval == 0
                     or i == self.iterations-1)):
                self.callback_func(
                    iteration=i,
                    reconstruction=best_output[0, 0, ...].cpu().numpy(),
                    loss=best_loss)

            if self.callback is not None:
                self.callback(self.reco_space.element(
                    best_output[0, 0, ...].cpu().numpy()))

        return self.reco_space.element(best_output[0, 0, ...].cpu().numpy())
