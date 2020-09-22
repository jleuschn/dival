from warnings import warn
from functools import partial
from tqdm import tqdm
import torch
import numpy as np

from torch.optim import Adam
from torch.nn import MSELoss

from odl.contrib.torch import OperatorModule
from odl.tomo import fbp_op

from dival.reconstructors import IterativeReconstructor
from dival.util.torch_losses import poisson_loss, tv_loss
from dival.util.constants import MU_MAX


class TVAdamCTReconstructor(IterativeReconstructor):
    """
    CT reconstructor minimizing a TV-functional with the Adam optimizer.
    """

    HYPER_PARAMS = {
        'lr':
            {'default': 1e-3,
             'range': [1e-5, 1e-1]},
        'gamma':
            {'default': 1e-4,
             'range': [1e-7, 1e-0],
             'grid_search_options': {'num_samples': 20}},
        'iterations':
            {'default': 5000,
             'range': [1, 50000]},
        'loss_function':
            {'default': 'mse',
             'choices': ['mse', 'poisson']},
        'photons_per_pixel':  # used by 'poisson' loss function
            {'default': 4096},
        'mu_max':  # used by 'poisson' loss function
            {'default': MU_MAX},
        'init_filter_type':
            {'default': 'Hann'},
        'init_frequency_scaling':
            {'default': 0.1}
    }

    def __init__(self, ray_trafo, callback_func=None,
                 callback_func_interval=100, show_pbar=True, **kwargs):
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

    def _reconstruct(self, observation, *args, **kwargs):
        self.fbp_op = fbp_op(
            self.ray_trafo, filter_type=self.init_filter_type,
            frequency_scaling=self.init_frequency_scaling)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.output = torch.tensor(self.fbp_op(observation))[None].to(device)
        self.output.requires_grad = True

        self.optimizer = Adam([self.output], lr=self.lr)

        y_delta = torch.tensor(np.asarray(observation), dtype=torch.float32)
        y_delta = y_delta.view(1, *y_delta.shape)
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

        best_loss = np.infty
        best_output = self.output.detach().clone()

        for i in tqdm(range(self.iterations),
                      desc='TV', disable=not self.show_pbar):
            self.optimizer.zero_grad()
            loss = criterion(self.ray_trafo_module(self.output),
                             y_delta) + self.gamma * tv_loss(self.output)
            loss.backward()

            self.optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_output = self.output.detach().clone()

            if (self.callback_func is not None and
                    (i % self.callback_func_interval == 0
                     or i == self.iterations-1)):
                self.callback_func(
                    iteration=i,
                    reconstruction=best_output[0, ...].cpu().numpy(),
                    loss=best_loss)

            if self.callback is not None:
                self.callback(self.reco_space.element(
                    best_output[0, ...].cpu().numpy()))

        return self.reco_space.element(best_output[0, ...].cpu().numpy())
