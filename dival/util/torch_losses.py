"""Provides custom loss functions for PyTorch."""
import torch
import numpy as np

from dival.util.constants import MU_MAX


def tv_loss(x):
    """
    Isotropic TV loss similar to the one in [1]_.

    Parameters
    ----------
    x : :class:`torch.Tensor`
        Tensor of which to compute the isotropic TV w.r.t. its last two axes.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Total_variation_denoising
    """
    dh = torch.abs(x[..., :, 1:] - x[..., :, :-1])
    dw = torch.abs(x[..., 1:, :] - x[..., :-1, :])
    return torch.sum(dh[..., :-1, :] + dw[..., :, :-1])


def poisson_loss(y_pred, y_true, photons_per_pixel=4096, mu_max=MU_MAX):
    """
    Loss corresponding to Poisson regression (cf. [2]_) for post-log CT data.
    The default parameters are based on the LoDoPaB dataset creation
    (cf. [3]_).

    :Authors:
        Sören Dittmer <sdittmer@math.uni-bremen.de>

    Parameters
    ----------
    y_pred : :class:`torch.Tensor`
        Predicted observation (post-log, normalized by `mu_max`).
    y_true : :class:`torch.Tensor`
        True observation (post-log, normalized by `mu_max`).
    photons_per_pixel : int or float, optional
        Mean number of photons per detector pixel for an unattenuated beam.
        Default: `4096`.
    mu_max : float, optional
        Normalization factor, by which `y_pred` and `y_true` have
        been divided (this function will multiply by it accordingly).
        Default: ``dival.util.constants.MU_MAX``.

    References
    ----------
    .. [2] https://en.wikipedia.org/wiki/Poisson_regression
    .. [3] https://github.com/jleuschn/lodopab_tech_ref/blob/master/create_dataset.py
    """

    def get_photons(y):
        y = torch.exp(-y * mu_max) * photons_per_pixel
        return y

    def get_photons_log(y):
        y = -y * mu_max + np.log(photons_per_pixel)
        return y

    y_true_photons = get_photons(y_true)
    y_pred_photons = get_photons(y_pred)
    y_pred_photons_log = get_photons_log(y_pred)

    return torch.sum(y_pred_photons - y_true_photons * y_pred_photons_log)
