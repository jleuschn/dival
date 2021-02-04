"""
Provides utilities related to PyTorch.

The classes and functions

    :class:`TorchRayTrafoParallel2DModule`
    :class:`TorchRayTrafoParallel2DAdjointModule`
    :func:`get_torch_ray_trafo_parallel_2d`
    :func:`get_torch_ray_trafo_parallel_2d_adjoint`.

in this module rely on the
`tomosipo <https://github.com/ahendriksen/tomosipo>`_ library and experimental
astra features available in version 1.9.9.dev4 using CUDA.
In order to instantiate or call these classes and functions, all of these
requirements need to be fulfilled, otherwise an :class:`ImportError` is raised.
"""
import numpy as np
import torch
try:
    import tomosipo as ts
except ImportError:
    TOMOSIPO_AVAILABLE = False
    MISSING_TOMOSIPO_MESSAGE = (
        'Missing optional dependency \'tomosipo\'. The latest development '
        'version can be installed via '
        '`pip install git+https://github.com/ahendriksen/tomosipo@develop`')
else:
    TOMOSIPO_AVAILABLE = True
    from tomosipo.odl import (
        from_odl, parallel_2d_to_3d_geometry, discretized_space_2d_to_3d)
    from tomosipo.torch_support import to_autograd
from odl.tomo.backends.astra_cuda import astra_cuda_bp_scaling_factor
try:
    import astra
except ImportError:
    ASTRA_AVAILABLE = False
else:
    ASTRA_AVAILABLE = True


class RandomAccessTorchDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, part, reshape=None,
                 transform=None):
        self.dataset = dataset
        self.part = part
        self.reshape = reshape or (
            (None,) * self.dataset.get_num_elements_per_sample())
        self.transform = transform

    def __len__(self):
        return self.dataset.get_len(self.part)

    def __getitem__(self, idx):
        arrays = self.dataset.get_sample(idx, part=self.part)
        mult_elem = isinstance(arrays, tuple)
        if not mult_elem:
            arrays = (arrays,)
        tensors = []
        for arr, s in zip(arrays, self.reshape):
            t = torch.from_numpy(np.asarray(arr))
            if s is not None:
                t = t.view(*s)
            tensors.append(t)
        sample = tuple(tensors) if mult_elem else tensors[0]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

class GeneratorTorchDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, part, reshape=None,
                 transform=None):
        self.part = part
        self.dataset = dataset
        self.reshape = reshape or (
            (None,) * dataset.get_num_elements_per_sample())
        self.transform = transform

    def __len__(self):
        return self.dataset.get_len(self.part)

    def __iter__(self):
        return self.generate()

    def generate(self):
        for arrays in self.dataset.generator(self.part):
            mult_elem = isinstance(arrays, tuple)
            if not mult_elem:
                arrays = (arrays,)
            tensors = []
            for arr, s in zip(arrays, self.reshape):
                t = torch.from_numpy(np.asarray(arr))
                if s is not None:
                    t = t.view(*s)
                tensors.append(t)
            sample = tuple(tensors) if mult_elem else tensors[0]
            if self.transform is not None:
                sample = self.transform(sample)
            yield sample


class TorchRayTrafoParallel2DModule(torch.nn.Module):
    """
    Torch module applying a 2D parallel-beam ray transform using tomosipo that
    calls the direct forward projection routine of astra, which avoids copying
    between GPU and CPU (available in 1.9.9.dev4).

    All 2D transforms are computed using a single 3D transform.
    To this end the used tomosipo operator is renewed in :meth:`forward`
    everytime the product of batch and channel dimensions of the current batch
    differs compared to the previous batch, or compared to the value of
    `init_z_shape` specified to :meth:`init` for the first batch.
    """
    def __init__(self, ray_trafo, init_z_shape=1):
        """
        Parameters
        ----------
        ray_trafo : :class:`odl.tomo.RayTransform`
            Ray transform
        init_z_shape : int, optional
            Initial guess for the number of 2D transforms per batch, i.e. the
            product of batch and channel dimensions.
        """
        if not TOMOSIPO_AVAILABLE:
            raise ImportError(MISSING_TOMOSIPO_MESSAGE)
        if not ASTRA_AVAILABLE:
            raise RuntimeError('Astra is not available.')
        if not astra.use_cuda():
            raise RuntimeError('Astra is not able to use CUDA.')
        super().__init__()
        self.ray_trafo = ray_trafo
        self._construct_operator(init_z_shape)

    def _construct_operator(self, z_shape):
        self.torch_ray_trafo = (
            get_torch_ray_trafo_parallel_2d(self.ray_trafo, z_shape=z_shape))
        self._z_shape = z_shape

    def forward(self, x):
        z_shape = x.shape[0] * x.shape[1]
        if self._z_shape != z_shape:
            self._construct_operator(z_shape)
        x = x.view(1, z_shape, *x.shape[2:])
        x = self.torch_ray_trafo(x)
        x = x.view(z_shape, 1, *x.shape[2:])
        return x

class TorchRayTrafoParallel2DAdjointModule(torch.nn.Module):
    """
    Torch module applying the adjoint of a 2D parallel-beam ray transform
    using tomosipo that calls the direct backward projection routine of astra,
    which avoids copying between GPU and CPU (available in 1.9.9.dev4).

    All 2D transforms are computed using a single 3D transform.
    To this end the used tomosipo operator is renewed in :meth:`forward`
    everytime the product of batch and channel dimensions of the current batch
    differs compared to the previous batch, or compared to the value of
    `init_z_shape` specified to :meth:`init` for the first batch.
    """
    def __init__(self, ray_trafo, init_z_shape=1):
        """
        Parameters
        ----------
        ray_trafo : :class:`odl.tomo.RayTransform`
            Ray transform
        init_z_shape : int, optional
            Initial guess for the number of 2D transforms per batch, i.e. the
            product of batch and channel dimensions.
        """
        if not TOMOSIPO_AVAILABLE:
            raise ImportError(MISSING_TOMOSIPO_MESSAGE)
        if not ASTRA_AVAILABLE:
            raise RuntimeError('Astra is not available.')
        if not astra.use_cuda():
            raise RuntimeError('Astra is not able to use CUDA.')
        super().__init__()
        self.ray_trafo = ray_trafo
        self._construct_operator(init_z_shape)

    def _construct_operator(self, z_shape):
        self.torch_ray_trafo_adjoint = (
            get_torch_ray_trafo_parallel_2d_adjoint(self.ray_trafo,
                                                    z_shape=z_shape))
        self._z_shape = z_shape

    def forward(self, x):
        z_shape = x.shape[0] * x.shape[1]
        if self._z_shape != z_shape:
            self._construct_operator(z_shape)
        x = x.view(1, z_shape, *x.shape[2:])
        x = self.torch_ray_trafo_adjoint(x)
        x = x.view(z_shape, 1, *x.shape[2:])
        return x

def get_torch_ray_trafo_parallel_2d(ray_trafo, z_shape=1):
    """
    Create a torch autograd-enabled function from a 2D parallel-beam
    :class:`odl.tomo.RayTransform` using tomosipo that calls the direct
    forward projection routine of astra, which avoids copying between GPU and
    CPU (available in 1.9.9.dev4).

    Parameters
    ----------
    ray_trafo : :class:`odl.tomo.RayTransform`
        Ray transform
    z_shape : int, optional
        Channel dimension.
        Default: ``1``.

    Returns
    -------
    torch_ray_trafo : callable
        Torch autograd-enabled function applying the parallel-beam forward
        projection.
        Input and output have a trivial leading batch dimension and a channel
        dimension specified by `z_shape` (default ``1``), i.e. the
        input shape is ``(1, z_shape) + ray_trafo.domain.shape`` and the
        output shape is ``(1, z_shape) + ray_trafo.range.shape``.
    """
    if not TOMOSIPO_AVAILABLE:
        raise ImportError(MISSING_TOMOSIPO_MESSAGE)
    if not ASTRA_AVAILABLE:
        raise RuntimeError('Astra is not available.')
    if not astra.use_cuda():
        raise RuntimeError('Astra is not able to use CUDA.')
    vg = from_odl(discretized_space_2d_to_3d(ray_trafo.domain,
                                             z_shape=z_shape))
    pg = from_odl(parallel_2d_to_3d_geometry(ray_trafo.geometry,
                                             det_z_shape=z_shape))
    ts_op = ts.operator(vg, pg)
    torch_ray_trafo = to_autograd(ts_op)
    return torch_ray_trafo

def get_torch_ray_trafo_parallel_2d_adjoint(ray_trafo, z_shape=1):
    """
    Create a torch autograd-enabled function from a 2D parallel-beam
    :class:`odl.tomo.RayTransform` using tomosipo that calls the direct
    backward projection routine of astra, which avoids copying between GPU and
    CPU (available in 1.9.9.dev4).

    Parameters
    ----------
    ray_trafo : :class:`odl.tomo.RayTransform`
        Ray transform
    z_shape : int, optional
        Batch dimension.
        Default: ``1``.

    Returns
    -------
    torch_ray_trafo_adjoint : callable
        Torch autograd-enabled function applying the parallel-beam backward
        projection.
        Input and output have a trivial leading batch dimension and a channel
        dimension specified by `z_shape` (default ``1``), i.e. the
        input shape is ``(1, z_shape) + ray_trafo.range.shape`` and the
        output shape is ``(1, z_shape) + ray_trafo.domain.shape``.
    """
    if not TOMOSIPO_AVAILABLE:
        raise ImportError(MISSING_TOMOSIPO_MESSAGE)
    if not ASTRA_AVAILABLE:
        raise RuntimeError('Astra is not available.')
    if not astra.use_cuda():
        raise RuntimeError('Astra is not able to use CUDA.')
    vg = from_odl(discretized_space_2d_to_3d(ray_trafo.domain,
                                             z_shape=z_shape))
    pg = from_odl(parallel_2d_to_3d_geometry(ray_trafo.geometry,
                                             det_z_shape=z_shape))
    ts_op = ts.operator(vg, pg)
    torch_ray_trafo_adjoint_ts = to_autograd(ts_op.T)
    scaling_factor = astra_cuda_bp_scaling_factor(
        ray_trafo.range, ray_trafo.domain, ray_trafo.geometry)
    def torch_ray_trafo_adjoint(y):
        return scaling_factor * torch_ray_trafo_adjoint_ts(y)
    return torch_ray_trafo_adjoint

def load_state_dict_convert_data_parallel(model, state_dict):
    """
    Load a state dict into a model, while automatically converting the weight
    names if :attr:`model` is a :class:`nn.DataParallel`-model but the stored
    state dict stems from a non-data-parallel model, or vice versa.

    Parameters
    ----------
    model : nn.Module
        Torch model that should load the state dict.
    state_dict : dict
        Torch state dict

    Raises
    ------
    RuntimeError
        If there are missing or unexpected keys in the state dict.
        This error is not raised when conversion of the weight names succeeds.
    """
    missing_keys, unexpected_keys = model.load_state_dict(
        state_dict, strict=False)
    if missing_keys or unexpected_keys:
        # since directly loading failed, assume now that state_dict's
        # keys are named in the other way compared to type(model)
        if isinstance(model, torch.nn.DataParallel):
            state_dict = {('module.' + k): v
                          for k, v in state_dict.items()}
            missing_keys2, unexpected_keys2 = (
                model.load_state_dict(state_dict, strict=False))
            if missing_keys2 or unexpected_keys2:
                if len(missing_keys2) < len(missing_keys):
                    raise RuntimeError(
                        'Failed to load learned weights. Missing keys (in '
                        'case of prefixing with \'module.\', which lead to '
                        'fewer missing keys):\n{}'
                        .format(', '.join(
                            ('"{}"'.format(k) for k in missing_keys2))))
                else:
                    raise RuntimeError(
                        'Failed to load learned weights (also when trying '
                        'with additional \'module.\' prefix). Missing '
                        'keys:\n{}'
                        .format(', '.join(
                            ('"{}"'.format(k) for k in missing_keys))))
        else:
            if all(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k[len('module.'):]: v
                              for k, v in state_dict.items()}
                missing_keys2, unexpected_keys2 = (
                    model.load_state_dict(state_dict,
                                               strict=False))
                if missing_keys2 or unexpected_keys2:
                    if len(missing_keys2) < len(missing_keys):
                        raise RuntimeError(
                            'Failed to load learned weights. Missing keys (in '
                            'case of removing \'module.\' prefix, which lead '
                            'to fewer missing keys):\n{}'
                            .format(', '.join(
                                ('"{}"'.format(k) for k in missing_keys2))))
                    else:
                        raise RuntimeError(
                            'Failed to load learned weights (also when '
                            'removing \'module.\' prefix). Missing keys:\n{}'
                            .format(', '.join(
                                ('"{}"'.format(k) for k in missing_keys))))
            else:
                raise RuntimeError(
                    'Failed to load learned weights. Missing keys:\n{}'
                    .format(', '.join(
                        ('"{}"'.format(k) for k in missing_keys))))
