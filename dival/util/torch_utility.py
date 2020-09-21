"""Provides utilities related to PyTorch."""
import torch

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
