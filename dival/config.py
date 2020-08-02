# -*- coding: utf-8 -*-
"""
Configuration of the library.

The module variable :obj:`CONFIG` holds the current configuration.

The library configuration is stored in the file ``~/.dival/config.json``.
If the config file does not exist when the library is imported, it is created
using default values.
Changes made in ``config.json`` are loaded when the library is (re-)imported.
"""
import os
import json

CONFIG = {
 'lodopab_dataset': {
  'data_path': os.path.normpath(
      os.path.expanduser('~/.dival/datasets/lodopab'))
 },
 'reference_params': {
  'data_path': os.path.normpath(
      os.path.expanduser('~/.dival/reference_params'))
 }
}
"""
Global configuration dict.

Holds the current configuration of the library. On ``import dival``, the
configuration is loaded from ``~/.dival/config.json``.
"""

# configuration loaded from config file, overrides hardcoded configuration
CONFIG_FILENAME = os.path.normpath(os.path.expanduser('~/.dival/config.json'))
"""
Path of the configuration file.
The value is given by ``'~/.dival/config.json'``, expanded and normalized.
"""

# automatically write config file if not existing (e.g., on first import)
if not os.path.isfile(CONFIG_FILENAME):
    os.makedirs(os.path.dirname(CONFIG_FILENAME), exist_ok=True)
    with open(CONFIG_FILENAME, 'w') as config_fp:
        json.dump(CONFIG, config_fp, indent=1)
with open(CONFIG_FILENAME, 'r') as config_fp:
    CONFIG.update(json.load(config_fp))


def get_config(key_path='/'):
    """
    Return (sub-)configuration stored in config file.
    Note that values may differ from the current ``CONFIG`` variable if it was
    manipulated directly.

    Parameters
    ----------
    key_path : str, optional
        ``'/'``-separated path to sub-configuration. Default is ``'/'``, which
        returns the full configuration dict.

    Returns
    -------
    sub_config
        (sub-)configuration, either a dict or a value
    """
    keys = [k for k in key_path.split('/') if k != '']
    with open(CONFIG_FILENAME, 'r') as config_fp:
        config = json.load(config_fp)
    sub_config = config
    for k in keys:
        sub_config = sub_config[k]
    return sub_config


def set_config(key_path, value, verbose=True):
    """
    Updates (sub-)configuration both in ``CONFIG`` variable and in config file.

    Parameters
    ----------
    key_path : str, optional
        ``'/'``-separated path to sub-configuration. Pass ``'/'`` to replace
        the full configuration dict.
    value : object
        (sub-)configuration value. Either a dict, which is copied, or a value.
    """
    global CONFIG
    c = get_config()
    keys = [k for k in key_path.split('/') if k != '']
    if isinstance(value, dict):
        value = value.copy()
    if len(keys) == 0:
        CONFIG = value
        c = value
    else:
        sub_config = CONFIG
        sub_c = c
        for k in keys[:-1]:
            sub_config = sub_config[k]
            sub_c = sub_c[k]
        sub_config[keys[-1]] = value
        sub_c[keys[-1]] = value
    with open(CONFIG_FILENAME, 'w') as config_fp:
        json.dump(c, config_fp, indent=1)
    if verbose:
        print("updated configuration in '{}':".format(CONFIG_FILENAME))
        print("'{}' = {}".format(key_path, value))
