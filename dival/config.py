# -*- coding: utf-8 -*-
import os
import json

# hardcoded configuration
CONFIG = {
 'lidc_idri_dival': {
  'data_path': '/localdata/LIDC-IDRI'
 }
}

# configuration loaded from config file, overrides hardcoded configuration
if 'APPDATA' in os.environ:
    _config_folder = os.environ['APPDATA']
elif 'XDG_CONFIG_HOME' in os.environ:
    _config_folder = os.environ['XDG_CONFIG_HOME']
else:
    _config_folder = os.path.join(os.environ['HOME'], '.config')
CONFIG_FILENAME = os.path.join(_config_folder, 'dival', 'config.json')

if not os.path.isfile(CONFIG_FILENAME):
    os.makedirs(os.path.dirname(CONFIG_FILENAME), exist_ok=True)
    with open(CONFIG_FILENAME, 'w') as config_fp:
        json.dump(CONFIG, config_fp, indent=1)
with open(CONFIG_FILENAME, 'r') as config_fp:
    CONFIG.update(json.load(config_fp))
