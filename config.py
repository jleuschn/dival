# -*- coding: utf-8 -*-
import os
import json

# hardcoded configuration
CONFIG = {}

# configuration loaded from config file
CONFIG_FILENAME = os.path.join(os.path.dirname(__file__), 'config.json')

if os.path.isfile(CONFIG_FILENAME):
    with open(CONFIG_FILENAME, 'r') as config_fp:
        CONFIG.update(json.load(config_fp))
