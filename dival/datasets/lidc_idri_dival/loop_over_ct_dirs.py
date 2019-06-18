# -*- coding: utf-8 -*-
"""
Spyder Editor
"""
import os
import json
import numpy as np
from pydicom.filereader import dcmread
from dival.datasets.lidc_idri_dival.create_file_list import get_dirs
from dival.config import CONFIG

np.random.seed(1)


DATA_PATH = CONFIG['lidc_idri_dival']['data_path']

ct_dirs = get_dirs()
scan_options = []
for ct_dir in ct_dirs:
    dcm_files = [f for f in os.listdir(os.path.join(DATA_PATH, ct_dir))
                 if (os.path.isfile(os.path.join(DATA_PATH, ct_dir, f))
                 and f.endswith('.dcm'))]
    dcm_files.sort()
    dataset = dcmread(os.path.join(DATA_PATH, ct_dir, dcm_files[0]))
    scan_options.append(str(dataset.ScanOptions) if 'ScanOptions' in dataset else '')
