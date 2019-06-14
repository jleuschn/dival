# -*- coding: utf-8 -*-
import os
import json
from tqdm import tqdm
from pydicom.filereader import dcmread
from create_file_list import get_dirs
from dival.config import CONFIG


DATA_PATH = CONFIG['lidc_idri_dival']['data_path']
FILE_LIST_FILE = os.path.join(os.path.dirname(__file__),
                              'lidc_idri_file_list.json')
SERIES_DICT_FILE = os.path.join(os.path.dirname(__file__),
                                'lidc_idri_series_dict.json')

with open(FILE_LIST_FILE, 'r') as json_file:
    file_dict = json.load(json_file)
    file_list = []
    for files in file_dict.values():
        file_list += files
ct_dirs = get_dirs()
series_dict = dict()
for ct_dir in tqdm(ct_dirs):
    dcm_files = [f for f in os.listdir(os.path.join(DATA_PATH, ct_dir))
                 if (os.path.isfile(os.path.join(DATA_PATH, ct_dir, f))
                 and f.endswith('.dcm'))]
    dcm_files.sort()
    dataset = dcmread(os.path.join(DATA_PATH, ct_dir, dcm_files[0]))
    file_dict = {i: f for i, f in enumerate(dcm_files) if
                 os.path.join(ct_dir, f) in file_list}
    series_dict[dataset.SeriesInstanceUID] = [ct_dir, file_dict]

with open(SERIES_DICT_FILE, 'w') as json_file:
    json.dump(series_dict, json_file, indent=True)
