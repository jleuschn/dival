# -*- coding: utf-8 -*-
import os
import json
from pydicom.filereader import dcmread
from dival.config import CONFIG


DATA_PATH = CONFIG['lidc_idri_dival']['data_path']
FILE_LIST_FILE = os.path.join(os.path.dirname(__file__),
                              'lidc_idri_file_list.json')
UID_LIST_FILE = os.path.join(os.path.dirname(__file__),
                             'lidc_idri_uid_list.json')

with open(FILE_LIST_FILE, 'r') as json_file:
    dcm_files_dict = json.load(json_file)

uid_dict = {'train': [], 'validation': [], 'test': []}
for part in ('train', 'validation', 'test'):
    for dcm_file in dcm_files_dict[part]:
        dataset = dcmread(os.path.join(DATA_PATH, dcm_file))
        uid_dict[part].append([dataset.SeriesInstanceUID,
                               dataset.SOPInstanceUID])

with open(UID_LIST_FILE, 'w') as json_file:
    json.dump(uid_dict, json_file, indent=True)
