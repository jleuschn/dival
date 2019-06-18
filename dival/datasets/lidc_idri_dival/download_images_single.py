# -*- coding: utf-8 -*-
import os
import json
from itertools import chain
from tqdm import tqdm
from tciaclient import TCIAClient

PATH = '/localdata/lidc_idri_dival'
FILE_LIST_FILE = os.path.join(os.path.dirname(__file__),
                              'lidc_idri_file_list.json')
UID_LIST_FILE = os.path.join(os.path.dirname(__file__),
                             'lidc_idri_uid_list.json')
API_KEY = '0b59e3b6-6baf-4e1f-a2ba-32fcaf0c82f8'


os.makedirs(PATH, exist_ok=True)

client = TCIAClient(
    apiKey=API_KEY,
    baseUrl='https://services.cancerimagingarchive.net/services/v3',
    resource='TCIA')

with open(FILE_LIST_FILE, 'r') as json_file:
    file_list_dict = json.load(json_file)
with open(UID_LIST_FILE, 'r') as json_file:
    uid_list_dict = json.load(json_file)
for (series_uid, uid), filepath in tqdm(
        chain(zip(uid_list_dict['train'], file_list_dict['train']),
              zip(uid_list_dict['validation'], file_list_dict['validation']),
              zip(uid_list_dict['test'], file_list_dict['test'])),
        desc='download LIDC-IDRI images',
        total=sum((len(v) for v in uid_list_dict.values()))):
    os.makedirs(os.path.join(PATH, os.path.dirname(filepath)), exist_ok=True)
    client.get_single_image(series_uid, uid, PATH, filepath)
