# -*- coding: utf-8 -*-
import os
import json
from tqdm import tqdm
from zipfile import ZipFile
from tciaclient import TCIAClient
from dival.config import CONFIG

FILE_LIST_FILE = os.path.join(os.path.dirname(__file__),
                              'lidc_idri_file_list.json')
SERIES_DICT_FILE = os.path.join(os.path.dirname(__file__),
                                'lidc_idri_series_dict.json')
API_KEY = '0b59e3b6-6baf-4e1f-a2ba-32fcaf0c82f8'


def download_images(path, file_list_file=FILE_LIST_FILE,
                    series_dict_file=SERIES_DICT_FILE):
    os.makedirs(path, exist_ok=True)

    client = TCIAClient(
        apiKey=API_KEY,
        baseUrl='https://services.cancerimagingarchive.net/services/v3',
        resource='TCIA')

    tmp_dir = os.path.join(path, 'tmp')
    os.makedirs(tmp_dir, exist_ok=True)
    with open(file_list_file, 'r') as json_file:
        file_list = []
        for files in json.load(json_file).values():
            file_list += files
    zip_filename = 'tmp.zip'
    with open(series_dict_file, 'r') as json_file:
        series_dict = json.load(json_file)
    for seriesInstanceUID, (ct_dir, file_dict) in tqdm(
            series_dict.items(), desc='download LIDC-IDRI images'):
        abs_ct_dir = os.path.join(path, ct_dir)
        client.get_image(seriesInstanceUid=seriesInstanceUID,
                         downloadPath=path, zipFileName=zip_filename)
        with ZipFile(os.path.join(path, zip_filename), 'r') as zip_file:
            files = [f for f in zip_file.namelist() if
                     not zip_file.getinfo(f).is_dir() and f.endswith('.dcm')]
            os.makedirs(abs_ct_dir, exist_ok=True)
            members = [files[int(i)] for i in file_dict.keys()]
            zip_file.extractall(tmp_dir, members=members)
            for file, f in zip(members, file_dict.values()):
                os.rename(os.path.join(tmp_dir, file),
                          os.path.join(abs_ct_dir, f))
    os.remove(os.path.join(path, zip_filename))
    os.rmdir(tmp_dir)


if __name__ == '__main__':
    DATA_PATH = CONFIG['lidc_idri_dival']['data_path']  # path to store the
#    relevant LIDC-IDRI images. See also lidc_idri_dival_dataset.py.
    download_images(DATA_PATH)
