#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create the image file list for `LIDCIDRIDivalDataset`.
"""
import os
import json
import numpy as np
from pydicom.filereader import dcmread
from dival.config import CONFIG

np.random.seed(1)


DATA_PATH = CONFIG['lidc_idri_dival']['data_path']
FILE_LIST_FILE = os.path.join(os.path.dirname(__file__),
                              'lidc_idri_file_list.json')

# directories with scans that are not valid in the rectangle to which they are
# cropped (used as a blacklist)
DIRS_WITH_TOO_SMALL_CIRCLES = [
 'LIDC-IDRI-0368/01-01-2000-98197/5445-45926',
 'LIDC-IDRI-0120/01-01-2000-71173/3221-29969',
 'LIDC-IDRI-0289/01-01-2000-CT THORAX WCONTRAST-50294/4-Recon 3 C-A-P-34169',
 'LIDC-IDRI-0798/01-01-2000-50632/5424-05549',
 'LIDC-IDRI-0418/01-01-2000-71464/3000651-51680',
 'LIDC-IDRI-0972/01-01-2000-95294/5447-10714',
 'LIDC-IDRI-1000/01-01-2000-96031/3128-29768',
 'LIDC-IDRI-0102/01-01-2000-76051/3000670-70157',
 'LIDC-IDRI-0032/01-01-2000-53482/3000537-91689',
 'LIDC-IDRI-0116/01-01-2000-34322/3000700-19196',
 'LIDC-IDRI-0004/01-01-2000-91780/3000534-58228',
 'LIDC-IDRI-0541/01-01-2000-89260/5515-39652',
 'LIDC-IDRI-0926/01-01-2000-39841/5343-06818',
]


def get_dirs():
    """Return the list of directories to include in the dataset.

    Each directory contains a 3D scan (multiple dicom files).
    """
    def is_valid_dataset(dataset):
        return (dataset.Modality == 'CT' and
                dataset.Rows == 512 and dataset.Columns == 512 and
                dataset.pixel_array.ndim == 2 and
                [float(a) for a in dataset.ImageOrientationPatient] ==
                [1., 0., 0., 0., 1., 0.])

    ct_dirs = []
    for root, _, files in os.walk(DATA_PATH):
        dcm_files = [f for f in files if f.endswith('.dcm')]
        if (len(dcm_files) > 1 and  # only series with multiple images
                (os.path.relpath(root, DATA_PATH) not in
                 DIRS_WITH_TOO_SMALL_CIRCLES)):
            dcm_file = dcm_files[0]
            dataset = dcmread(os.path.join(root, dcm_file))
            if is_valid_dataset(dataset):
                ct_dirs.append(os.path.relpath(root, DATA_PATH))
    ct_dirs.sort()

    return ct_dirs


def get_files(ct_dirs, shuffle=True):
    """Return the list of dicom files to include in the dataset.
    """
    selected_files = []
    for ct_dir in ct_dirs:
        dcm_files = [f for f in os.listdir(os.path.join(DATA_PATH, ct_dir))
                     if (os.path.isfile(os.path.join(DATA_PATH, ct_dir, f))
                     and f.endswith('.dcm'))]
        dcm_files.sort()
        dataset = dcmread(os.path.join(DATA_PATH, ct_dir, dcm_files[0]))
        every_n = int(7.5//dataset.SliceThickness)
        for dcm_file in dcm_files[::every_n]:
            selected_files.append(os.path.join(ct_dir, dcm_file))

    if shuffle:
        np.random.shuffle(selected_files)

    return selected_files


if __name__ == '__main__':
    ct_dirs = get_dirs()

    NUM_TEST_PATIENTS = 100  # 50
    NUM_VALIDATION_IMAGES = 5000
    patients = np.unique([s.split('/')[0] for s in ct_dirs])
    np.random.shuffle(patients)
    test_patients = patients[-NUM_TEST_PATIENTS:]
    train_dirs = []
    test_dirs = []
    for ct_dir in ct_dirs:
        if ct_dir.split('/')[0] in test_patients:
            test_dirs.append(ct_dir)
        else:
            train_dirs.append(ct_dir)
    train = get_files(train_dirs)
    test = get_files(test_dirs)

    num_add_test_images = 0  # len(test)
    test += train[len(train)-num_add_test_images:]
    np.random.shuffle(test)
    validation = train[-NUM_VALIDATION_IMAGES-num_add_test_images:
                       -num_add_test_images]
    train = train[:-NUM_VALIDATION_IMAGES-num_add_test_images]

    json_dict = {'train': train,
                 'validation': validation,
                 'test': test}

    with open(FILE_LIST_FILE, 'w') as json_file:
        json.dump(json_dict, json_file, indent=True)
