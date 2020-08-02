# -*- coding: utf-8 -*-
import os
import requests
from dival.util.input import input_yes_no
from dival.util.download import download_file


def download_zenodo_record(record_id, base_path='', md5sum_check=True):
    """
    Download a zenodo record.

    Parameters
    ----------
    record_id : str
        Record id.
    base_path : str, optional
        Path to store the downloaded files in. Default is the current folder.
    md5sum_check : bool, optional
        Whether to check the MD5 sum of each downloaded file.

    Returns
    -------
    success : bool
        If ``md5sum_check=True``, whether all sums matched. Otherwise the
        returned value is always ``True``.
    """
    r = requests.get('https://zenodo.org/api/records/{}'.format(record_id))
    files = r.json()['files']
    success = True
    for i, f in enumerate(files):
        url = f['links']['self']
        filename = f['key']
        size_kb = f['size'] / 1000
        checksum = f['checksum']
        print("downloading file {:d}/{:d}: '{}', {}KB".format(
            i+1, len(files), filename, size_kb))
        if md5sum_check:
            retry = True
            md5sum_matches = False
            while retry and not md5sum_matches:
                md5sum = download_file(url, os.path.join(base_path, filename),
                                       md5sum=True)
                md5sum_matches = (md5sum == checksum.split(':')[1])
                if not md5sum_matches:
                    print("md5 checksum does not match for file '{}'. Retry "
                          "downloading? (y)/n".format(filename))
                    retry = input_yes_no()
            if not md5sum_matches:
                success = False
                print('record download aborted')
                break
        else:
            download_file(url, os.path.join(base_path, filename), md5sum=False)
    return success
