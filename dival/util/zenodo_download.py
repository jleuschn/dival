# -*- coding: utf-8 -*-
import os
import requests
from dival.util.input import input_yes_no
from dival.util.download import download_file, compute_md5sum


def download_zenodo_record(record_id, base_path='', md5sum_check=True,
                           auto_yes=False):
    """
    Download a zenodo record.
    Unfortunately, downloads cannot be resumed, so this method is only
    recommended for stable internet connections.

    Parameters
    ----------
    record_id : str
        Record id.
    base_path : str, optional
        Path to store the downloaded files in. Default is the current folder.
    md5sum_check : bool, optional
        Whether to check the MD5 sum of each downloaded file.
        Default: `True`
    auto_yes : bool, optional
        Whether to answer user input questions with "y" by default.
        User input questions are:
        If ``md5sum_check=True``, in case of a checksum mismatch: whether to
        retry downloading.
        If ``md5sum_check=False``, in case of an existing file of correct size:
        whether to re-download.
        Default: `False`

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
        path = os.path.join(base_path, filename)
        size = f['size']
        size_kb = size / 1000
        checksum = f['checksum']
        try:
            size_existing = os.stat(path).st_size
        except OSError:
            size_existing = -1
        if size_existing == size:
            if md5sum_check:
                print("File {:d}/{:d}, '{}', {}KB already exists with correct "
                      "size. Will check md5 sum now.".format(
                          i+1, len(files), filename, size_kb))
                md5sum_existing = compute_md5sum(path)
                md5sum_matches = (md5sum_existing == checksum.split(':')[1])
                if md5sum_matches:
                    print("skipping file {}, md5 checksum matches".format(
                        filename))
                    continue
                else:
                    print("existing file {} will be overwritten, md5 checksum "
                          "does not match".format(filename))
            else:
                print("File {:d}/{:d}, '{}', {}KB already exists with correct "
                      "size. Re-download this file? (y)/n".format(
                          i+1, len(files), filename, size_kb))
                if auto_yes:
                    print("y")
                    download = True
                else:
                    download = input_yes_no()
                if not download:
                    print("skipping existing file {}".format(filename))
                    continue
        print("downloading file {:d}/{:d}: '{}', {}KB".format(
            i+1, len(files), filename, size_kb))
        if md5sum_check:
            retry = True
            md5sum_matches = False
            while retry and not md5sum_matches:
                md5sum = download_file(url, path, md5sum=True)
                md5sum_matches = (md5sum == checksum.split(':')[1])
                if not md5sum_matches:
                    print("md5 checksum does not match for file '{}'. Retry "
                          "downloading? (y)/n".format(filename))
                    if auto_yes:
                        print("y")
                        retry = True
                    else:
                        retry = input_yes_no()
            if not md5sum_matches:
                success = False
                print('record download aborted')
                break
        else:
            download_file(url, os.path.join(base_path, filename), md5sum=False)
    return success

