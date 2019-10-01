# -*- coding: utf-8 -*-
import os
import requests
import tqdm
import hashlib


def download_file(url, filename=False, chunk_size=1024, verbose=False,
                  md5sum=True):
    """
    Download file with progressbar.

    Based on
    `<https://gist.github.com/ruxi/5d6803c116ec1130d484a4ab8c00c603>`_
    (MIT License).

    Parameters
    ----------
    url : str
        URL to download from.
    filename : str, optional
        Local destination filename (including path).
        By default, the file is stored under the current folder and the name
        given by the last part of `url`.
    chunk_size : int, optional
        Number of bytes in a chunk.
    verbose : bool, optional
        Whether to print additional information.
    md5sum : bool, optional
        Whether to compute and return the MD5 checksum (hex-digest).

    Returns
    -------
    md5sum : str or `None`
        Hex-digest of the MD5 hash, if ``md5sum=True``, otherwise `None`.
    """
    if not filename:
        local_filename = os.path.join(".", url.split('/')[-1])
    else:
        local_filename = filename
    r = requests.get(url, stream=True)
    file_size = int(r.headers['Content-Length'])
    chunk = 1
    num_bars = int(file_size / chunk_size)
    if verbose:
        print(dict(file_size=file_size))
        print(dict(num_bars=num_bars))

    if md5sum:
        hash_md5 = hashlib.md5()

    with open(local_filename, 'wb') as fp:
        for chunk in tqdm.tqdm(r.iter_content(chunk_size=chunk_size),
                               total=num_bars,
                               unit='chunks',
                               desc=local_filename,
                               leave=True):
            fp.write(chunk)
            if md5sum:
                hash_md5.update(chunk)
    if md5sum:
        return hash_md5.hexdigest()


def input_yes_no(default='y'):
    """
    Demand user input y[es] or n[o].

    The user is asked repeatedly, until the input is valid.

    Parameters
    ----------
    default : {``'y'``, ``'n'``}, optional
        The output if the user enters empty input.

    Returns
    -------
    inp : {``'y'``, ``'n'``}
        The users input (or `default`).
    """
    def _input():
        inp = input()
        inp = inp.lower()
        if inp in ['y', 'yes']:
            inp = 'y'
        elif inp in ['n', 'no']:
            inp = 'n'
        elif inp == '':
            inp = default
        else:
            print('please input y[es] or n[o]')
            return None
        return inp

    inp = _input()
    while inp not in ['y', 'n']:
        inp = _input()

    return inp == 'y'


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
