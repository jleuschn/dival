# -*- coding: utf-8 -*-
import os
import requests
from math import ceil
import tqdm
import hashlib


def download_file(url, filename=None, chunk_size=1024, verbose=False,
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
        Default: `1024`
    verbose : bool, optional
        Whether to print additional information.
        Default: `False`
    md5sum : bool, optional
        Whether to compute and return the MD5 checksum (hex-digest).
        Default: `True`

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
    num_chunks = ceil(file_size / chunk_size)
    if verbose:
        print("downloading {:d} bytes".format(file_size))

    if md5sum:
        hash_md5 = hashlib.md5()

    with open(local_filename, 'wb') as fp:
        for chunk in tqdm.tqdm(r.iter_content(chunk_size=chunk_size),
                               total=num_chunks,
                               unit='chunks',
                               desc=local_filename,
                               leave=True):
            if chunk:
                fp.write(chunk)
                if md5sum:
                    hash_md5.update(chunk)
    if md5sum:
        return hash_md5.hexdigest()


def compute_md5sum(filename, show_pbar=True):
    """
    Compute MD5 checksum of an existing file.

    Parameters
    ----------
    filename : str
        Filename including path.
    show_pbar : str
        Show tqdm progress bar.
        Default: `True`

    Returns
    -------
    md5sum : str
        Hex-digest of the MD5 hash.
    """
    CHUNK_SIZE = 1024
    hash_md5 = hashlib.md5()
    file_size = os.stat(filename).st_size
    num_chunks = ceil(file_size / CHUNK_SIZE)
    with open(filename, 'rb') as fp:
        for _ in tqdm.tqdm(range(num_chunks), unit='chunks', desc='md5sum',
                           disable=not show_pbar):
            chunk = fp.read(CHUNK_SIZE)
            if chunk is not None:
                hash_md5.update(chunk)
    return hash_md5.hexdigest()
