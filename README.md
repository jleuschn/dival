# Deep Inversion Validation Library

Library for testing and comparing deep learning based methods for inverse
problems, written in python.

## Standard datasets

One main goal of this library is to provide public standard datasets suitable
for deep learning.
Currently, the following datasets are included:

* ``'ellipses'``:
    A typical synthetical CT dataset with ellipse phantoms.
* ``'lidc_idri_dival'``:
    A dataset based on real CT reconstructions from the [LIDC-IDRI](
    https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI) dataset.

These datasets can be accessed by calling ``dival.get_standard_dataset(name)``.

### Downloading LIDC-IDRI images
For the real CT dataset ``'lidc_idri_dival'`` images from the [LIDC-IDRI](
https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI) dataset are
used. They must be stored in a directory, that can be configured by editing
``config.json`` (under package root). If you have not downloaded the data yet,
there are two options:

1. use the [NBIA Data Retriever](https://wiki.cancerimagingarchive.net/display
   /NBIA/Downloading+the+NBIA+Data+Retriever+7.0) (requires free user login)
2. run the script ``datasets/lidc_idri_dival/download_images.py``

The script will only store the files that are used by the ``'lidc_idri_dival'``
dataset (~30 GB instead of ~135 GB).
