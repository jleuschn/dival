# Deep Inversion Validation Library

Library for testing and comparing deep learning based methods for inverse
problems, written in python.

## Standard datasets

One main goal of this library is to provide public standard datasets suitable for
deep learning.
Currently, the following datasets are included:

* ``'ellipses'``:
    A typical synthetical CT dataset with ellipse phantoms.
* ``'lidc_idri_dival'``:
    A dataset based on real CT reconstructions from the [LIDC-IDRI](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI) dataset.

These datasets can be accessed by calling ``dival.get_standard_dataset(name)``.
