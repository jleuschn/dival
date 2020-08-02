# Deep Inversion Validation Library

Library for testing and comparing deep learning based methods for inverse
problems, written in python.

See the [documentation](https://jleuschn.github.io/docs.dival/).

The project is also available on [PyPI](https://pypi.org/project/dival/).

## Standard datasets

One main goal of this library is to provide public standard datasets suitable
for deep learning.
Currently, the following datasets are included:

* ``'ellipses'``:
    A typical synthetical CT dataset with ellipse phantoms.
* ``'lodopab'``:
    The public [LoDoPaB-CT dataset](https://doi.org/10.5281/zenodo.3384092),
    based on real CT reconstructions from the public
    [LIDC-IDRI](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI)
    dataset.

These datasets can be accessed by calling ``dival.get_standard_dataset(name)``.

## Contribute

We would like to include more reconstruction methods. If you know of classical
or state-of-the-art methods that should not be missing in our library, please
let us know!

Also, bug reports and suggestions on improving our library are welcome.
Please file an issue for such a purpose.
