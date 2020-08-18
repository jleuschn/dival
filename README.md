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

*Note on astra and CUDA:* The CT datasets come with a ``ray_trafo`` attribute providing
the forward operator.
There are different backend implementations for it, the default is ``'astra_cuda'``.
This requires both the [astra-toolbox](https://www.astra-toolbox.com/) and
a CUDA-enabled GPU being available.
In order to use the (slow) scikit-image backend instead, you can pass ``impl='skimage'`` to ``get_standard_dataset``. If astra is
available but CUDA is not, ``impl='astra_cpu'`` is preferred.
The latest development version of astra can be installed with
``conda install astra-toolbox -c astra-toolbox/label/dev``.

## Contribute

We would like to include more reconstruction methods. If you know of classical
or state-of-the-art methods that should not be missing in our library, please
let us know!

Also, bug reports and suggestions on improving our library are welcome.
Please file an issue for such a purpose.
