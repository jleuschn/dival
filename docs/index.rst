Deep Inversion Validation Library's documentation
=================================================

This is the documentation of `DIVal <https://github.com/jleuschn/dival>`_, a
library for testing and comparing deep learning based methods for inverse
problems written in python.

To get started, we recommend having a look at the `example scripts
<https://github.com/jleuschn/dival/tree/master/dival/examples>`_.

Below is a list of some fundamental functions and classes.

.. autosummary::
    ~dival.datasets.standard.get_standard_dataset
    ~dival.datasets.dataset.Dataset
    ~dival.reference_reconstructors.get_reference_reconstructor
    ~dival.reconstructors.reconstructor.Reconstructor
    ~dival.measure.Measure
    ~dival.evaluation.TaskTable

Standard datasets
-----------------

One main goal of this library is to provide public standard datasets suitable
for deep learning.
Currently, the following datasets are included:

* ``'ellipses'``:
    A typical synthetical CT dataset with ellipse phantoms.
* ``'lodopab'``:
    The public `LoDoPaB-CT dataset <https://doi.org/10.5281/zenodo.3384092>`_,
    based on real CT reconstructions from the public
    `LIDC-IDRI <https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI>`_
    dataset. See also the
    :class:`~dival.datasets.lodopab_dataset.LoDoPaBDataset` class.

These datasets can be accessed by calling :func:`dival.get_standard_dataset`,
e.g.

.. code-block::

    from dival import get_standard_dataset
    dataset = get_standard_dataset('ellipses')

*Note on astra and CUDA:* The CT datasets come with a ``ray_trafo`` attribute
providing the forward operator.
There are different backend implementations for it, the default is
``'astra_cuda'``. This requires both the
`astra-toolbox <https://www.astra-toolbox.com/>`_ and a CUDA-enabled GPU being
available. In order to use the (slow) scikit-image backend instead, you can
pass ``impl='skimage'`` to ``get_standard_dataset``. If astra is available but
CUDA is not, ``impl='astra_cpu'`` is preferred.
The latest development version of astra can be installed with
``conda install astra-toolbox -c astra-toolbox/label/dev``.

Reference reconstructors
------------------------
For some reconstructor classes on some standard datasets reference
configurations are provided.
These configurations include hyper parameters and pretrained learned parameters
for a :class:`~dival.reconstructors.reconstructor.LearnedReconstructor`.

A reference reconstructor can be retrieved by calling
:func:`dival.get_reference_reconstructor`, e.g.

.. code-block::

    from dival import get_reference_reconstructor
    reconstructor = get_reference_reconstructor('fbp', 'ellipses')

We are happy to receive your feedback (e.g. via
`github issues <https://github.com/jleuschn/dival/issues>`_) if you think a
reference configuration should be improved or another reconstructor should be
included.

Metrics for image reconstruction
--------------------------------
To evaluate image reconstruction performance on datasets that include ground
truth images, full-reference methods like :obj:`~dival.measure.PSNR` and
:obj:`~dival.measure.SSIM` can be used.

Both PSNR and SSIM depend on the data range, which is the difference between
the maximum and the minimum possible values.
While e.g. `scikit-image <https://scikit-image.org/>`_
uses a global data range of ``1.0`` for PSNR and ``2.0`` for SSIM by default
(in case of a floating point data type), an image-dependent default value is
used in dival. If no data range is specified explicitly, it is determined as
``np.max(gt)-np.min(gt)`` from the respective ground truth image ``gt``.
While we consider it to be a drawback of this approach that the metric differs
for different ground truth images, it has the benefit to be quite flexible
in absence of a priori knowledge. If known, we recommend specifying the data
range explicitly.

.. toctree::
    :glob:
    :maxdepth: 3
    :caption: Contents:
    
    dival.*
