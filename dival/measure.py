# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from warnings import warn
import numpy as np
from skimage.measure import compare_ssim
from odl.operator.operator import Operator


def gen_unique_name(name_orig):
    i = 1
    while True:
        yield '{}_{}'.format(name_orig, str(i))
        i += 1


class Measure(ABC):
    """Abstract base class for measures used for evaluation.

    Attributes
    ----------
    measure_type : {'distance', 'quality'}
        The measure type.
        Measures with type ``'distance'`` should attain small values if the
        reconstruction is good. Measures with type ``'quality'`` should attain
        large values if the reconstruction is good.
    short_name : str
        Short name of the measure, used as identifier (key in `measure_dict`).
    name : str
        Name of the measure.
    description : str
        Description of the measure.
    measure_dict : dict, class attribute
        Registry of all measures with their `short_name` as key.

    Methods
    -------
    apply(reconstruction, ground_truth)
        Calculate the value of this measure.

    Class methods
    -------------
    get_by_short_name(short_name)
        Return `Measure` instance with given short name by registry lookup.
    """
    measure_type = None
    short_name = ''
    name = ''
    description = ''

    measure_dict = {}

    def __init__(self, short_name=None):
        if short_name is not None:
            self.short_name = short_name
        elif self.short_name is None:
            self.short_name = self.__class__.__name__
        if self.short_name in self.__class__.measure_dict:
            old_short_name = self.short_name
            unique_name = gen_unique_name(self.short_name)
            while self.short_name in self.__class__.measure_dict:
                self.short_name = next(unique_name)
            warn("Measure `short_name` '{}' already exists, changed to '{}'"
                 .format(old_short_name, self.short_name))
        self.__class__.measure_dict[self.short_name] = self

    @abstractmethod
    def apply(self, reconstruction, ground_truth):
        """Calculate the value of this measure.

        Returns
        -------
        float
            The value of this measure for the given `reconstruction` and
            `ground_truth`.
        """

    def __call__(self, reconstruction, ground_truth):
        """Calculate the value of this measure by calling `apply`.

        Returns
        -------
        float
            The value of this measure for the given `reconstruction` and
            `ground_truth`.
        """
        return self.apply(self, reconstruction, ground_truth)

    @classmethod
    def get_by_short_name(cls, short_name):
        return cls.measure_dict.get(short_name)

    class _OperatorForFixedGroundTruth(Operator):
        def __init__(self, measure, ground_truth):
            super().__init__(ground_truth.space, ground_truth.space.field)
            self.measure = measure
            self.ground_truth = ground_truth

        def _call(self, x):
            return self.measure.apply(x, self.ground_truth)

    def as_operator_for_fixed_ground_truth(self, ground_truth):
        return self._OperatorForFixedGroundTruth(self, ground_truth)


class L2Measure(Measure):
    """The euclidean (l2) distance measure."""
    measure_type = 'distance'
    short_name = 'l2'
    name = 'euclidean distance'
    description = ('distance given by '
                   'sqrt(sum((reconstruction-ground_truth)**2))')

    def apply(self, reconstruction, ground_truth):
        return np.linalg.norm((reconstruction.asarray() -
                               ground_truth.asarray()).flat)


L2 = L2Measure()


class MSEMeasure(Measure):
    """The mean squared error distance measure."""
    measure_type = 'distance'
    short_name = 'mse'
    name = 'mean squared error'
    description = ('distance given by '
                   '1/n * sum((reconstruction-ground_truth)**2)')

    def apply(self, reconstruction, ground_truth):
        return np.mean((reconstruction.asarray() - ground_truth.asarray())**2)


MSE = MSEMeasure()


class PSNRMeasure(Measure):
    """The peak signal-to-noise ratio (PSNR) measure.

    Attributes
    ----------
    max_value : float
        Maximum image value that is possible.
        If `max_value` is ``None``, ``np.max(ground_truth)`` is used in
        `apply`.
    """
    measure_type = 'quality'
    short_name = 'psnr'
    name = 'peak signal-to-noise ratio'
    description = 'quality given by 10*log10(MAX**2/MSE)'

    def __init__(self, max_value=None, short_name=None):
        self.max_value = max_value
        if self.max_value is not None and short_name is None:
            short_name = '{}_max{}'.format(self.__class__.short_name,
                                           self.max_value)
        super().__init__(short_name=short_name)

    def apply(self, reconstruction, ground_truth):
        gt = ground_truth.asarray()
        mse = np.mean((reconstruction.asarray() - gt)**2)
        if mse == 0.:
            return float('inf')
        max_value = self.max_value or np.max(gt)
        return 20*np.log10(max_value) - 10*np.log10(mse)


PSNR = PSNRMeasure()


class SSIMMeasure(Measure):
    """The structural similarity index measure."""
    measure_type = 'quality'
    short_name = 'ssim'
    name = 'structural similarity index'
    description = ('The (M)SSIM like described in `Wang et al. 2014 '
                   '<https://doi.org/10.1109/TIP.2003.819861>`_.')

    def __init__(self, *args, short_name=None, **kwargs):
        """Construct a new SSIM measure.

        This is a wrapper for `skimage.measure.compare_ssim`.

        Parameters
        ----------
        args : list
            Arguments that will be passed to `compare_ssim`.
        kwargs : dict
            Keyword arguments that will be passed to `compare_ssim`.
        """
        self.args = args
        self.kwargs = kwargs
        super().__init__(short_name=short_name)

    def apply(self, reconstruction, ground_truth):
        return compare_ssim(reconstruction, ground_truth, *self.args,
                            **self.kwargs)


SSIM = SSIMMeasure()
