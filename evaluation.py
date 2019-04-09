# -*- coding: utf-8 -*-
"""Provides classes and methods useful for evaluation of methods."""
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from util.plot import plot_image


class TestData:
    """
    Bundles an `observation` with a `ground_truth`.

    Attributes
    ----------
    observation : `Data`
        The observation, possibly distorted or low-dimensional.
    ground_truth : `Data`
        The ground truth. May be replaced with a good quality reference.
        Reconstructors will be evaluated by comparing their reconstructions
        with this value. May also be ``None`` if no evaluation based on
        ground truth shall be performed.
    """
    def __init__(self, observation, ground_truth=None,
                 name='', short_name='', description=''):
        self.observation = observation
        self.ground_truth = ground_truth
        self.name = name
        self.short_name = short_name
        self.description = description


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
        Short name of the measure, used as identifier (e.g. dict key).
    name : str
        Name of the measure.
    description : str
        Description of the measure.

    Methods
    -------
    apply(reconstruction, ground_truth)
        Calculate the value of this measure.
    """
    measure_type = None
    short_name = ''
    name = ''
    description = ''

    @abstractmethod
    def apply(self, reconstruction, ground_truth):
        """Calculate the value of this measure.

        Returns
        -------
        float
            The value of this measure for the given `reconstruction` and
            `ground_truth`.
        """


def measure_by_short_name(short_name):
    """Return a measure object by giving a short name.

    Parameters
    ----------
    short_name : str
        The `short_name` of the `Measure` subclass.

    Returns
    -------
    subclass of `Measure`
        Object of the measure class given by `short_name`.
    """
    measure_classes = [L2Measure, MSEMeasure, PSNRMeasure]
    measure_dict = {m.short_name: m for m in measure_classes}
    try:
        measure_class = measure_dict[short_name.lower()]
    except KeyError:
        raise ValueError('unknown measure name \'{}\''.format(short_name))
    return measure_class()


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


class MSEMeasure(Measure):
    """The mean squared error distance measure."""
    measure_type = 'distance'
    short_name = 'mse'
    name = 'mean squared error'
    description = ('distance given by '
                   '1/n * sum((reconstruction-ground_truth)**2)')

    def apply(self, reconstruction, ground_truth):
        return np.mean((reconstruction.asarray() - ground_truth.asarray())**2)


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

    def __init__(self, max_value=None):
        self.max_value = max_value

    def apply(self, reconstruction, ground_truth):
        gt = ground_truth.asarray()
        mse = np.mean((reconstruction.asarray() - gt)**2)
        if mse == 0.:
            return float('inf')
        max_value = self.max_value or np.max(gt)
        return 20*np.log10(max_value) - 10*np.log10(mse)


class EvaluationTaskTable:
    """Task table containing reconstruction tasks to evaluate."""
    def __init__(self, tasks=None, name=''):
        self.tasks = tasks or []
        self.name = name

    def run(self, save_reconstructions=True):
        """Run all tasks and return the results.

        Parameters
        ----------
        save_reconstructions : bool, optional
            Whether the reconstructions should be saved in the results.
            If measures shall be applied after this method returns,
            it must be ``True``.
        """
        results = EvaluationResultTable()
        row_list = []
        for task in self.tasks:
            test_data = task['test_data']
            reconstruction = task['reconstructor'].reconstruct(
                test_data.observation)
            measure_values = {}
            for measure in task['measures']:
                measure_values[measure.short_name] = measure.apply(
                    reconstruction, test_data.ground_truth)
            row = {'reconstruction': None,
                   'test_data': test_data,
                   'measure_values': measure_values}
            if save_reconstructions:
                row['reconstruction'] = reconstruction
            row_list.append(row)
        results.results = pd.concat([results.results, pd.DataFrame(row_list)],
                                    ignore_index=True, sort=False)
        return results

    def append(self, test_data, reconstructor, measures=None):
        """Append a task."""
        if measures is None:
            measures = []
        self.tasks.append({'test_data': test_data,
                           'reconstructor': reconstructor,
                           'measures': measures})

    def append_all_combinations(self, test_data, reconstructors,
                                measures=None):
        """Append all combinations of the passed parameter lists as tasks."""
        for test_data_ in test_data:
            for reconstructor in reconstructors:
                self.append(test_data_, reconstructor, measures)

    def __repr__(self):
        return "EvaluationTaskTable(name='{}', tasks={})".format(
            self.name, self.tasks.__repr__())


class EvaluationResultTable:
    """Result table of running an evaluation task table.

    Attributes
    ----------
    results : `pandas.DataFrame`
        The results.
        It has at least the columns ``"reconstructions"``, ``"test_data"`` and
        ``"measure_values"``.
    """
    def __init__(self, reconstructions=None, test_data=None,
                 measure_values=None):
        if reconstructions is None:
            reconstructions = []
        if test_data is None:
            test_data = []
        if measure_values is None:
            measure_values = []
        data_dict = {'reconstruction': reconstructions,
                     'test_data': test_data,
                     'measure_values': measure_values}
        self.results = pd.DataFrame.from_dict(data_dict)

    def apply_measures(self, measures, index=None):
        """Apply (additional) measures to reconstructions.

        Only possible if the reconstructions were saved.

        Parameters
        ----------
        measures : list of Measure
            Measures to apply.
        index : int or sequence of ints, optional
            Indexes of results to which the measures shall be applied.
            If `index` is ``None``, this is interpreted as "all results".
        """
        if index is None:
            indexes = range(len(self.results))
        elif np.isscalar(index):
            indexes = [index]
        elif isinstance(index, list):
            indexes = index
        else:
            raise ValueError('index must be a scalar, a list of ints or '
                             '``None``')
        for i in indexes:
            row = self.results.iloc[i]
            for measure in measures:
                row['measure_values'][measure.short_name] = measure.apply(
                    row['reconstruction'], row['test_data'].ground_truth)

    def plot_reconstruction(self, index):
        """Plot the reconstruction at the specified index.

        Parameters
        ----------
        index : int
            Index of the reconstruction.
        """
        reconstruction = self.results.iloc[index].at['reconstruction']
        if reconstruction is None:
            raise ValueError('reconstruction is ``None``')
        if reconstruction.asarray().ndim == 1:
            plt.plot(reconstruction)
        elif reconstruction.asarray().ndim == 2:
            plot_image(reconstruction)
        else:
            print('only 1d and 2d reconstructions can be plotted (currently)')

    def plot_all_reconstructions(self):
        """Plot all reconstructions."""
        for i in range(len(self.results)):
            self.plot_reconstruction(i)

    def to_string(self, **kwargs):
        """Convert to string.

        kwargs : dict
            Keyword arguments passed to `self.results.to_string`.
        """
        def test_data_formatter(test_data):
            if test_data.name:
                return test_data.name
            else:
                return test_data.__str__()
        return "EvaluationResultTable(results=\n{}\n)".\
            format(self.results.to_string(formatters={'test_data':
                                                      test_data_formatter},
                                          **kwargs))

    def __repr__(self):
        return "EvaluationResultTable(results=\n{}\n)".format(self.results)

    def __str__(self):
        return self.to_string()
