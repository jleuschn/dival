# -*- coding: utf-8 -*-
"""Provides classes and methods useful for evaluation of methods."""
from abc import ABC, abstractmethod
import numpy as np
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
        with this value.
    """
    def __init__(self, observation, ground_truth,
                 short_name='', name='', description=''):
        self.observation = observation
        self.ground_truth = ground_truth
        self.short_name = short_name
        self.name = name
        self.description = description


class Measure(ABC):
    """
    Abstract base class for measures used for evaluation.

    Attributes
    ----------
    measure_type : str
        The measure type, e.g. 'distance' or 'quality'.

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


def measure_by_str(string):
    """Return a measure object by giving a short name.

    Parameters
    ----------
    s : str
        The `short_name` of the `Measure` subclass.

    Returns
    -------
    subclass of `Measure`
        Object of the measure class given by `short_name`.
    """
    measure_classes = [L2Measure]
    measure_dict = {m.short_name: m for m in measure_classes}
    try:
        measure_class = measure_dict[string.lower()]
    except KeyError:
        raise ValueError('unknown measure name \'{}\''.format(string))
    return measure_class()


class L2Measure(Measure):
    """The euclidean (l2) distance measure."""
    measure_type = 'distance'
    short_name = 'l2'
    name = 'euclidean distance'
    description = ('distance given by '
                   'sqrt(sum((reconstruction-ground_truth)**2))')

    def apply(self, reconstruction, ground_truth):
        return np.linalg.norm((reconstruction-ground_truth).flat)


class EvaluationTaskTable:
    """Task table containing reconstruction tasks to evaluate."""
    def __init__(self, name='', tasks=None):
        self.name = name
        self.tasks = tasks or []

    def run(self):
        """Run all tasks and return the results."""
        results = EvaluationResultTable()
        for task in self.tasks:
            test_data = task['test_data']
            reconstruction = task['reconstructor'].reconstruct(
                test_data.observation)
            results.reconstructions.append(reconstruction)
            measure_values = []
            for measure in task['measures']:
                measure_values.append(measure.apply(test_data.observation,
                                                    test_data.ground_truth))
            results.measure_values.append(measure_values)
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
        return "EvaluationTaskTable(name='{}', tasks='{}')".format(
            self.name, self.tasks.__repr__)


class EvaluationResultTable:
    """Result table of running an evaluation task table."""
    def __init__(self, reconstructions=None, measure_values=None):
        self.reconstructions = reconstructions or []
        self.measure_values = measure_values or (
            [[]] * len(self.reconstructions))

    def plot_reconstruction(self, index):
        """Plot the reconstruction at the specified index."""
        reconstruction = self.reconstructions[index]
        if reconstruction.asarray().ndim == 1:
            plt.plot(reconstruction)
        elif reconstruction.asarray().ndim == 2:
            plot_image(self.reconstructions[index])
        else:
            print('only 1d and 2d reconstructions can be plotted (currently)')
