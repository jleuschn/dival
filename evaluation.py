# -*- coding: utf-8 -*-
"""Provides classes and methods useful for evaluation of methods."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
from odl.solvers.util.callback import CallbackStore
from dival.util.plot import plot_image, plot_images
from dival.util.std_out_err_redirect_tqdm import std_out_err_redirect_tqdm
from dival.measure import Measure
from dival.data import TestData
from dival import LearnedReconstructor
from dival.hyper_param_optimization import optimize_hyper_params


class TaskTable:
    """Task table containing reconstruction tasks to evaluate.

    Attributes
    ----------
    name : str
        Name of the task table.
    tasks : list of dict
        Tasks that shall be run. The fields of each dict are set from the
        parameters to `append` (or `append_all_combinations`). See the
        documentation of `append` for documentation of these fields.
    """
    def __init__(self, name=''):
        self.name = name
        self.tasks = []

    def run(self, save_reconstructions=True, show_progress='text'):
        """Run all tasks and return the results.

        Parameters
        ----------
        save_reconstructions : bool, optional
            Whether the reconstructions should be saved in the results.
            The default is ``True``.

            If measures shall be applied after this method returns, it must be
            ``True``.

            If ``False``, no iterates (intermediate reconstructions) will be
            saved, even if ``task['options']['save_iterates']`` is ``True``.
        show_progress : str, optional
            Whether and how to show progress. Options are:

                ``'text'`` (default)
                    print a line before running each task
                ``'tqdm'``
                    show a progress bar with ``tqdm``
                ``None``
                    do not show progress
        """
        results = ResultTable()
        row_list = []
        with std_out_err_redirect_tqdm(None if show_progress == 'tqdm' else
                                       sys.stdout) as orig_stdout:
            for i, task in enumerate(tqdm(self.tasks,
                                          desc='task',
                                          file=orig_stdout,
                                          disable=(show_progress != 'tqdm'))):
                if show_progress == 'text':
                    print('running task {i}/{num_tasks} ...'.format(
                        i=i, num_tasks=len(self.tasks)))
                test_data = task['test_data']
                reconstructor = task['reconstructor']
                if test_data.ground_truth is None and task['measures']:
                    raise ValueError('missing ground truth, cannot apply '
                                     'measures')
                measures = [(measure if isinstance(measure, Measure) else
                             Measure.get_by_short_name(measure))
                            for measure in task['measures']]
                options = task['options']

                hp_opt = options.get('hyper_param_search')
                if hp_opt:
                    kwargs = {}
                    for k in ('hyperopt_max_evals',
                              'hyperopt_max_evals_retrain'):
                        v = hp_opt.get(k)
                        if v is not None:
                            kwargs[k] = v
                    optimize_hyper_params(
                        reconstructor, test_data, hp_opt['measure'],
                        dataset=task.get('dataset'),
                        HYPER_PARAMS_override=hp_opt.get('HYPER_PARAMS'),
                        hyperopt_rstate=hp_opt.get('hyperopt_rstate'),
                        show_progressbar=hp_opt.get('show_progress',
                                                    show_progress == 'text'),
                        tqdm_file=orig_stdout,
                        **kwargs)

                reconstructions = []
                if save_reconstructions and options.get('save_iterates'):
                    iterates = []
                    if options.get('save_iterates_measure_values'):
                        iterates_measure_values = {m.short_name: []
                                                   for m in measures}

                for observation, ground_truth in zip(test_data.observations,
                                                     test_data.ground_truth):
                    callbacks = []
                    if reconstructor.callback is not None:
                        callbacks.append(reconstructor.callback)
                    if save_reconstructions and options.get('save_iterates'):
                        iters = []
                        iterates.append(iters)
                        callback_save_iterates = CallbackStore(
                            iters, step=options.get('save_iterates_step', 1))
                        callbacks.append(callback_save_iterates)
                    if options.get('save_iterates_measure_values'):
                        for measure in measures:
                            iters_mvs = []
                            iterates_measure_values[measure.short_name].append(
                                iters_mvs)
                            callback_store = CallbackStore(
                                iters_mvs,
                                step=options.get('save_iterates_step', 1))
                            callbacks.append(
                                callback_store *
                                measure.as_operator_for_fixed_ground_truth(
                                    ground_truth))
                    if len(callbacks) > 0:
                        reconstructor.callback = callbacks[-1]
                        for callback in callbacks[-2::-1]:
                            reconstructor.callback &= callback

                    reconstructions.append(reconstructor.reconstruct(
                        observation))

                measure_values = {}
                for measure in measures:
                    measure_values[measure.short_name] = [
                        measure.apply(r, g) for r, g in zip(
                            reconstructions, test_data.ground_truth)]
                misc = {}
                if save_reconstructions and options.get('save_iterates'):
                    misc['iterates'] = iterates
                if options.get('save_iterates_measure_values'):
                    misc['iterates_measure_values'] = iterates_measure_values
                row = {'reconstructions': None,
                       'reconstructor': reconstructor,
                       'test_data': test_data,
                       'measure_values': measure_values,
                       'misc': misc}
                if save_reconstructions:
                    row['reconstructions'] = reconstructions
                row_list.append(row)
        results.results = pd.concat([results.results, pd.DataFrame(row_list)],
                                    ignore_index=True, sort=False)
        return results

    def append(self, reconstructor, test_data, measures=None, dataset=None,
               options=None):
        """Append a task.

        Parameters
        ----------
        reconstructor : `Reconstructor`
            The reconstructor.
        test_data : `TestData`
            The test data.
        measures : sequence of (`Measure` or str)
            Measures that will be applied. Either `Measure` objects or their
            short names can be passed.
        dataset : `Dataset`
            The dataset that will be passed to `reconstructor.train` if it is a
            `LearnedReconstructor`.
        options : dict
            Options that will be used. Options are:

            ``'save_iterates'`` : bool, optional
                Whether to save the intermediate reconstructions of iterative
                reconstructors (the default is ``False``).
                Will be ignored if ``save_reconstructions=False`` is passed to
                `run`. Requires the reconstructor to call its `callback`
                attribute after each iteration.
            ``'save_iterates_measure_values'`` : bool, optional
                Whether to compute and save the measure values for each
                intermediate reconstruction of iterative reconstructors
                (the default is ``False``). Requires the reconstructor to call
                its `callback` attribute after each iteration.
            ``'save_iterates_step'`` : int, optional
                Step size for ``'save_iterates'`` and
                ``'save_iterates_measure_values'`` (the default is 1).
            ``'hyper_param_search'`` : dict, optional
                Options for hyper parameter search. If ``None``, the default
                hyper parameter values are used. If given, it must specify the
                following fields:

                    ``'measure'`` : `Measure`
                        The measure used for hyper parameter optimization.
                    ``'dataset'`` : `Dataset`, optional
                        Dataset for training the reconstructor. Only needs to
                        be specified if the reconstructor is a
                        `LearnedReconstructor`.
                    ``'HYPER_PARAMS'`` : dict, optional
                        Hyper parameter specification overriding the defaults
                        in ``type(reconstructor).HYPER_PARAMS``.
                        The structure of this dict is the same as the structure
                        of ``Reconstructor.HYPER_PARAMS``, except that all
                        fields are optional.
                        Here, each value of a dict for one parameter is treated
                        as an entity, i.e. specifying the dict
                        ``HYPER_PARAMS[...]['grid_search_options']`` overrides
                        the whole dict, not only the specified keys in it.
        """
        if isinstance(reconstructor, LearnedReconstructor) and dataset is None:
            raise ValueError('in order to use a learned reconstructor you '
                             'must specify a `dataset` for training')
        if measures is None:
            measures = []
        if options is None:
            options = {}
        self.tasks.append({'reconstructor': reconstructor,
                           'test_data': test_data,
                           'measures': measures,
                           'dataset': dataset,
                           'options': options})

    def append_all_combinations(self, reconstructors, test_data, measures=None,
                                datasets=None, options=None):
        """Append tasks of all combinations of test data, reconstructors and
        optionally datasets.
        The order is taken from the lists, with test data changing slowest
        and reconstructor changing fastest.

        Parameters
        ----------
        reconstructors : list of `Reconstructor`
            Reconstructor list.
        test_data : list of `TestData`
            Test data list.
        measures : sequence of (`Measure` or str)
            Measures that will be applied. The same measures are used for all
            combinations of test data and reconstructors. Either `Measure`
            objects or their short names can be passed.
        datasets : list of `Dataset`, optional
            Dataset list. Required if `reconstructors` contains at least one
            `LearnedReconstructor`.
        options : dict
            Options that will be used. The same options are used for all
            combinations of test data and reconstructors. See `append` for
            documentation of the options.
        """
        if datasets is None:
            datasets = [None]
        for test_data_ in test_data:
            for dataset in datasets:
                for reconstructor in reconstructors:
                    self.append(reconstructor=reconstructor,
                                test_data=test_data_, measures=measures,
                                dataset=dataset, options=options)

    def __repr__(self):
        return "TaskTable(name='{name}', tasks={tasks})".format(
            name=self.name,
            tasks=self.tasks)


class ResultTable:
    """Result table of running an evaluation task table.

    Attributes
    ----------
    results : `pandas.DataFrame`
        The results.
        It has the columns ``'reconstructions'``, ``'reconstructor'``,
        ``'test_data'``, ``'measure_values'`` and ``'misc'``.
    """
    def __init__(self, reconstructions=None, reconstructor=None,
                 test_data=None, measure_values=None, misc=None):
        if reconstructions is None:
            reconstructions = []
        if reconstructor is None:
            reconstructor = []
        if test_data is None:
            test_data = []
        if measure_values is None:
            measure_values = []
        if misc is None:
            misc = []
        data_dict = {'reconstructions': reconstructions,
                     'reconstructor': reconstructor,
                     'test_data': test_data,
                     'measure_values': measure_values,
                     'misc': misc}
        self.results = pd.DataFrame.from_dict(data_dict)

    def apply_measures(self, measures, index=None):
        """Apply (additional) measures to reconstructions.

        This is not possible if the reconstructions were not saved, in which
        case a `ValueError` is raised.

        Parameters
        ----------
        measures : list of Measure
            Measures to apply.
        index : int or sequence of ints, optional
            Indexes of results to which the measures shall be applied.
            If `index` is ``None``, this is interpreted as "all results".

        Raises
        ------
        ValueError
            If reconstructions are missing or `index` is not valid.
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
            if row['reconstructions'] is None:
                raise ValueError('reconstructions missing in row {i}'.format(
                    i=i))
            for measure in measures:
                if isinstance(measure, str):
                    measure = Measure.get_by_short_name(measure)
                row['measure_values'][measure.short_name] = [
                    measure.apply(r, g) for r, g in zip(
                        row['reconstructions'], row['test_data'].ground_truth)]

    def plot_reconstruction(self, index, test_index=0,
                            plot_ground_truth=True, **kwargs):
        """Plot the reconstruction at the specified index.
        Supports only 1d and 2d reconstructions.

        Parameters
        ----------
        index : int
            Index of the task.
        test_index : int
            Index in test data.
        plot_ground_truth : bool, optional
            Whether to show the ground truth next to the reconstruction.
            The default is ``True``.
        kwargs : dict
            Keyword arguments that are passed to `plot_image` if the
            reconstruction is 2d.

        Returns
        -------
        ax : `matplotlib.axes.Axes`
            The axes the reconstruction was plotted in.
        """
        row = self.results.iloc[index]
        reconstruction = row.at['reconstructions'][test_index]
        reconstructor = row.at['reconstructor']
        test_data = row.at['test_data']
        ground_truth = test_data.ground_truth[test_index]
        if reconstruction is None:
            raise ValueError('reconstruction is ``None``')
        if reconstruction.asarray().ndim > 2:
            print('only 1d and 2d reconstructions can be plotted (currently)')
            return
        if reconstruction.asarray().ndim == 1:
            x = reconstruction.space.points()
            ax = plt.subplot()
            ax.plot(x, reconstruction, label=reconstructor.name)
            if plot_ground_truth:
                ax.plot(x, ground_truth, label='ground truth')
            ax.legend()
        elif reconstruction.asarray().ndim == 2:
            if plot_ground_truth:
                _, ax = plot_images([reconstruction, ground_truth], **kwargs)
                ax[1].set_title('ground truth')
                ax = ax[0]
            else:
                _, ax = plot_image(reconstruction, **kwargs)
            ax.set_title(reconstructor.name)
        return ax

    def plot_all_reconstructions(self, **kwargs):
        """Plot all reconstructions.

        Parameters
        ----------
        kwargs : dict
            Keyword arguments that are forwarded to `self.plot_reconstruction`.

        Returns
        -------
        ax : ndarray of `matplotlib.axes.Axes`
            List of the axes the reconstructions were plotted in.
        """
        ax = []
        for i in range(len(self.results)):
            ax_ = self.plot_reconstruction(i, **kwargs)
            ax.append(ax_)
        return np.array(ax)

    def plot_convergence(self, index, measures=None, fig_size=None,
                         gridspec_kw=None):
        """
        Plot measure values for saved iterates.

        This shows the convergence behavior with respect to the measures.

        Parameters
        ----------
        index : int
            Row index of the result.
        measures : list of `Measure` or `Measure`, optional
            Measures to apply. Each measure is plotted in a subplot.
            If ``None`` is passed, all measures in ``result['measure_values']``
            are used.

        Returns
        -------
        ax : ndarray of matplotlib.axes.Axes
            The axes the measure values were plotted in.
        """
        row = self.results.iloc[index]
        iterates_measure_values = row['misc'].get('iterates_measure_values')
        if not iterates_measure_values:
            iterates = row['misc'].get('iterates')
            if not iterates:
                raise ValueError(
                    "no 'iterates_measure_values' or 'iterates' in results "
                    "row {}".format(index))
        if measures is None:
            measures = row['measure_values'].keys()
        elif isinstance(measures, Measure):
            measures = [measures]
        fig, ax = plt.subplots(len(measures), 1, gridspec_kw=gridspec_kw)
        if fig_size is not None:
            fig.set_size_inches(fig_size)
        fig.suptitle('convergence of {}'.format(row['reconstructor'].name))
        for measure, ax_ in zip(measures, ax.flat):
            if isinstance(measure, str):
                measure = Measure.get_by_short_name(measure)
            if iterates_measure_values:
                errors = np.mean([iters_mvs[measure.short_name] for iters_mvs
                                  in iterates_measure_values], axis=0)
            else:
                ground_truth = row['test_data'].ground_truth
                errors = np.mean([[measure.apply(x, g) for x in iters] for
                                  iters, g in zip(iterates, ground_truth)],
                                 axis=0)
            ax_.plot(errors)
            ax_.set_title(measure.short_name)
        return ax

    def plot_performance(self, measure, reconstructors=None, test_data=None,
                         weighted_average=False, **kwargs):
        """
        Plot average measure values for different reconstructors.
        The values have to be computed previously, e.g. by
        `self.apply_measures`.

        The average is computed over all rows of `self.results` with the
        specified `test_data` that store the requested `measure` value.

        Parameters
        ----------
        measure : `Measure` or str
            The measure to plot (or its ``short_name``).
        reconstructors : sequence of `Reconstructor`, optional
            The reconstructors to compare. If ``None`` (default), all
            reconstructors that are found in the results are compared.
        test_data : sequence of `TestData` or `TestData`, optional
            Test data to take into account for computing the mean value.
        weighted_average : bool, optional
            Whether to weight the rows according to the number of test data
            elements.
            Default: ``False``, i.e. all rows are weighted equally.
            If ``True``, all test data elements are weighted equally.

        Returns
        -------
        ax : `matplotlib.axes.Axes`
            The axes the performance was plotted in.
        """
        if not isinstance(measure, Measure):
            measure = Measure.get_by_short_name(measure)
        if reconstructors is None:
            reconstructors = self.results['reconstructor'].unique()
        if isinstance(test_data, TestData):
            test_data = [test_data]
        mask = [measure.short_name in row['measure_values'].keys() and
                row['reconstructor'] in reconstructors and
                (test_data is None or row['test_data'] in test_data)
                for _, row in self.results.iterrows()]
        rows = self.results[mask]
        v = []
        for reconstructor in reconstructors:
            r_rows = rows[rows['reconstructor'] == reconstructor]
            values = [mvs[measure.short_name] for mvs in
                      r_rows['measure_values']]
            weights = None
            if weighted_average:
                weights = [len(test_data.observations) for test_data in
                           r_rows['test_data']]
            v.append(np.average(values, weights=weights))
        fig, ax = plt.subplots(**kwargs)
        ax.bar(range(len(v)), v)
        ax.set_xticks(range(len(v)))
        ax.set_xticklabels([r.name for r in reconstructors], rotation=30)
        ax.set_title('{measure_name}'.format(measure_name=measure.name))
        return ax

    def to_string(self, max_colwidth=70, formatters=None, **kwargs):
        """Convert to string. Used by `self.__str__`.

        Parameters
        ----------
        max_colwidth : int, optional
            Maximum width of each column, c.f. pandas' option
            ``'display.max_colwidth'``. Default is 70.
        formatters : dict of functions, optional
            Formatter functions for the columns of `self.results`, passed to
            `self.results.to_string`.
        kwargs : dict
            Keyword arguments passed to `self.results.to_string`.

        Returns
        -------
        str
            The string.
        """
        def test_data_formatter(test_data):
            return test_data.name or test_data.__repr__()
        formatters_ = {}
        formatters_['test_data'] = test_data_formatter
        if formatters is not None:
            formatters_.update(formatters)
        with pd.option_context('display.max_colwidth', max_colwidth):
            return "ResultTable(results=\n{}\n)".format(
                self.results.to_string(formatters=formatters_, **kwargs))

    def print_summary(self):
        """Prints a summary of the results.
        """
        print('ResultTable with {:d} rows.'.format(len(self.results)))
        test_data_list = pd.unique(self.results['test_data'])
        if len(test_data_list) == 1:
            print('Test data: {}'.format(test_data_list[0]))

    def __repr__(self):
        return "ResultTable(results=\n{results})".format(results=self.results)

    def __str__(self):
        return self.to_string()
