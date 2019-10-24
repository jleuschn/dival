# -*- coding: utf-8 -*-
"""
Optimization of hyper parameters.

Both grid search and random search using the ``hyperopt`` library are
supported.

The hyper parameter specification of a reconstructor class, optionally
including default options for optimization, are specified in the class
attribute :attr:`~dival.Reconstructor.HYPER_PARAMS`.
"""
from itertools import product
from warnings import warn
import numpy as np
from hyperopt import hp, fmin, tpe, Trials, space_eval
from tqdm import tqdm
from dival.util.std_out_err_redirect_tqdm import std_out_err_redirect_tqdm
from dival.measure import Measure
from dival import LearnedReconstructor


def optimize_hyper_params(reconstructor, validation_data, measure,
                          dataset=None,
                          HYPER_PARAMS_override=None,
                          hyperopt_max_evals=1000,
                          hyperopt_max_evals_retrain=1000,
                          hyperopt_rstate=None,
                          show_progressbar=True,
                          tqdm_file=None):
    """Optimize hyper parameters of a reconstructor.

    Parameters
    ----------
    reconstructor : :class:`.Reconstructor`
        The reconstructor.
    validation_data : :class:`.DataPairs`
        The test data on which the performance is measured.
    measure : :class:`.Measure` or str
        The measure to use as the objective. The sign is chosen automatically
        depending on the measures :attr:`~Measure.measure_type`.
    dataset : :class:`.Dataset`, optional
        The dataset used for training `reconstructor` if it is a
        :class:`LearnedReconstructor`.
    HYPER_PARAMS_override : dict, optional
        Hyper parameter specification overriding the defaults
        in ``type(reconstructor).HYPER_PARAMS``.
        The structure of this dict is the same as the structure
        of :attr:`Reconstructor.HYPER_PARAMS`, except that all
        fields are optional.
        Here, each value of a dict for one parameter is treated
        as an entity, i.e. specifying the dict
        ``HYPER_PARAMS[...]['grid_search_options']`` overrides
        the whole dict, not only the specified keys in it.
    hyperopt_max_evals : int, optional
        Number of evaluations for different combinations of the parameters that
        are optimized by ``hyperopt`` and that do not require retraining.
        Should be chosen depending on the complexity of dependence and the
        number of such parameters.
    hyperopt_max_evals_retrain : int, optional
        Number of evaluations for different combinations of the parameters that
        are optimized by ``hyperopt`` and that require retraining.
        Should be chosen depending on the complexity of dependence and the
        number of such parameters.
    hyperopt_rstate : :class:`np.random.RandomState`, optional
        Random state for the random searches performed by ``hyperopt``.
    show_progressbar : bool, optional
        Whether to show a progress bar for the optimization. Default: ``True``.
    tqdm_file : file-like object
        File/stream to pass to ``tqdm``.
    """
    if isinstance(measure, str):
        measure = Measure.get_by_short_name(measure)
    if dataset is None and isinstance(reconstructor, LearnedReconstructor):
        raise ValueError('dataset required for training of '
                         '`LearnedReconstructor`')

    if HYPER_PARAMS_override is None:
        HYPER_PARAMS_override = {}
    for k in HYPER_PARAMS_override.keys():
        if k not in type(reconstructor).HYPER_PARAMS.keys():
            warn("unknown hyper param '{}' for reconstructor of type '{}'"
                 .format(k, type(reconstructor)))

    params = {}
    params_retrain = {}
    for k, v in type(reconstructor).HYPER_PARAMS.items():
        param = v.copy()
        param.update(HYPER_PARAMS_override.get(k, {}))
        param.setdefault('method', 'grid_search')
        retrain = v.get('retrain', False)
        if retrain:
            params_retrain[k] = param
        else:
            params[k] = param

    loss_sign = 1 if measure.measure_type == 'distance' else -1

    def fn(x):
        reconstructor.hyper_params.update(x)
        reconstructions = [reconstructor.reconstruct(observation) for
                           observation in validation_data.observations]
        measure_values = [measure.apply(r, g) for r, g in
                          zip(reconstructions, validation_data.ground_truth)]
        loss = loss_sign * np.mean(measure_values)

        return {'status': 'ok',
                'loss': loss}

    def fn_retrain(x):
        reconstructor.hyper_params.update(x)
        reconstructor.train(dataset)

        best_sub_hp = _optimize_hyper_params_impl(
            reconstructor, fn, params,
            hyperopt_max_evals=hyperopt_max_evals,
            hyperopt_rstate=hyperopt_rstate, show_progressbar=False)

        reconstructions = [reconstructor.reconstruct(observation) for
                           observation in validation_data.observations]
        measure_values = [measure.apply(r, g) for r, g in
                          zip(reconstructions, validation_data.ground_truth)]
        loss = loss_sign * np.mean(measure_values)

        return {'status': 'ok',
                'loss': loss,
                'best_sub_hp': best_sub_hp}

    if params_retrain:
        best_hyper_params = _optimize_hyper_params_impl(
            reconstructor, fn_retrain, params_retrain,
            hyperopt_max_evals=hyperopt_max_evals_retrain,
            hyperopt_rstate=hyperopt_rstate,
            show_progressbar=show_progressbar, tqdm_file=tqdm_file)
    else:
        best_hyper_params = _optimize_hyper_params_impl(
            reconstructor, fn, params,
            hyperopt_max_evals=hyperopt_max_evals,
            hyperopt_rstate=hyperopt_rstate,
            show_progressbar=show_progressbar, tqdm_file=tqdm_file)

    return best_hyper_params


def _optimize_hyper_params_impl(reconstructor, fn, params,
                                hyperopt_max_evals=1000, hyperopt_rstate=None,
                                show_progressbar=True, tqdm_file=None):
    grid_search_params = []
    grid_search_param_choices = []
    hyperopt_space = {}
    for k, param in params.items():
        method = param['method']
        if method == 'grid_search':
            grid_search_options = param.get('grid_search_options', {})
            choices = param.get('choices')
            if choices is None:
                range_ = param.get('range')
                if range_ is not None:
                    grid_type = grid_search_options.get('type', 'linear')
                    if grid_type == 'linear':
                        n = grid_search_options.get('num_samples', 10)
                        choices = np.linspace(range_[0], range_[1], n)
                    elif grid_type == 'logarithmic':
                        n = grid_search_options.get('num_samples', 10)
                        b = grid_search_options.get('log_base', 10.)
                        choices = np.logspace(range_[0], range_[1], n, base=b)
                    else:
                        raise ValueError(
                            "unknown grid type '{grid_type}' in {reco_cls}."
                            "HYPER_PARAMS['{k}']['grid_search_options']".
                            format(
                                grid_type=grid_type,
                                reco_cls=reconstructor.__class__.__name__,
                                k=k))
                else:
                    raise ValueError(
                        "neither 'choices' nor 'range' is specified in "
                        "{reco_cls}.HYPER_PARAMS['{k}'], one of them must be "
                        "specified for grid search".format(
                            reco_cls=reconstructor.__class__.__name__, k=k))
            grid_search_params.append(k)
            grid_search_param_choices.append(choices)
        elif method == 'hyperopt':
            hyperopt_options = param.get('hyperopt_options', {})
            space = hyperopt_options.get('space')
            if space is None:
                choices = param.get('choices')
                if choices is None:
                    range_ = param.get('range')
                    if range_ is not None:
                        space_type = hyperopt_options.get('type', 'uniform')
                        if space_type == 'uniform':
                            space = hp.uniform(k, range_[0], range_[1])
                        else:
                            raise ValueError(
                                "unknown hyperopt space type '{space_type}' "
                                "in {reco_cls}.HYPER_PARAMS['{k}']"
                                "['hyperopt_options']".format(
                                    space_type=space_type,
                                    reco_cls=reconstructor.__class__.__name__,
                                    k=k))
                    else:
                        raise ValueError(
                            "neither 'choices' nor 'range' is specified in "
                            "{reco_cls}.HYPER_PARAMS['{k}']"
                            "['hyperopt_options']. One of these or "
                            "{reco_cls}.HYPER_PARAMS['{k}']"
                            "['hyperopt_options']['space'] must be specified "
                            "for hyperopt param search".format(
                                reco_cls=reconstructor.__class__.__name__,
                                k=k))
                else:
                    space = hp.choice(k, choices)
            hyperopt_space[k] = space
        else:
            raise ValueError("unknown method '{method}' for "
                             "{reco_cls}.HYPER_PARAMS['{k}']".format(
                                 method=method,
                                 reco_cls=reconstructor.__class__.__name__,
                                 k=k))

    best_loss = np.inf

    best_hyper_params = None
    with std_out_err_redirect_tqdm(tqdm_file) as orig_stdout:
        grid_search_total = np.prod([len(c) for c in
                                     grid_search_param_choices])
        for grid_search_values in tqdm(
                product(*grid_search_param_choices),
                desc='hyper param opt. for {reco_cls}'.format(
                    reco_cls=type(reconstructor).__name__),
                total=grid_search_total,
                file=orig_stdout,
                leave=False,
                disable=not show_progressbar):
            grid_search_param_dict = dict(zip(grid_search_params,
                                              grid_search_values))
            reconstructor.hyper_params.update(grid_search_param_dict)
            if len(hyperopt_space) == 0:
                result = fn({})
                if result['loss'] < best_loss:
                    best_loss = result['loss']
                    best_hyper_params = result.get('best_sub_hp', {})
                    best_hyper_params.update(grid_search_param_dict)
            else:
                trials = Trials()
                argmin = fmin(fn=fn, space=hyperopt_space, algo=tpe.suggest,
                              max_evals=hyperopt_max_evals, trials=trials,
                              rstate=hyperopt_rstate,
                              show_progressbar=False)
                best_trial = trials.best_trial
                if best_trial['result']['loss'] < best_loss:
                    best_loss = best_trial['result']['loss']
                    best_hyper_params = best_trial['result'].get('best_sub_hp',
                                                                 {})
                    best_hyper_params.update(grid_search_param_dict)
                    best_hyper_params.update(space_eval(hyperopt_space,
                                                        argmin))

    if best_hyper_params is not None:
        reconstructor.hyper_params.update(best_hyper_params)

    return best_hyper_params
