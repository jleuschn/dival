# -*- coding: utf-8 -*-
"""Provides the abstract reconstructor base class."""
import os
from inspect import signature, Parameter
import json
from warnings import warn


class Reconstructor:
    """Abstract reconstructor base class.

    There are two ways of implementing a `Reconstructor` subclass:

        * Implement :meth:`reconstruct`. It has to support optional in-place
          and out-of-place evaluation.
        * Implement :meth:`_reconstruct`. It must have one of the following
          signatures:

            - ``_reconstruct(self, observation, out)`` (in-place)
            - ``_reconstruct(self, observation)`` (out-of-place)
            - ``_reconstruct(self, observation, out=None)`` (optional
              in-place)

    Attributes
    ----------
    reco_space : :class:`odl.discr.DiscretizedSpace`, optional
        Reconstruction space.
    observation_space : :class:`odl.discr.DiscretizedSpace`, optional
        Observation space.
    name : str
        Name of the reconstructor.
    hyper_params : dict
        Current hyper parameter values.
        Initialized automatically using the default values from
        :attr:`HYPER_PARAMS` (but may be overridden by `hyper_params` passed
        to :meth:`__init__`).
    """

    HYPER_PARAMS = {}
    """Specification of hyper parameters.

    This class attribute is a dict that lists the hyper parameter of the
    reconstructor.
    It should not be hidden by an instance attribute of the same name (i.e. by
    assigning a value to `self.HYPER_PARAMS` in an instance of a subtype).

    The keys of this dict are the names of the hyper parameters, and each
    value is a dict with the following fields.

        Standard fields:

            ``'default'``
                Default value.
            ``'retrain'`` : bool, optional
                Whether training depends on the parameter. Default: ``False``.
                Any custom subclass of `LearnedReconstructor` must set this
                field to ``True`` if training depends on the parameter value.

        Hyper parameter search fields:

            ``'range'`` : (float, float), optional
                Interval of valid values. If this field is set, the parameter
                is taken to be real-valued.
                Either ``'range'`` or ``'choices'`` has to be set.
            ``'choices'`` : sequence, optional
                Sequence of valid values of any type. If this field is set,
                ``'range'`` is ignored. Can be used to perform manual grid
                search. Either ``'range'`` or ``'choices'`` has to be set.
            ``'method'`` : {'grid_search', 'hyperopt'}, optional
                 Optimization method for the parameter.
                 Default: ``'grid_search'``.
                 Options are:

                    ``'grid_search'``
                        Grid search over a sequence of fixed values. Can be
                        configured by the dict ``'grid_search_options'``.
                    ``'hyperopt'``
                        Random search using the ``hyperopt`` package. Can be
                        configured by the dict ``'hyperopt_options'``.

            ``'grid_search_options'`` : dict
                Option dict for grid search.

                The following fields determine how ``'range'`` is sampled
                (in case it is specified and no ``'choices'`` are specified):

                    ``'num_samples'`` : int, optional
                        Number of values. Default: ``10``.
                    ``'type'`` : {'linear', 'logarithmic'}, optional
                        Type of grid, i.e. distribution of the values.
                        Default: ``'linear'``.
                        Options are:

                            ``'linear'``
                                Equidistant values in the ``'range'``.
                            ``'logarithmic'``
                                Values in the ``'range'`` that are equidistant
                                in the log scale.
                    ``'log_base'`` : int, optional
                        Log-base that is used if ``'type'`` is
                        ``'logarithmic'``. Default: ``10.``.

            ``'hyperopt_options'`` : dict
                Option dict for ``'hyperopt'`` method with the fields:

                    ``'space'`` : hyperopt space, optional
                        Custom hyperopt search space. If this field is set,
                        ``'range'`` and ``'type'`` are ignored.
                    ``'type'`` : {'uniform'}, optional
                        Type of the space for sampling. Default: ``'uniform'``.
                        Options are:

                            ``'uniform'``
                                Uniform distribution over the ``'range'``.
    """

    def __init__(self, reco_space=None, observation_space=None,
                 name='', hyper_params=None):
        self.reco_space = reco_space
        self.observation_space = observation_space
        self.name = name or self.__class__.__name__
        self.hyper_params = {k: v['default']
                             for k, v in self.HYPER_PARAMS.items()}
        if hyper_params is not None:
            self.hyper_params.update(hyper_params)

    def reconstruct(self, observation, out=None):
        """Reconstruct input data from observation data.

        The default implementation calls `_reconstruct`, automatically choosing
        in-place or out-of-place evaluation.

        Parameters
        ----------
        observation : :attr:`observation_space` element-like
            The observation data.
        out : :attr:`reco_space` element-like, optional
            Array to which the result is written (in-place evaluation).
            If `None`, a new array is created (out-of-place evaluation).

        Returns
        -------
        reconstruction : :attr:`reco_space` element or `out`
            The reconstruction.
        """
        parameters = signature(self._reconstruct).parameters
        if 'out' in parameters:
            if out is not None:
                self._reconstruct(observation, out)
                reco = out
            elif parameters['out'].default == Parameter.empty:
                reco = self.reco_space.zero()
                self._reconstruct(observation, reco)
            else:
                reco = self._reconstruct(observation)
        else:
            reco = self._reconstruct(observation)
            if out is not None:
                out[:] = reco
        return reco

    def _reconstruct(self, observation, *args, **kwargs):
        """Reconstruct input data from observation data.

        This method must have one of the following signatures:
            - ``_reconstruct(self, observation, out)`` (in-place)
            - ``_reconstruct(self, observation)`` (out-of-place)
            - ``_reconstruct(self, observation, out=None)`` (optional
              in-place)

        The parameters and return value are documented in :meth:`reconstruct`.
        """
        raise NotImplementedError("'_reconstruct' not implemented by class "
                                  "'{}'. Reconstructor subclasses must "
                                  "implement either 'reconstruct' or "
                                  "'_reconstruct'.".format(type(self)))

    def save_hyper_params(self, path):
        """Save hyper parameters to JSON file.
        See also :meth:`load_hyper_params`.

        Parameters
        ----------
        path : str
            Path of the file in which the hyper parameters should be saved.
            The ending ``'.json'`` is automatically appended if not included.
        """
        path = os.path.splitext(path)[0] + '.json'
        with open(path, 'w') as f:
            json.dump(self.hyper_params, f, indent=1)

    def load_hyper_params(self, path):
        """Load hyper parameters from JSON file.
        See also :meth:`save_hyper_params`.

        Parameters
        ----------
        path : str
            Path of the file in which the hyper parameters are stored.
            The ending ``'.json'`` is automatically appended if not included.
        """
        path = os.path.splitext(path)[0] + '.json'
        with open(path, 'r') as f:
            hyper_params = json.load(f)
            for k, v in hyper_params.items():
                if k not in self.HYPER_PARAMS:
                    warn("loading value for unknown hyper parameter '{}'"
                         .format(k))
            self.hyper_params.update(hyper_params)

    def save_params(self, path=None, hyper_params_path=None):
        """Save all parameters to file.
        E.g. for learned reconstructors, both hyper parameters and learned
        parameters should be included.
        The purpose of this method, together with :meth:`load_params`, is to
        define a unified way of saving and loading any kind of reconstructor.
        The default implementation calls :meth:`save_hyper_params`.
        Subclasses must reimplement this method in order to include non-hyper
        parameters.

        Parameters
        ----------
        path : str[, optional]
            Path at which all non-hyper parameters should be saved.
            This argument is required if the reconstructor has non-hyper
            parameters.
            Implementations should either use it as a file path or as a
            directory path for multiple files (the dir should be created by
            this method if it does not exist).
        hyper_params_path : str, optional
            Path of the file in which the hyper parameters should be saved.
            The ending ``'.json'`` is automatically appended if not included.
            If not specified, it should be determined from `path`.
            The default implementation saves to the file
            ``path + '_hyper_params.json'``.
        """
        hp_path = hyper_params_path
        if hp_path is None:
            if path is None:
                raise ValueError(
                    'either a directory `path` or a filename '
                    '`hyper_params_path` required (in default implementation '
                    'of `Reconstructor.save_params`)')
            hp_path = path + '_hyper_params.json'
        else:
            hp_path = (hyper_params_path if hyper_params_path.endswith('.json')
                       else hyper_params_path + '.json')
        self.save_hyper_params(hp_path)

    def load_params(self, path=None, hyper_params_path=None):
        """Load of parameters from file.
        E.g. for learned reconstructors, both hyper parameters and learned
        parameters should be included.
        The purpose of this method, together with :meth:`save_params`, is to
        define a unified way of saving and loading any kind of reconstructor.
        The default implementation calls :meth:`load_hyper_params`.
        Subclasses must reimplement this method in order to include non-hyper
        parameters.

        Parameters
        ----------
        path : str[, optional]
            Path at which the parameters are stored.
            This argument is required if the reconstructor has non-hyper
            parameters.
            Depending on the implementation, this may be a file path or a
            directory path for multiple files.
        hyper_params_path : str, optional
            Path of the file in which the hyper parameters are stored.
            The ending ``'.json'`` is automatically appended if not included.
            If not specified, it should be determined from `path`. If `path`
            is interpreted as a directory, the default for `hyper_params_path`
            should be some file in this directory; otherwise, if `path` is used
            as a single file name, the default for `hyper_params_path` should
            be a file in the same directory as this file.
            The default implementation reads from the file
            ``path + '_hyper_params.json'``.
        """
        hp_path = hyper_params_path
        if hp_path is None:
            if path is None:
                raise ValueError(
                    'either a directory `path` or a filename '
                    '`hyper_params_path` required (in default implementation '
                    'of `Reconstructor.save_params`)')
            hp_path = path + '_hyper_params.json'
        else:
            hp_path = (hyper_params_path if hyper_params_path.endswith('.json')
                       else hyper_params_path + '.json')
        self.load_hyper_params(hp_path)


class LearnedReconstructor(Reconstructor):
    def train(self, dataset):
        """Train the reconstructor with a dataset by adapting its parameters.

        Should only use the training data from `dataset`.

        Parameters
        ----------
        dataset : :class:`.Dataset`
            The dataset from which the training data should be used.
        """
        raise NotImplementedError

    def save_params(self, path, hyper_params_path=None):
        """Save all parameters to file.

        Calls :meth:`save_hyper_params` and :meth:`save_learned_params`, where
        :meth:`save_learned_params` should be implemented by the subclass.

        Parameters
        ----------
        path : str
            Path at which the learned parameters should be saved.
            Passed to :meth:`save_learned_params`.
        hyper_params_path : str, optional
            Path of the file in which the hyper parameters should be saved.
            The ending ``'.json'`` is automatically appended if not included.
            If not specified, it should be determined from `path`.
            The default implementation saves to the file
            ``path + '_hyper_params.json'``.
        """
        hp_path = hyper_params_path
        if hp_path is None:
            hp_path = path + '_hyper_params.json'
        else:
            hp_path = (hyper_params_path if hyper_params_path.endswith('.json')
                       else hyper_params_path + '.json')
        self.save_hyper_params(hp_path)
        self.save_learned_params(path)

    def load_params(self, path, hyper_params_path=None):
        """Load all parameters from file.

        Calls :meth:`load_hyper_params` and :meth:`load_learned_params`, where
        :meth:`load_learned_params` should be implemented by the subclass.

        Parameters
        ----------
        path : str
            Path at which the parameters are stored.
            Passed to :meth:`load_learned_params`.
        hyper_params_path : str, optional
            Path of the file in which the hyper parameters are stored.
            The ending ``'.json'`` is automatically appended if not included.
            If not specified, it should be determined from `path`.
            The default implementation reads from the file
            ``path + '_hyper_params.json'``.
        """
        hp_path = hyper_params_path
        if hp_path is None:
            hp_path = path + '_hyper_params.json'
        else:
            hp_path = (hyper_params_path if hyper_params_path.endswith('.json')
                       else hyper_params_path + '.json')
        self.load_hyper_params(hp_path)
        self.load_learned_params(path)

    def save_learned_params(path):
        """Save learned parameters to file.

        Parameters
        ----------
        path : str
            Path at which the learned parameters should be saved.
            Implementations may interpret this as a file path or as a directory
            path for multiple files (which then should be created if it does
            not exist).
        """
        raise NotImplementedError

    def load_learned_params(path):
        """Load learned parameters from file.

        Parameters
        ----------
        path : str
            Path at which the learned parameters are stored.
            Implementations may interpret this as a file path or as a directory
            path for multiple files.
        """
        raise NotImplementedError


class IterativeReconstructor(Reconstructor):
    """Iterative reconstructor base class.

    Subclasses should call :attr:`callback` after each iteration in
    ``self.reconstruct``.

    Attributes
    ----------
    callback : ``odl.solvers.util.callback.Callback`` or `None`
        Callback to be called after each iteration.
    """
    def __init__(self, callback=None, **kwargs):
        """
        Parameters
        ----------
        callback : ``odl.solvers.util.callback.Callback``, optional
            Callback to be called after each iteration.
        """
        self.callback = callback
        super().__init__(**kwargs)

    def reconstruct(self, observation, out=None, callback=None):
        """Reconstruct input data from observation data.

        Same as :meth:`Reconstructor.reconstruct`, but with additional optional
        `callback` parameter.

        Parameters
        ----------
        observation : :attr:`observation_space` element-like
            The observation data.
        out : :attr:`reco_space` element-like, optional
            Array to which the result is written (in-place evaluation).
            If `None`, a new array is created (out-of-place evaluation).
        callback : ``odl.solvers.util.callback.Callback``, optional
            New value of :attr:`callback`. If `None`, the value of
            :attr:`callback` is not modified.

        Returns
        -------
        reconstruction : :attr:`reco_space` element or `out`
            The reconstruction.
        """
        if callback is not None:
            self.callback = callback
        return super().reconstruct(observation, out=out)


class FunctionReconstructor(Reconstructor):
    """Reconstructor defined by a callable.

    Attributes
    ----------
    function : callable
        Callable that is used in `reconstruct`.
    fun_args : list
        Arguments to be passed to `function`.
    fun_kwargs : dict
        Keyword arguments to be passed to `function`.
    """
    def __init__(self, function, *args, fun_args=None, fun_kwargs=None,
                 **kwargs):
        """
        Parameters
        ----------
        function : callable
            Callable that is used in :meth:`reconstruct`.
        fun_args : list, optional
            Arguments to be passed to `function`.
        fun_kwargs : dict, optional
            Keyword arguments to be passed to `function`.
        """
        super().__init__(*args, **kwargs)
        self.function = function
        self.fun_args = fun_args or []
        self.fun_kwargs = fun_kwargs or {}

    def _reconstruct(self, observation):
        return self.function(observation, *self.fun_args, **self.fun_kwargs)
