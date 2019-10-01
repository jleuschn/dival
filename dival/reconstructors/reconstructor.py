# -*- coding: utf-8 -*-
"""Provides the abstract reconstructor base class."""
from inspect import signature, Parameter


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
                Option dict for grid search with the fields:

                    ``'num_samples'`` : int, optional
                        Number of values. Default: 10.
                        Only relevant if ``'range'`` is specified and
                        ``'choices'`` is not.
                    ``'type'`` : {'linear'}, optional
                        Type of grid, i.e. distribution of the values.
                        Default: ``'linear'``.
                        Only relevant if ``'range'`` is specified and
                        ``'choices'`` is not.
                        Options are:

                            ``'linear'``
                                Equidistant values in the ``'range'``.

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

    def save_params(self, path):
        """Save parameters to path.

        Parameters
        ----------
        path : str
            Path at which the parameters should be saved.
            Implementations should either use it as a file path (w/ or w/o
            extension) or as a directory path for multiple files (the dir
            should be created by this method if it does not exist).
        """
        raise NotImplementedError

    def load_params(self, path):
        """Load parameters from path.

        Parameters
        ----------
        path : str
            Path at which the parameters are stored. C.f. `save_params`.
        """
        raise NotImplementedError


class IterativeReconstructor(Reconstructor):
    """Iterative reconstructor base class.

    Subclasses should call ``self.callback()`` after each iteration in
    ``self.reconstruct``.

    Attributes
    ----------
    callback : ``odl.solvers.util.callback.Callback`` or `None`
        Callback to be called after each iteration.
    """
    def __init__(self, callback=None, **kwargs):
        self.callback = None
        super().__init__(**kwargs)


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
