# -*- coding: utf-8 -*-
"""Provides the abstract reconstructor base class."""
from abc import ABC, abstractmethod
from inspect import signature
from odl.solvers.util.callback import Callback, CallbackApply


class Reconstructor(ABC):
    """Abstract reconstructor base class.

    Subclasses must implement the abstract method `reconstruct`.

    Attributes
    ----------
    name : str
        Name of the reconstructor.
    callback : odl.solvers.util.callback.Callback
        Object that should be called at some point(s) in `reconstruct`.
        E.g. an iterative reconstructor could call it at each iteration.
    """

    HYPER_PARAMS = {}
    """Specification of hyper parameters.

    This class attribute is a dict that lists the hyper parameter of the
    reconstructor.
    It should not be hidden by an instance attribute of the same name (i.e. by
    assigning a value to `self.HYPER_PARAMS` in an instance of a subtype).

    The keys of this dict are the names of the hyper parameters, and each
    value is a dict with the following fields:

        ``'default'``
            Default value.
        ``'retrain'`` : bool, optional
            Whether training depends on the parameter. Default: ``False``.
            If ``True``, the reconstructor is retrained while optimizing
            the parameter value.
        ``'range'`` : (float, float), optional
            Interval of valid values. If this field is set, the parameter is
            taken to be real-valued.
            Either ``'range'`` or ``'choices'`` has to be set.
        ``'choices'`` : sequence, optional
            Sequence of valid values of any type. If this field is set,
            ``'range'`` is ignored. Can be used to perform manual grid search.
            Either ``'range'`` or ``'choices'`` has to be set.
        ``'method'`` : {'grid_search', 'hyperopt'}, optional
             Optimization method for the parameter. Default: ``'grid_search'``.
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
                    Only relevant if ``'range'`` is specified and ``'choices'``
                    is not.
                ``'type'`` : {'linear'}, optional
                    Type of grid, i.e. distribution of the values.
                    Default: ``'linear'``.
                    Only relevant if ``'range'`` is specified and ``'choices'``
                    is not.
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
                 name='', hyper_params=None, callback=None):
        self.reco_space = reco_space
        self.observation_space = observation_space
        self.name = name or self.__class__.__name__
        if callable(callback) and not isinstance(callback, Callback):
            callback = CallbackApply(callback)
        self.hyper_params = {k: v['default']
                             for k, v in self.HYPER_PARAMS.items()}
        if hyper_params is not None:
            self.hyper_params.update(hyper_params)
        self.callback = callback

    @abstractmethod
    def reconstruct(self, observation_data):
        """Reconstruct input data from observation data.

        This method must be implemented by subclasses.

        Parameters
        ----------
        observation_data : `observation_space` element
            The observation data.

        Returns
        -------
        reconstruction : `reco_space` element
            The reconstruction.
        """

    def reset(self):
        """Reset the reconstructor.

        The default implementation resets `callback` if present.
        """
        if self.callback is not None:
            self.callback.reset()


class FunctionReconstructor(Reconstructor):
    """Reconstructor defined by a callable calculating the reconstruction.

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
        """Construct a reconstructor by specifying a callable.

        Parameters
        ----------
        function : callable
            Callable that is used in `reconstruct`.
        fun_args : list, optional
            Arguments to be passed to `function`.
        fun_kwargs : dict, optional
            Keyword arguments to be passed to `function`.
        """
        super().__init__(*args, **kwargs)
        self.function = function
        self.fun_args = fun_args or []
        self.fun_kwargs = fun_kwargs or {}

    def reconstruct(self, observation_data):
        if 'callback' in signature(self.function).parameters:
            self.kwargs['callback'] = self.callback
        return self.function(observation_data, *self.fun_args,
                             **self.fun_kwargs)


class LearnedReconstructor(Reconstructor):
    @abstractmethod
    def train(self, dataset):
        """Train the reconstructor with a dataset by adapting its parameters.

        Should only use the training data from `dataset`.

        Parameters
        ----------
        dataset : `Dataset`
            The dataset from which the training data should be used.
        """
