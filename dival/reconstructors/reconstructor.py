# -*- coding: utf-8 -*-
"""Provides the abstract reconstructor base class."""
import os
from inspect import signature, Parameter
import json
from warnings import warn
from copy import deepcopy


class _ReconstructorMeta(type):
    def __init__(cls, name, bases, dct):
        def get_fget(k):
            def fget(self):
                return self.hyper_params[k]
            return fget

        def get_fset(k):
            def fset(self, v):
                self.hyper_params[k] = v
            return fset

        for k in cls.HYPER_PARAMS.keys():
            if k.isidentifier():
                fget = get_fget(k)
                fset = get_fset(k)
                setattr(cls, '_fget_{}'.format(k), fget)
                setattr(cls, '_fset_{}'.format(k), fset)
                setattr(cls, k, property(fget, fset))


class Reconstructor(metaclass=_ReconstructorMeta):
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

    The class attribute :attr:`HYPER_PARAMS` defines the hyper parameters of
    the reconstructor class.
    The current values for a reconstructor instance are given by the attribute
    :attr:`hyper_params`.
    Properties wrapping :attr:`hyper_params` are automatically created by the
    metaclass (for hyper parameter names that are valid identifiers), such that
    the hyper parameters can be written and read like instance attributes.

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
        It is expected to have the same keys as :attr:`HYPER_PARAMS`.
        The values for these keys in this dict are wrapped by properties with
        the key as identifier (if possible), so an assignment to the property
        changes the value in this dict and vice versa.
    """

    HYPER_PARAMS = {}
    """Specification of hyper parameters.

    This class attribute is a dict that lists the hyper parameter of the
    reconstructor.
    It should not be hidden by an instance attribute of the same name (i.e. by
    assigning a value to `self.HYPER_PARAMS` in an instance of a subtype).

    *Note:* in order to inherit :attr:`HYPER_PARAMS` from a super class, the
    subclass should create a deep copy of it, i.e. execute
    ``HYPER_PARAMS = copy.deepcopy(SuperReconstructorClass.HYPER_PARAMS)`` in
    the class body.

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
            If `None`, the new array is initialized with zero before calling
            :meth:`_reconstruct`.

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
                reco = out
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

        Implementations should derive a sensible default for
        `hyper_params_path` from `path`, such that all parameters can be saved
        and loaded by specifying only `path`.
        Recommended patterns are:

            - if non-hyper parameters are stored in a single file and `path`
              specifies it without file ending:
              ``hyper_params_path=path + '_hyper_params.json'``
            - if non-hyper parameters are stored in a directory:
              ``hyper_params_path=os.path.join(path, 'hyper_params.json')``.
            - if there are no non-hyper parameters, this default implementation
              can be used:
              ``hyper_params_path=path + '_hyper_params.json'``

        Parameters
        ----------
        path : str[, optional]
            Path at which all (non-hyper) parameters should be saved.
            This argument is required if the reconstructor has non-hyper
            parameters or hyper_params_path is omitted.
            If the reconstructor has non-hyper parameters, the implementation
            may interpret it as a file path or as a directory path for multiple
            files (the dir should be created by this method if it does not
            exist).
            If the implementation expects a file path, it should accept it
            without file ending.
        hyper_params_path : str, optional
            Path of the file in which the hyper parameters should be saved.
            The ending ``'.json'`` is automatically appended if not included.
            If not specified, it should be determined from `path` (see
            method description above).
            The default implementation saves to the file
            ``path + '_hyper_params.json'``.
        """
        hp_path = hyper_params_path
        if hp_path is None:
            if path is None:
                raise ValueError(
                    'either `path` or `hyper_params_path` required (in '
                    'default implementation of `Reconstructor.save_params`)')
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

        See :meth:`save_params` for recommended patterns to derive a default
        `hyper_params_path` from `path`.

        Parameters
        ----------
        path : str[, optional]
            Path at which all (non-hyper) parameters are stored.
            This argument is required if the reconstructor has non-hyper
            parameters or hyper_params_path is omitted.
            If the reconstructor has non-hyper parameters, the implementation
            may interpret it as a file path or as a directory path for multiple
            files.
            If the implementation expects a file path, it should accept it
            without file ending.
        hyper_params_path : str, optional
            Path of the file in which the hyper parameters are stored.
            The ending ``'.json'`` is automatically appended if not included.
            If not specified, it should be determined from `path` (see
            description of :meth:`save_params`).
            The default implementation reads from the file
            ``path + '_hyper_params.json'``.
        """
        hp_path = hyper_params_path
        if hp_path is None:
            if path is None:
                raise ValueError(
                    'either `path` or `hyper_params_path` required (in '
                    'default implementation of `Reconstructor.save_params`)')
            hp_path = path + '_hyper_params.json'
        else:
            hp_path = (hyper_params_path if hyper_params_path.endswith('.json')
                       else hyper_params_path + '.json')
        self.load_hyper_params(hp_path)


class LearnedReconstructor(Reconstructor):
    def train(self, dataset):
        """Train the reconstructor with a dataset by adapting its parameters.

        Should only use the training and validation data from `dataset`.

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

        This implementation assumes that `path` is interpreted as a single
        file name, preferably specified without file ending.
        If `path` is a directory, the subclass needs to reimplement this method
        in order to follow the recommended default value pattern:
        ``hyper_params_path=os.path.join(path, 'hyper_params.json')``.

        Parameters
        ----------
        path : str
            Path at which the learned parameters should be saved.
            Passed to :meth:`save_learned_params`.
            If the implementation interprets it as a file path, it is
            preferred to exclude the file ending (otherwise the default
            value of `hyper_params_path` is suboptimal).
        hyper_params_path : str, optional
            Path of the file in which the hyper parameters should be saved.
            The ending ``'.json'`` is automatically appended if not included.
            If not specified, this implementation saves to the file
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

        This implementation assumes that `path` is interpreted as a single
        file name, preferably specified without file ending.
        If `path` is a directory, the subclass needs to reimplement this method
        in order to follow the recommended default value pattern:
        ``hyper_params_path=os.path.join(path, 'hyper_params.json')``.

        Parameters
        ----------
        path : str
            Path at which the parameters are stored.
            Passed to :meth:`load_learned_params`.
            If the implementation interprets it as a file path, it is
            preferred to exclude the file ending (otherwise the default
            value of `hyper_params_path` is suboptimal).
        hyper_params_path : str, optional
            Path of the file in which the hyper parameters are stored.
            The ending ``'.json'`` is automatically appended if not included.
            If not specified, this implementation reads from the file
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

    def save_learned_params(self, path):
        """Save learned parameters to file.

        Parameters
        ----------
        path : str
            Path at which the learned parameters should be saved.
            Implementations may interpret this as a file path or as a directory
            path for multiple files (which then should be created if it does
            not exist).
            If the implementation expects a file path, it should accept it
            without file ending.
        """
        raise NotImplementedError

    def load_learned_params(self, path):
        """Load learned parameters from file.

        Parameters
        ----------
        path : str
            Path at which the learned parameters are stored.
            Implementations may interpret this as a file path or as a directory
            path for multiple files.
            If the implementation expects a file path, it should accept it
            without file ending.
        """
        raise NotImplementedError


class IterativeReconstructor(Reconstructor):
    """Iterative reconstructor base class.
    It is recommended to use :class:`StandardIterativeReconstructor` as a base
    class for iterative reconstructors if suitable, which provides some default
    implementation.

    Subclasses must call :attr:`callback` after each iteration in
    ``self.reconstruct``. This is e.g. required by the :mod:`~dival.evaluation`
    module.

    Attributes
    ----------
    callback : ``odl.solvers.util.callback.Callback`` or `None`
        Callback to be called after each iteration.
    """

    HYPER_PARAMS = deepcopy(Reconstructor.HYPER_PARAMS)
    HYPER_PARAMS.update({
        'iterations': {
            'default': 100,
            'retrain': False
        }
    })

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
            Additional callback for this reconstruction that is temporarily
            composed with :attr:`callback`, i.e. also called after each
            iteration.
            If `None`, just :attr:`callback` is called.

        Returns
        -------
        reconstruction : :attr:`reco_space` element or `out`
            The reconstruction.
        """
        if callback is not None:
            orig_callback = self.callback
            self.callback = (callback if self.callback is None else
                             self.callback & callback)
        reconstruction = super().reconstruct(observation, out=out)
        if callback is not None:
            self.callback = orig_callback
        return reconstruction


class StandardIterativeReconstructor(IterativeReconstructor):
    """Standard iterative reconstructor base class.

    Provides a default implementation that only requires subclasses to
    implement :meth:`_compute_iterate` and optionally :meth:`_setup`.

    Attributes
    ----------
    x0 : :attr:`reco_space` element-like or `None`
        Default initial value for the iterative reconstruction.
        Can be overridden by passing a different ``x0`` to :meth:`reconstruct`.
    callback : ``odl.solvers.util.callback.Callback`` or `None`
        Callback that is called after each iteration.
    """

    def __init__(self, x0=None, callback=None, **kwargs):
        """
        Parameters
        ----------
        x0 : :attr:`reco_space` element-like, optional
            Default initial value for the iterative reconstruction.
            Can be overridden by passing a different ``x0`` to
            :meth:`reconstruct`.
        callback : ``odl.solvers.util.callback.Callback``, optional
            Callback that is called after each iteration.
        """
        self.x0 = x0
        super().__init__(callback=callback, **kwargs)

    def _setup(self, observation):
        """Setup before iteration process.
        Called by the default implementation of :meth:`_reconstruct` in the
        beginning, i.e. before computing the first iterate.

        Parameters
        ----------
        observation : :attr:`observation_space` element-like
            The observation data (forwarded from :meth:`reconstruct`).
        """
        pass

    def _compute_iterate(self, observation, reco_previous, out):
        """Compute next iterate.
        This method implements the iteration step in the default
        implementation of :meth:`_reconstruct`.

        Parameters
        ----------
        observation : :attr:`observation_space` element-like
            The observation data (forwarded from :meth:`reconstruct`).
        reco_previous : :attr:`reco_space` element-like
            The previous iterate value.
        out : :attr:`reco_space` element-like
            Array to which the iterate value is written.
        """
        raise NotImplementedError

    def reconstruct(self, observation, out=None, x0=None, last_iter=0,
                    callback=None):
        """Reconstruct input data from observation data.

        Same as :meth:`Reconstructor.reconstruct`, but with additional
        options for iterative reconstructors.

        Parameters
        ----------
        observation : :attr:`observation_space` element-like
            The observation data.
        out : :attr:`reco_space` element-like, optional
            Array to which the result is written (in-place evaluation).
            If `None`, a new array is created (out-of-place evaluation).
        x0 : :attr:`reco_space` element-like, optional
            Initial value for the iterative reconstruction.
            Overrides the attribute :attr:`x0`, which can be set when calling
            :meth:`__init__`.
            If both :attr:`x0` and this argument are `None`, the default
            implementation uses the value of `out` if called in-place, or zero
            if called out-of-place.
        last_iter : int, optional
            If `x0` is the result of an iteration by this method,
            this can be used to specify the number of iterations so far.
            The number of iterations for the current call is
            ``self.hyper_params['iterations'] - last_iter``.
        callback : ``odl.solvers.util.callback.Callback``, optional
            Additional callback for this reconstruction that is temporarily
            composed with :attr:`callback`, i.e. also called after each
            iteration.
            If `None`, just :attr:`callback` is called.

        Returns
        -------
        reconstruction : :attr:`reco_space` element or `out`
            The reconstruction.
        """
        self._x0_override = x0
        self._last_iter = last_iter
        return super().reconstruct(observation, out=out, callback=callback)

    def _reconstruct(self, observation, out):
        self._setup(observation)
        x = out
        if self._x0_override is not None:
            x[:] = self._x0_override  # override for specific reconstruction
        elif self.x0 is not None:
            x[:] = self.x0  # default init value
        # keep value of `out` if no `x0` was specified
        for i in range(self.hyper_params['iterations'] - self._last_iter):
            self._compute_iterate(observation, reco_previous=x.copy(), out=x)
            if self.callback is not None:
                self.callback(x)


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
