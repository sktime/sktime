# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""
Extension template for transformers, SIMPLE version.

Contains only bare minimum of implementation requirements for a functional transformer.
Also assumes *no composition*, i.e., no transformer or other estimator components.
For advanced cases (inverse transform, composition, etc),
    see full extension template in forecasting.py

Purpose of this implementation template:
    quick implementation of new estimators following the template
    NOT a concrete class to import! This is NOT a base class or concrete class!
    This is to be used as a "fill-in" coding template.

How to use this implementation template to implement a new estimator:
- make a copy of the template in a suitable location, give it a descriptive name.
- work through all the "todo" comments below
- fill in code for mandatory methods, and optionally for optional methods
- you can add more private methods, but do not override BaseEstimator's private methods
    an easy way to be safe is to prefix your methods with "_custom"
- change docstrings for functions and the file
- ensure interface compatibility by sktime.utils.estimator_checks.check_estimator
- once complete: use as a local library, or contribute to sktime via PR
- more details: https://www.sktime.org/en/stable/developer_guide/add_estimators.html

Mandatory implements:
    fitting         - _fit(self, X, y=None)
    transformation  - _transform(self, X, y=None)

Testing - implement if sktime transformer (not needed locally):
    get default parameters for test instance(s) - get_test_params()
"""
# todo: write an informative docstring for the file or module, remove the above
# todo: add an appropriate copyright notice for your estimator
#       estimators contributed to sktime should have the copyright notice at the top
#       estimators of your own do not need to have permissive or BSD-3 copyright

# todo: uncomment the following line, enter authors' GitHub IDs
# __author__ = [authorGitHubID, anotherAuthorGitHubID]

import numpy as np

from sktime.transformations.base import BaseTransformer
from sktime.utils.validation._dependencies import _check_soft_dependencies

# todo: add any necessary sktime external imports here


_check_soft_dependencies("pykalman", severity="warning")
_check_soft_dependencies("filterpy", severity="warning")


# todo: add any necessary sktime internal imports here


class BaseKalmanFilter:
    """todo: write docstring.

    todo: describe your custom transformer here
        fill in sections appropriately
        docstring must be numpydoc compliant

    Parameters
    ----------
    state_dim : int
        descriptive explanation of parama
    state_transition : numpy array, optional (default=None)
        descriptive explanation of paramb
    process_noise : int
        descriptive explanation of parama
    measurement_noise : numpy array, optional (default=None)
        descriptive explanation of paramb
    measurement_function : numpy array, optional (default=None)
        descriptive explanation of paramc and so on
    initial_state : numpy array, optional (default=None)
        descriptive explanation of paramc and so on
    initial_state_covariance : numpy array
        descriptive explanation of parama
    """

    def __init__(
        self,
        state_dim,
        state_transition=None,
        process_noise=None,
        measurement_noise=None,
        measurement_function=None,
        initial_state=None,
        initial_state_covariance=None,
    ):
        # todo: write any hyper-parameters to self
        self.state_dim = state_dim

        # F/A
        self.state_transition = state_transition
        # Q
        self.process_noise = process_noise
        # R
        self.measurement_noise = measurement_noise
        # H/C
        self.measurement_function = measurement_function
        # X0
        self.initial_state = initial_state
        # P0
        self.initial_state_covariance = initial_state_covariance

        # important: no checking or other logic should happen here
        super(BaseKalmanFilter, self).__init__()

    def _init_matrix(self, matrices, transform_func, default_val):
        if matrices is None:
            return default_val
        return transform_func(matrices)

    def _get_init_values(self, measurement_dim, state_dim):
        F = self._init_matrix(
            matrices=self.state_transition,
            transform_func=np.atleast_2d,
            default_val=np.eye(state_dim),
        )
        Q = self._init_matrix(
            matrices=self.process_noise,
            transform_func=np.atleast_2d,
            default_val=np.eye(state_dim),
        )
        R = self._init_matrix(
            matrices=self.measurement_noise,
            transform_func=np.atleast_2d,
            default_val=np.eye(measurement_dim),
        )
        H = self._init_matrix(
            matrices=self.measurement_function,
            transform_func=np.atleast_2d,
            default_val=np.eye(measurement_dim, state_dim),
        )
        X0 = self._init_matrix(
            matrices=self.initial_state,
            transform_func=np.atleast_1d,
            default_val=np.zeros(state_dim),
        )
        P0 = self._init_matrix(
            matrices=self.initial_state_covariance,
            transform_func=np.atleast_2d,
            default_val=np.eye(state_dim),
        )

        return F, Q, R, H, X0, P0

    def _state_dim(self, t):
        dim_list = []
        for (matrices, transform_func, dim_index) in t:
            if matrices is not None:
                dim_list.append(transform_func(matrices).shape[dim_index])

        if len(dim_list) == 0:
            return self.state_dim

        if not (dim_list.count(dim_list[0]) == len(dim_list)):
            raise ValueError(
                "There's an inconsistency in the dimensions of input matrices. "
            )

        return dim_list[0]

    def _infer_state_dim(self):
        t = [
            (self.state_transition, np.atleast_2d, 1),
            (self.process_noise, np.atleast_2d, 1),
            (self.initial_state, np.asarray, 0),
            (self.initial_state_covariance, np.atleast_2d, 1),
            (self.measurement_function, np.atleast_2d, 1),
        ]

        return self._state_dim(t)

    def _get_t_matrix(self, time, matrices, matrices_dims):
        """Extract matrix to be used at time step t.

        Parameters
        ----------
        time : int, The required time step.
        matrices : np.array
        matrices_dims : int

        Returns
        -------
        matrix or vector to be used at time step t
        """
        matrices = np.asarray(matrices)
        if matrices.ndim == matrices_dims:
            return matrices[time]
        elif matrices.ndim == matrices_dims - 1:
            return matrices
        else:
            raise ValueError(
                "dimensions of `matrices` does not match "
                "dimensions of a single object or a list of objects."
            )


class KalmanFilterPykalmanAdapter(BaseKalmanFilter, BaseTransformer):
    """todo: write docstring.

    todo: describe your custom transformer here
        fill in sections appropriately
        docstring must be numpydoc compliant

    Parameters
    ----------
    state_dim : int
        descriptive explanation of parama
    state_transition : numpy array, optional (default=None)
        descriptive explanation of paramb
    process_noise : int
        descriptive explanation of parama
    measurement_noise : numpy array, optional (default=None)
        descriptive explanation of paramb
    measurement_function : numpy array, optional (default=None)
        descriptive explanation of paramc and so on
    initial_state : numpy array, optional (default=None)
        descriptive explanation of paramc and so on
    initial_state_covariance : numpy array
        descriptive explanation of parama
    smoothing : bool, optional (default=None)
        This parameter affects transform. If False, then this kf will be inferring
        hidden state (given u, z, produce x). If True, uses
        `pykalman` `smooth()`
        denoising (given u, z, produce denoised z).
    estimate_matrices : optional, subset of [‘state_transition’,
        ‘measurement_function’, ‘process_noise’, ‘measurement_noise’,
        ‘initial_state’, ‘initial_state_covariance’] or - ‘all’.
        If estimate_matrices is an iterable of strings, only matrices in
        estimate_matrices will be estimated using EM algorithm,
        like described in `pykalman`. If estimate_matrices is ‘all’,
        then all matrices will be estimated.
    """

    # todo: fill out estimator tags here
    #  tags are inherited from parent class if they are not set
    #
    # todo: define the transformer scitype by setting the tags
    #   scitype:transform-input - the expected input scitype of X
    #   scitype:transform-output - the output scitype that transform produces
    #   scitype:transform-labels - whether y is used and if yes which scitype
    #   scitype:instancewise - whether transform uses all samples or acts by instance
    #
    # todo: define internal types for X, y in _fit/_transform by setting the tags
    #   X_inner_mtype - the internal mtype used for X in _fit and _transform
    #   y_inner_mtype - if y is used, the internal mtype used for y; usually "None"
    #   setting this guarantees that X, y passed to _fit, _transform are of above types
    #   for possible mtypes see datatypes.MTYPE_REGISTER, or the datatypes tutorial
    #
    #  when scitype:transform-input is set to Panel:
    #   X_inner_mtype must be changed to one or a list of sktime Panel mtypes
    #  when scitype:transform-labels is set to Series or Panel:
    #   y_inner_mtype must be changed to one or a list of compatible sktime mtypes
    #  the other tags are "safe defaults" which can usually be left as-is

    _tags = {
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:transform-labels": "None",
        # what is the scitype of y: None (not needed), Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "capability:inverse_transform": False,  # can the transformer inverse transform?
        "univariate-only": False,  # can the transformer handle multivariate X?
        "X_inner_mtype": "np.ndarray",  # which mtypes do _fit/_predict support for X?
        # this can be a Panel mtype even if transform-input is Series, vectorized
        "y_inner_mtype": "np.ndarray",  # which mtypes do _fit/_predict support for y?
        "requires_y": False,  # does y need to be passed in fit?
        "enforce_index_type": None,  # index type that needs to be enforced in X/y
        "fit_is_empty": False,  # is fit empty and can be skipped? Yes = True
        "X-y-must-have-same-index": False,  # can estimator handle different X/y index?
        "transform-returns-same-time-index": False,
        # does transform return have the same time index as input X
        "skip-inverse-transform": False,  # is inverse-transform skipped when called?
        "capability:unequal_length": True,
        # can the transformer handle unequal length time series (if passed Panel)?
        "capability:unequal_length:removes": False,
        # is transform result always guaranteed to be equal length (and series)?
        #   not relevant for transformers that return Primitives in transform-output
        "handles-missing-data": True,  # can estimator handle missing data?
        # todo: rename to capability:missing_values
        "capability:missing_values:removes": False,
        # is transform result always guaranteed to contain no missing values?
    }

    def __init__(
        self,
        state_dim,
        state_transition=None,
        transition_offsets=None,
        measurement_offsets=None,
        process_noise=None,
        measurement_noise=None,
        measurement_function=None,
        initial_state=None,
        initial_state_covariance=None,
        estimate_matrices=None,
        smoothing=False,
    ):
        _check_soft_dependencies("pykalman", severity="error", object=self)

        super(KalmanFilterPykalmanAdapter, self).__init__(
            state_dim=state_dim,
            state_transition=state_transition,
            process_noise=process_noise,
            measurement_noise=measurement_noise,
            measurement_function=measurement_function,
            initial_state=initial_state,
            initial_state_covariance=initial_state_covariance,
        )

        self.transition_offsets = transition_offsets
        self.measurement_offsets = measurement_offsets
        self.estimate_matrices = estimate_matrices
        self.smoothing = smoothing

    # todo: implement this, mandatory (except in special case below)
    def _fit(self, X, y=None):
        """Fit transformer to X and y.

        private _fit containing the core logic, called from fit

        Parameters
        ----------
        X : Series or Panel of mtype X_inner_mtype
            if X_inner_mtype is list, _fit must support all types in it
            Data to fit transform to
        y : Series or Panel of mtype y_inner_mtype, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        self: reference to self
        """
        # implement here
        # X, y passed to this function are always of X_inner_mtype, y_inner_mtype
        # IMPORTANT: avoid side effects to X, y
        #
        # any model parameters should be written to attributes ending in "_"
        #  attributes set by the constructor must not be overwritten
        #
        # special case: if no fitting happens before transformation
        #  then: delete _fit (don't implement)
        #   set "fit_is_empty" tag to True
        #
        # Note: when interfacing a model that has fit, with parameters
        #   that are not data (X, y) or data-like
        #   but model parameters, *don't* add as arguments to fit, but treat as follows:
        #   1. pass to constructor,  2. write to self in constructor,
        #   3. read from self in _fit,  4. pass to interfaced_model.fit in _fit

        measurement_dim = X.shape[1]
        state_dim_ = self._infer_state_dim()

        if self.estimate_matrices is None:
            (
                self.F_,
                self.Q_,
                self.R_,
                self.H_,
                self.X0_,
                self.P0_,
            ) = self._get_init_values(measurement_dim, state_dim_)

            self.transition_offsets_ = self._init_matrix(
                matrices=self.transition_offsets,
                transform_func=np.atleast_1d,
                default_val=np.zeros(state_dim_),
            )
            self.measurement_offsets_ = self._init_matrix(
                matrices=self.measurement_offsets,
                transform_func=np.atleast_1d,
                default_val=np.zeros(measurement_dim),
            )

            return self

        (F, H, Q, R, transition_offsets, measurement_offsets, X0, P0) = self._em(
            X=X, measurement_dim=measurement_dim, state_dim=state_dim_
        )

        self.F_ = np.copy(F)
        self.H_ = np.copy(H)
        self.Q_ = np.copy(Q)
        self.R_ = np.copy(R)
        self.transition_offsets_ = np.copy(transition_offsets)
        self.measurement_offsets_ = np.copy(measurement_offsets)
        self.X0_ = np.copy(X0)
        self.P0_ = np.copy(P0)

        return self

    def _em(self, X, measurement_dim, state_dim):
        from pykalman import KalmanFilter

        X_masked = np.ma.masked_invalid(X)
        estimate_matrices_ = self._get_estimate_matrices()

        kf = KalmanFilter(
            transition_matrices=self.state_transition,
            observation_matrices=self.measurement_function,
            transition_covariance=self.process_noise,
            observation_covariance=self.measurement_noise,
            transition_offsets=self.transition_offsets,
            observation_offsets=self.measurement_offsets,
            initial_state_mean=self.initial_state,
            initial_state_covariance=self.initial_state_covariance,
            n_dim_obs=measurement_dim,
            n_dim_state=state_dim,
        )

        kf = kf.em(X=X_masked, em_vars=estimate_matrices_)

        F = kf.transition_matrices
        H = kf.observation_matrices
        Q = kf.transition_covariance
        R = kf.observation_covariance
        transition_offsets = kf.transition_offsets
        measurement_offsets = kf.observation_offsets
        X0 = kf.initial_state_mean
        P0 = kf.initial_state_covariance

        return F, H, Q, R, transition_offsets, measurement_offsets, X0, P0

    # todo: implement this, mandatory
    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Parameters
        ----------
        X : Series or Panel of mtype X_inner_mtype
            if X_inner_mtype is list, _transform must support all types in it
            Data to be transformed
        y : Series or Panel of mtype y_inner_mtype, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        transformed version of X
        """
        # implement here
        # X, y passed to this function are always of X_inner_mtype, y_inner_mtype
        # IMPORTANT: avoid side effects to X, y
        #
        # if transform-output is "Primitives":
        #  return should be pd.DataFrame, with as many rows as instances in input
        #  if input is a single series, return should be single-row pd.DataFrame
        # if transform-output is "Series":
        #  return should be of same mtype as input, X_inner_mtype
        #  if multiple X_inner_mtype are supported, ensure same input/output
        # if transform-output is "Panel":
        #  return a multi-indexed pd.DataFrame of Panel mtype pd_multiindex
        #
        # todo: add the return mtype/scitype to the docstring, e.g.,
        #  Returns
        #  -------
        #  X_transformed : Series of mtype pd.DataFrame
        #       transformed version of X

        from pykalman import KalmanFilter

        X_masked = np.ma.masked_invalid(X)

        kf = KalmanFilter(
            transition_matrices=self.F_,
            observation_matrices=self.H_,
            transition_covariance=self.Q_,
            observation_covariance=self.R_,
            transition_offsets=self.transition_offsets_,
            observation_offsets=self.measurement_offsets_,
            initial_state_mean=self.X0_,
            initial_state_covariance=self.P0_,
        )
        if self.smoothing:
            (state_means, state_covariances) = kf.smooth(X_masked)
        else:
            (state_means, state_covariances) = kf.filter(X_masked)
        return state_means

    # todo: return default parameters, so that a test instance can be created
    #   required for automated unit and integration testing of estimator
    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for transformers.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        # todo: set the testing parameters for the estimators
        # Testing parameters can be dictionary or list of dictionaries.
        # Testing parameter choice should cover internal cases well.
        #   for "simple" extension, ignore the parameter_set argument.
        #
        # example 1: specify params as dictionary
        # any number of params can be specified
        # params = {"est": value0, "parama": value1, "paramb": value2}
        #
        # example 2: specify params as list of dictionary
        # note: Only first dictionary will be used by create_test_instance
        # params = [{"est": value1, "parama": value2},
        #           {"est": value3, "parama": value4}]
        #
        # return params
        params = {"state_dim": 2}
        return params

    def _get_estimate_matrices(self):
        params_mapping = {
            "state_transition": "transition_matrices",
            "process_noise": "transition_covariance",
            "measurement_offsets": "observation_offsets",
            "transition_offsets": "transition_offsets",
            "measurement_noise": "observation_covariance",
            "measurement_function": "observation_matrices",
            "initial_state": "initial_state_mean",
            "initial_state_covariance": "initial_state_covariance",
        }

        if isinstance(self.estimate_matrices, str):
            if self.estimate_matrices == "all":
                return list(params_mapping.values())
            if self.estimate_matrices in params_mapping:
                return list(params_mapping[self.estimate_matrices])

            raise ValueError(
                f"If `estimate_matrices` is passed as a "
                f"string, "
                f"it must be `all` / one of: "
                f"{list(params_mapping.keys())}, but found: "
                f"{self.estimate_matrices}"
            )

        em_vars = []
        for _matrix in self.estimate_matrices:
            if _matrix not in params_mapping:
                raise ValueError(
                    f"Elements of `estimate_matrices` "
                    f"must be a subset of "
                    f"{list(params_mapping.keys())}, but found: "
                    f"{_matrix}"
                )
            em_vars.append(params_mapping[_matrix])

        return em_vars


class KalmanFilterFilterPyAdapter(BaseKalmanFilter, BaseTransformer):
    """todo: write docstring.

    todo: describe your custom transformer here
        fill in sections appropriately
        docstring must be numpydoc compliant

    Parameters
    ----------
    state_dim : int
        descriptive explanation of parama
    state_transition : numpy array, optional (default=None)
        descriptive explanation of paramb
    control_transition : numpy array, optional (default=None)
        descriptive explanation of paramc and so on
    process_noise : int
        descriptive explanation of parama
    measurement_noise : numpy array, optional (default=None)
        descriptive explanation of paramb
    measurement_function : numpy array, optional (default=None)
        descriptive explanation of paramc and so on
    initial_state : numpy array, optional (default=None)
        descriptive explanation of paramc and so on
    initial_state_covariance : numpy array
        descriptive explanation of parama
    smoothing : bool, optional (default=None)
        This parameter affects transform. If False, then this kf will be inferring
        hidden state (given u, z, produce x). If True, uses
        `FilterPy` Rauch-Tung-Striebal Kalman smoother for
        denoising (given u, z, produce denoised z).
    estimate_matrices : optional, subset of [‘state_transition’,
        ‘measurement_function’, ‘process_noise’, ‘measurement_noise’,
        ‘initial_state’, ‘initial_state_covariance’] or - ‘all’.
        If estimate_matrices is an iterable of strings, only matrices in
        estimate_matrices will be estimated using EM algorithm,
        like described in `pykalman`. If estimate_matrices is ‘all’,
        then all matrices will be estimated.
        Note - ‘control_transition’ cannot be estimated.
    """

    # todo: fill out estimator tags here
    #  tags are inherited from parent class if they are not set
    #
    # todo: define the transformer scitype by setting the tags
    #   scitype:transform-input - the expected input scitype of X
    #   scitype:transform-output - the output scitype that transform produces
    #   scitype:transform-labels - whether y is used and if yes which scitype
    #   scitype:instancewise - whether transform uses all samples or acts by instance
    #
    # todo: define internal types for X, y in _fit/_transform by setting the tags
    #   X_inner_mtype - the internal mtype used for X in _fit and _transform
    #   y_inner_mtype - if y is used, the internal mtype used for y; usually "None"
    #   setting this guarantees that X, y passed to _fit, _transform are of above types
    #   for possible mtypes see datatypes.MTYPE_REGISTER, or the datatypes tutorial
    #
    #  when scitype:transform-input is set to Panel:
    #   X_inner_mtype must be changed to one or a list of sktime Panel mtypes
    #  when scitype:transform-labels is set to Series or Panel:
    #   y_inner_mtype must be changed to one or a list of compatible sktime mtypes
    #  the other tags are "safe defaults" which can usually be left as-is
    _tags = {
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:transform-labels": "None",
        # what is the scitype of y: None (not needed), Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "capability:inverse_transform": False,  # can the transformer inverse transform?
        "univariate-only": False,  # can the transformer handle multivariate X?
        "X_inner_mtype": "np.ndarray",  # which mtypes do _fit/_predict support for X?
        # this can be a Panel mtype even if transform-input is Series, vectorized
        "y_inner_mtype": "np.ndarray",  # which mtypes do _fit/_predict support for y?
        "requires_y": False,  # does y need to be passed in fit?
        "enforce_index_type": None,  # index type that needs to be enforced in X/y
        "fit_is_empty": False,  # is fit empty and can be skipped? Yes = True
        "X-y-must-have-same-index": False,  # can estimator handle different X/y index?
        "transform-returns-same-time-index": False,
        # does transform return have the same time index as input X
        "skip-inverse-transform": False,  # is inverse-transform skipped when called?
        "capability:unequal_length": True,
        # can the transformer handle unequal length time series (if passed Panel)?
        "capability:unequal_length:removes": False,
        # is transform result always guaranteed to be equal length (and series)?
        #   not relevant for transformers that return Primitives in transform-output
        "handles-missing-data": True,  # can estimator handle missing data?
        # todo: rename to capability:missing_values
        "capability:missing_values:removes": False,
        # is transform result always guaranteed to contain no missing values?
    }

    def __init__(
        self,
        state_dim,
        state_transition=None,
        control_transition=None,
        process_noise=None,
        measurement_noise=None,
        measurement_function=None,
        initial_state=None,
        initial_state_covariance=None,
        estimate_matrices=None,
        smoothing=False,
    ):
        _check_soft_dependencies("filterpy", severity="error", object=self)

        super(KalmanFilterFilterPyAdapter, self).__init__(
            state_dim=state_dim,
            state_transition=state_transition,
            process_noise=process_noise,
            measurement_noise=measurement_noise,
            measurement_function=measurement_function,
            initial_state=initial_state,
            initial_state_covariance=initial_state_covariance,
        )

        self.control_transition = control_transition
        self.estimate_matrices = estimate_matrices
        self.smoothing = smoothing

    # todo: implement this, mandatory (except in special case below)
    def _fit(self, X, y=None):
        """Fit transformer to X and y.

        private _fit containing the core logic, called from fit

        Parameters
        ----------
        X : Series or Panel of mtype X_inner_mtype
            if X_inner_mtype is list, _fit must support all types in it
            Data to fit transform to
        y : Series or Panel of mtype y_inner_mtype, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        self: reference to self
        """
        # implement here
        # X, y passed to this function are always of X_inner_mtype, y_inner_mtype
        # IMPORTANT: avoid side effects to X, y
        #
        # any model parameters should be written to attributes ending in "_"
        #  attributes set by the constructor must not be overwritten
        #
        # special case: if no fitting happens before transformation
        #  then: delete _fit (don't implement)
        #   set "fit_is_empty" tag to True
        #
        # Note: when interfacing a model that has fit, with parameters
        #   that are not data (X, y) or data-like
        #   but model parameters, *don't* add as arguments to fit, but treat as follows:
        #   1. pass to constructor,  2. write to self in constructor,
        #   3. read from self in _fit,  4. pass to interfaced_model.fit in _fit

        measurement_dim = X.shape[1]
        self.state_dim_ = self._infer_state_dim()

        if self.estimate_matrices is None:
            (
                self.F_,
                self.Q_,
                self.R_,
                self.H_,
                self.X0_,
                self.P0_,
            ) = self._get_init_values(measurement_dim, self.state_dim_)
            return self

        if isinstance(self.estimate_matrices, str) and self.estimate_matrices == "all":
            estimate_matrices_ = [
                "state_transition",
                "process_noise",
                "measurement_noise",
                "measurement_function",
                "initial_state",
                "initial_state_covariance",
            ]
        else:
            estimate_matrices_ = self.estimate_matrices

        transformer_ = KalmanFilterPykalmanAdapter(
            state_dim=self.state_dim,
            state_transition=self.state_transition,
            process_noise=self.process_noise,
            measurement_noise=self.measurement_noise,
            measurement_function=self.measurement_function,
            initial_state=self.initial_state,
            initial_state_covariance=self.initial_state_covariance,
            estimate_matrices=estimate_matrices_,
        )
        transformer_ = transformer_.fit(X, y=y)

        self.F_ = np.copy(transformer_.F_)
        self.H_ = np.copy(transformer_.H_)
        self.Q_ = np.copy(transformer_.Q_)
        self.R_ = np.copy(transformer_.R_)
        self.X0_ = np.copy(transformer_.X0_)
        self.P0_ = np.copy(transformer_.P0_)

        return self

    # todo: implement this, mandatory
    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Parameters
        ----------
        X : Series or Panel of mtype X_inner_mtype
            if X_inner_mtype is list, _transform must support all types in it
            Data to be transformed
        y : Series or Panel of mtype y_inner_mtype, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        transformed version of X
        """
        # implement here
        # X, y passed to this function are always of X_inner_mtype, y_inner_mtype
        # IMPORTANT: avoid side effects to X, y
        #
        # if transform-output is "Primitives":
        #  return should be pd.DataFrame, with as many rows as instances in input
        #  if input is a single series, return should be single-row pd.DataFrame
        # if transform-output is "Series":
        #  return should be of same mtype as input, X_inner_mtype
        #  if multiple X_inner_mtype are supported, ensure same input/output
        # if transform-output is "Panel":
        #  return a multi-indexed pd.DataFrame of Panel mtype pd_multiindex
        #
        # todo: add the return mtype/scitype to the docstring, e.g.,
        #  Returns
        #  -------
        #  X_transformed : Series of mtype pd.DataFrame
        #       transformed version of X

        import filterpy.kalman.kalman_filter as filterpy_kf

        time_steps = X.shape[0]

        x_priori = np.zeros((time_steps, self.state_dim_))
        p_priori = np.zeros((time_steps, self.state_dim_, self.state_dim_))

        x_posteriori = np.zeros((time_steps, self.state_dim_))
        p_posteriori = np.zeros((time_steps, self.state_dim_, self.state_dim_))

        if y is not None:
            u = np.atleast_1d(y)
            dim_u = u.shape[1] if u.ndim == 2 else u.shape[0]
            G = self._init_matrix(
                matrices=self.control_transition,
                transform_func=np.atleast_2d,
                default_val=np.eye(self.state_dim_, dim_u),
            )
        else:
            u = [0.0]
            G = np.array([[0.0]])

        x = self.X0_
        p = self.P0_

        for t in range(time_steps):
            (zt, Ft, Gt, Qt, Rt, Ht, ut) = self._get_iter_t_matrices(X=X, G=G, u=u, t=t)

            x, p = filterpy_kf.predict(x, p, u=ut, B=Gt, F=Ft, Q=Qt)
            x_priori[t, :] = x
            p_priori[t, :, :] = p

            x, p = filterpy_kf.update(x, p, zt, R=Rt, H=Ht)
            x_posteriori[t, :] = x
            p_posteriori[t, :, :] = p

        if self.smoothing:
            Fs = [self.F_] * time_steps if self.F_.ndim == 2 else self.F_
            Qs = [self.Q_] * time_steps if self.Q_.ndim == 2 else self.Q_
            return filterpy_kf.rts_smoother(
                Xs=x_posteriori, Ps=p_posteriori, Fs=Fs, Qs=Qs
            )[0]
        return x_posteriori

    # todo: return default parameters, so that a test instance can be created
    #   required for automated unit and integration testing of estimator
    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for transformers.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        # todo: set the testing parameters for the estimators
        # Testing parameters can be dictionary or list of dictionaries.
        # Testing parameter choice should cover internal cases well.
        #   for "simple" extension, ignore the parameter_set argument.
        #
        # example 1: specify params as dictionary
        # any number of params can be specified
        # params = {"est": value0, "parama": value1, "paramb": value2}
        #
        # example 2: specify params as list of dictionary
        # note: Only first dictionary will be used by create_test_instance
        # params = [{"est": value1, "parama": value2},
        #           {"est": value3, "parama": value4}]
        #
        # return params
        params = {"state_dim": 2}
        return params

    def _get_iter_t_matrices(self, X, G, u, t):
        zt = None if any(np.isnan(X[t])) else X[t]
        Ft = self._get_t_matrix(t, self.F_, 3)
        Gt = self._get_t_matrix(t, G, 3)
        Qt = self._get_t_matrix(t, self.Q_, 3)
        Rt = self._get_t_matrix(t, self.R_, 3)
        Ht = self._get_t_matrix(t, self.H_, 3)
        ut = self._get_t_matrix(t, u, 2)

        return zt, Ft, Gt, Qt, Rt, Ht, ut
