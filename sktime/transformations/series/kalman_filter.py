# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""
Kalman Filter Transformer.

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


class KalmanFilter(BaseTransformer):
    """Custom transformer. todo: write docstring.

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
    smoothing_technic : string, optional (default=None)
        This parameter affects transform. If `None`, then this kf will be inferring
        hidden state (given u, z, produce x). Another option is 'rts' which uses
        `FilterPy` Rauch-Tung-Striebal Kalman smoother for
        denoising (given u, z, produce denoised z).
    estimate_matrices : optional, subset of [‘state_transition’,
        ‘measurement_function’, ‘process_noise’, ‘measurement_noise’,
        ‘initial_state’, ‘initial_state_covariance’] or - ‘all’/'None'.
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

    # todo: add any hyper-parameters and components to constructor

    # state_transition == A/F, numpy 2D or 3D ndarray
    # control_transition == B/G, numpy 2D or 3D ndarray
    # process_noise == Q, numpy 2D or 3D ndarray
    # measurement_noise == R, numpy 2D or 3D ndarray
    # measurement_function == H, numpy 2D or 3D ndarray
    # initial_state == X0, numpy 1D array
    # initial_state_covariance == P0, numpy 1D array

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
        smoothing_technic=None,
        estimate_matrices=None,
    ):
        _check_soft_dependencies("pykalman", severity="error", object=self)
        _check_soft_dependencies("filterpy", severity="error", object=self)

        # todo: write any hyper-parameters to self
        # if estimate_matrices is None:
        #     estimate_matrices =
        #     ["process_noise", "initial_state", "initial_state_covariance"]
        self.state_dim = state_dim

        # F/A
        self.state_transition = state_transition
        # G/B
        self.control_transition = control_transition
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

        self.smoothing_technic = smoothing_technic
        self.estimate_matrices = estimate_matrices

        # important: no checking or other logic should happen here
        # todo: change "MyTransformer" to the name of the class
        super(KalmanFilter, self).__init__()

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

        # state_transition == A/F
        # control_transition == B/G
        # process_noise == Q
        # measurement_noise == R
        # measurement_function == H
        # initial_state == X0
        # initial_state_covariance == P0

        self.measurement_dim_ = X.shape[1]
        estimate_matrices_ = self._get_estimate_matrices()

        if estimate_matrices_ is None:
            (
                self.F_,
                self.G_,
                self.Q_,
                self.R_,
                self.H_,
                self.X0_,
                self.P0_,
            ) = self._set_params(self.measurement_dim_)
            return self

        from pykalman import KalmanFilter

        X_masked = np.ma.masked_invalid(X)

        kf = KalmanFilter(
            transition_matrices=self.state_transition,
            observation_matrices=self.measurement_function,
            transition_covariance=self.process_noise,
            observation_covariance=self.measurement_noise,
            initial_state_mean=self.initial_state,
            initial_state_covariance=self.initial_state_covariance,
            n_dim_obs=self.measurement_dim_,
            n_dim_state=self.state_dim,
        )

        kf = kf.em(X=X_masked, em_vars=estimate_matrices_)

        self.F_ = kf.transition_matrices
        self.H_ = kf.observation_matrices
        self.Q_ = kf.transition_covariance
        self.R_ = kf.observation_covariance
        self.X0_ = kf.initial_state_mean
        self.P0_ = kf.initial_state_covariance

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

        # state_transition == A/F
        # control_transition == B/G of size [n, state_dim, u_dim] or [state_dim, u_dim]
        # process_noise == Q
        # measurement_noise == R
        # measurement_function == H
        # initial_state == X0
        # initial_state_covariance == P0

        # X_ = np.array(X, copy=True)
        # X_ = np.where(np.isnan(X_), None, X_)
        from pykalman import KalmanFilter

        X_masked = np.ma.masked_invalid(X)

        b = self._get_offset(u=y)
        kf = KalmanFilter(
            transition_matrices=self.F_,
            observation_matrices=self.H_,
            transition_covariance=self.Q_,
            observation_covariance=self.R_,
            transition_offsets=b,
            initial_state_mean=self.X0_,
            initial_state_covariance=self.P0_,
        )

        if isinstance(self.smoothing_technic, str):
            if self.smoothing_technic == "rts":
                import filterpy.kalman.kalman_filter as filterpy_kf

                (state_means, state_covariances) = kf.filter(X_masked)
                time_steps = len(state_means)
                Fs = [self.F_] * time_steps if self.F_.ndim == 2 else self.F_
                Qs = [self.Q_] * time_steps if self.Q_.ndim == 2 else self.Q_
                (Xs, Ps, _, _) = filterpy_kf.rts_smoother(
                    Xs=state_means, Ps=state_covariances, Fs=Fs, Qs=Qs
                )
                return Xs

            # give different name
            if self.smoothing_technic == "pykalman_smoothing":
                (smoothed_state_means, smoothed_state_covariances) = kf.smooth(X_masked)
                return smoothed_state_means

        return kf.filter(X_masked)[0]

    def _set_params(self, measurement_dim, dim_u=0):
        F = (
            np.eye(self.state_dim)
            if self.state_transition is None
            else np.atleast_2d(self.state_transition)
        )
        G = self._get_G(dim_u=dim_u)
        Q = (
            np.eye(self.state_dim)
            if self.process_noise is None
            else np.atleast_2d(self.process_noise)
        )
        R = (
            np.eye(measurement_dim)
            if self.measurement_noise is None
            else np.atleast_2d(self.measurement_noise)
        )
        H = (
            np.eye(measurement_dim, self.state_dim)
            if self.measurement_function is None
            else np.atleast_2d(self.measurement_function)
        )
        X0 = (
            np.zeros(self.state_dim)
            if self.initial_state is None
            else np.atleast_1d(self.initial_state)
        )
        P0 = (
            np.eye(self.state_dim)
            if self.initial_state_covariance is None
            else np.atleast_2d(self.initial_state_covariance)
        )

        return F, G, Q, R, H, X0, P0

    def _get_G(self, dim_u=0, must_3d=False):
        if self.control_transition is not None:
            G = np.atleast_2d(self.control_transition)
        else:
            G = np.eye(self.state_dim, dim_u) if dim_u > 0 else None
        if must_3d and len(G.shape) == 2:
            G = np.atleast_2d([G])
        return G

    def _get_estimate_matrices(self):
        if self.estimate_matrices is None:
            return None

        params_mapping = {
            "state_transition": "transition_matrices",
            "control_transition": "transition_offsets",
            "process_noise": "transition_covariance",
            "measurement_noise": "observation_covariance",
            "measurement_function": "observation_matrices",
            "initial_state": "initial_state_mean",
            "initial_state_covariance": "initial_state_covariance",
        }

        if isinstance(self.estimate_matrices, str):
            if self.estimate_matrices == "all":
                return list(params_mapping.values())
            if self.estimate_matrices == "None" or self.estimate_matrices == "none":
                return None
            if self.estimate_matrices in params_mapping:
                return list(params_mapping[self.estimate_matrices])

            raise ValueError(
                f"If `estimate_matrices` is passed as a "
                f"string, "
                f"it must be `all` / `None` / one of: "
                f"{list(params_mapping.keys())}, but found: "
                f"{self.estimate_matrices}"
            )

        else:
            for _matrix in self.estimate_matrices:
                if _matrix not in params_mapping:
                    raise ValueError(
                        f"Elements of `estimate_matrices` "
                        f"must be a subset of "
                        f"{list(params_mapping.keys())}, but found: "
                        f"{_matrix}"
                    )
            return list(np.vectorize(params_mapping.get)(self.estimate_matrices))

    def _get_offset(self, u=None):
        # transition_offsets: [n_timesteps - 1, n_dim_state]
        # or [n_dim_state] array - like.
        # Also known as b.state. Offsets for times[0...n_timesteps - 2]
        if u is None:
            return None

        u_ = np.atleast_2d(u)
        time_steps_u, dim_u = u_.shape

        G = self._get_G(dim_u=dim_u, must_3d=True)
        time_steps_G, _, _ = G.shape

        n = max(time_steps_u, time_steps_G)
        offsets = np.zeros((n, self.state_dim))
        for t in range(n):
            ut = u_[0] if time_steps_u == 1 else u_[t]
            Gt = G[0] if time_steps_G == 1 else G[t]
            offsets[t] = np.dot(Gt, ut)

        if n == 1:
            return offsets[0]
        return offsets

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
