# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Kalman Filter Transformers.

Series based transformers based on Kalman Filter algorithm. Contains Base class
and two transformers which are each Adapters for external packages pykalman and FilterPy.
"""

# todo: add an appropriate copyright notice for your estimator
#       estimators contributed to sktime should have the copyright notice at the top
#       estimators of your own do not need to have permissive or BSD-3 copyright


__author__ = ["NoaBenAmi", "lielleravid"]
__all__ = ["BaseKalmanFilter", "KalmanFilterPykalmanAdapter", "KalmanFilterFilterPyAdapter"]

import numpy as np

from sktime.transformations.base import BaseTransformer
from sktime.utils.validation._dependencies import _check_soft_dependencies

_check_soft_dependencies("pykalman", severity="warning")
_check_soft_dependencies("filterpy", severity="warning")


class BaseKalmanFilter:
    """Kalman Filter is used for denoising data and/or inferring the hidden state of data given.

    The Kalman filter is an unsupervised algorithm consisting of several mathematical equations which are used to create
    an estimate of the state of a process.
    The algorithm does this efficiently and recursively in a way where the mean squared error is minimal.
    The Kalman Filter has the ability to support estimations of past, present and future states.
    The strength of the Kalman Filter is in its ability to infer the state of a system even when the exact nature of the
    system is not known.
    When given time series data the Kalman filter creates a denoising effect,
    by removing noise from the data and recovering the true state of the underlying object we are tracking within the
    data.
    The Kalman Filter computations are based on five equations.
    Two prediction equations:
    •	State Extrapolation Equation - prediction or estimation of the future state, based on the known present estimation.
    •	Covariance Extrapolation Equation - the measure of uncertainty in our prediction.
    Two update equations:
    •	State Update Equation - estimation of the current state, based on the known past estimation and present measurement.
    •	Covariance Update Equation - the measure of uncertainty in our estimation.
    Kalman Gain Equation – this is a required argument for the update equations.
    It acts as a weighting parameter for the past estimations and the given measurement.
    It defines the weight of the past estimation and the weight of the measurement in estimating the current state.

    Parameters
    ----------
    state_dim : int
        system state feature dimension
    state_transition : [timesteps, state_dim, state_dim] or [state_dim, state_dim] array-like, optional (default=None)
        state transition matrix also referred to as F is a matrix which describes the way the underlying series moves
        through successive time periods.
    process_noise : [timesteps, state_dim, state_dim] or [state_dim, state_dim] array-like, optional (default=None)
        process noise matrix also referred to as Q the uncertainty of the dynamic model
    measurement_noise : [timesteps, measurement_dim, measurement_dim] or [measurement_dim, measurement_dim] array-like,
        optional (default=None)
        measurement noise matrix also referred to as R represents the uncertainty of the measurements
    measurement_function : [timesteps, measurement_dim, state_dim] or [measurement_dim, state_dim], optional (default=None)
        measurement equation also referred to as H adjusts dimensions of samples to match dimensions of the state matrix
    initial_state : [state_dim] array-like, optional (default=None)
        initial estimated system state also referred to as x_0
    initial_state_covariance : [state_dim, state_dim] array-like, optional (default=None)
        initial_state_covariance also referred to as P_0

    References
    ----------
    .. [1] Greg Welch and Gary Bishop, "An Introduction to the Kalman Filter", 2006
       https://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf
    .. [2] R.H.Shumway and D.S.Stoffer "An Approach to time Series Smoothing and Forecasting Using the EM Algorithm", 1982
       https://www.stat.pitt.edu/stoffer/dss_files/em.pdf
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
        """Initialize default value if matrix is None or transform input matrix to be np.ndarray

        Parameters
        ----------
        matrices : array-like
        transform_func : transformation function from array-like to ndarray
        default_val : np.ndarray

        Returns
        -------
        matrix as np.ndarray
        """
        if matrices is None:
            return default_val
        return transform_func(matrices)

    def _get_init_values(self, measurement_dim, state_dim):
        """Initializes parameter matrices to default values and returns them.

        Parameters
        ----------
        measurement_dim : int
        state_dim : int

        Returns
        -------
        Six parameter matrices F,Q,R,H,X0,P0 as np.ndarrays
        """
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
        """Infer the state_dim from input parameter matrices if not None. Validate the size of input matrices.

        Validate input matrices by making sure they are all of corresponding sizes

        Parameters
        ----------
        t : list of tuples

        Returns
        -------
        state_dim : int
        """
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
        """Infer the state_dim from input parameter matrices if not None.

        Returns
        -------
        state_dim : int
        """
        t = [
            (self.state_transition, np.atleast_2d, 1),
            (self.process_noise, np.atleast_2d, 1),
            (self.initial_state, np.asarray, 0),
            (self.initial_state_covariance, np.atleast_2d, 1),
            (self.measurement_function, np.atleast_2d, 1),
        ]

        return self._state_dim(t)

    def _get_t_matrix(self, time, matrices, matrices_dims):
        """Extract matrix to be used at time step 'time' of the Kalman filter iterations.

        Parameters
        ----------
        time : int, The required time step.
        matrices : np.ndarray
        matrices_dims : int

        Returns
        -------
        matrix or vector to be used at time step 'time'
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
    """Kalman Filter is used for denoising data and/or inferring the hidden state of data given.

    This class is the adapter for the PyKalman package into Sktime.
    It wraps the PyKalman functions while implementing the Sktime transformer class and inheriting from BaseKalmanFilter class.
    The transformer implements hidden inferred state and denoising depending on the boolean input given by the
    user called 'denoising'.

    Parameters
    ----------
    state_dim : int
        system state feature dimension
    state_transition : [timesteps, state_dim, state_dim] or [state_dim, state_dim] array-like, optional (default=None)
        state transition matrix also referred to as F is a matrix which describes the way the underlying series moves
        through successive time periods.
    process_noise : [timesteps, state_dim, state_dim] or [state_dim, state_dim] array-like, optional (default=None)
        process noise matrix also referred to as Q the uncertainty of the dynamic model
    measurement_noise : [timesteps, measurement_dim, measurement_dim] or [measurement_dim, measurement_dim] array-like,
        optional (default=None)
        measurement noise matrix also referred to as R represents the uncertainty of the measurements
    measurement_function : [timesteps, measurement_dim, state_dim] or [measurement_dim, state_dim], optional (default=None)
        measurement equation also referred to as H adjusts dimensions of samples to match dimensions of the state matrix
    initial_state : [state_dim] array-like, optional (default=None)
        initial estimated system state also referred to as x_0
    initial_state_covariance : [state_dim, state_dim] array-like, optional (default=None)
        initial_state_covariance also referred to as P_0
    transition_offsets : [timesteps, state_dim] or [state_dim] array-like, optional (default=None)
        state offsets as described in PyKalman
    measurement_offsets : [timesteps, measurement_dim] or [measurement_dim] array-like, optional (default=None)
        observation offsets as described in PyKalman
    denoising : bool, optional (default=None)
        This parameter affects transform. If False, then transform will be inferring
        hidden state. If True, uses `pykalman` `smooth()` for denoising.
    estimate_matrices : optional, subset of [‘state_transition’,
        ‘measurement_function’, ‘process_noise’, ‘measurement_noise’,
        ‘initial_state’, ‘initial_state_covariance’] or - ‘all’.
        If estimate_matrices is an iterable of strings, only matrices in
        estimate_matrices will be estimated using EM algorithm,
        like described in `pykalman`. If estimate_matrices is ‘all’,
        then all matrices will be estimated using EM algorithm.

    References
    ----------
    .. [1] https://github.com/pykalman/pykalman
    todo: add pykalman license https://github.com/pykalman/pykalman/blob/master/COPYING?
    """
    _tags = {
        "X_inner_mtype": "np.ndarray",  # which mtypes do _fit/_predict support for X?
        "requires_y": False,  # does y need to be passed in fit?
        "fit_is_empty": False,  # is fit empty and can be skipped? Yes = True
        "capability:unequal_length": False,
        # can the transformer handle unequal length time series (if passed Panel)?
        "handles-missing-data": True,  # can estimator handle missing data?
        # todo: rename to capability:missing_values
        "capability:missing_values:removes": False,
        # is transform result always guaranteed to contain no missing values?
        "scitype:instancewise": True,  # is this an instance-wise transform?
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
            denoising=False,
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
        self.denoising = denoising

    def _fit(self, X, y=None):
        """Fit transformer to X and y.

        This method prepares the transformer.
        The matrix initializations or estimations if requested by user are calculated here.

        Parameters
        ----------
        X : 2D np.ndarray
            Data (measurements) to be transformed. Missing values must be represented as np.NaN or np.nan.
        y : ignored argument for interface compatibility

        Returns
        -------
        self: reference to self
        """
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
        """Estimate matrices algorithm if requested by user.

        If input matrices are specified in 'estimate_matrices' this method will use the pykalman em algorithm function
        to estimate said matrices needed to calculate the Kalman Filter. Algorithm explained in Reference[2] in
        BaseKalmanFilter class.
        If 'estimate_matrices' is None no matrices will be estimated.

        Parameters
        ----------
        X : 2D np.ndarray
            Data (measurements) to be transformed. Missing values must be represented as np.NaN or np.nan.
        measurement_dim : int
            measurement feature dimensions
        state_dim : int
            state feature dimensions

        Returns
        -------
        self: reference to self
        """
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

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        This method performs the transformation of the input data according to the constructor input parameter 'denoising'.
        If 'denoising' is True - then denoise data using pykalman's 'smooth' function.
        Else, infer hidden state using pykalman's 'filter' function.

        Parameters
        ----------
        X : 2D np.ndarray
            Data (measurements) to be transformed. Missing values must be represented as np.NaN or np.nan.
        y : ignored argument for interface compatibility

        Returns
        -------
        X_transformed : 2D np.ndarray
            transformed version of X
        """
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
        if self.denoising:
            (state_means, state_covariances) = kf.smooth(X_masked)
        else:
            (state_means, state_covariances) = kf.filter(X_masked)
        return state_means

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
        """map matrices names to pykalman matrices names for use of pykalman estimate matrices function.

        Returns
        -------
        em_vars : list
            mapped matirces names
        """
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
    """Kalman Filter is used for denoising data and/or inferring the hidden state of data given.

    This class is the adapter for the FilterPy package into Sktime.
    It wraps the FilterPy functions while implementing the Sktime transformer class and inheriting from BaseKalmanFilter class.
    The transformer implements hidden inferred state and denoising depending on the boolean input given by the
    user called 'denoising'.

    Parameters
    ----------
    state_dim : int
        system state feature dimension
    state_transition : [timesteps, state_dim, state_dim] or [state_dim, state_dim] array-like, optional (default=None)
        state transition matrix also referred to as F is a matrix which describes the way the underlying series moves
        through successive time periods.
    process_noise : [timesteps, state_dim, state_dim] or [state_dim, state_dim] array-like, optional (default=None)
        process noise matrix also referred to as Q the uncertainty of the dynamic model
    measurement_noise : [timesteps, measurement_dim, measurement_dim] or [measurement_dim, measurement_dim] array-like,
        optional (default=None)
        measurement noise matrix also referred to as R represents the uncertainty of the measurements
    measurement_function : [timesteps, measurement_dim, state_dim] or [measurement_dim, state_dim] array-like, optional (default=None)
        measurement equation also referred to as H adjusts dimensions of samples to match dimensions of the state matrix
    initial_state : [state_dim] array-like, optional (default=None)
        initial estimated system state also referred to as x_0
    initial_state_covariance : [state_dim, state_dim] array-like, optional (default=None)
        initial_state_covariance also referred to as P_0
    control_transition : [timesteps, state_dim, control_variable_dim] or [state_dim, control_variable_dim] array-like,
        optional (default=None). control transition is also referred to as G.
        'control_variable_dim' is the dimension of 'control variable' also referred to as U.
        Control variable is an optional parameter for fit and transform functions.
    denoising : bool, optional (default=None)
        This parameter affects transform. If False, then transform will be inferring
        hidden state. If True, uses `pykalman` `smooth()` for denoising.
    estimate_matrices : optional, subset of [‘state_transition’,
        ‘measurement_function’, ‘process_noise’, ‘measurement_noise’,
        ‘initial_state’, ‘initial_state_covariance’] or - ‘all’.
        If estimate_matrices is an iterable of strings, only matrices in
        estimate_matrices will be estimated using EM algorithm,
        like described in `pykalman`. If estimate_matrices is ‘all’,
        then all matrices will be estimated using EM algorithm. 'control_transition'
        matrix cannot be estimated.

    References
    ----------
    .. [1] https://github.com/rlabbe/filterpy/tree/a437893597957764fb6b415bfb5640bb117f5b99
    todo: add pykalman license https://github.com/rlabbe/filterpy/blob/master/LICENSE?
    """
    _tags = {
        "scitype:transform-labels": "Series",
        # what is the scitype of y: None (not needed), Primitives, Series, Panel
        "X_inner_mtype": "np.ndarray",  # which mtypes do _fit/_predict support for X?
        # this can be a Panel mtype even if transform-input is Series, vectorized
        "y_inner_mtype": "np.ndarray",  # which mtypes do _fit/_predict support for y?
        "fit_is_empty": False,  # is fit empty and can be skipped? Yes = True
        "capability:unequal_length": False,
        # can the transformer handle unequal length time series (if passed Panel)?
        "handles-missing-data": True,  # can estimator handle missing data?
        # todo: rename to capability:missing_values
        "capability:missing_values:removes": False,
        # is transform result always guaranteed to contain no missing values?
        "scitype:instancewise": True,  # is this an instance-wise transform?
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
            denoising=False,
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
        self.denoising = denoising

    def _fit(self, X, y=None):
        """Fit transformer to X and y.

        This method prepares the transformer.
        The matrix initializations or estimations if requested by user are calculated here.

        Parameters
        ----------
        X : 2D np.ndarray
            Data (measurements) to be transformed. Missing values must be represented as np.NaN or np.nan.
        y : ignored argument for interface compatibility

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

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        This method performs the transformation of the input data according to the constructor input parameter 'denoising'.
        If 'denoising' is True - then denoise data using filterpy's 'rts_smoother' function.
        Else, infer hidden state using filterpy's 'predict' and 'update' functions.

        Parameters
        ----------
        X : 2D np.ndarray
            Data (measurements) to be transformed. Missing values must be represented as np.NaN or np.nan.
        y : 1D or 2D np.ndarray
            'control variable' also referred to as U. if 2D must be same length as X.

        Returns
        -------
        X_transformed : 2D np.ndarray
            transformed version of X
        """
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

        if self.denoising:
            Fs = [self.F_] * time_steps if self.F_.ndim == 2 else self.F_
            Qs = [self.Q_] * time_steps if self.Q_.ndim == 2 else self.Q_
            return filterpy_kf.rts_smoother(
                Xs=x_posteriori, Ps=p_posteriori, Fs=Fs, Qs=Qs
            )[0]
        return x_posteriori

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
        """Extract matrices and relevant data to be used at time step 't' of the Kalman filter iterations.

        Parameters
        ----------
        X : 2D np.ndarray
            Data (measurements) to be transformed. Missing values must be represented as np.NaN or np.nan.
        G : 2D np.ndarray or list of 2d np.ndarray
            control_transition matrix
        u : 1D or 2D np.ndarray - if control variable not given in _transform default value is [0]
            control variable
        t : int
            time step

        Returns
        -------
        Seven matrices to be used at time step t
        """
        zt = None if any(np.isnan(X[t])) else X[t]
        Ft = self._get_t_matrix(t, self.F_, 3)
        Gt = self._get_t_matrix(t, G, 3)
        Qt = self._get_t_matrix(t, self.Q_, 3)
        Rt = self._get_t_matrix(t, self.R_, 3)
        Ht = self._get_t_matrix(t, self.H_, 3)
        ut = self._get_t_matrix(t, u, 2)

        return zt, Ft, Gt, Qt, Rt, Ht, ut
