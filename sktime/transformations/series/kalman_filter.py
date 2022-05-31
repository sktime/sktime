# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Kalman Filter Transformers.

Series based transformers, based on Kalman Filter algorithm. Contains Base class
and two transformers which are each Adapters for external packages
pykalman and FilterPy.
"""

__author__ = ["NoaBenAmi", "lielleravid"]
__all__ = [
    "BaseKalmanFilter",
    "KalmanFilterTransformerPK",
    "KalmanFilterTransformerFP",
]

import numpy as np

from sktime.transformations.base import BaseTransformer
from sktime.utils.validation._dependencies import _check_soft_dependencies

_check_soft_dependencies("pykalman", severity="warning")
_check_soft_dependencies("filterpy", severity="warning")


def _get_t_matrix(time_t, matrices, shape, time_steps):
    """Extract matrix to be used at iteration 'time_t' of the Kalman filter iterations.

    Parameters
    ----------
    time_t : int
        The required time step.
    matrices : np.ndarray
    shape : tuple
        The shape of a single matrix.
    time_steps : int
        Number of iterations.

    Returns
    -------
        matrix : np.ndarray
            matrix to be used at iteration `time_t`
    """
    matrices = np.asarray(matrices)
    if matrices.shape == shape:
        return matrices
    if matrices.shape == (time_steps, *shape):
        return matrices[time_t]
    raise ValueError(
        f"Shape of `matrices` {matrices.shape}, does not match single matrix"
        f"`shape` {shape}, nor shape of list of matrices {(time_steps, *shape)}."
    )


def _check_param_dims(param_name, matrix_shape, actual_shape, time_steps=None):
    if time_steps is None:
        if actual_shape != matrix_shape:
            raise ValueError(
                f"Shape of parameter `{param_name}` is: {actual_shape}, "
                f"but should be: {matrix_shape}."
            )
    else:
        matrices_shape = (time_steps, *matrix_shape)
        if not (actual_shape == matrix_shape or actual_shape == matrices_shape):
            raise ValueError(
                f"Shape of parameter `{param_name}` is: {actual_shape}, but should be: "
                f"{matrix_shape} or {matrices_shape}."
            )


def _init_matrix(matrices, transform_func, default_val):
    """Initialize default value if matrix is None, or transform input matrix to np.ndarray.

    Parameters
    ----------
    matrices : np.ndarray
    transform_func : transformation function from array-like to ndarray
    default_val : np.ndarray

    Returns
    -------
        transformed_matrices : np.ndarray
            matrices as np.ndarray
    """
    if matrices is None:
        return default_val
    return transform_func(matrices)


class BaseKalmanFilter:
    """Kalman Filter is used for denoising data, or inferring the hidden state of data.

    Note - this class is a base class and should not be used directly.

    The Kalman filter is an unsupervised algorithm consisting of
    several mathematical equations which are used to create
    an estimate of the state of a process.
    The algorithm does this efficiently and recursively in a
    way where the mean squared error is minimal.
    The Kalman Filter has the ability to support estimations
    of past, present and future states.
    The strength of the Kalman Filter is in its ability to
    infer the state of a system even when the exact nature of the
    system is not known.
    When given time series data, the Kalman filter creates a denoising effect
    by removing noise from the data, and recovering the true
    state of the underlying object we are tracking within the
    data.
    The Kalman Filter computations are based on five equations.

    Two prediction equations:

    -   State Extrapolation Equation - prediction or estimation of the future state,
        based on the known present estimation.
    -	Covariance Extrapolation Equation - the measure of uncertainty in our prediction.

    Two update equations:

    -   State Update Equation - estimation of the current state,
        based on the known past estimation and present measurement.
    -	Covariance Update Equation - the measure of uncertainty in our estimation.

    Kalman Gain Equation – this is a required argument for the update equations.
    It acts as a weighting parameter for the past estimations and the given measurement.
    It defines the weight of the past estimation and
    the weight of the measurement in estimating the current state.

    Parameters
    ----------
    state_dim : int
        System state feature dimension
    state_transition : np.ndarray, optional (default=None)
        of shape (state_dim, state_dim) or (time_steps, state_dim, state_dim).
        State transition matrix, also referred to as F, is a matrix
        which describes the way the underlying series moves
        through successive time periods.
    process_noise : np.ndarray, optional (default=None)
        of shape (state_dim, state_dim) or (time_steps, state_dim, state_dim).
        Process noise matrix, also referred to as Q,
        the uncertainty of the dynamic model.
    measurement_noise : np.ndarray, optional (default=None)
        of shape (measurement_dim, measurement_dim) or
        (time_steps, measurement_dim, measurement_dim).
        Measurement noise matrix, also referred to as R,
        represents the uncertainty of the measurements.
    measurement_function : np.ndarray, optional (default=None)
        of shape (measurement_dim, state_dim) or
        (time_steps, measurement_dim, state_dim).
        Measurement equation matrix, also referred to as H, adjusts
        dimensions of measurements to match dimensions of state.
    initial_state : np.ndarray, optional (default=None)
        of shape (state_dim,).
        Initial estimated system state, also referred to as x0.
    initial_state_covariance : np.ndarray, optional (default=None)
        of shape (state_dim, state_dim).
        Initial estimated system state covariance, also referred to as P0.

    References
    ----------
    .. [1] Greg Welch and Gary Bishop, "An Introduction to the Kalman Filter", 2006
           https://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf
    .. [2] R.H.Shumway and D.S.Stoffer "An Approach to time
           Series Smoothing and Forecasting Using the EM Algorithm", 1982
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

        super(BaseKalmanFilter, self).__init__()

    def _get_shapes(self, state_dim, measurement_dim):
        """Return dictionary with default shape of each matrix parameter.

        Parameters
        ----------
        state_dim : int
            Dimension of `state`.
        measurement_dim : int
            Dimension of measurements.

        Returns
        -------
            shapes : dict
                Dictionary with default shape of each matrix parameter
        """
        shapes = {
            "F": (state_dim, state_dim),
            "Q": (state_dim, state_dim),
            "R": (measurement_dim, measurement_dim),
            "H": (measurement_dim, state_dim),
            "X0": (state_dim,),
            "P0": (state_dim, state_dim),
        }
        return shapes

    def _get_init_values(self, measurement_dim, state_dim):
        """Initialize parameter matrices to default values and returns them.

        Parameters
        ----------
        measurement_dim : int
        state_dim : int

        Returns
        -------
            Six parameter matrices F,Q,R,H,X0,P0 as np.ndarray
        """
        shapes = self._get_shapes(state_dim=state_dim, measurement_dim=measurement_dim)
        F = _init_matrix(
            matrices=self.state_transition,
            transform_func=np.atleast_2d,
            default_val=np.eye(*shapes["F"]),
        )
        Q = _init_matrix(
            matrices=self.process_noise,
            transform_func=np.atleast_2d,
            default_val=np.eye(*shapes["Q"]),
        )
        R = _init_matrix(
            matrices=self.measurement_noise,
            transform_func=np.atleast_2d,
            default_val=np.eye(*shapes["R"]),
        )
        H = _init_matrix(
            matrices=self.measurement_function,
            transform_func=np.atleast_2d,
            default_val=np.eye(*shapes["H"]),
        )
        X0 = _init_matrix(
            matrices=self.initial_state,
            transform_func=np.atleast_1d,
            default_val=np.zeros(*shapes["X0"]),
        )
        P0 = _init_matrix(
            matrices=self.initial_state_covariance,
            transform_func=np.atleast_2d,
            default_val=np.eye(*shapes["P0"]),
        )

        return F, Q, R, H, X0, P0

    def _set_param(self, param_name, inner_name, value, matrix_shape, time_steps=None):
        _check_param_dims(
            param_name=param_name,
            matrix_shape=matrix_shape,
            actual_shape=value.shape,
            time_steps=time_steps,
        )
        setattr(self, inner_name, value)


class KalmanFilterTransformerPK(BaseKalmanFilter, BaseTransformer):
    """Kalman Filter is used for denoising data, or inferring the hidden state of data.

    The Kalman Filter is an unsupervised algorithm, consisting of
    several mathematical equations which are used to create
    an estimate of the state of a process.


    This class is the adapter for the `pykalman` package into `sktime`.
    KalmanFilterTransformerPK implements hidden inferred states and
    denoising, depending on the boolean input parameter 'denoising'.
    In addition, KalmanFilterTransformerPK provides parameter
    optimization via Expectation-Maximization (EM) algorithm [2]_,
    implemented by `pykalman`.

    Parameters
    ----------
    state_dim : int
        System state feature dimension.
    state_transition : np.ndarray, optional (default=None)
        of shape (state_dim, state_dim) or (time_steps, state_dim, state_dim).
        State transition matrix, also referred to as F, is a matrix
        which describes the way the underlying series moves
        through successive time periods.
    process_noise : np.ndarray, optional (default=None)
        of shape (state_dim, state_dim) or
        (time_steps, state_dim, state_dim).
        Process noise matrix, also referred to as Q,
        the uncertainty of the dynamic model.
    measurement_noise : np.ndarray, optional (default=None)
        of shape (measurement_dim, measurement_dim) or
        (time_steps, measurement_dim, measurement_dim).
        Measurement noise matrix, also referred to as R,
        represents the uncertainty of the measurements.
    measurement_function : np.ndarray, optional (default=None)
        of shape (measurement_dim, state_dim) or
        (time_steps, measurement_dim, state_dim).
        Measurement equation matrix, also referred to as H, adjusts
        dimensions of measurements to match dimensions of state.
    initial_state : np.ndarray, optional (default=None)
        of shape (state_dim,).
        Initial estimated system state, also referred to as x0.
    initial_state_covariance : np.ndarray, optional (default=None)
        of shape (state_dim, state_dim).
        Initial estimated system state covariance, also referred to as P0.
    transition_offsets : np.ndarray, optional (default=None)
        of shape (state_dim,) or (time_steps, state_dim).
        State offsets, also referred to as b, as described in `pykalman`.
    measurement_offsets : np.ndarray, optional (default=None)
        of shape (measurement_dim,) or (time_steps, measurement_dim).
        Observation (measurement) offsets, also referred to as d,
        as described in `pykalman`.
    denoising : bool, optional (default=False).
        This parameter affects `transform`. If False, then `transform` will be inferring
        hidden state. If True, uses `pykalman` `smooth` for denoising.
    estimate_matrices : str or list of str, optional (default=None).
        Subset of [‘state_transition’, ‘measurement_function’,
        ‘process_noise’, ‘measurement_noise’, ‘initial_state’,
        ‘initial_state_covariance’, ‘transition_offsets’, ‘measurement_offsets’]
        or - ‘all’. If estimate_matrices is an iterable of strings,
        only matrices in estimate_matrices will be estimated using EM algorithm,
        like described in `pykalman`. If estimate_matrices is ‘all’,
        then all matrices will be estimated using EM algorithm.
        Note - parameters estimated by EM algorithm assumed to be constant.

    See Also
    --------
    KalmanFilterTransformerFP :
        Kalman Filter transformer, adapter for the `FilterPy` package into `sktime`.

    Notes
    -----
    `pykalman` KalmanFilter documentation :
        https://pykalman.github.io/#kalmanfilter

    References
    ----------
    .. [1] Greg Welch and Gary Bishop, "An Introduction to the Kalman Filter", 2006
           https://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf
    .. [2] R.H.Shumway and D.S.Stoffer "An Approach to time
           Series Smoothing and Forecasting Using the EM Algorithm", 1982
           https://www.stat.pitt.edu/stoffer/dss_files/em.pdf

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.transformations.series.kalman_filter import (
    >>>     KalmanFilterTransformerPK
    >>> )
    >>> time_steps = 10
    >>> state_dim = 2
    >>> measurement_dim = 2

        Basic example:
    >>> X = np.random.rand(time_steps, measurement_dim)
    >>> transformer = KalmanFilterTransformerPK(state_dim=state_dim)
    >>> X_transformed = transformer.fit_transform(X=X)

        Example of - denoising, estimate_matrices and missing values:
    >>> X = np.random.rand(time_steps, measurement_dim)
    >>> # missing value
    >>> X[0][0] = np.nan
    >>>
    >>> # Note - if `estimate_matrices` is not None, then assume matrix parameters
    >>> # given as elements of `estimate_matrices` are constants
    >>> # (not lists of matrices).
    >>> transformer = KalmanFilterTransformerPK(state_dim=state_dim,
    >>>                               state_transition=np.eye(state_dim),
    >>>                               measurement_offsets=np.zeros(measurement_dim),
    >>>                               denoising=True,
    >>>                               estimate_matrices=['state_transition',
    >>>                                                  'measurement_offsets',
    >>>                                                  'process_noise'])
    >>> X_transformed = transformer.fit_transform(X=X)

        Example of - matrix for each time_step, denoising and missing values:
    >>> X = np.random.rand(time_steps, measurement_dim) * 10
    >>> # missing values
    >>> X[0] = [np.NaN, np.NaN]
    >>>
    >>> # matrix for each time_step
    >>> state_transition = [np.eye(state_dim)] * time_steps
    >>> measurement_noise = np.asarray(([np.eye(measurement_dim)] * time_steps))
    >>>
    >>> # Note - if `estimate_matrices` is not None, then assume matrix parameters
    >>> # given as elements of `estimate_matrices` are constants
    >>> # (not lists of matrices).
    >>> transformer = KalmanFilterTransformerPK(state_dim=state_dim,
    >>>                                  state_transition=state_transition,
    >>>                                  measurement_noise=measurement_noise,
    >>>                                  denoising=True,
    >>>                                  estimate_matrices=['initial_state',
    >>>                                                     'initial_state_covariance'])
    >>> X_transformed = transformer.fit_transform(X=X)
    """

    _tags = {
        "X_inner_mtype": "np.ndarray",  # which mtypes do _fit/_predict support for X?
        "requires_y": False,  # does y need to be passed in fit?
        "fit_is_empty": False,  # is fit empty and can be skipped? Yes = True
        "capability:unequal_length": False,
        # can the transformer handle unequal length time series (if passed Panel)?
        "handles-missing-data": True,  # can estimator handle missing data?
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

        super(KalmanFilterTransformerPK, self).__init__(
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
        The matrix initializations or estimations
        (if requested by user) are calculated here.

        Parameters
        ----------
        X : np.ndarray
            of shape (time_steps, measurement_dim).
            Data (measurements) to be transformed.
            Missing values must be represented as np.NaN or np.nan.
        y : ignored argument for interface compatibility

        Returns
        -------
            self: reference to self
        """
        measurement_dim = X.shape[1]
        time_steps = X.shape[0]
        shapes = self._get_shapes(
            state_dim=self.state_dim, measurement_dim=measurement_dim
        )

        if self.estimate_matrices is None:
            (F_, Q_, R_, H_, X0_, P0_) = self._get_init_values(
                measurement_dim, self.state_dim
            )

            transition_offsets_ = _init_matrix(
                matrices=self.transition_offsets,
                transform_func=np.atleast_1d,
                default_val=np.zeros(*shapes["b"]),
            )

            measurement_offsets_ = _init_matrix(
                matrices=self.measurement_offsets,
                transform_func=np.atleast_1d,
                default_val=np.zeros(*shapes["d"]),
            )

        else:
            (
                F_,
                H_,
                Q_,
                R_,
                transition_offsets_,
                measurement_offsets_,
                X0_,
                P0_,
            ) = self._em(X=X, measurement_dim=measurement_dim, state_dim=self.state_dim)

        self._set_param(
            param_name="state_transition",
            inner_name="F_",
            value=F_,
            matrix_shape=shapes["F"],
            time_steps=time_steps,
        )
        self._set_param(
            param_name="process_noise",
            inner_name="Q_",
            value=Q_,
            matrix_shape=shapes["Q"],
            time_steps=time_steps,
        )
        self._set_param(
            param_name="measurement_noise",
            inner_name="R_",
            value=R_,
            matrix_shape=shapes["R"],
            time_steps=time_steps,
        )
        self._set_param(
            param_name="measurement_function",
            inner_name="H_",
            value=H_,
            matrix_shape=shapes["H"],
            time_steps=time_steps,
        )
        self._set_param(
            param_name="initial_state",
            inner_name="X0_",
            value=X0_,
            matrix_shape=shapes["X0"],
        )
        self._set_param(
            param_name="initial_state_covariance",
            inner_name="P0_",
            value=P0_,
            matrix_shape=shapes["P0"],
        )

        _check_param_dims(
            param_name="transition_offsets",
            matrix_shape=shapes["b"],
            actual_shape=transition_offsets_.shape,
            time_steps=time_steps,
        )
        self.transition_offsets_ = np.copy(transition_offsets_)

        _check_param_dims(
            param_name="measurement_offsets",
            matrix_shape=shapes["d"],
            actual_shape=measurement_offsets_.shape,
            time_steps=time_steps,
        )
        self.measurement_offsets_ = np.copy(measurement_offsets_)

        return self

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        This method performs the transformation of the input data
        according to the constructor input parameter `denoising`.
        If `denoising` is True - then denoise data using `pykalman`'s `smooth` function.
        Else, infer hidden state using `pykalman`'s `filter` function.

        Parameters
        ----------
        X : np.ndarray
            of shape (time_steps, measurement_dim).
            Data (measurements) to be transformed.
            Missing values must be represented as np.NaN or np.nan.
        y : ignored argument for interface compatibility

        Returns
        -------
            X_transformed : np.ndarray
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
        params = {"state_dim": 2}
        return params

    def _em(self, X, measurement_dim, state_dim):
        """Estimate matrices algorithm if requested by user.

        If input matrices are specified in 'estimate_matrices',
        this method will use the `pykalman` EM algorithm function
        to estimate said matrices needed to calculate the Kalman Filter.
        Algorithm explained in References[2].
        If 'estimate_matrices' is None no matrices will be estimated.

        Parameters
        ----------
        X : np.ndarray
            of shape (time_steps, measurement_dim).
            Data (measurements). Missing values must be represented as np.NaN or np.nan.
        measurement_dim : int
            Measurement feature dimensions.
        state_dim : int
            `state` feature dimensions.

        Returns
        -------
            Eight parameter matrices -
            F, H, Q, R, transition_offsets, measurement_offsets,
            X0, P0 as np.ndarray.
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

    def _get_estimate_matrices(self):
        """Map parameters names to `pykalman` parameters names for use of `pykalman` `em`.

        Returns
        -------
            em_vars : list
                mapped parameters names
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

    def _get_shapes(self, state_dim, measurement_dim):
        """Return a dictionary with default shape of each matrix parameter.

        Parameters
        ----------
        state_dim : int
            `state` feature dimensions.
        measurement_dim : int
            Measurement (data) feature dimensions.

        Returns
        -------
            Dictionary with default shape of each matrix parameter
        """
        shapes = super()._get_shapes(state_dim, measurement_dim)
        shapes["b"] = (state_dim,)
        shapes["d"] = (measurement_dim,)
        return shapes


class KalmanFilterTransformerFP(BaseKalmanFilter, BaseTransformer):
    """Kalman Filter is used for denoising data or inferring the hidden state of data given.

    The Kalman Filter is an unsupervised algorithm, consisting of
    several mathematical equations which are used to create
    an estimate of the state of a process.

    This class is the adapter for the `FilterPy` package into `sktime`.
    KalmanFilterTransformerFP implements hidden inferred states and
    denoising, depending on the boolean input parameter `denoising`.
    In addition, KalmanFilterTransformerFP provides parameter
    optimization via Expectation-Maximization (EM) algorithm.

    Parameters
    ----------
    state_dim : int
        System state feature dimension.
    state_transition : np.ndarray, optional (default=None)
        of shape (state_dim, state_dim) or (time_steps, state_dim, state_dim).
        State transition matrix, also referred to as F, is a matrix
        which describes the way the underlying series moves
        through successive time periods.
    process_noise : np.ndarray, optional (default=None)
        of shape (state_dim, state_dim) or (time_steps, state_dim, state_dim).
        Process noise matrix, also referred to as Q,
        the uncertainty of the dynamic model.
    measurement_noise : np.ndarray, optional (default=None)
        of shape (measurement_dim, measurement_dim) or
        (time_steps, measurement_dim, measurement_dim).
        Measurement noise matrix, also referred to as R,
        represents the uncertainty of the measurements.
    measurement_function : np.ndarray, optional (default=None)
        of shape (measurement_dim, state_dim) or
        (time_steps, measurement_dim, state_dim).
        Measurement equation matrix, also referred to as H, adjusts
        dimensions of measurements to match dimensions of state.
    initial_state : np.ndarray, optional (default=None)
        of shape (state_dim,).
        Initial estimated system state, also referred to as x0.
    initial_state_covariance : np.ndarray, optional (default=None)
        of shape (state_dim, state_dim).
        Initial estimated system state covariance, also referred to as P0.
    control_transition : np.ndarray, optional (default=None)
        of shape (state_dim, control_variable_dim) or
        (time_steps, state_dim, control_variable_dim).
        Control transition matrix, also referred to as G.
        'control_variable_dim' is the dimension of 'control variable',
        also referred to as `u`.
        `control variable` is an optional parameter for `fit` and `transform` functions.
    denoising : bool, optional (default=False).
        This parameter affects `transform`. If False, then `transform` will be inferring
        hidden state. If True, uses `FilterPy` `rts_smoother` for denoising.
    estimate_matrices : str or list of str, optional (default=None).
        Subset of [‘state_transition’, ‘measurement_function’,
        ‘process_noise’, ‘measurement_noise’, ‘initial_state’,
        ‘initial_state_covariance’]
        or - ‘all’. If `estimate_matrices` is an iterable of strings,
        only matrices in `estimate_matrices` will be estimated using EM algorithm.
        If `estimate_matrices` is ‘all’,
        then all matrices will be estimated using EM algorithm.
        Note -
            - parameters estimated by EM algorithm assumed to be constant.
            - 'control_transition' matrix cannot be estimated.

    See Also
    --------
    KalmanFilterTransformerPK :
        Kalman Filter transformer, adapter for the `pykalman` package into `sktime`.

    Notes
    -----
    `FilterPy` KalmanFilter documentation :
        https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html

    References
    ----------
    .. [1] Greg Welch and Gary Bishop, "An Introduction to the Kalman Filter", 2006
           https://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf
    .. [2] R.H.Shumway and D.S.Stoffer "An Approach to time
           Series Smoothing and Forecasting Using the EM Algorithm", 1982
           https://www.stat.pitt.edu/stoffer/dss_files/em.pdf

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.transformations.series.kalman_filter import (
    >>>     KalmanFilterTransformerFP
    >>> )
    >>> time_steps = 10
    >>> state_dim = 2
    >>> measurement_dim = 2
    >>> control_variable_dim = 1

        Basic example:
    >>> X = np.random.rand(time_steps, measurement_dim)
    >>> transformer = KalmanFilterTransformerFP(state_dim=state_dim)
    >>> X_transformed = transformer.fit_transform(X=X)

        Example of - denoising, estimate_matrices, missing values and transform with y:
    >>> X = np.random.rand(time_steps, measurement_dim)
    >>> # missing value
    >>> X[0][0] = np.nan
    >>>
    >>> # y
    >>> control_variable = np.random.rand(time_steps, control_variable_dim)
    >>>
    >>> # Note - if `estimate_matrices` is not None, then assume matrix parameters
    >>> # given as elements of `estimate_matrices` are constants
    >>> # (not lists of matrices).
    >>> transformer = KalmanFilterTransformerFP(
    >>>                    state_dim=state_dim,
    >>>                    state_transition=np.eye(state_dim),
    >>>                    control_transition=np.eye(state_dim, control_variable_dim),
    >>>                    denoising=True,
    >>>                    estimate_matrices='all')
    >>> X_transformed = transformer.fit_transform(X=X, y=control_variable)

        Example of - matrix for each time_step, denoising, missing values and
                     transform with y.
    >>> X = np.random.rand(time_steps, measurement_dim) * 10
    >>> # missing values
    >>> X[0] = [np.NaN, np.NaN]
    >>>
    >>> # y
    >>> control_variable = np.random.rand(control_variable_dim)
    >>>
    >>> # matrix for each time_step
    >>> state_transition = [np.eye(state_dim)] * time_steps
    >>> measurement_noise = np.asarray(([np.eye(measurement_dim)] * time_steps))
    >>>
    >>> # Note - if `estimate_matrices` is not None, then assume matrix parameters
    >>> # given as elements of `estimate_matrices` are constants
    >>> # (not lists of matrices).
    >>> transformer = KalmanFilterTransformerFP(state_dim=state_dim,
    >>>                                  state_transition=state_transition,
    >>>                                  measurement_noise=measurement_noise,
    >>>                                  denoising=True,
    >>>                                  estimate_matrices=['initial_state',
    >>>                                                     'initial_state_covariance'])
    >>> X_transformed = transformer.fit_transform(X=X, y=control_variable)
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

        super(KalmanFilterTransformerFP, self).__init__(
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
        The matrix initializations or estimations
        (if requested by user) are calculated here.

        Parameters
        ----------
        X : np.ndarray
            of shape (time_steps, measurement_dim).
            Data (measurements) to be transformed.
            Missing values must be represented as np.NaN or np.nan.
        y : ignored argument for interface compatibility

        Returns
        -------
            self: reference to self
        """
        # The below call to `_check_conditioned_dependency` checks the installation
        # of `pykalman` package, if needed. `pykalman` is used when the user requires
        # matrices estimation (`estimate_matrices` is not None).
        # This conditioned dependency check can be performed in
        # `__init__` for early user feedback.
        self._check_conditioned_dependency(
            condition=(self.estimate_matrices is not None)
        )

        measurement_dim = X.shape[1]
        time_steps = X.shape[0]
        shapes = self._get_shapes(
            state_dim=self.state_dim, measurement_dim=measurement_dim
        )

        if self.estimate_matrices is None:
            (F_, Q_, R_, H_, X0_, P0_) = self._get_init_values(
                measurement_dim, self.state_dim
            )

        else:
            if (
                isinstance(self.estimate_matrices, str)
                and self.estimate_matrices == "all"
            ):
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

            transformer_ = KalmanFilterTransformerPK(
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

            F_ = transformer_.F_
            H_ = transformer_.H_
            Q_ = transformer_.Q_
            R_ = transformer_.R_
            X0_ = transformer_.X0_
            P0_ = transformer_.P0_

        self._set_param(
            param_name="state_transition",
            inner_name="F_",
            value=F_,
            matrix_shape=shapes["F"],
            time_steps=time_steps,
        )
        self._set_param(
            param_name="process_noise",
            inner_name="Q_",
            value=Q_,
            matrix_shape=shapes["Q"],
            time_steps=time_steps,
        )
        self._set_param(
            param_name="measurement_noise",
            inner_name="R_",
            value=R_,
            matrix_shape=shapes["R"],
            time_steps=time_steps,
        )
        self._set_param(
            param_name="measurement_function",
            inner_name="H_",
            value=H_,
            matrix_shape=shapes["H"],
            time_steps=time_steps,
        )
        self._set_param(
            param_name="initial_state",
            inner_name="X0_",
            value=X0_,
            matrix_shape=shapes["X0"],
        )
        self._set_param(
            param_name="initial_state_covariance",
            inner_name="P0_",
            value=P0_,
            matrix_shape=shapes["P0"],
        )

        return self

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        This method performs the transformation of the input data
        according to the constructor input parameter `denoising`.
        If `denoising` is True - then denoise data using
        `FilterPy`'s `rts_smoother` function.
        Else, infer hidden state using `FilterPy`'s `predict` and `update` functions.

        Parameters
        ----------
        X : np.ndarray
            of shape (time_steps, measurement_dim).
            Data (measurements) to be transformed.
            Missing values must be represented as np.NaN or np.nan.
        y : np.ndarray, optional (default=None).
            of shape (control_variable_dim,) or (time_steps, control_variable_dim).
            `control variable`, also referred to as `u`.
            if 2D, must be same length as X.

        Returns
        -------
            X_transformed : 2D np.ndarray
                transformed version of X
        """
        import filterpy.kalman.kalman_filter as filterpy_kf

        time_steps = X.shape[0]
        measurement_dim = X.shape[1]

        if y is None:
            y = np.zeros(1)
        u_dim = y.shape[1] if y.ndim == 2 else y.shape[0]

        shapes = self._get_shapes(
            state_dim=self.state_dim, measurement_dim=measurement_dim, u_dim=u_dim
        )
        G = _init_matrix(
            matrices=self.control_transition,
            transform_func=np.atleast_2d,
            default_val=np.eye(*shapes["G"]),
        )
        _check_param_dims(
            param_name="control_transition",
            matrix_shape=shapes["G"],
            actual_shape=G.shape,
            time_steps=time_steps,
        )

        x_priori = np.zeros((time_steps, *shapes["X0"]))
        p_priori = np.zeros((time_steps, *shapes["P0"]))

        x_posteriori = np.zeros((time_steps, *shapes["X0"]))
        p_posteriori = np.zeros((time_steps, *shapes["P0"]))

        x = self.X0_
        p = self.P0_

        for t in range(time_steps):
            (zt, Ft, Gt, Qt, Rt, Ht, ut) = self._get_iter_t_matrices(
                X=X, G=G, u=y, t=t, time_steps=time_steps, shapes=shapes
            )

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
        params = {"state_dim": 2}
        return params

    def _get_iter_t_matrices(self, X, G, u, t, time_steps, shapes):
        """Extract data to be used at time step 't' of the Kalman filter iterations.

        Parameters
        ----------
        X : np.ndarray
            of shape (time_steps, measurement_dim).
            Data (measurements).
        G : np.ndarray
            of shape (state_dim, control_variable_dim) or
            (time_steps, state_dim, control_variable_dim).
            control_transition matrix
        u : np.ndarray
            of shape (control_variable_dim,) or (time_steps, control_variable_dim).
            u is the `control variable`. If `u` not given in _transform default
            value is [0].
        t : int
            time step

        Returns
        -------
            Seven matrices to be used at time step t
        """
        zt = None if any(np.isnan(X[t])) else X[t]
        Ft = _get_t_matrix(
            time_t=t, matrices=self.F_, shape=shapes["F"], time_steps=time_steps
        )
        Gt = _get_t_matrix(
            time_t=t, matrices=G, shape=shapes["G"], time_steps=time_steps
        )
        Qt = _get_t_matrix(
            time_t=t, matrices=self.Q_, shape=shapes["Q"], time_steps=time_steps
        )
        Rt = _get_t_matrix(
            time_t=t, matrices=self.R_, shape=shapes["R"], time_steps=time_steps
        )
        Ht = _get_t_matrix(
            time_t=t, matrices=self.H_, shape=shapes["H"], time_steps=time_steps
        )
        ut = _get_t_matrix(
            time_t=t, matrices=u, shape=shapes["u"], time_steps=time_steps
        )

        return zt, Ft, Gt, Qt, Rt, Ht, ut

    def _get_shapes(self, state_dim, measurement_dim, u_dim=1):
        """Return dictionary with default shape of each matrix parameter.

        Parameters
        ----------
        state_dim : int
            Dimension of `state`.
        measurement_dim : int
            Dimension of measurements.
        u_dim : int
            Dimension of control variable u (y).

        Returns
        -------
            Dictionary with default shape of each matrix parameter
        """
        shapes = super()._get_shapes(state_dim, measurement_dim)
        shapes["u"] = (u_dim,)
        shapes["G"] = (state_dim, u_dim)
        return shapes

    def _check_conditioned_dependency(
        self, condition, msg=None, package="pykalman", severity="error"
    ):
        """If `condition` applies, check the soft dependency `package` installation.

        Call _check_soft_dependencies.
        If `package` is not installed, raise ModuleNotFoundError with
        `msg` as the error message.

        Parameters
        ----------
        condition : bool
            The condition to perform the soft dependency check.
        msg : str
            Error message to attach to ModuleNotFoundError.
        package : str
            Package name for soft dependency check.
        severity : str
            'error' or 'warning'.

        Raises
        ------
        ModuleNotFoundError
            error with informative message, asking to install required soft dependencies
        """
        if condition:
            if msg is None:
                msg = (
                    f"{self.__class__.__name__}'s parameter matrices estimation "
                    f"is performed when `estimate_matrices` "
                    f"is {self.estimate_matrices}, "
                    f"and requires {package} installed. Please run: "
                    f"`pip install {package}` to "
                    f"install the {package} package. "
                )
            try:
                _check_soft_dependencies(package, severity=severity, object=self)
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(msg) from e
