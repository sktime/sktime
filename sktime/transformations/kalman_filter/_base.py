# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Shared code between different KalmanFilterTransformers."""

__author__ = ["NoaWegerhoff", "lielleravid", "oseiskar"]

import numpy as np


def _init_matrix(matrices, transform_func, default_val):
    """Initialize default value if matrix is None, or transform to np.ndarray.

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


def _validate_param_shape(param_name, matrix_shape, actual_shape, time_steps=None):
    """Validate shape of matrix parameter.

    Assert ``actual_shape`` equals to:
        -  'shape' of a single matrix or
        -  'shape' of time_steps matrices.
    If neither, raise an informative ``ValueError`` that includes the parameter's name.

    Parameters
    ----------
    param_name : str
        The name of the matrix-parameter.
    matrix_shape : tuple
        The supposed shape of a single matrix.
    actual_shape : tuple
        The actual shape of matrix-parameter.
    time_steps : int
        actual_shape[0] if matrix-parameter is dynamic (matrix per time-step).

    Raises
    ------
        ValueError
            error with an informative message that includes parameter's name,
            and the shape that parameter should have.
    """
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

    -  State Extrapolation Equation - prediction or estimation of the future state,
        based on the known present estimation.
    -  Covariance Extrapolation Equation - the measure of uncertainty in our prediction.

    Two update equations:

    -  State Update Equation - estimation of the current state,
        based on the known past estimation and present measurement.
    -  Covariance Update Equation - the measure of uncertainty in our estimation.

    Kalman Gain Equation - this is a required argument for the update equations.
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

    _tags = {
        "authors": ["NoaWegerhoff", "lielleravid"],
        "maintainers": ["NoaWegerhoff"],
    }

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

        super().__init__()

    def _get_shapes(self, state_dim, measurement_dim):
        """Return dictionary with default shape of each matrix parameter.

        Parameters
        ----------
        state_dim : int
            Dimension of ``state``.
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
        """Initialize matrix parameters to default values and returns them.

        Parameters
        ----------
        measurement_dim : int
        state_dim : int

        Returns
        -------
            Six matrix parameters F,Q,R,H,X0,P0 as np.ndarray
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

    def _set_attribute(
        self, param_name, attr_name, value, matrix_shape, time_steps=None
    ):
        """Validate the shape of parameter and set as attribute if no error.

        Parameters
        ----------
        param_name : str
            The name of matrix-parameter.
        attr_name : str
            The name of corresponding attribute.
        value : np.ndarray
            The value of corresponding attribute.
        matrix_shape : tuple
            The supposed shape of a single matrix.
        time_steps : int, optional (default=None)
        """
        _validate_param_shape(
            param_name=param_name,
            matrix_shape=matrix_shape,
            actual_shape=value.shape,
            time_steps=time_steps,
        )
        setattr(self, attr_name, value)
