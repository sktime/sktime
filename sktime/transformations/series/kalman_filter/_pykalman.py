"""KalmanFilterTransformerPK - Kalman Filter transformer using PyKalman"""

__author__ = ["NoaWegerhoff", "lielleravid"]

import numpy as np
from ._base import (
    _BaseKalmanFilter,
    _validate_param_shape,
    _init_matrix,
    _validate_estimate_matrices,
)
from sktime.transformations.base import BaseTransformer


class KalmanFilterTransformerPK(_BaseKalmanFilter, BaseTransformer):
    """Kalman Filter, from pykalman (sktime native maintenance fork).

    The Kalman Filter is an unsupervised algorithm, consisting of
    several mathematical equations which are used to create
    an estimate of the state of a process.

    The Kalman Filter is typically used for denoising data,
    or inferring the hidden state of data.

    This class is the adapter for the ``pykalman`` package into ``sktime``.
    ``KalmanFilterTransformerPK`` implements hidden inferred states and
    denoising, depending on the boolean input parameter ``denoising``.
    In addition, ``KalmanFilterTransformerPK`` provides parameter
    optimization via Expectation-Maximization (EM) algorithm [2]_,
    implemented by ``pykalman``.

    As the ``pykalman`` package is no longer maintained, ``sktime`` now contains
    an up-to-date maintenance fork of the ``pykalman`` package.

    The maintenance fork can also be directly accessed in
    ``sktime.libs.pykalman``.

    Parameters
    ----------
    state_dim : int
        System state feature dimension.
    state_transition : np.ndarray, optional (default=None)
        of shape (state_dim, state_dim) or (time_steps, state_dim, state_dim).
        State transition matrix, also referred to as ``F``, is a matrix
        which describes the way the underlying series moves
        through successive time periods.
    process_noise : np.ndarray, optional (default=None)
        of shape (state_dim, state_dim) or
        (time_steps, state_dim, state_dim).
        Process noise matrix, also referred to as ``Q``,
        the uncertainty of the dynamic model.
    measurement_noise : np.ndarray, optional (default=None)
        of shape (measurement_dim, measurement_dim) or
        (time_steps, measurement_dim, measurement_dim).
        Measurement noise matrix, also referred to as ``R``,
        represents the uncertainty of the measurements.
    measurement_function : np.ndarray, optional (default=None)
        of shape (measurement_dim, state_dim) or
        (time_steps, measurement_dim, state_dim).
        Measurement equation matrix, also referred to as ``H``, adjusts
        dimensions of measurements to match dimensions of state.
    initial_state : np.ndarray, optional (default=None)
        of shape (state_dim,).
        Initial estimated system state, also referred to as ``X0``.
    initial_state_covariance : np.ndarray, optional (default=None)
        of shape (state_dim, state_dim).
        Initial estimated system state covariance, also referred to as ``P0``.
    transition_offsets : np.ndarray, optional (default=None)
        of shape (state_dim,) or (time_steps, state_dim).
        State offsets, also referred to as ``b``, as described in ``pykalman``.
    measurement_offsets : np.ndarray, optional (default=None)
        of shape (measurement_dim,) or (time_steps, measurement_dim).
        Observation (measurement) offsets, also referred to as ``d``,
        as described in ``pykalman``.
    denoising : bool, optional (default=False).
        This parameter affects ``transform``. If False, then ``transform`` will be
        inferring
        hidden state. If True, uses ``pykalman`` ``smooth`` for denoising.
    estimate_matrices : str or list of str, optional (default=None).
        Subset of [``state_transition``, ``measurement_function``,
        ``process_noise``, ``measurement_noise``, ``initial_state``,
        ``initial_state_covariance``, ``transition_offsets``, ``measurement_offsets``]
        or - ``all``. If ``estimate_matrices`` is an iterable of strings,
        only matrices in ``estimate_matrices`` will be estimated using EM algorithm,
        like described in ``pykalman``. If ``estimate_matrices`` is ``all``,
        then all matrices will be estimated using EM algorithm.

        Note - parameters estimated by EM algorithm assumed to be constant.

    See Also
    --------
    KalmanFilterTransformerFP :
        Kalman Filter transformer, adapter for the ``FilterPy`` package into ``sktime``.
    KalmanFilterTransformerSIMD :
        Kalman Filter transformer, adapter for the ``simdkalman`` package into ``sktime``.

    Notes
    -----
    ``pykalman`` KalmanFilter documentation :
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
        Basic example:

    >>> import numpy as np  # doctest: +SKIP
    >>> import sktime.transformations.series.kalman_filter as kf
    >>> time_steps, state_dim, measurement_dim = 10, 2, 3
    >>>
    >>> X = np.random.rand(time_steps, measurement_dim) * 10
    >>> transformer = kf.KalmanFilterTransformerPK(state_dim=state_dim) # doctest: +SKIP
    >>> X_transformed = transformer.fit_transform(X=X)  # doctest: +SKIP

        Example of - denoising, matrix estimation and missing values:

    >>> import numpy as np  # doctest: +SKIP
    >>> import sktime.transformations.series.kalman_filter as kf
    >>> time_steps, state_dim, measurement_dim = 10, 2, 2
    >>>
    >>> X = np.random.rand(time_steps, measurement_dim)
    >>> # missing value
    >>> X[0][0] = np.nan
    >>>
    >>> # If matrices estimation is required, elements of ``estimate_matrices``
    >>> # are assumed to be constants.
    >>> transformer = kf.KalmanFilterTransformerPK(  # doctest: +SKIP
    ...     state_dim=state_dim,
    ...     measurement_noise=np.eye(measurement_dim),
    ...     denoising=True,
    ...     estimate_matrices=['measurement_noise']
    ...     )
    >>>
    >>> X_transformed = transformer.fit_transform(X=X)  # doctest: +SKIP

        Example of - dynamic inputs (matrix per time-step) and missing values:

    >>> import numpy as np  # doctest: +SKIP
    >>> import sktime.transformations.series.kalman_filter as kf
    >>> time_steps, state_dim, measurement_dim = 10, 4, 4
    >>>
    >>> X = np.random.rand(time_steps, measurement_dim)
    >>> # missing values
    >>> X[0] = [np.NaN for i in range(measurement_dim)]
    >>>
    >>> # Dynamic input -
    >>> # ``state_transition`` provide different matrix for each time step.
    >>> transformer = kf.KalmanFilterTransformerPK(  # doctest: +SKIP
    ...     state_dim=state_dim,
    ...     state_transition=np.random.rand(time_steps, state_dim, state_dim),
    ...     estimate_matrices=['initial_state', 'initial_state_covariance']
    ...     )
    >>>
    >>> X_transformed = transformer.fit_transform(X=X)  # doctest: +SKIP
    """

    _tags = {
        # packaging info
        # --------------
        "authors": [
            "duckworthd",
            "NoaWegerhoff",
            "lielleravid",
            "mbalatsko",
            "gliptak",
        ],
        # duckworthd for the original pykalman package (abandoned later)
        # mbalatsko, gliptak for fixes and updates
        # NoaWegerhoff, lielleravid for the sktime adapter
        "maintainers": ["NoaWegerhoff"],
        # estimator type
        # --------------
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
        super().__init__(
            state_dim=state_dim,
            state_transition=state_transition,
            process_noise=process_noise,
            measurement_noise=measurement_noise,
            measurement_function=measurement_function,
            initial_state=initial_state,
            initial_state_covariance=initial_state_covariance,
        )
        # b
        self.transition_offsets = transition_offsets
        # d
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

        self._set_attribute(
            param_name="state_transition",
            attr_name="F_",
            value=F_,
            matrix_shape=shapes["F"],
            time_steps=time_steps,
        )
        self._set_attribute(
            param_name="process_noise",
            attr_name="Q_",
            value=Q_,
            matrix_shape=shapes["Q"],
            time_steps=time_steps,
        )
        self._set_attribute(
            param_name="measurement_noise",
            attr_name="R_",
            value=R_,
            matrix_shape=shapes["R"],
            time_steps=time_steps,
        )
        self._set_attribute(
            param_name="measurement_function",
            attr_name="H_",
            value=H_,
            matrix_shape=shapes["H"],
            time_steps=time_steps,
        )
        self._set_attribute(
            param_name="initial_state",
            attr_name="X0_",
            value=X0_,
            matrix_shape=shapes["X0"],
        )
        self._set_attribute(
            param_name="initial_state_covariance",
            attr_name="P0_",
            value=P0_,
            matrix_shape=shapes["P0"],
        )

        _validate_param_shape(
            param_name="transition_offsets",
            matrix_shape=shapes["b"],
            actual_shape=transition_offsets_.shape,
            time_steps=time_steps,
        )
        self.transition_offsets_ = np.copy(transition_offsets_)

        _validate_param_shape(
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
        according to the constructor input parameter ``denoising``.
        If ``denoising`` is True - then denoise data using
        ``pykalman``'s ``smooth`` function.
        Else, infer hidden state using ``pykalman``'s ``filter`` function.

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
        from sktime.libs.pykalman import KalmanFilter

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
            special parameters are defined for a value, will return ``"default"`` set.
            There are currently no reserved values for transformers.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        params1 = {"state_dim": 2}
        params2 = {
            "state_dim": 2,
            "initial_state": np.array([0, 0]),
            "initial_state_covariance": np.array([[0.1, 0], [0.1, 0]]),
            "state_transition": np.array([[1, 0.1], [0, 1]]),
            "process_noise": np.array(
                [
                    [1 / 4 * (0.1**4), 1 / 2 * (0.1**3)],
                    [1 / 2 * (0.1**3), 0.1**2],
                ]
            )
            * 0.1,
            "denoising": True,
            "estimate_matrices": ["measurement_noise"],
        }
        return [params1, params2]

    def _em(self, X, measurement_dim, state_dim):
        """Estimate matrices algorithm if requested by user.

        If input matrices are specified in ``estimate_matrices``,
        this method will use the ``pykalman`` EM algorithm function
        to estimate said matrices needed to calculate the Kalman Filter.
        Algorithm explained in References[2].
        If ``estimate_matrices`` is None no matrices will be estimated.

        Parameters
        ----------
        X : np.ndarray
            of shape (time_steps, measurement_dim).
            Data (measurements). Missing values must be represented as np.NaN or np.nan.
        measurement_dim : int
            Measurement feature dimensions.
        state_dim : int
            ``state`` feature dimensions.

        Returns
        -------
            Eight matrix parameters -
            F, H, Q, R, transition_offsets, measurement_offsets,
            X0, P0 as np.ndarray.
        """
        from sktime.libs.pykalman import KalmanFilter

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
        """Map parameter names to ``pykalman`` names for use of ``em``.

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
        valid_ems = _validate_estimate_matrices(
            input_ems=self.estimate_matrices, all_ems=list(params_mapping.keys())
        )

        em_vars = [params_mapping[em_var] for em_var in valid_ems]
        return em_vars

    def _get_shapes(self, state_dim, measurement_dim):
        """Return a dictionary with default shape of each matrix parameter.

        Parameters
        ----------
        state_dim : int
            ``state`` feature dimensions.
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
