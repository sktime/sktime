"""KalmanFilterTransformerSIMD - Kalman Filter transformer using simdkalman."""

__author__ = ["oseiskar"]

import numpy as np

from sktime.transformations.base import BaseTransformer

from ._base import BaseKalmanFilter


class KalmanFilterTransformerSIMD(BaseKalmanFilter, BaseTransformer):
    """Vectorized Kalman Filter from simdkalman.

    The Kalman Filter is an unsupervised algorithm, consisting of
    several mathematical equations which are used to create
    an estimate of the state of a process.

    The Kalman Filter is typically used for denoising data,
    or inferring the hidden state of data.

    This class is the adapter for the ``simdkalman`` package into ``sktime``.
    ``KalmanFilterTransformerSIMD`` implements hidden inferred states and
    denoising, depending on the boolean input parameter ``hidden``.
    In addition, filtering (forward pass only) and smoothing (forward and backward
    pass) options can be selected with the ``denoising`` parameter.

    The ``simdkalman`` package is ideal for Panels where similar
    Kalman Filters are applied in to multiple time series. The package
    applies multi-dimensional matrix operations, which can be an order
    of magnitude faster than the non-vectorized implementations.

    This version does not currently support the EM algorithm or dynamic inputs,
    i.e., list of matrices per time-step.

    Parameters
    ----------
    state_dim : int
        System state feature dimension.
    state_transition : np.ndarray, optional (default=None)
        of shape (state_dim, state_dim).
        State transition matrix, also referred to as ``F``, is a matrix
        which describes the way the underlying series moves
        through successive time periods. Called ``A`` in ``simdkalman``.
    process_noise : np.ndarray, optional (default=None)
        of shape (state_dim, state_dim).
        Process noise matrix, also referred to as ``Q``,
        the uncertainty of the dynamic model.
    measurement_noise : np.ndarray, optional (default=None)
        of shape (measurement_dim, measurement_dim).
        Measurement noise matrix, also referred to as ``R``,
        represents the uncertainty of the measurements.
    measurement_function : np.ndarray, optional (default=None)
        of shape (measurement_dim, state_dim).
        Measurement equation matrix, also referred to as ``H``, adjusts
        dimensions of measurements to match dimensions of state.
    initial_state : np.ndarray, optional (default=None)
        of shape (state_dim,).
        Initial estimated system state, also referred to as ``X0``.
    initial_state_covariance : np.ndarray, optional (default=None)
        of shape (state_dim, state_dim).
        Initial estimated system state covariance, also referred to as ``P0``.
    denoising : bool, optional (default=False).
        This parameter affects ``transform``. If False, then ``transform`` will
        use a Kalman filter (forward pass only). If true, uses a Kalman smoother.
    hidden : bool, optional (default=True).
        This parameter affects ``transform``. If True, then ``transform`` will be
        inferring hidden state. If False, returns smoothed/filtered observations
        (see also ``denoising``), which always has the same dimensions as the
        input data, independent of the hidden state dimension.

    See Also
    --------
    KalmanFilterTransformerPK :
        Kalman Filter transformer, adapter for the ``pykalman`` package
        into ``sktime``.
    KalmanFilterTransformerFP :
        Kalman Filter transformer, adapter for the ``filterpy`` package
        into ``sktime``.

    Notes
    -----
    ``simdkalman`` documentation :
        https://simdkalman.readthedocs.io/


    Examples
    --------
        Timing example:

    >>> import numpy as np
    >>> import time
    >>> from sktime.utils._testing.panel import make_transformer_problem
    >>> from sktime.transformations.series.kalman_filter import (
    ...     KalmanFilterTransformerPK,
    ...     KalmanFilterTransformerSIMD,
    ... )
    >>>
    >>> # Test data
    >>> X = make_transformer_problem(  # doctest: +SKIP
    ...     n_instances=200,
    ...     n_columns=2,
    ...     n_timepoints=500
    ... )
    >>>
    >>> kf_params = dict(  # doctest: +SKIP
    ...     state_dim=2,
    ...     state_transition=np.array([[1, 1], [0, 1]]),
    ...     process_noise=np.diag([1e-6, 0.01]),
    ...     measurement_function=np.array([[1, 0], [0, 1]]),
    ...     measurement_noise=np.diag([10, 10]),
    ...     initial_state=np.array([0, -1]),
    ...     initial_state_covariance=np.diag([1, 1]),
    ... )
    >>>
    >>> t0 = time.time()  # doctest: +SKIP
    >>> kf_SIMD = KalmanFilterTransformerSIMD(**kf_params)  # doctest: +SKIP
    >>> X_SIMD = kf_SIMD.fit_transform(X)  # doctest: +SKIP
    >>> T_SIMD = time.time() - t0  # doctest: +SKIP
    >>>
    >>> t0 = time.time()  # doctest: +SKIP
    >>> kf_PK = KalmanFilterTransformerPK(**kf_params)  # doctest: +SKIP
    >>> X_PK = kf_PK.fit_transform(X)  # doctest: +SKIP
    >>> T_PK = time.time() - t0  # doctest: +SKIP
    >>>
    >>> print("SIMD Kalman: %.03f s" % T_SIMD)  # doctest: +SKIP
    SIMD Kalman: 0.260 s
    >>> print("PyKalman:    %.03f s" % T_PK)  # doctest: +SKIP
    PyKalman:    13.934 s
    >>>
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["oseiskar"],
        "maintainers": ["oseiskar"],
        "python_dependencies": "simdkalman",
        # estimator type
        # --------------
        "scitype:transform-labels": "Series",
        # what is the scitype of y: None (not needed), Primitives, Series, Panel
        "X_inner_mtype": [
            "np.ndarray",
            "numpy3D",
        ],  # which mtypes do _fit/_predict support for X?
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
        process_noise=None,
        measurement_noise=None,
        measurement_function=None,
        initial_state=None,
        initial_state_covariance=None,
        denoising=False,
        hidden=True,
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

        self.hidden = hidden
        self.denoising = denoising

    def _fit(self, X, y=None):
        """Fit transformer to X and y.

        This method prepares the transformer.
        The matrix initializations or estimations
        (if requested by user) are calculated here.

        Parameters
        ----------
        X : np.ndarray
            of shape (time_steps, measurement_dim) or
            (instance, time_steps, measurement_dim).
            Data (measurements) to be transformed.
            Missing values must be represented as np.NaN or np.nan.
        y : ignored argument for interface compatibility

        Returns
        -------
            self: reference to self
        """
        measurement_dim = X.shape[1]
        mat_tup = self._get_init_values(measurement_dim, self.state_dim)
        (F_, Q_, R_, H_, X0_, P0_) = mat_tup

        for m in mat_tup:
            if len(m.shape) > 2:
                raise ValueError("Dynamic inputs are not supported by simdkalman")

        from simdkalman import KalmanFilter as simdkalman_KalmanFilter

        self._X0 = X0_[:, np.newaxis]
        self._P0 = P0_
        self._kalman_filter = simdkalman_KalmanFilter(
            state_transition=F_,
            process_noise=Q_,
            observation_model=H_,
            observation_noise=R_,
        )

        # TODO: EM algorithm
        # which requires changes in the simdkalman package for full support

        return self

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        This method performs the transformation of the input data
        according to the constructor input parameter ``denoising``.
        If ``denoising`` is True - then denoise data by returning
        ``simdkalman``'s smoothed *observations*.
        Else, infer hidden state using ``simdkalman`` smoothed
        *states*.

        Parameters
        ----------
        X : np.ndarray
            of shape (time_steps, measurement_dim) or
            (instance, time_steps, measurement_dim).
            Data (measurements) to be transformed.
            Missing values must be represented as np.NaN or np.nan.
        y : ignored argument for interface compatibility

        Returns
        -------
            X_transformed : np.ndarray
                transformed version of X
        """
        multiple_instances = len(X.shape) == 3  # (instance, variable, time point)
        if multiple_instances:
            X_transposed = X.transpose(0, 2, 1)  # (instance, time point, variable)
        else:
            X_transposed = X[np.newaxis, ...]

        smooth = self.denoising
        r = self._kalman_filter.compute(
            X_transposed,
            n_test=0,
            initial_value=self._X0,
            initial_covariance=self._P0,
            observations=not self.hidden,
            states=self.hidden,
            covariances=False,
            filtered=not smooth,
            smoothed=smooth,
        )

        if smooth:
            result = r.smoothed
        else:
            result = r.filtered

        if self.hidden:
            result = result.states
        else:
            result = result.observations

        result = result.mean

        # undo auto-flatten in simdkalman
        if len(result.shape) < 3:
            result = result[..., np.newaxis]

        assert len(result.shape) == 3

        if multiple_instances:
            result = result.transpose(0, 2, 1)
        else:
            assert result.shape[0] == 1
            result = result[0, ...]

        return result

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

        def get_params2(**kwargs):
            base = {
                # two hidden states, one observed variable
                "state_dim": 2,
                "state_transition": np.array([[1, 1], [0, 1]]),
                "process_noise": np.diag([0.1, 0.01]),
                "measurement_function": np.array([[1, 0]]),
                "measurement_noise": 1.0,
            }
            r = {}
            for k, v in base.items():
                r[k] = v
            for k, v in kwargs.items():
                r[k] = v
            return r

        # copied from KalmanFilterTransformerFP
        params1 = {
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
            "measurement_function": np.array([[1, 0]]),
            "measurement_noise": np.array([[0.1]]),
            "denoising": True,
        }
        return [
            params1,
            get_params2(denoising=False),
            get_params2(hidden=False),
            get_params2(initial_state=np.array([10, 10])),
        ]
