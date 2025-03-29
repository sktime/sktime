"""KalmanFilterTransformerFP - Kalman Filter transformer using FilterPy"""

__author__ = ["NoaWegerhoff", "lielleravid"]

import numpy as np
from ._base import (
    _BaseKalmanFilter,
    _validate_param_shape,
    _init_matrix,
    _validate_estimate_matrices,
)
from ._pykalman import KalmanFilterTransformerPK
from sktime.transformations.base import BaseTransformer
from sktime.utils.warnings import warn


def _get_t_matrix(time_t, matrices, shape, time_steps):
    """Extract matrix to be used at iteration ``time_t`` of Kalman filter iterations.

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
            matrix to be used at iteration ``time_t``
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


class KalmanFilterTransformerFP(_BaseKalmanFilter, BaseTransformer):
    """Kalman Filter is used for denoising or inferring the hidden state of given data.

    The Kalman Filter is an unsupervised algorithm, consisting of
    several mathematical equations which are used to create
    an estimate of the state of a process.

    This class is the adapter for the ``FilterPy`` package into ``sktime``.
    ``KalmanFilterTransformerFP`` implements hidden inferred states and
    denoising, depending on the boolean input parameter ``denoising``.
    In addition, ``KalmanFilterTransformerFP`` provides parameter
    optimization via Expectation-Maximization (EM) algorithm.

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
        of shape (state_dim, state_dim) or (time_steps, state_dim, state_dim).
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
    control_transition : np.ndarray, optional (default=None)
        of shape (state_dim, control_variable_dim) or
        (time_steps, state_dim, control_variable_dim).
        Control transition matrix, also referred to as ``G``.
        ``control_variable_dim`` is the dimension of ``control variable``,
        also referred to as ``u``.
        ``control variable`` is an optional parameter for ``fit`` and ``transform``
        functions.
    denoising : bool, optional (default=False).
        This parameter affects ``transform``. If False, then ``transform`` will be
        inferring
        hidden state. If True, uses ``FilterPy`` ``rts_smoother`` for denoising.
    estimate_matrices : str or list of str, optional (default=None).
        Subset of [``state_transition``, ``measurement_function``,
        ``process_noise``, ``measurement_noise``, ``initial_state``,
        ``initial_state_covariance``]
        or - ``all``. If ``estimate_matrices`` is an iterable of strings,
        only matrices in ``estimate_matrices`` will be estimated using EM algorithm.
        If ``estimate_matrices`` is ``all``,
        then all matrices will be estimated using EM algorithm.

        Note -
            - parameters estimated by EM algorithm assumed to be constant.
            - ``control_transition`` matrix cannot be estimated.

    See Also
    --------
    KalmanFilterTransformerPK :
        Kalman Filter transformer, adapter for the ``pykalman`` package
        into ``sktime``.
    KalmanFilterTransformerSIMD :
        Kalman Filter transformer, adapter for the ``simdkalman`` package
        into ``sktime``.

    Notes
    -----
    ``FilterPy`` KalmanFilter documentation :
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
        Basic example:

    >>> import numpy as np  # doctest: +SKIP
    >>> import sktime.transformations.series.kalman_filter as kf
    >>> time_steps, state_dim, measurement_dim = 10, 2, 3
    >>>
    >>> X = np.random.rand(time_steps, measurement_dim) * 10
    >>> transformer = kf.KalmanFilterTransformerFP(state_dim=state_dim) # doctest: +SKIP
    >>> Xt = transformer.fit_transform(X=X)  # doctest: +SKIP

        Example of - denoising, matrix estimation, missing values and transform with y:
    >>> import numpy as np  # doctest: +SKIP
    >>> import sktime.transformations.series.kalman_filter as kf
    >>> time_steps, state_dim, measurement_dim = 10, 3, 3
    >>> control_variable_dim = 2
    >>>
    >>> X = np.random.rand(time_steps, measurement_dim)
    >>> # missing value
    >>> X[0][0] = np.nan
    >>>
    >>> # y
    >>> control_variable = np.random.rand(time_steps, control_variable_dim)
    >>>
    >>> # If matrices estimation is required, elements of ``estimate_matrices``
    >>> # are assumed to be constants.
    >>> transformer = kf.KalmanFilterTransformerFP(  # doctest: +SKIP
    ...     state_dim=state_dim,
    ...     measurement_noise=np.eye(measurement_dim),
    ...     denoising=True,
    ...     estimate_matrices='measurement_noise'
    ...     )
    >>> Xt = transformer.fit_transform(X=X, y=control_variable)  # doctest: +SKIP

        Example of - dynamic inputs (matrix per time-step), missing values:

    >>> import numpy as np  # doctest: +SKIP
    >>> import sktime.transformations.series.kalman_filter as kf
    >>> time_steps, state_dim, measurement_dim = 10, 4, 4
    >>> control_variable_dim = 4
    >>>
    >>> X = np.random.rand(time_steps, measurement_dim)
    >>> # missing values
    >>> X[0] = [np.NaN for i in range(measurement_dim)]
    >>>
    >>> # y
    >>> control_variable = np.random.rand(control_variable_dim)
    >>>
    >>> # Dynamic input -
    >>> # ``state_transition`` provide different matrix for each time step.
    >>> transformer = kf.KalmanFilterTransformerFP(  # doctest: +SKIP
    ...     state_dim=state_dim,
    ...     state_transition=np.random.rand(time_steps, state_dim, state_dim),
    ...     estimate_matrices=['initial_state', 'initial_state_covariance']
    ...     )
    >>> Xt = transformer.fit_transform(X=X, y=control_variable)  # doctest: +SKIP
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["NoaWegerhoff", "lielleravid"],
        "maintainers": ["NoaWegerhoff"],
        "python_dependencies": "filterpy",
        # estimator type
        # --------------
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
        super().__init__(
            state_dim=state_dim,
            state_transition=state_transition,
            process_noise=process_noise,
            measurement_noise=measurement_noise,
            measurement_function=measurement_function,
            initial_state=initial_state,
            initial_state_covariance=initial_state_covariance,
        )
        # G/B
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
            estimate_matrices_ = _validate_estimate_matrices(
                input_ems=self.estimate_matrices,
                all_ems=[
                    "state_transition",
                    "process_noise",
                    "measurement_noise",
                    "measurement_function",
                    "initial_state",
                    "initial_state_covariance",
                ],
            )

            transformer_ = KalmanFilterTransformerPK(
                state_dim=self.state_dim,
                state_transition=self.state_transition,
                transition_offsets=np.zeros(self.state_dim),
                measurement_offsets=np.zeros(measurement_dim),
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

        return self

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        This method performs the transformation of the input data
        according to the constructor input parameter ``denoising``.
        If ``denoising`` is True - then denoise data using
        ``FilterPy``'s ``rts_smoother`` function.
        Else, infer hidden state using ``FilterPy``'s ``predict`` and ``update``
        functions.

        Parameters
        ----------
        X : np.ndarray
            of shape (time_steps, measurement_dim).
            Data (measurements) to be transformed.
            Missing values must be represented as np.NaN or np.nan.
        y : np.ndarray, optional (default=None).
            of shape (control_variable_dim,) or (time_steps, control_variable_dim).
            ``control variable``, also referred to as ``u``.
            if 2D, must be same length as X.

        Returns
        -------
            X_transformed : np.ndarray
                transformed version of X
        """
        import filterpy.kalman.kalman_filter as filterpy_kf

        time_steps = X.shape[0]
        measurement_dim = X.shape[1]

        if y is None:
            if self.control_transition is None:
                y_dim = 1
            else:
                y_dim = np.atleast_2d(self.control_transition).shape[-1]
                warn(
                    "Class parameter `control_transition` was initiated with user data "
                    "but received no data through `transform` argument, `y`. "
                    "Therefore, omitting `control_transition` "
                    "when calculating the result. ",
                    obj=self,
                )
            y = np.zeros(y_dim)

        shapes = self._get_shapes(
            state_dim=self.state_dim, measurement_dim=measurement_dim, u_dim=y.shape[-1]
        )
        G = _init_matrix(
            matrices=self.control_transition,
            transform_func=np.atleast_2d,
            default_val=np.eye(*shapes["G"]),
        )
        _validate_param_shape(
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

        # kalman filter iterations
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
            "measurement_function": np.array([[1, 0]]),
            "measurement_noise": np.array([[0.1]]),
            "denoising": True,
            "estimate_matrices": ["measurement_noise"],
        }
        return [params1, params2]

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
            u is the ``control variable``. If ``u`` not given in _transform default
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
            Dimension of ``state``.
        measurement_dim : int
            Dimension of measurements.
        u_dim : int
            Dimension of control variable u (y).

        Returns
        -------
            shapes : dict
                Dictionary with default shape of each matrix parameter
        """
        shapes = super()._get_shapes(state_dim, measurement_dim)
        shapes["u"] = (u_dim,)
        shapes["G"] = (state_dim, u_dim)
        return shapes
