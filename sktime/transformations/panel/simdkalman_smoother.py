# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Vectorized Kalman Filter/Smoother transformer implemented using the simdkalman library"""

__author__ = ["oseiskar"]

import numpy as np
import pandas as pd

from sktime.transformations.base import BaseTransformer
from sktime.utils.warnings import warn


class SIMDKalmanSmoother(BaseTransformer):
    """Vectorized Kalman Filter/Smoother transformer from simdkalman.

    The Kalman Filter is an unsupervised algorithm, consisting of
    several mathematical equations which are used to create
    an estimate of the state of a process. The Kalman Filter is typically
    used for denoising data,  or inferring the hidden state of data.

    This class is the adapter for the ``simdkalman`` package into ``sktime``.
    ``SIMDKalmanSmoother`` implements hidden inferred states and
    denoising, depending on the boolean input parameter ``hidden``.
    In addition, filtering (forward pass only) and smoothing (forward and backward
    pass) options can be selected with the ``denoising`` parameter.

    The ``simdkalman`` package is ideal for Panels where similar
    Kalman Filters are applied in to multiple time series. The package
    applies multi-dimensional matrix operations, which can be an order
    of magnitude faster than the non-vectorized implementations.

    As long as the shapes of the parameters match reasonably according
    to the rules of matrix multiplication, this class is flexible in their
    exact nature accepting

     * scalars: ``process_noise = 0.1``
     * (2d) numpy matrices: ``process_noise = numpy.eye(2)``
     * 2d arrays: ``observation_model = [[1,2]]``

    See https://simdkalman.readthedocs.io/ for the mathematical definitions
    of the parameters.

    Parameters
    ----------
    state_transition : np.ndarray
        of shape (state_dim, state_dim).
        State transition matrix, also referred to as ``F``, is a matrix
        which describes the way the underlying series moves
        through successive time periods. Called ``A`` in ``simdkalman``.
    process_noise : np.ndarray
        of shape (state_dim, state_dim).
        Process noise matrix, also referred to as ``Q``,
        the uncertainty of the dynamic model.
    measurement_noise : np.ndarray
        of shape (measurement_dim, measurement_dim).
        Measurement noise matrix, also referred to as ``R``,
        represents the uncertainty of the measurements.
    measurement_function : np.ndarray
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
    state_dim : int, optional (default=None).
        If provided, the state dimension is checked against the state transition matrix.
        Provided for compatibility with ``KalmanFilterTransformerSIMD``.

    See Also
    --------
    sktime.transformations.series.KalmanFilterTransformerSIMD :
        ``simdkalman``-based Kalman Filter transformer for Series data.

    Notes
    -----
    ``simdkalman`` documentation :
        https://simdkalman.readthedocs.io/


    Examples
    --------
        Basic example:

    >>> import numpy as np  # doctest: +SKIP
    >>> import sktime.transformations.series.kalman_filter as kf
    >>> time_steps, state_dim, measurement_dim = 10, 2, 1
    >>>
    >>> X = np.random.rand(time_steps, measurement_dim) * 10
    >>> transformer = kf.SIMDKalmanSmoother(  # doctest: +SKIP
    ...     state_transition=np.array([[1,1],[0,1]]),
    ...     process_noise=np.diag([0.1, 0.01]),
    ...     measurement_function=np.array([[1,0]]),
    ...     measurement_noise=np.eye(measurement_dim) * 2.0,
    ...     )
    >>> Xt = transformer.fit_transform(X=X)  # doctest: +SKIP

    """

    _tags = {
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Series",
        "scitype:instancewise": True,
        "scitype:transform-labels": "None",
        "X_inner_mtype": ["pd.Series", "pd.DataFrame", "numpy3D"],
        "y_inner_mtype": "None",
        "capability:unequal_length": False,
        "univariate-only": False,  # TODO: can also support 3D data
        "requires_y": False,
        "capability:inverse_transform": False,
        "fit_is_empty": False,
        "handles-missing-data": True,
        "authors": ["oseiskar"],
        "maintainers": ["oseiskar"],
        "python_dependencies": ["simdkalman"],
    }

    def __init__(
        self,
        state_transition,
        process_noise,
        measurement_noise,
        measurement_function,
        initial_state=None,
        initial_state_covariance=None,
        denoising=False,
        hidden=True,
        state_dim=None,
    ):
        self.state_transition = state_transition
        self.process_noise = process_noise
        self.measurement_function = measurement_function
        self.measurement_noise = measurement_noise
        self.initial_state = initial_state
        self.initial_state_covariance = initial_state_covariance
        self.denoising = denoising
        self.hidden = hidden
        self.state_dim = state_dim

        if state_dim is not None:
            assert state_dim == np.atleast_2d(state_transition).shape[0]

        super().__init__()

    def _fit(self, X, y=None):
        from sktime.transformations.panel._simdkalman_adapter import _SIMDKalmanAdapter

        self._adapter = _SIMDKalmanAdapter(
            state_transition=self.state_transition,
            process_noise=self.process_noise,
            measurement_noise=self.measurement_noise,
            measurement_function=self.measurement_function,
            initial_state=self.initial_state,
            initial_state_covariance=self.initial_state_covariance,
            hidden=self.hidden,
            denoising=self.denoising,
        )

        # TODO: EM algorithm

        return self

    def _transform(self, X, y=None):
        if isinstance(X, pd.Series):
            X_numpy = X.to_numpy()[np.newaxis, :, np.newaxis]
            meta = dict(index=X.index)
            if not self.hidden:
                meta["name"] = X.name
        elif isinstance(X, pd.DataFrame):
            X_numpy = X.to_numpy().transpose()[np.newaxis, ...]
            meta = dict(index=X.index)
        else:
            assert len(X.shape) == 3  # (instance, variable, time point)
            X_numpy = X.transpose(0, 2, 1)  # (instance, time point, variable)

        if not self.hidden:
            # NOTE: this is questionable but is required for check_estimator
            original_shape = X_numpy.shape
            if self.measurement_function.shape[0] == 1 and X_numpy.shape[2] != 1:
                original_shape = X_numpy.shape
                X_numpy = X_numpy.reshape(X_numpy.shape[0], -1, 1)
                warn(
                    "SIMDKalmanSmoother vectorizing over wrong dimension for compatibility"
                )
                meta = {}

        assert X_numpy.shape[2] == self.measurement_function.shape[0]
        Xt_numpy = self._adapter.compute(X_numpy, multiple_instances=True)
        assert len(Xt_numpy.shape) == 3

        if not self.hidden and Xt_numpy.shape != original_shape:
            Xt_numpy = Xt_numpy.reshape(original_shape)

        r = Xt_numpy.transpose(0, 2, 1)

        if isinstance(X, pd.Series):
            assert r.shape[0] == 1
            if r.shape[1] == 1:
                r = pd.Series(r[0, 0, :], **meta)
            else:
                r = pd.DataFrame(r[0, ...].transpose(), index=X.index)
        elif isinstance(X, pd.DataFrame):
            r = pd.DataFrame(r[0, ...].transpose(), **meta)
        return r

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        def get_params(**kwargs):
            base = {
                # two hidden states, one observed variable
                "state_transition": np.array([[1, 1], [0, 1]]),
                "process_noise": np.diag([0.1, 0.01]),
                "measurement_function": np.array([[1, 0]]),
                "measurement_noise": 1.0,
                "hidden": False,
            }
            r = {}
            for k, v in base.items():
                r[k] = v
            for k, v in kwargs.items():
                r[k] = v
            return r

        return [
            get_params(),
            get_params(denoising=False),
            get_params(initial_state=np.array([10, 10])),
        ]
