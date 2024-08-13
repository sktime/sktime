# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Kalman Filter Forecaster.

Forecaster based on Kalman Filter algorithm.
"""
__author__ = ["NoaBenAmi", "lielleravid"]
__all__ = ["KalmanFilterForecaster"]

import numpy as np
import pandas as pd

from sktime.forecasting.base import BaseForecaster
from sktime.transformations.series.kalman_filter import KalmanFilterTransformerFP


def _split(to_split, split_point, nd_function=np.atleast_2d, ndim=3):
    if to_split is None:
        return None, None

    _to_split = nd_function(to_split)
    if _to_split.ndim == ndim:
        if split_point < _to_split.shape[0]:
            return _to_split[:split_point], _to_split[split_point:]
    return _to_split, None


def _set_data_atleast_nd(data, default, n=3):
    f = {1: np.atleast_1d, 2: np.atleast_2d, 3: np.atleast_3d}[n]
    if data is None:
        data = default

    data = np.asarray(data)
    if len(data.shape) == n - 1:
        return f([data])
    return f(data)


def _forecast_matrices(fm, tm, split_point):
    if fm is None:
        l, r = _split(tm, split_point=split_point)
        return l if r is None else r
    return fm


class KalmanFilterForecaster(BaseForecaster):
    """Kalman Filter can forecast, denoise, and estimate the hidden state of a process.

    The Kalman Filter is an unsupervised algorithm, consisting of
    several mathematical equations which are used to estimate past,
    present, and future states a process.

    The strength of the Kalman Filter is in its ability to efficiently infer
    the state of a system, even when its exact nature is unknown.

    Kalman Filter Forecaster
    Estimate the series of `future` states - :math:`x_t` for
    :math:`t = n+1, n+2, ..., n+fh`
    When given input data measurements - :math:`m_t` for :math:`t = 1, 2, ..., n` and
    `forecast horizon` :math:`fh`

    In-sample predictions are calculated by `KalmanFilterTransformerFP`.

    `time_steps` is referring to the length of training data.

    ============================  ========================
           Parameter Name           Literature Reference
    ============================  ========================
      state_transition                  `F` or `A`
      control_transition                `G` or `B`
      process_noise                     `Q`
      measurement_noise                 `R`
      measurement_function              `H` or `C`
      initial_state                     `X0`
      initial_state_covariance          `P0`
    ============================  ========================

    Parameters
    ----------
    state_dim : int, optional (default=None)
        System state feature dimension. If None, infer the size
        of the dimension from input data.
    #todo: rewrite state_transition
    state_transition : np.ndarray, optional (default=None)
        of shape (state_dim, state_dim) or (n, state_dim, state_dim),
        s.t :math:`n >= time_steps`.
        State transition matrix, also referred to as `F`, is a matrix
        which describes the way the underlying series moves
        through successive time periods.
    process_noise : np.ndarray, optional (default=None)
        of shape (state_dim, state_dim) or (time_steps, state_dim, state_dim).
        Process noise matrix, also referred to as `Q`,
        the uncertainty of the dynamic model.
    measurement_noise : np.ndarray, optional (default=None)
        of shape (measurement_dim, measurement_dim) or
        (time_steps, measurement_dim, measurement_dim).
        Measurement noise matrix, also referred to as `R`,
        represents the uncertainty of the measurements.
    measurement_function : np.ndarray, optional (default=None)
        of shape (measurement_dim, state_dim) or
        (time_steps, measurement_dim, state_dim).
        Measurement equation matrix, also referred to as `H`, adjusts
        dimensions of measurements to match dimensions of state.
    initial_state : np.ndarray, optional (default=None)
        of shape (state_dim,).
        Initial estimated system state, also referred to as `X0`.
    initial_state_covariance : np.ndarray, optional (default=None)
        of shape (state_dim, state_dim).
        Initial estimated system state covariance, also referred to as `P0`.
    #todo: rewrite control_transition
    control_transition : np.ndarray, optional (default=None)
        of shape (state_dim, control_variable_dim) or
        (time_steps, state_dim, control_variable_dim).
        Control transition matrix, also referred to as `G` or `B`.
        `control_variable_dim` is the dimension of `control variable`,
        also referred to as `u`.
        `control variable` is an optional parameter for `fit` and `transform` functions.
    denoising : bool, optional (default=True).
        This parameter affects in-sample predictions.
        `KalmanFilterTransformerFP` is initiated with the value of `denoising` and
        then used to calculate in-sample predictions.
        See `KalmanFilterTransformerFP` documentation for more details.
    estimate_matrices : str or list of str, optional (default=None).
        Subset of [`state_transition`, `measurement_function`,
        `process_noise`, `measurement_noise`, `initial_state`,
        `initial_state_covariance`]
        or - `all`. If `estimate_matrices` is an iterable of strings,
        only matrices in `estimate_matrices` will be estimated using EM algorithm.

        Note -
            - parameters estimated by EM algorithm assumed to be constant.
            - `control_transition` matrix cannot be estimated.
            - To avoid constant and repetitive results, `state_transition`
              will be estimated by the EM algorithm if its given value is None.

    See Also
    --------
    KalmanFilterTransformerPK :
        Kalman Filter transformer, adapter for the `pykalman` package into `sktime`.
    KalmanFilterTransformerFP :
        Kalman Filter transformer, adapter for the `FilterPy` package into `sktime`.

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
    """

    _tags = {
        "scitype:y": "both",  # which y are fine? univariate/multivariate/both
        "ignores-exogeneous-X": False,  # does estimator ignore the exogeneous X?
        "handles-missing-data": True,  # can estimator handle missing data?
        "y_inner_mtype": "pd.DataFrame",  # which types do _fit, _predict, assume for y?
        "X_inner_mtype": "pd.DataFrame",  # which types do _fit, _predict, assume for X?
        "requires-fh-in-fit": True,  # is forecasting horizon already required in fit?
        "X-y-must-have-same-index": True,  # can estimator handle different X/y index?
        "enforce_index_type": None,  # index type that needs to be enforced in X/y
        "capability:pred_int": False,  # does forecaster implement predict_quantiles?
    }

    def __init__(
        self,
        state_dim=None,
        state_transition=None,
        control_transition=None,
        process_noise=None,
        measurement_noise=None,
        measurement_function=None,
        initial_state=None,
        initial_state_covariance=None,
        estimate_matrices=None,
        denoising=True,
    ):

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

        self.estimate_matrices = estimate_matrices
        self.denoising = denoising

        super(KalmanFilterForecaster, self).__init__()

    def _fit(self, y, X=None, fh=None):
        """Fit forecaster to training data.

        private _fit containing the core logic, called from fit
        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        y : pd.DataFrame
            Time series to which to fit the forecaster.
            Input data can be univariate or multivariate.
        fh : ForecastingHorizon
            The forecasting horizon with the steps ahead to to predict.
            Required (non-optional).
        X : pd.DataFrame, optional (default=None)
            of shape (control_variable_dim,) or (time_steps, control_variable_dim).
            `control variable`, also referred to as `u`.
            if 2D, must be same length as `y`.

        Returns
        -------
        self : reference to self
        """
        y = y.to_numpy()
        X = X.to_numpy() if X is not None else None
        time_steps = y.shape[0]
        self.state_dim_ = y.shape[-1] if self.state_dim is None else self.state_dim

        # split control_transition matrices to transform and forecast.
        B_transform, B_forecast = _split(
            to_split=self.control_transition, split_point=time_steps
        )
        # split state_transition matrices to transform and forecast.
        F_transform, F_forecast = _split(
            to_split=self.state_transition, split_point=time_steps
        )
        # split X (control variable, u) matrices to transform and forecast.
        us_transform, self.us_forecast = _split(
            to_split=X, split_point=time_steps, nd_function=np.asarray, ndim=2
        )

        # if `state_transition` is None and not in `estimate_matrices`,
        # then `state_transition` will be estimated with EM.
        # This is done in order to avoid the unwanted behavior -
        # the default value of `state_transition` is np.eye(state_dim),
        # so when calling `predict` with future fh,
        # predictions will be of a constant value.
        em = [] if self.estimate_matrices is None else self.estimate_matrices
        if F_transform is not None or em == "all" or "state_transition" in em:
            estimate_matrices_ = self.estimate_matrices
        else:
            estimate_matrices_ = em + ["state_transition"]

        transformer_ = KalmanFilterTransformerFP(
            state_dim=self.state_dim_,
            state_transition=F_transform,
            control_transition=B_transform,
            process_noise=self.process_noise,
            measurement_noise=self.measurement_noise,
            measurement_function=self.measurement_function,
            initial_state=self.initial_state,
            initial_state_covariance=self.initial_state_covariance,
            estimate_matrices=estimate_matrices_,
            denoising=self.denoising,
        )
        transformer_ = transformer_.fit(X=y, y=us_transform)
        y_transformed = transformer_.transform(X=y, y=us_transform)

        # set last value of y_transformed as x_hat_ (the prediction for t = time_steps)
        self.x_hat_ = y_transformed[-1]
        # Extract values of in-sample predictions and set as an attribute for later use.
        self.in_samples_preds_ = y_transformed[self._relative_indices()]

        self.B_ = _forecast_matrices(
            fm=B_forecast, tm=B_transform, split_point=time_steps - 1
        )
        self.F_ = _forecast_matrices(
            fm=F_forecast, tm=transformer_.F_, split_point=time_steps - 1
        )

        return self

    def _predict(self, fh, X=None):
        """Forecast time series at future horizon.

        private _predict containing the core logic, called from predict
        State required:
            Requires state to be "fitted".
        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : ForecastingHorizon, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            If not passed in _fit, guaranteed to be passed here
        X : pd.DataFrame, optional (default=None)
            Exogenous time series

        Returns
        -------
        y_pred : pd.Series
            Point predictions
        """
        # X = 0 if X is None else X.to_numpy()
        us_dim_hint = 1 if self.B_ is None else self.B_.shape[-1]
        X = self.us_forecast if X is None else X.to_numpy()
        # transform X to 2D np.array
        us = _set_data_atleast_nd(data=X, default=np.zeros(us_dim_hint), n=2)
        # transform B_ to 3D np.array
        Bs = _set_data_atleast_nd(
            data=self.B_, default=np.eye(self.state_dim_, us.shape[-1])
        )
        # transform F_ to 3D np.array
        Fs = _set_data_atleast_nd(data=self.F_, default=np.eye(self.state_dim_))

        future_pred_indices = self._relative_indices(in_sample=False)
        preds = []

        x = self.x_hat_
        # Kalman Filter iteration
        for t in range(np.amax(np.asarray(future_pred_indices), initial=0)):
            Ft = Fs[t] if len(Fs) > t else Fs[-1]
            Bt = Bs[t] if len(Bs) > t else Bs[-1]
            ut = us[t] if len(us) > t else us[-1]

            x = np.dot(Ft, x) + np.dot(Bt, ut)

            if t + 1 in future_pred_indices:
                preds.append(x)

        output_predictions = [pred for pred in self.in_samples_preds_] + preds
        return pd.DataFrame(
            data=output_predictions, index=self.fh.to_absolute(self.cutoff).to_numpy()
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for forecasters.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        params = {"denoising": True}
        return params

    def _relative_indices(self, in_sample=True):
        fh_relative = self.fh.to_relative(self.cutoff)
        if in_sample:
            return [i - 1 for i in fh_relative[fh_relative < 1]]
        return fh_relative[fh_relative > 0]
