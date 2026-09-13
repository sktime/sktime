# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements Next generation reservoir computing estimator."""

__all__ = ["NextRC"]
__author__ = ["frthjf", "fkiraly"]

import numpy as np
import sklearn.linear_model
from sklearn.base import clone
from sklearn.utils.validation import check_array

from sktime.forecasting.base import BaseForecaster


class NextRC(BaseForecaster):
    r"""Next generation reservoir based forecasts of time series data.

    Uses a `sklearn` regressor `regressor` to regress values of time series on index:

    In `fit`, for input time series :math:`(v_i, t_i), i = 1, \dots, T`,
    where :math:`v_i` are values and :math:`t_i` are time stamps,
    fits an `sklearn` model :math:`v_i = f(t_i) + \epsilon_i`, where `f` is
    the model fitted when `regressor.fit` is passed `X` = vector of :math:`t_i`,
    and `y` = vector of :math:`v_i`.

    In `predict`, for a new time point :math:`t_*`, predicts :math:`f(t_*)`,
    where :math:`f` is the function as fitted above in `fit`.

    Default for `regressor` is linear regression = `sklearn` `Ridge` default.

    If time stamps are `pd.DatetimeIndex`, fitted coefficients are in units
    of days since start of 1970. If time stamps are `pd.PeriodIndex`,
    coefficients are in units of (full) periods since start of 1970.

    Parameters
    ----------
    regressor : estimator object, default = None
        Define the regression model type. If not set, will default to
         sklearn.linear_model.Ridge

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.reservoir import NextRC
    >>> y = load_airline()
    >>> forecaster = NextRC()
    >>> forecaster.fit(y)
    NextRC(...)
    >>> y_pred = forecaster.predict(fh=[1,2,3])
    """

    _tags = {
        "y_inner_mtype": "np.ndarray",
        "X_inner_mtype": "np.ndarray",
        "requires-fh-in-fit": False,
        "handles-missing-data": True,
        "scitype:y": "univariate",
        "ignores-exogeneous-X": True,
    }

    def __init__(self, regressor=None):
        self.regressor = regressor
        self.X_ = None
        self.y_ = None
        self.delay_ = 1
        self.d_linear_ = None
        self.d_nonlinear_ = None
        self.fitted_y = False
        super(NextRC, self).__init__()

    def _feature_vector(self, y, k=None):
        if k is None:
            k = self.delay_

        (n_samples, n_features) = y.shape

        # size of the linear part of the feature vector
        self.d_linear_ = d_linear = k * n_features
        # size of the non-linear part of feature vector
        self.d_nonlinear_ = d_nonlinear = int(
            d_linear * (d_linear + 1) * (d_linear + 2) / 6
        )
        # total size
        d = d_linear + d_nonlinear

        # create an array to hold the linear part of the feature vector
        features = np.zeros((d_linear, n_samples))

        # fill in the linear part of the complete feature vector
        for delay in range(k):
            for j in range(delay, n_samples):
                features[n_features * delay : n_features * (delay + 1), j] = y[
                    j - delay, :
                ]

        # create full feature vector
        O_features = np.zeros((d, n_samples))

        # copy over the linear part
        O_features[0:d_linear, :] = features

        # fill in the non-linear part
        cnt = 0
        for row in range(d_linear):
            for column in range(row, d_linear):
                for span in range(column, d_linear):
                    O_features[d_linear + cnt] = (
                        O_features[row, :] * O_features[column, :] * O_features[span, :]
                    )
                    cnt += 1

        return O_features

    def _fit(self, y, X=None, fh=None):
        """Fit to training data.

        Parameters
        ----------
        y : np.ndarray
            Target time series with which to fit the forecaster.
        X : np.ndarray, default=None
            Exogenous variables are ignored
        fh : int, list or np.array, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.

        Returns
        -------
        self : returns an instance of self.
        """
        self.X_ = X
        self.y_ = y
        self.delay_ = 2

        if self.regressor is None:
            self.regressor_ = sklearn.linear_model.Ridge(
                alpha=1.0e-3, fit_intercept=False
            )
        else:
            self.regressor_ = clone(self.regressor)

        n_features = y.shape[1]

        # use relative fh as time index to predict
        fh = self.fh.to_absolute_int(self.cutoff)

        # print(y[: fh[-1] + 1, :])

        X_features = self._feature_vector(y[: fh[-1] + 1, :])

        # print(X_features[:, 19 : 20])

        if X is None:
            X = (
                X_features[0:n_features, fh[0] - 2 : fh[-1] - 1]
                - X_features[0:n_features, fh[0] - 1 : fh[-1]]
            )
            self.fitted_X = False
        else:
            X = X[fh[0] : fh[-1], :].T
            self.fitted_X = True

        # print(X)

        self.regressor_.fit(
            X_features[:, fh[0] - 1 : fh[-1]].T,
        )

        return self

    def _predict(self, fh=None, X=None):
        """Make forecasts for the given forecast horizon.

        Parameters
        ----------
        fh : int, list or np.array
            The forecast horizon with the steps ahead to predict
        X : np.ndarray, default=None
            Exogenous variables (ignored)

        Returns
        -------
        y_pred : np.ndarray
            Point predictions for the forecast
        """
        self.check_is_fitted()

        X = check_array(self.y_)

        n_features = X.shape[1]

        # use relative fh as time index to predict
        fh = self.fh.to_absolute_int(self.cutoff)

        X_features = self._feature_vector(X[: fh[-1] + 1, :])

        y_pred = self.regressor_.predict(X_features[:, fh[0] - 1 : fh[-1]].T)

        # print(y_pred)

        if self.fitted_y:
            return y_pred
        return y_pred.T + X_features[0:n_features, fh[0] - 1 : fh[-1]]

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
        """
        from sklearn.ensemble import RandomForestRegressor

        params_list = [{}, {"regressor": RandomForestRegressor()}]

        return params_list
