# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements TrendForecaster."""

__author__ = ["tensorflow-as-tf", "mloning", "aiwalter", "fkiraly"]
__all__ = ["TrendForecaster"]

import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression

from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.trend._util import _get_X_numpy_int_from_pandas


class TrendForecaster(BaseForecaster):
    r"""Trend based forecasts of time series data, regressing values on index.

    Uses an ``sklearn`` regressor specified by the ``regressor`` parameter
    to perform regression on time series values against their corresponding indices,
    providing trend-based forecasts:

    In ``fit``, for input time series :math:`(v_i, t_i), i = 1, \dots, T`,
    where :math:`v_i` are values and :math:`t_i` are time stamps,
    fits an ``sklearn`` model :math:`v_i = f(t_i) + \epsilon_i`, where :math:`f` is
    the model fitted when ``regressor.fit`` is passed ``X`` = vector of :math:`t_i`,
    and ``y`` = vector of :math:`v_i`.

    In ``predict``, for a new time point :math:`t_*`, predicts :math:`f(t_*)`,
    where :math:`f` is the function as fitted above in ``fit``.

    Default for ``regressor`` is linear regression = ``sklearn`` ``LinearRegression``,
    with default parameters.

    If time stamps are ``pd.DatetimeIndex``, fitted coefficients are in units
    of days since start of 1970. If time stamps are ``pd.PeriodIndex``,
    coefficients are in units of (full) periods since start of 1970.

    Parameters
    ----------
    regressor : estimator object, default = None
        Define the regression model type. If not set, will default to
         sklearn.linear_model.LinearRegression

    Attributes
    ----------
    regressor_ : sklearn regressor estimator object
        The fitted regressor object. Clone of ``regressor``.

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.trend import TrendForecaster
    >>> y = load_airline()
    >>> forecaster = TrendForecaster()
    >>> forecaster.fit(y)
    TrendForecaster(...)
    >>> y_pred = forecaster.predict(fh=[1,2,3])
    """

    _tags = {
        "authors": ["tensorflow-as-tf", "mloning", "aiwalter", "fkiraly"],
        "maintainers": ["tensorflow-as-tf"],
        "ignores-exogeneous-X": True,
        "requires-fh-in-fit": False,
        "handles-missing-data": False,
    }

    def __init__(self, regressor=None):
        # for default regressor, set fit_intercept=True
        self.regressor = regressor
        super().__init__()

    def _fit(self, y, X, fh):
        """Fit to training data.

        Parameters
        ----------
        y : pd.Series
            Target time series with which to fit the forecaster.
        X : pd.DataFrame, default=None
            Exogenous variables are ignored
        fh : int, list or np.array, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.

        Returns
        -------
        self : returns an instance of self.
        """
        if self.regressor is None:
            self.regressor_ = LinearRegression(fit_intercept=True)
        else:
            self.regressor_ = clone(self.regressor)

        # we regress index on series values
        # the sklearn X is obtained from the index of y
        # the sklearn y can be taken as the y seen here
        X_sklearn = _get_X_numpy_int_from_pandas(y.index)

        # fit regressor
        self.regressor_.fit(X_sklearn, y)
        return self

    def _predict(self, fh=None, X=None):
        """Make forecasts for the given forecast horizon.

        Parameters
        ----------
        fh : int, list or np.array
            The forecast horizon with the steps ahead to predict
        X : pd.DataFrame, default=None
            Exogenous variables (ignored)

        Returns
        -------
        y_pred : pd.Series
            Point predictions for the forecast
        """
        # use relative fh as time index to predict
        fh = self.fh.to_absolute_index(self.cutoff)
        X_sklearn = _get_X_numpy_int_from_pandas(fh)
        y_pred_sklearn = self.regressor_.predict(X_sklearn)
        y_pred = pd.Series(y_pred_sklearn, index=fh)
        y_pred.name = self._y.name
        return y_pred

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        from sklearn.ensemble import RandomForestRegressor

        params_list = [{}, {"regressor": RandomForestRegressor()}]

        return params_list
