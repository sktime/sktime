# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements SplineTrendForecaster."""

__author__ = ["jgyasu", "tensorflow-as-tf", "mloning", "aiwalter", "fkiraly"]
__all__ = ["SplineTrendForecaster"]

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.forecasting.trend._util import _get_X_numpy_int_from_pandas


class SplineTrendForecaster(PolynomialTrendForecaster):
    """
    Forecast time series data with a spline trend.

    Parameters
    ----------
    regressor : sklearn regressor estimator object, default=None
        Define the regression model type. If not set, defaults to
        sklearn.linear_model.LinearRegression.
    degree : int, default=1
        Degree of the polynomial function.
    with_intercept : bool, default=True
        If True, includes a feature in which all polynomial powers are
        zero (i.e., a column of ones, acting as an intercept term in a linear
        model).
    prediction_intervals : bool, default=False
        Whether to compute prediction intervals. If True, additional
        calculations are done during fit to enable prediction intervals
        to be calculated during predict. The prediction intervals are
        based on an OLS regression model fitted to the data and calculated
        according to Section 7.9 in [1]. Formulas are modified appropriately
        if `with_intercept` is False.
    n_knots : int, default=5
        Number of knots of the splines if `knots` equals one of {'uniform', 'quantile'}.
        Must be at least 2. Ignored if `knots` is array-like.
    knots : {'uniform', 'quantile'}or array-like of shape (n_knots, n_features),
        default='uniform'
        Determines knot positions such that first knot <= features <= last knot.
        - 'uniform': `n_knots` are distributed uniformly between the
        min and max values of the features.
        - 'quantile': `n_knots` are distributed uniformly along the quantiles
        of the features.
        - array-like: Specifies sorted knot positions, including the boundary knots.
        Internally, additional knots are added before the first knot and after
        the last knot based on the spline degree.
    extrapolation : {'error', 'constant', 'linear', 'continue', 'periodic'},
        default='constant'
        Determines how to handle values outside the min and max values of the
        training features:
        - 'error': Raises a ValueError.
        - 'constant': Uses the spline value at the minimum or maximum feature as
        constant extrapolation.
        - 'linear': Applies linear extrapolation.
        - 'continue': Extrapolates as is (equivalent to `extrapolate=True` in
        `scipy.interpolate.BSpline`).
        - 'periodic': Uses periodic splines with a periodicity equal to the distance
        between the first and last knot, enforcing equal function values and
        derivatives at these knots.
    include_bias : bool, default=True
        If False, the last spline element inside the feature range is dropped.
        B-splines sum to one over the basis functions, implicitly including
        a bias term, i.e., a column of ones.

    References
    ----------
    .. [1] Hyndman, Rob J., and George Athanasopoulos. Forecasting: principles
    and practice, 3rd edition. OTexts: Melbourne, Australia. OTexts.com/fpp3.

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.trend import SplineTrendForecaster
    >>> y = load_airline()
    >>> forecaster = SplineTrendForecaster(
    ...     degree=1,
    ...     n_knots=5,
    ...     knots="uniform",
    ...     extrapolation="constant"
    ... )
    >>> forecaster.fit(y)
    SplineTrendForecaster()
    >>> y_pred = forecaster.predict(fh=[1, 2, 3])
    """

    _tags = {
        "authors": ["jgyasu", "tensorflow-as-tf", "mloning", "aiwalter", "fkiraly"],
        "ignores-exogeneous-X": True,
        "requires-fh-in-fit": False,
        "handles-missing-data": False,
        "capability:pred_int": True,
    }

    def __init__(
        self,
        regressor=None,
        degree=1,
        with_intercept=True,
        prediction_intervals=False,
        n_knots=5,
        knots="uniform",
        extrapolation="constant",
    ):
        super().__init__(
            regressor=regressor,
            degree=degree,
            with_intercept=with_intercept,
            prediction_intervals=prediction_intervals,
        )
        self.n_knots = n_knots
        self.knots = knots
        self.extrapolation = extrapolation
        # prediction_intervals : bool, default=False
        # By default, the extra information needed to later generate the prediction
        # intervals is not calculated. If set to True, the extra information is
        # calculated and stored in the forecaster.

        self.set_tags(**{"capability:pred_int": prediction_intervals})

    def _fit(self, y, X, fh):
        """Fit to training data.

        Parameters
        ----------
        y : pd.Series
            Target time series with which to fit the forecaster.
        X : pd.DataFrame, default=None
            Exogenous variables are ignored
        fh : int, list or np.array, default=None
            The forecasters horizon with the steps ahead to to predict.

        Returns
        -------
        self : returns an instance of self.
        """
        if self.regressor is None:
            regressor = LinearRegression(fit_intercept=False)
        else:
            regressor = clone(self.regressor)

        # make pipeline with spline features
        self.regressor_ = make_pipeline(
            SplineTransformer(
                degree=self.degree,
                n_knots=self.n_knots,
                knots=self.knots,
                extrapolation=self.extrapolation,
                include_bias=False,
            ),
            regressor,
        )

        # we regress index on series values
        # the sklearn X is obtained from the index of y
        # the sklearn y can be taken as the y seen here
        X_sklearn = _get_X_numpy_int_from_pandas(y.index)

        # fit regressor
        self.regressor_.fit(X_sklearn, y)

        if self.prediction_intervals:
            # calculate and save values needed for the prediction interval method
            fitted_values = self.regressor_.predict(X_sklearn)
            residuals = y - fitted_values
            p = self.degree + int(self.with_intercept)
            self.s_squared_ = np.sum(residuals**2) / (len(y) - p)
            self.train_index_ = y.index

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

    def _predict_var(self, fh=None, X=None, cov=False):
        """Compute the variance at each forecast horizon."""
        if not self.prediction_intervals:
            raise ValueError(
                "Prediction intervals were not calculated during fit. \
                Set prediction_intervals=True at initialization."
            )

        # 1. get X (design matrix) and M = (X^t X)^-1
        t_train = _get_X_numpy_int_from_pandas(self.train_index_).flatten()
        X = np.polynomial.polynomial.polyvander(t_train, self.degree)
        if not self.with_intercept:
            X = X[:, 1:]  # remove the column of 1's that handles the intercept

        M = np.linalg.inv(X.T @ X)

        # 2. get time vector t for the forecast horizons
        if fh.is_relative:
            fh = fh.to_absolute(cutoff=self.train_index_[-1])

        t_fh = fh.to_pandas()
        fh_periods = _get_X_numpy_int_from_pandas(t_fh)
        t = np.array(fh_periods)

        # 3. calculate (half-) range of PI (1 + sqrt(x_0^t M x_0)) (up to scaling)
        start = 0 if self.with_intercept else 1
        v = []

        for _, z in enumerate(t):
            w = np.array([z**j for j in range(start, self.degree + 1)])
            v.append(w.T @ M @ w)

        v = (1 + np.array(v)).flatten()  # see Hyndman FPP3 Section 7.9

        l_var = v * self.s_squared_  # see Hyndman FPP3 Section 7.9
        pred_var = pd.DataFrame(l_var, columns=[self._y.name])
        return pred_var

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

        params_list = [
            {},
            {
                "regressor": RandomForestRegressor(),
                "degree": 2,
                "with_intercept": False,
                "prediction_intervals": False,
                "n_knots": 5,
                "knots": "uniform",
                "extrapolation": "constant",
            },
            {
                "regressor": RandomForestRegressor(),
                "degree": 2,
                "with_intercept": True,
                "prediction_intervals": True,
                "n_knots": 5,
                "knots": "uniform",
                "extrapolation": "constant",
            },
            {
                "regressor": RandomForestRegressor(),
                "degree": 2,
                "with_intercept": False,
                "prediction_intervals": True,
                "n_knots": 5,
                "knots": "uniform",
                "extrapolation": "constant",
            },
        ]

        return params_list
