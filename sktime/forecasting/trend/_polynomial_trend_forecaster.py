# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements PolynomialTrendForecaster."""

__author__ = ["tensorflow-as-tf", "mloning", "aiwalter", "fkiraly"]
__all__ = ["PolynomialTrendForecaster"]

import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.trend._util import _get_X_numpy_int_from_pandas


class PolynomialTrendForecaster(BaseForecaster):
    r"""Forecast time series data with a polynomial trend.

    Uses an ``sklearn`` regressor specified by the ``regressor`` parameter
    to perform regression on time series values against their corresponding indices,
    after extraction of polynomial features.
    Same as ``TrendForecaster`` where ``regressor`` is pipelined with transformation
    step ``PolynomialFeatures(degree, with_intercept)`` applied to time index,
    at the start.

    In ``fit``, for input time series :math:`(v_i, p(t_i)), i = 1, \dots, T`,
    where :math:`v_i` are values, :math:`t_i` are time stamps,
    and :math:`p` is the polynomial feature transform with degree ``degree``,
    and with/without intercept depending on ``with_intercept``,
    fits an ``sklearn`` model :math:`v_i = f(p(t_i)) + \epsilon_i`, where :math:`f` is
    the model fitted when ``regressor.fit`` is passed ``X`` = vector of :math:`p(t_i)`,
    and ``y`` = vector of :math:`v_i`.

    In ``predict``, for a new time point :math:`t_*`, predicts :math:`f(p(t_*))`,
    where :math:`f` is the function as fitted above in ``fit``,
    and :math:`p` is the same polynomial feature transform as above.

    Default for ``regressor`` is linear regression = ``sklearn`` ``LinearRegression``,
    with default parameters. Default for ``degree`` is 1.

    If time stamps are ``pd.DatetimeIndex``, fitted coefficients are in units
    of days since start of 1970. If time stamps are ``pd.PeriodIndex``,
    coefficients are in units of (full) periods since start of 1970.

    Parameters
    ----------
    regressor : sklearn regressor estimator object, default = None
        Define the regression model type. If not set, will default to
        sklearn.linear_model.LinearRegression
    degree : int, default = 1
        Degree of polynomial function
    with_intercept : bool, default=True
        If true, then include a feature in which all polynomial powers are
        zero. (i.e. a column of ones, acts as an intercept term in a linear
        model)

    Attributes
    ----------
    regressor_ : sklearn regressor estimator object
        The fitted regressor object.
        This is a fitted ``sklearn`` pipeline with steps
        ``PolynomialFeatures(degree, with_intercept)``,
        followed by a clone of ``regressor``.

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.trend import PolynomialTrendForecaster
    >>> y = load_airline()
    >>> forecaster = PolynomialTrendForecaster(degree=1)
    >>> forecaster.fit(y)
    PolynomialTrendForecaster(...)
    >>> y_pred = forecaster.predict(fh=[1,2,3])
    """

    _tags = {
        "authors": ["tensorflow-as-tf", "mloning", "aiwalter", "fkiraly"],
        "maintainers": ["tensorflow-as-tf"],
        "ignores-exogeneous-X": True,
        "requires-fh-in-fit": False,
        "handles-missing-data": False,
        "capability:pred_var": True,
        "capability:pred_int": True,
    }

    def __init__(self, regressor=None, degree=1, with_intercept=True, prediction_intervals=False):
        self.regressor = regressor
        self.degree = degree
        self.with_intercept = with_intercept
        self.regressor_ = self.regressor
        self.prediction_intervals = prediction_intervals   
            # prediction_intervals : bool, default=False
            # By default, the extra information needed to later generate the prediction intervals is not
            # calculated. If set to True, the extra information is calculated and stored in the forecaster.
        super().__init__()

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
        import numpy as np
        
        # for default regressor, set fit_intercept=False as we generate a
        # dummy variable in polynomial features
        if self.regressor is None:
            regressor = LinearRegression(fit_intercept=False)
        else:
            regressor = clone(self.regressor)

        # make pipeline with polynomial features
        self.regressor_ = make_pipeline(
            PolynomialFeatures(degree=self.degree, include_bias=self.with_intercept),
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
            p = self.degree + int(self.get_params()['with_intercept'])
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
        """Computes the variance at each forecast horizon"""
        import numpy as np
        from scipy.stats import norm
        
        if self.prediction_intervals is False:
            raise ValueError("Prediction intervals were not calculated during fit. Set prediction_intervals=True at initialization.")
        
        # 1. get forecasts
        pred_values = self.predict(fh)
        
        # 2. get X (design matrix) and M = (X^t X)^-1
        t_train = _get_X_numpy_int_from_pandas(self.train_index_).flatten()
        X = np.polynomial.polynomial.polyvander(t_train, self.degree)
        if not self.get_params()['with_intercept']:
            X = X[:, 1:] # remove the column of 1's that handles the intercept
        
        M = np.linalg.inv(X.T @ X)

        # 3. get time vector t for the forecast horizons
        if fh.is_relative:
            fh = fh.to_absolute(cutoff = self.train_index_[-1])
            
        t_fh = fh.to_pandas()
        fh_periods = _get_X_numpy_int_from_pandas(t_fh)
        t = np.array(fh_periods)
      
        # 4. calculate (half-) range of prediction interval (1 + sqrt(x_0^t M x_0)) (up to scaling)
        start = 0 if self.get_params()['with_intercept'] else 1
        v = []
        for i in range(len(t)):
            z = t[i]
            w = np.array([z**j for j in range(start, self.degree + 1)])
            v.append(w.T @ M @ w)

        v = np.sqrt(1 + np.array(v)).flatten() # see Hyndman FPP3 Section 7.9
        
        s = np.sqrt(self.s_squared_).item()
        
        l_var = (1 + np.array(v)) * self.s_squared_  # see Hyndman FPP3 Section 7.9
        pred_var = pd.DataFrame(l_var, columns=[self._y.name])
        return pred_var        

    def _predict_quantiles(self, fh, X=None, alpha=[0.5]):
        """Compute/return prediction quantiles for a forecast.

        private _predict_quantiles containing the core logic,
            called from predict_quantiles and default _predict_interval

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon
            The forecasting horizon with the steps ahead to to predict.
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Exogeneous time series to predict from.
        alpha : list of float, optional (default=[0.5])
            A list of probabilities at which quantile forecasts are computed.

        Returns
        -------
        quantiles : pd.DataFrame
            Column has multi-index: first level is variable name from y in fit,
                second level being the values of alpha passed to the function.
            Row index is fh, with additional (upper) levels equal to instance levels,
                    from y seen in fit, if y_inner_mtype is Panel or Hierarchical.
            Entries are quantile forecasts, for var in col index,
                at quantile probability in second col index, for the row index.
        """
        
        import numpy as np
        from scipy.stats import norm
        
        #get forecasts
        pred_values = self.predict(fh).values.flatten()
        
        l_var = self._predict_var(fh=fh, X=X, cov=False)
        
        all_quantiles = []
        
        for a in alpha:
            z_alpha = norm.ppf(a)
            l_quant = np.sqrt(l_var.values) * z_alpha
            all_quantiles.append(l_quant.flatten() + pred_values)
            
        df = pd.DataFrame(all_quantiles).transpose()
        df.index = fh.to_absolute(self.cutoff).to_pandas()
        
        multi_index = pd.MultiIndex.from_product([["y"], alpha], names=['variable', 'alpha'])
        df.columns = multi_index

        return df
    
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
            },
        ]

        return params_list
