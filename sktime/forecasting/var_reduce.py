# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements ReducedVAR where user can choose the type of Regressor to use."""

# __author__ = [meraldoantonio]

import numpy as np
import pandas as pd

from sktime.forecasting.base import BaseForecaster


class VARReduce(BaseForecaster):
    """
    VAR-like reduced forecaster.

    It reduces multivariate time series to tabular regression and trains a regressor.
    Any scikit-learn compatible regressor can be used; by default, `LinearRegression`,
    is used, making it behave like a traditional VAR. Users can specify other
    regressors such as `ElasticNet` to introduce regularization and potentially
    enhance performance with large datasets or in the presence of multicollinearity.

    Parameters
    ----------
    lags : int
        The number of lagged values to include in the model.
    regressor : object, optional (default=LinearRegression())
        The regressor to use for fitting the model. Must be scikit-learn-compatible.

    Attributes
    ----------
    coefficients : np.ndarray, shape (lags, num_series, num_series)
        The estimated coefficients of the model
    intercept : np.ndarray, shape (num_series,)
        The intercept term of the model for each time series.
    num_series : int
        The number of time series being modeled.

    Examples
    --------
    >>> from sktime.forecasting.var_reduce import VARReduce
    >>> from sktime.datasets import load_longley
    >>> _, y = load_longley()
    >>> forecaster = VARReduce()  # doctest: +SKIP
    >>> forecaster.fit(y)  # doctest: +SKIP
    VARReduce(...)
    >>> y_pred = forecaster.predict(fh=[1,2,3])  # doctest: +SKIP
    """

    _tags = {
        "scitype:y": "both",
        "authors": ["meraldoantonio"],
        "python_dependencies": "cvxpy",
        "y_inner_mtype": "pd.DataFrame",
        "X_inner_mtype": "pd.DataFrame",
        "ignores-exogeneous-X": True,
        "requires-fh-in-fit": False,
    }

    def __init__(self, lags, regressor=None):
        from sklearn.base import RegressorMixin
        from sklearn.linear_model import LinearRegression

        self.lags = lags
        self.regressor = regressor if regressor is not None else LinearRegression()
        assert isinstance(
            self.regressor, RegressorMixin
        ), "The regressor must be a scikit-learn compatible regressor."
        self.coefficients = None
        self.intercept = None
        self.num_series = None
        super().__init__()

    def prepare_var_data(self, data, return_as_ndarray=True):
        """
        Prepare the data for the VAR(p) model by creating lagged predictors.

        Parameters
        ----------
        data : pd.DataFrame
            The input multivariate time series data.
        return_as_ndarray : bool, optional (default=True)
            If True, returns the data as numpy arrays.
            If False, returns the data as pandas DataFrames.

        Returns
        -------
        X : np.ndarray or pd.DataFrame
            The lagged values as predictors.
            with shape (num_samples, num_series * lags)
        y : np.ndarray or pd.DataFrame
            with shape (num_samples, num_series).
        """
        df = pd.concat([data.shift(i) for i in range(self.lags + 1)], axis=1)
        df.columns = [
            f"{col}_lag{i}" for i in range(self.lags + 1) for col in data.columns
        ]
        df = df.dropna()
        y = df[[f"{col}_lag0" for col in data.columns]]
        X = df.drop(columns=[f"{col}_lag0" for col in data.columns])

        if return_as_ndarray:
            return X.values, y.values
        else:
            return X, y

    def _fit(self, y, X=None, fh=None):
        """Fit forecaster to training data.

        Parameters
        ----------
        y : pd.DataFrame
            Guaranteed to have a single column if scitype:y=="univariate".
        fh : ForecastingHorizon, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
        X : pd.DataFrame, optional (default=None)
            Exogenous time series to fit to; will be ignored

        Returns
        -------
        self : reference to self
        """
        X, y = self.prepare_var_data(y)
        n, k = X.shape
        num_series = y.shape[1]
        self.num_series = num_series

        # Initialize placeholders for coefficients and intercepts
        coefficients = np.zeros((self.lags, num_series, num_series))
        intercepts = np.zeros(num_series)

        # Fit the chosen regressor model for each series
        for i in range(num_series):
            model = self.regressor
            model.fit(X, y[:, i])
            intercepts[i] = model.intercept_
            # Reshape the coefficients to match with statsmodels VAR
            coefficients[:, :, i] = model.coef_.reshape(self.lags, num_series)

        # Transpose coefficients to match the desired shape
        self.coefficients = np.transpose(coefficients, (0, 2, 1))
        self.intercept = intercepts
        return self

    def _predict(self, fh, X=None):
        """Forecast time series at future horizon.

        Parameters
        ----------
        fh : ForecastingHorizon
            The forecasting horizon with the steps ahead to to predict.
        X : pd.DataFrame, optional (default=None)
            Exogenous time series

        Returns
        -------
        y_pred : pd.DataFrame
            Point predictions
        """
        # Get the last observed values
        y_last = self._y.iloc[-self.lags :].values
        steps = len(fh)

        # Produce forecasts
        y_pred = self.forecast(y_last, steps)

        # Convert to DataFrame with the correct index
        row_idx = fh.to_absolute(self.cutoff).to_pandas()
        y_pred = pd.DataFrame(y_pred, index=row_idx, columns=self._y.columns)

        return y_pred

    def forecast(self, y, steps):
        """
        Produce steps-ahead forecast, adapted from statsmodels VAR's forecast.

        Parameters
        ----------
        y : np.ndarray (k_ar x neqs)
            The most recent observations to base the forecast on.
        steps : int
            The number of steps to forecast.

        Returns
        -------
        forecasts : np.ndarray (steps x neqs)
            Forecasted values for the next `steps` time points.
        """
        # Extract coefficients and intercept
        coefs = self.coefficients
        intercept = self.intercept

        # Number of lags (p) and number of series (k)
        p = len(coefs)
        k = coefs.shape[1]

        # Ensure there are enough observations to base the forecast on
        if y.shape[0] < p:
            raise ValueError(
                f"y must have at least {p} observations. Got {y.shape[0]}."
            )

        # Initialize the forecast array
        forecasts = np.zeros((steps, k))

        # Add the intercept to the forecast if available
        if intercept is not None:
            forecasts += intercept

        # Iteratively calculate forecasts for each step
        for h in range(1, steps + 1):
            # Initialize the forecast for the current step
            current_forecast = forecasts[h - 1]

            # Sum the contributions from the lagged observations
            for lag in range(1, p + 1):
                # Determine the prior observation based on the lag
                if h - lag <= 0:
                    prior_y = y[h - lag - 1]  # Use the original observations
                else:
                    prior_y = forecasts[
                        h - lag - 1
                    ]  # Use the previously forecasted values

                # Update the current forecast with the contribution from the current lag
                current_forecast += np.dot(coefs[lag - 1], prior_y)

            # Store the forecast for the current step
            forecasts[h - 1] = current_forecast

        return forecasts

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
        from sklearn.linear_model import Ridge

        params = {"lags": 2, "regressor": Ridge(alpha=1.0)}
        return params
