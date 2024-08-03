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
    lags : int, optional, default=1
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
        "y_inner_mtype": "pd.DataFrame",
        "X_inner_mtype": "pd.DataFrame",
        "ignores-exogeneous-X": True,
        "requires-fh-in-fit": False,
    }

    def __init__(self, lags=1, regressor=None):
        from sklearn.base import RegressorMixin, clone
        from sklearn.linear_model import LinearRegression

        self.lags = lags
        if regressor is None:
            self.regressor = LinearRegression()
            self.regressor_ = LinearRegression()
        else:
            self.regressor = clone(regressor)
            self.regressor_ = clone(regressor)

        # a dictionary of var_name: fitted regressor, to be filled in during fitting
        self.regressors = {}

        self.coefficients_ = None
        self.intercept_ = None
        self.num_series = None
        super().__init__()

    def prepare_var_data(self, data, return_as_ndarray=True):
        """
        Prepare the data for VAR fitting.

        This function transforms the provided training data into
        a tabular format suitable for regression. Specifically,
        The predictors X consist of lagged values of the multivariate time series, while
        the target variables y are the current, unlagged values of the multivariate time series.

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
        from copy import deepcopy

        var_names = y.columns
        X, y = self.prepare_var_data(y) # note - from this line on, 'y' changes meaning!
        n, k = X.shape
        num_series = y.shape[1]
        self.num_series = num_series

        # Initialize placeholders for coefficients and intercepts
        coefficients = np.zeros((self.lags, num_series, num_series))
        intercepts = np.zeros(num_series)

        # Fit the chosen regressor model for each series
        for i, var_name in enumerate(var_names):
            model = deepcopy(self.regressor_)
            model.fit(X, y[:, i])
            self.regressors[var_name] = model
            intercepts[i] = model.intercept_
            # Reshape the coefficients to match with statsmodels VAR
            coefficients[:, :, i] = model.coef_.reshape(self.lags, num_series)

        # Transpose coefficients to match the desired shape
        self.coefficients_ = np.transpose(coefficients, (0, 2, 1))
        self.intercept_ = intercepts
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
        from sktime.forecasting.base import ForecastingHorizon

        if not isinstance(fh, ForecastingHorizon):
            raise ValueError("`fh` must be a ForecastingHorizon object")

        fh = fh.to_relative(self.cutoff)
        fh = fh.to_numpy()

        # Get the last available values for prediction
        last_values = self._y.iloc[-self.lags:].values.T

        # List to store ALL predictions across all time steps and all time series
        predictions = []

        # Iterate over the steps in fh
        for step in range(1, max(fh) + 1):

            # Prepare X_pred by concatenating the lagged values
            X_pred = np.concatenate(
                [last_values[:, -i].reshape(-1, 1) for i in range(1, self.lags + 1)],
                axis=1,
            ).T

            # List to store predictions for all time series just for this time step
            y_pred_step = []

            # Iterate over each time series and its corresponding regressor
            for var_name, model in self.regressors.items():
                y_pred_step.append(model.predict(X_pred.reshape(1, -1))[0])

            # Convert the list of predictions to a numpy array
            y_pred_step = np.array(y_pred_step)
            predictions.append(y_pred_step)

            # Update last_values by appending the current predictions
            # and removing the oldest set of lagged values
            last_values = np.concatenate([last_values[:, 1:], y_pred_step.reshape(-1, 1)], axis=1)

        predictions = np.array(predictions)

        # Convert predictions to DataFrame
        y_pred = pd.DataFrame(predictions, index=fh, columns=self._y.columns)

        return y_pred

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
        params = {"lags": 2}
        return params
