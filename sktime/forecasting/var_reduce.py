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
        from sklearn.base import clone
        from sklearn.linear_model import LinearRegression

        self.regressor = regressor
        self.lags = lags
        if regressor is None:
            self.regressor_ = LinearRegression()
        else:
            self.regressor_ = clone(regressor)

        assert hasattr(self.regressor_, "fit"), "Regressor must have 'fit'"
        assert hasattr(self.regressor_, "predict"), "Regressor must have 'predict'"

        # a dictionary of var_name: fitted regressor, to be filled in during fitting
        self.regressors = {}
        self.coefficients_ = None
        self.intercept_ = None
        self.num_series = None
        super().__init__()

    def prepare_for_fit(self, data, return_as_ndarray=True):
        """
        Prepare the data for VAR fitting though tabularization.

        This function transforms the provided training data into
        a tabular format suitable for regression. Specifically,
        the predictors X consist of lagged values of the multivariate time series, while
        the target variables y are the current values of the multivariate time series.

        Parameters
        ----------
        data : pd.DataFrame
            The input multivariate time series data to be transformed
            with shape (num_samples, num_series)
        return_as_ndarray : bool, optional (default=True)
            If True, returns the transformed data as numpy arrays.
            If False, returns the transformed data as pandas DataFrames.

        Returns
        -------
        X : np.ndarray or pd.DataFrame
            The lagged values as predictors.
            with shape (num_samples, num_series * lags)
        Y : np.ndarray or pd.DataFrame
            The original, unlagged data
            with shape (num_samples, num_series).
        """
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data, columns=self.var_names)

        # Generate the lagged values and rename the columns
        df = pd.concat([data.shift(i) for i in range(self.lags + 1)], axis=1)
        df.columns = [
            f"{col}_lag{i}" for i in range(self.lags + 1) for col in data.columns
        ]
        df = df.dropna()

        # Separate df into X and Y
        Y = df[[f"{col}_lag0" for col in data.columns]]
        Y.columns = data.columns
        X = df.drop(columns=[f"{col}_lag0" for col in data.columns])

        if return_as_ndarray:
            return X.values, Y.values
        else:
            return X, Y

    def prepare_for_predict(self, data, return_as_ndarray=True):
        """
        Prepare the data for VAR prediction through tabularization.

        This function extracts the last `lags` rows of the provided data
        and linearizes them into a single row suitable for input into the regressor.
        The number of lags is inferred from the `self.lags` attribute.

        Parameters
        ----------
        data : pd.DataFrame
            The input multivariate time series data to be transformed
            with shape (num_samples, num_series)
            Usually is training data

        return_as_ndarray : bool, optional (default=True)
            If True, returns the transformed data as a NumPy array.
            If False, returns the transformed data as a pandas DataFrame

        Returns
        -------
        np.ndarray or pd.DataFrame
            A single row of the transformed data, with shape (1, num_series * lags).
            If returned as a DataFrame, the columns are named according to the original
            column names with the lag number appended (e.g., 'A_lag1', 'B_lag2').
        """
        # Take the last lags rows of the input data
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data, columns=self.var_names)
        lags = self.lags
        lagged_data = data.tail(lags)

        # Invert the data
        inverted_data = lagged_data[::-1]

        # Linearize it
        linearized_data = inverted_data.values.flatten().reshape(1, -1)

        # Return as np.array or pd.DataFrame
        if return_as_ndarray:
            return np.array(linearized_data)
        else:
            # Generate column names
            columns = []
            for lag in range(1, lags + 1):
                columns += [f"{col}_lag{lag}" for col in data.columns]
            return pd.DataFrame(linearized_data, columns=columns)

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
        self.var_names = var_names

        X, Y = self.prepare_for_fit(y, return_as_ndarray=False)
        n, k = X.shape
        num_series = y.shape[1]
        self.num_series = num_series

        # Initialize placeholders for coefficients and intercepts
        coefficients = np.zeros((self.lags, num_series, num_series))
        intercepts = np.zeros(num_series)

        # Fit the chosen regressor model for each series
        for i, var_name in enumerate(var_names):
            model = deepcopy(self.regressor_)
            y = Y[var_name]
            model.fit(X, y)
            self.regressors[var_name] = model

            # if the model has `.intercept_` and `.coef_` attributes, extract them
            if hasattr(model, "intercept_") and hasattr(model, "coef_"):
                intercepts[i] = model.intercept_
                # Reshape the coefficients to match with statsmodels VAR
                coefficients[:, :, i] = model.coef_.reshape(self.lags, num_series)
            else:
                pass

        # if the model has `.intercept_` and `.coef_` attributes, extract them
        if hasattr(model, "intercept_") and hasattr(model, "coef_"):
            # Transpose coefficients to match the order of statsmodels' VAR
            self.coefficients_ = np.transpose(coefficients, (0, 2, 1))
            self.intercept_ = intercepts
        else:
            self.coefficients_ = None
            self.intercept_ = None
        return self

    def _predict(self, fh, X=None):
        """Forecast time series at future horizon.

        Parameters
        ----------
        fh : ForecastingHorizon
            The forecasters horizon with the steps ahead to to predict.
            Default is one-step ahead forecast,
            i.e. np.array([1])
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored.

        Returns
        -------
        y_pred : pd.DataFrame
            Series of predicted values
        """
        # Get the last available values for prediction
        y_last = self._y.iloc[-self.lags :]

        # Initialize a list to store predictions
        y_pred = []

        for _ in range(1, len(fh) + 1):
            # Prepare X
            X_last = self.prepare_for_predict(y_last, return_as_ndarray=False)

            # Make timestep specific predictions for each variable using its regressor
            y_pred_step = []
            for var_name in self.var_names:
                model = self.regressors[var_name]
                y_pred_step_var = model.predict(X_last).item()  # is a float
                y_pred_step.append(y_pred_step_var)

            # Convert y_pred_step into a one-row DataFrame
            y_pred_step = pd.DataFrame([y_pred_step], columns=self.var_names)
            y_pred.append(y_pred_step)

            # Append the new predictions (y_pred_step) at the end of y_last
            y_last = pd.concat([y_last, y_pred_step], axis=0)

        # Concatenate the list of Series into the final DataFrame
        y_pred = pd.concat(y_pred)
        y_pred.index = fh.to_absolute_index()

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
