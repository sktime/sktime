# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements VARReduce, a VAR-like model combining tabularization with regression."""

__author__ = ["meraldoantonio"]

import numpy as np
import pandas as pd

from sktime.forecasting.base import BaseForecaster, ForecastingHorizon


class VARReduce(BaseForecaster):
    """
    A flexible VAR-like forecaster that combines tabularization with regression.

    The input data `Y_in` is a multivariate time series data containing `n` time series.
    An example with `n` = 2:

    +---------+-----+-----+
    |  index  | ts1 | ts2 |
    +---------+-----+-----+
    |    1    | 11  | 6   |
    |    2    | 12  | 7   |
    |    3    | 13  | 8   |
    |    4    | 14  | 9   |
    |    5    | 15  | 10  |
    +---------+-----+-----+

    Fitting proceeds in two steps:
    1. Tabularization:
        For each time step and each time series within `Y_in`,
        lagged values `X` are generated. The number of lagged values
        are determined by the `lags` parameters.

        Below is the `X` for the sample `Y_in` with `lags` = 2.
        Note the absence of the earliest 2 timesteps as no
        corresponding lag value is available.

        +---------+----------+----------+----------+----------+
        |  index  | ts1_lag1 | ts2_lag1 | ts1_lag2 | ts2_lag2 |
        +---------+----------+----------+----------+----------+
        |    3    | 12       | 7        | 11       | 6        |
        |    4    | 13       | 8        | 12       | 7        |
        |    5    | 14       | 9        | 13       | 8        |
        +---------+----------+----------+----------+----------+

    2. Regression:
        If a regressor that supports multioutput is chosen, it is
        fitted with `Y_in` as a target and `X` as predictors.

        Otherwise, for each series in `Y_in`, a separate regressor is
        trained using the common predictors `X` created earlier.
        The target is the series itself (i.e. the current values).
        For example, below is the target for `ts1`.
        Note the removal of the first 2 timesteps for alignment with `X`.

        +---------+-----+
        |  index  | ts1 |
        +---------+-----+
        |    3    | 13  |
        |    4    | 14  |
        |    5    | 15  |
        +---------+-----+

    For forecasting, the last `lags` observations in `Y_in` are
    reframed as lagged predictors `X_forecast` and passed to the
    trained regressors to obtain the forecasts.
    `X_forecast` is shown below.

    +---------+----------+----------+----------+----------+
    |  index  | ts1_lag1 | ts2_lag1 | ts1_lag2 | ts2_lag2 |
    +---------+----------+----------+----------+----------+
    |    6    | 15       | 10       | 14       | 9        |
    +---------+----------+----------+----------+----------+

    By default, `LinearRegression` is used, yielding results equivalent to
    a traditional VAR model. Alternatively, any scikit-learn compatible regressor can
    be used to introduce regularization and/or non-linearity.

    For example:
        - VARReduce(regressor = Ridge()) is equivalent to VAR with L2 regularization;
        - VARReduce(regressor = Lasso()) is equivalent to VAR with L1 regularization.
    These two models can be used to incorporate regularization and prevent overfitting
    when the input data contain a large number of individual time series relative to
    data points.

    Parameters
    ----------
    lags : int, optional, default=1
        The number of lagged values to include in the model.
    regressor : object, optional (default=LinearRegression())
        The regressor to use for fitting the model. Must be scikit-learn-compatible.

    Attributes
    ----------
    coefficients_ : np.ndarray, shape (lags, num_series, num_series)
        The estimated coefficients of the model;
        only available if the regressor has `coef_` attribute
    intercept_ : np.ndarray, shape (num_series,)
        The intercept for each time series;
        only available if the regressor has `coef_` attribute
    num_series : int
        The number of time series being modeled.
    var_names : list of str
        The names of the time series being modeled

    Examples
    --------
    >>> from sktime.forecasting.var_reduce import VARReduce
    >>> from sklearn.linear_model import Lasso
    >>> from sktime.datasets import load_longley
    >>> _, y = load_longley()
    >>> forecaster = VARReduce(regressor=Lasso())  # doctest: +SKIP
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

        self.regressor = regressor  # not used/modified
        self.lags = lags
        if regressor is None:
            self.regressor_ = LinearRegression()
        else:
            self.regressor_ = clone(regressor)

        assert hasattr(self.regressor_, "fit"), "Regressor must have 'fit'"
        assert hasattr(self.regressor_, "predict"), "Regressor must have 'predict'"

        # dictionary of var_name: fitted regressor;
        # filled in during fitting
        self.regressors = {}
        self.multioutput_regressor = None
        self.coefficients_ = None
        self.intercept_ = None
        self.num_series = None
        self.var_names = None
        super().__init__()

    def _prepare_for_fit(self, data, return_as_ndarray=True):
        """
        Prepare the data for VAR fitting though tabularization.

        This function transforms the provided training data into
        a tabular format suitable for regression. Specifically,
        the predictors X consist of lagged values of the multivariate time series, while
        the target variables Y are the current values of the multivariate time series.

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

    def _prepare_for_predict(self, data, return_as_ndarray=True):
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

        X, Y = self._prepare_for_fit(y, return_as_ndarray=False)
        n, k = X.shape
        num_series = y.shape[1]
        self.num_series = num_series

        # Initialize placeholders for coefficients and intercepts
        coefficients = np.zeros((self.lags, num_series, num_series))
        intercepts = np.zeros(num_series)

        # Try multioutput fitting
        try:
            model = deepcopy(self.regressor_)
            model.fit(X, Y)
            self.multioutput_regressor = model
            # if the model has `.intercept_` and `.coef_` attributes, extract them
            if hasattr(model, "intercept_") and hasattr(model, "coef_"):
                self.coefficients_ = model.coef_.reshape(
                    (self.num_series, self.lags, self.num_series)
                )
                self.coefficients_ = np.transpose(self.coefficients_, (1, 0, 2))
                self.intercept_ = model.intercept_
        except ValueError:
            # Exception means regressor doesn't support multioutput;
            # for each target series `y` in `Y`, a separate regressor is
            # trained using `y` and the common predictors `X`.
            # TODO: use sklearn's MultiOutputRegressor to unify API
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
        y_pred_outsample = pd.DataFrame()
        y_pred_insample = pd.DataFrame()
        fh_int = fh.to_relative(self.cutoff)

        # ---- insample forecasts  -----
        if fh_int.min() <= 0:
            # Reproduce the original X we used for fitting
            X, _ = self._prepare_for_fit(self._y, return_as_ndarray=False)

            # Initialize a placeholder for in-sample predictons
            self._y_pred_insample = pd.DataFrame(index=self._y.index[self.lags :])

            if self.multioutput_regressor is not None:
                self._y_pred_insample = self.multioutput_regressor.predict(X)
            else:
                # Extract the fitted regressor for each series
                for i, var_name in enumerate(self.var_names):
                    model = self.regressors[var_name]
                    # Use the model to perform in-sample prediction and save them
                    self._y_pred_insample[var_name] = model.predict(X)

                # Create a new `fh` object with only in-sample values
                # and use it to filter previously-created in-sample predictions
                fh_insample = fh_int.to_in_sample(cutoff=self.cutoff)
                insample_index = fh_insample.to_absolute_index(cutoff=self.cutoff)
                y_pred_insample = self._y_pred_insample.loc[insample_index]

        # ---- outsample forecasts ----
        if fh_int.max() > 0:
            # Get the last available values for prediction
            y_last = self._y.iloc[-self.lags :]

            # Initialize a list to store out-of-sample predictions
            y_pred_outsample = []

            # Generate as many future forecast steps as the maximum number in fh_int
            for _ in range(0, fh_int[-1]):
                # Prepare X
                X_last = self._prepare_for_predict(y_last, return_as_ndarray=False)

                if self.multioutput_regressor is not None:
                    y_pred_step = self.multioutput_regressor.predict(X_last)
                    y_pred_step = pd.DataFrame(y_pred_step, columns=self.var_names)
                else:
                    # One timestep prediction for each variable using its regressor
                    y_pred_step = []
                    for var_name in self.var_names:
                        model = self.regressors[var_name]
                        y_pred_step_var = model.predict(X_last).item()  # is a float
                        y_pred_step.append(y_pred_step_var)

                    # Convert y_pred_step into a one-row DataFrame
                    y_pred_step = pd.DataFrame([y_pred_step], columns=self.var_names)

                y_pred_outsample.append(y_pred_step)

                # Append the new predictions (y_pred_step) at the end of y_last
                y_last = pd.concat([y_last, y_pred_step], axis=0)

            # Concatenate the list of Series into the final DataFrame
            y_pred_outsample = pd.concat(y_pred_outsample)

            # Create an index for all generated forecasts
            fh_outsample_all = ForecastingHorizon(
                range(1, fh_int[-1] + 1), is_relative=True, freq=fh.freq
            )
            fh_outsample_all = fh_outsample_all.to_absolute(cutoff=self.cutoff)
            y_pred_outsample.index = fh_outsample_all.to_absolute_index()

            # Create a new `fh` object with only the required outsample indices
            fh_outsample = fh_int.to_out_of_sample(cutoff=self.cutoff)
            fh_outsample = fh_outsample.to_absolute(cutoff=self.cutoff)

            # Use it to filter the previously generated outsample forecasts
            outsample_index = fh_outsample.to_absolute_index()
            y_pred_outsample = y_pred_outsample.loc[outsample_index]

        # Concatenate filtered y_pred_insample and y_pred_outsample
        y_pred = pd.concat([y_pred_insample, y_pred_outsample])
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
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.svm import SVR

        params1 = {"lags": 2, "regressor": RandomForestRegressor()}
        params2 = {"lags": 3, "regressor": SVR(kernel="rbf")}

        return [params1, params2]
