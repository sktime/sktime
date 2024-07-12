# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements Regularized (L1/L2) VAR using cvxpy."""

# __author__ = [meraldoantonio]

import numpy as np
import pandas as pd

from sktime.forecasting.base import BaseForecaster


class VARReduce(BaseForecaster):
    """Custom forecaster using VAR with regularization.

    Parameters
    ----------
    lags : int
        The number of lagged values to include in the model.
    L1_penalty : float, optional (default=0.0)
        The L1 regularization penalty.
    L2_penalty : float, optional (default=0.0)
        The L2 regularization penalty.
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

    def __init__(self, lags, L1_penalty=0.0, L2_penalty=0.0):
        self.lags = lags
        self.L1_penalty = L1_penalty
        self.L2_penalty = L2_penalty
        self.coefficients = None
        self.intercept = None
        self.num_series = None
        super().__init__()

    def create_var_data(self, data):
        """
        Prepare the data for VAR(p) model.

        Parameters
        ----------
        data : pd.DataFrame
            The input time series data.

        Returns
        -------
        X : np.ndarray (n_samples, n_features * lags)
            The lagged values as predictors.
        y : np.ndarray (n_samples, n_features)
            The current values as response variable.
        """
        df = pd.concat([data.shift(i) for i in range(self.lags + 1)], axis=1)
        df.columns = [
            f"{col}_lag{i}" for i in range(self.lags + 1) for col in data.columns
        ]
        df = df.dropna()
        y = df[[f"{col}_lag0" for col in data.columns]].values
        X = df.drop(columns=[f"{col}_lag0" for col in data.columns]).values
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
            Exogeneous time series to fit to.

        Returns
        -------
        self : reference to self
        """
        import cvxpy as cp

        X, y = self.create_var_data(y)
        n, k = X.shape
        num_series = y.shape[1]
        self.num_series = num_series

        # Define the optimization problem
        coefficients = cp.Variable((k, num_series))
        intercept = cp.Variable((1, num_series))

        # Define the objective function with regularization
        objective = cp.Minimize(
            cp.sum_squares(y - X @ coefficients - intercept)
            + self.L1_penalty * cp.norm(coefficients, 1)
            + self.L2_penalty * cp.norm(coefficients, "fro")
        )
        problem = cp.Problem(objective)
        problem.solve()

        # Store the estimated coefficients and intercept
        # Reshape and rearrange the coefficients to match with statsmodels VAR
        self.coefficients = coefficients.value.reshape(
            self.lags, num_series, num_series
        )
        self.coefficients = np.transpose(self.coefficients, (0, 2, 1))
        self.intercept = intercept.value.reshape(num_series)

        self._is_fitted = True
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
        params = {"lags": 2, "L1_penalty": 0.1, "L2_penalty": 0.1}
        return params
