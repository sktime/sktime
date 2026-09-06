# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

"""Recursive Moving Average Forecaster implementation.

Implements a forecaster that predicts using a recursive moving average,
where forecasts beyond horizon 1 use previous forecasts recursively.
"""

__author__ = ["architmittal"]
__all__ = ["RecursiveMovingAverageForecaster"]

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from sktime.forecasting.base._delegate import _DelegatedForecaster
from sktime.forecasting.compose import make_reduction


class _AverageXEstimator(BaseEstimator):
    """Simple sklearn estimator that predicts the mean of input features.

    This is a helper class for RecursiveMovingAverageForecaster.
    It simply returns the row-wise mean of X as predictions.
    """

    def fit(self, X, y):
        """Fit the estimator (no-op, just records fitted state).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values (ignored).

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X, y = check_X_y(X, y, accept_sparse=True)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        """Predict by taking the row-wise mean of X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Row-wise mean of X.
        """
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, "is_fitted_")
        return X.mean(axis=1)


class RecursiveMovingAverageForecaster(_DelegatedForecaster):
    """Recursive Moving Average Forecaster.

    This forecaster predicts future values as the recursive moving average
    of past observations. For each forecast horizon step, it computes the
    mean of the most recent `window_length` values, where values beyond
    the training data use previously computed forecasts.

    This is different from NaiveForecaster(strategy="mean") because:

    - NaiveForecaster(strategy="mean") predicts a constant (the mean of
      the last window) for all future horizons.
    - RecursiveMovingAverageForecaster uses recursive application where
      forecasts for horizon h use forecasts from horizons 1 to h-1.

    This model is commonly used as a simple baseline in demand forecasting,
    where it is sometimes called "Simple Moving Average" (SMA).

    Parameters
    ----------
    window_length : int, default=3
        The number of past observations to use when computing the moving average.
        The forecast for each step is the mean of the previous `window_length` values.

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.moving_average import RecursiveMovingAverageForecaster
    >>> from sktime.split import temporal_train_test_split
    >>>
    >>> y = load_airline()
    >>> y_train, y_test = temporal_train_test_split(y, test_size=12)
    >>> forecaster = RecursiveMovingAverageForecaster(window_length=3)
    >>> forecaster.fit(y_train)
    RecursiveMovingAverageForecaster(...)
    >>> y_pred = forecaster.predict(fh=[1, 2, 3])

    Notes
    -----
    For multi-step ahead forecasting, this forecaster applies the moving average
    recursively. For example, with window_length=3:

    - Forecast for h=1 uses the last 3 actual observations
    - Forecast for h=2 uses the last 2 actual observations + forecast for h=1
    - Forecast for h=3 uses the last 1 actual observation + forecasts for h=1 and h=2
    - And so on...

    This is achieved internally by using `make_reduction` with strategy="recursive".

    References
    ----------
    .. [1] Discussion on GitHub: https://github.com/sktime/sktime/issues/3992

    See Also
    --------
    NaiveForecaster : Naive forecasting strategies (last, mean, drift).
    make_reduction : General reduction of forecasting to regression.
    """

    _tags = {
        "authors": ["architmittal"],
        "maintainers": ["architmittal"],
        "scitype:y": "univariate",
        "y_inner_mtype": "pd.Series",
        "requires-fh-in-fit": False,
        "capability:pred_int": False,
        "capability:missing_values": False,
    }

    # attribute for _DelegatedForecaster, which then delegates
    #     all non-overridden methods are same as of getattr(self, _delegate_name)
    #     see further details in _DelegatedForecaster docstring
    _delegate_name = "estimator_"

    def __init__(self, window_length=3):
        if not isinstance(window_length, int) or window_length < 1:
            raise ValueError(
                f"window_length must be a positive integer >= 1, "
                f"but found: {window_length}"
            )
        self.window_length = window_length

        super().__init__()

    def _fit(self, y, X=None, fh=None):
        """Fit forecaster to training data.

        Parameters
        ----------
        y : pd.Series
            Target time series to which to fit the forecaster.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables (ignored).
        fh : ForecastingHorizon, optional (default=None)
            The forecasting horizon.

        Returns
        -------
        self : reference to self
        """
        self.estimator_ = make_reduction(
            estimator=_AverageXEstimator(),
            scitype="tabular-regressor",
            window_length=self.window_length,
            strategy="recursive",
        )

        self.estimator_.fit(y=y, X=X, fh=fh)

        return self

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
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`.
        """
        return [
            {"window_length": 3},
            {"window_length": 5},
        ]
