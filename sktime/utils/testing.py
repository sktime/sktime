import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

from sktime.forecasting.base import _BaseForecaster
from sktime.forecasting.reduction import _ReducedTabularRegressorMixin
from sktime.forecasting.reduction import _ReducedTimeSeriesRegressorMixin
from sktime.transformers.compose import Tabulariser
from sktime.utils.data_container import detabularise
from sktime.forecasting.model_selection import SlidingWindowSplitter


def _construct_instance(Estimator):
    """Construct Estimator instance if possible"""
    required_parameters = getattr(Estimator, "_required_parameters", [])
    if len(required_parameters) > 0:
        # if estimator requires parameters for construction,
        # set default ones for testing
        if issubclass(Estimator, _BaseForecaster):
            if "regressor" in required_parameters:
                if issubclass(Estimator, _ReducedTabularRegressorMixin):
                    kwargs = {"regressor": LinearRegression()}

                elif issubclass(Estimator, _ReducedTimeSeriesRegressorMixin):
                    kwargs = {"regressor": make_pipeline(Tabulariser(), LinearRegression())}

            if "cv" in required_parameters:
                kwargs["cv"] = SlidingWindowSplitter(fh=1, window_length=10)
            
            estimator = Estimator(**kwargs)

        else:
            raise NotImplementedError()
    else:
        # construct without kwargs if no parameters are required
        estimator = Estimator()

    return estimator


def generate_df_from_array(array, n_rows=10, n_cols=1):
    return pd.DataFrame([[pd.Series(array) for _ in range(n_cols)] for _ in range(n_rows)],
                        columns=[f'col{c}' for c in range(n_cols)])


def generate_polynomial_series(n, order, coefs=None):
    """Helper function to generate polynomial series of given order and coefficients"""
    if coefs is None:
        coefs = np.ones((order + 1, 1))

    x = np.vander(np.arange(n), N=order + 1).dot(coefs)
    return x.ravel()


def generate_time_series_data_with_trend(n_instances=1, n_timepoints=100, order=0, coefs=None, noise=False):
    """Helper function to generate time series/panel data with polynomial trend"""
    samples = []
    for i in range(n_instances):
        s = generate_polynomial_series(n_timepoints, order=order, coefs=coefs)

        if noise:
            s = s + np.random.normal(size=n_timepoints)

        index = np.arange(n_timepoints)
        y = pd.Series(s, index=index)

        samples.append(y)

    X = pd.DataFrame(samples)
    assert X.shape == (n_instances, n_timepoints)
    return detabularise(X)


def generate_seasonal_time_series_data_with_trend(n_samples=1, n_obs=100, order=0, sp=1, model='additive'):
    """Helper function to generate time series/panel data with polynomial trend and seasonal component"""
    if sp == 1:
        return generate_time_series_data_with_trend(n_instances=n_samples, n_timepoints=n_obs, order=order)

    samples = []
    for i in range(n_samples):
        # coefs = np.random.normal(scale=0.01, size=(order + 1, 1))
        s = generate_polynomial_series(n_obs, order)

        if model == 'additive':
            s[::sp] = s[::sp] + 0.1
        else:
            s[::sp] = s[::sp] * 1.1

        index = np.arange(n_obs)
        y = pd.Series(s, index=index)
        samples.append(y)

    X = pd.DataFrame(samples)
    assert X.shape == (n_samples, n_obs)
    return detabularise(X)
