import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.composition import TransformedTargetForecaster
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.reduction import DirectRegressionForecaster
from sktime.forecasting.reduction import DirectTimeSeriesRegressionForecaster
from sktime.forecasting.reduction import RecursiveRegressionForecaster
from sktime.forecasting.reduction import RecursiveTimeSeriesRegressionForecaster
from sktime.forecasting.theta import ThetaForecaster
from sktime.transformers.compose import Tabulariser
from sktime.transformers.detrend import Detrender
from sktime.utils.data_container import detabularise
from sktime.utils.validation.forecasting import check_y

# look up table for estimators which require arguments during constructions,
# links base classes with the default constructor arguments
REGRESSOR = LinearRegression()

DEFAULT_INSTANTIATIONS = {
    DirectRegressionForecaster: {"regressor": REGRESSOR},
    RecursiveRegressionForecaster: {"regressor": REGRESSOR},
    DirectTimeSeriesRegressionForecaster: {"regressor": make_pipeline(Tabulariser(), REGRESSOR)},
    RecursiveTimeSeriesRegressionForecaster: {"regressor": make_pipeline(Tabulariser(), REGRESSOR)},
    TransformedTargetForecaster: {"forecaster": NaiveForecaster(), "transformer": Detrender(ThetaForecaster())}
}


def _construct_instance(Estimator):
    """Construct Estimator instance if possible"""
    required_parameters = getattr(Estimator, "_required_parameters", [])
    if len(required_parameters) > 0:
        # if estimator requires parameters for construction,
        # set default ones for testing
        if issubclass(Estimator, BaseForecaster):
            kwargs = {}
            if Estimator in DEFAULT_INSTANTIATIONS:
                kwargs = DEFAULT_INSTANTIATIONS[Estimator]
            if not kwargs:
                raise ValueError(f"No default instantiation has been found "
                                 f"for estimator: {Estimator}")
        else:
            raise NotImplementedError()

        estimator = Estimator(**kwargs)

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


def compute_expected_index_from_update_predict(y, fh, step_length):
    """Helper function to compute expected time index from `update_predict`"""
    # time points at which to make predictions
    y = check_y(y)
    index = y.index.values

    predict_at_all = np.arange(index[0] - 1, index[-1], step_length)

    # only predict at time points if all steps in fh can be predicted within y_test
    predict_at = predict_at_all[np.isin(predict_at_all + max(fh), index)]
    n_predict_at = len(predict_at)

    # all time points predicted, including duplicates from overlapping fhs
    broadcast_fh = np.repeat(fh, n_predict_at).reshape(len(fh), n_predict_at)
    points_predicted = predict_at + broadcast_fh

    # return only unique time points
    return np.unique(points_predicted)
