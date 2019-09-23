import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
import pytest

from sktime.forecasters import ARIMAForecaster
from sktime.forecasters.compose import TransformedTargetForecaster, ReducedTimeSeriesRegressionForecaster
from sktime.transformers.forecasting import Deseasonaliser, Detrender
from sktime.datasets import load_shampoo_sales
from sktime.utils.validation.forecasting import check_consistent_time_indices
from sktime.pipeline import Pipeline
from sktime.transformers.compose import Tabulariser


@pytest.mark.parametrize("trend_order", [0, 1, 2])
@pytest.mark.parametrize("arima_order", [(2, 1, 0), (4, 2, 0)])
def test_TransformedTargetForecaster_fit_predict(trend_order, arima_order):
    # forecasting horizon
    fh = np.arange(3) + 1
    len_fh = len(fh)

    # load data and split into train/test series
    y = load_shampoo_sales()
    train = pd.Series([y.iloc[0].iloc[:-len_fh]])
    test = pd.Series([y.iloc[0].iloc[-len_fh:]])

    transformer = Detrender(order=trend_order)
    forecaster = ARIMAForecaster(order=arima_order)

    # using meta-estimator
    forecaster = TransformedTargetForecaster(forecaster, transformer)
    forecaster.fit(train)
    actual = forecaster.predict(fh=fh)
    check_consistent_time_indices(actual, test.iloc[0])

    # checking against manual transform-inverse-transform
    train = pd.DataFrame(train)
    traint = transformer.fit_transform(train)
    traint = traint.iloc[:, 0]

    forecaster.fit(traint)
    pred = forecaster.predict(fh=fh)

    pred = pd.DataFrame(pd.Series([pred]))
    pred = transformer.inverse_transform(pred)
    expected = pred.iloc[0, 0]
    check_consistent_time_indices(expected, test.iloc[0])

    np.testing.assert_allclose(actual, expected)


tsr = Pipeline([  # time series regressor
    ('tabularise', Tabulariser()),
    ('regress', DummyRegressor())
])

@pytest.mark.parametrize("window_length", [3, 5, 7])
@pytest.mark.parametrize("recursive", [True, False])
@pytest.mark.parametrize("fh", [np.array([1]), np.array([1, 2]), np.array([5, 6])])
def test_ReducedForecastingRegressor(window_length, recursive, fh):
    # define setting
    # forecasting horizon
    len_fh = len(fh)

    # load data and split into train/test series
    y = load_shampoo_sales()
    train = pd.Series([y.iloc[0].iloc[:-len_fh]])
    test = pd.Series([y.iloc[0].iloc[-len_fh:]])

    forecaster = ReducedTimeSeriesRegressionForecaster(tsr, window_length=window_length, recursive=recursive)
    # check if error is raised when recursive is set to true but fh is not specified
    if not recursive:
        with pytest.raises(ValueError):
            forecaster.fit(train)

    forecaster.fit(train, fh=fh)
    pred = forecaster.predict(fh=fh)
    assert len(pred) == len(test.iloc[0])


@pytest.mark.parametrize("window_length", [3, 5, 7])
@pytest.mark.parametrize("recursive", [True, False])
@pytest.mark.parametrize("fh", [np.array([1]), np.array([1, 2]), np.array([5, 6])])
def test_ReducedForecastingRegressor_with_TransformedTargetRegressor(window_length, recursive, fh):
    # define setting
    # forecasting horizon
    len_fh = len(fh)

    # load data and split into train/test series
    y = load_shampoo_sales()
    train = pd.Series([y.iloc[0].iloc[:-len_fh]])
    test = pd.Series([y.iloc[0].iloc[-len_fh:]])

    forecaster = ReducedTimeSeriesRegressionForecaster(tsr, window_length=window_length, recursive=recursive)
    transformer = Pipeline([
        ('deseasonalise', Deseasonaliser(sp=12)),
        ('detrend', Detrender(order=1))
    ])
    m = TransformedTargetForecaster(forecaster, transformer)

    # check if error is raised when recursive is set to true but fh is not specified
    if not recursive:
        with pytest.raises(ValueError):
            m.fit(train)

    m.fit(train, fh=fh)
    pred = m.predict(fh=fh)
    assert len(pred) == len(test.iloc[0])

