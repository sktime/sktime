import pandas as pd

from skpro.distributions.base import BaseDistribution
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.conformal import ConformalIntervals
from sktime.forecasting.naive import NaiveForecaster
from sklearn.preprocessing import StandardScaler
from sktime.forecasting.compose import ForecastingPipeline
from sktime.transformations.series.adapt import TabularToSeriesAdaptor

def test_predict_proba_vectorized_panel():
    index = pd.MultiIndex.from_product(
        [
            ["A", "B"],
            pd.period_range("2020-01", periods=12, freq="M"),
        ],
        names=["instance", "time"],
    )

    y = pd.DataFrame(
        {
            "value": [
                10, 12, 13, 15, 14, 16, 18, 17, 19, 21, 20, 22,
                20, 21, 23, 22, 24, 25, 27, 26, 28, 30, 29, 31,
            ]
        },
        index=index,
    )

    forecaster = ConformalIntervals(
        forecaster=NaiveForecaster(strategy="last"),
        initial_window=6,
    )
    forecaster.fit(y)

    fh = ForecastingHorizon([1, 2, 3], is_relative=True)
    expected = forecaster.predict(fh)
    result = forecaster.predict_proba(fh)

    assert isinstance(result, BaseDistribution)
    assert result.index.equals(expected.index)

    expected_df = expected

    assert result.columns.equals(expected_df.columns)

    pd.testing.assert_frame_equal(
        result.mean(),
        expected_df,
    )

    assert result.shape == expected_df.shape

def test_predict_proba_vectorized_hierarchical():
    index = pd.MultiIndex.from_product(
        [
            ["North", "South"],
            ["Store_A", "Store_B"],
            pd.period_range("2020-01", periods=12, freq="M"),
        ],
        names=["region", "store", "time"],
    )

    y = pd.DataFrame(
        {"value": list(range(48))},
        index=index,
    )

    forecaster = ConformalIntervals(
        forecaster=NaiveForecaster(strategy="last"),
        initial_window=6,
    )
    forecaster.fit(y)

    fh = ForecastingHorizon([1, 2, 3], is_relative=True)

    expected = forecaster.predict(fh)
    result = forecaster.predict_proba(fh)

    assert isinstance(result, BaseDistribution)
    assert result.index.equals(expected.index)
    assert result.columns.equals(expected.columns)

    pd.testing.assert_frame_equal(
        result.mean(),
        expected,
    )

    assert result.shape == expected.shape

def test_predict_proba_vectorized_multivariate():
    index = pd.period_range("2020-01", periods=12, freq="M")

    y = pd.DataFrame(
        {
            "value_a": [10, 12, 13, 15, 14, 16, 18, 17, 19, 21, 20, 22],
            "value_b": [20, 21, 23, 22, 24, 25, 27, 26, 28, 30, 29, 31],
        },
        index=index,
    )

    forecaster = ConformalIntervals(
        forecaster=NaiveForecaster(strategy="last"),
        initial_window=6,
    )
    forecaster.fit(y)

    fh = ForecastingHorizon([1, 2, 3], is_relative=True)

    expected = forecaster.predict(fh)
    result = forecaster.predict_proba(fh)

    assert isinstance(result, BaseDistribution)
    assert result.index.equals(expected.index)
    assert result.columns.equals(expected.columns)

    pd.testing.assert_frame_equal(
        result.mean(),
        expected,
    )

    assert result.shape == expected.shape

def test_predict_proba_forecasting_pipeline():
    index = pd.period_range("2020-01", periods=12, freq="M")

    y = pd.Series(
        [10, 12, 13, 15, 14, 16, 18, 17, 19, 21, 20, 22],
        index=index,
    )

    X = pd.DataFrame(
        {"feature": range(12)},
        index=index,
    )

    forecaster = ForecastingPipeline(
        steps=[
            ("scaler", TabularToSeriesAdaptor(StandardScaler())),
            (
                "forecaster",
                ConformalIntervals(
                    forecaster=NaiveForecaster(strategy="last"),
                    initial_window=6,
                ),
            ),
        ]
    )

    forecaster.fit(y, X=X)

    fh = ForecastingHorizon([1, 2, 3], is_relative=True)

    expected = forecaster.predict(fh, X=X)
    result = forecaster.predict_proba(fh, X=X)

    assert isinstance(result, BaseDistribution)
    assert result.index.equals(expected.index)
    expected_df = expected.to_frame()

    assert result.columns.equals(expected_df.columns)

    pd.testing.assert_frame_equal(
        result.mean(),
        expected_df,
    )

    assert result.shape == expected_df.shape

 
