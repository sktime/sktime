import itertools

import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.compose import GroupbyCategoryForecaster
from sktime.forecasting.naive import NaiveForecaster
from sktime.transformations.base import BaseTransformer


class PredefinedCategory(BaseTransformer):
    _tags = {
        "scitype:transform-input": "Panel",
        "scitype:transform-output": "Panel",
    }

    def __init__(self, transform_output):
        self.transform_output = transform_output
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return self.transform_output


@pytest.fixture
def timeseries_index():
    series_ids = [
        ("A", "A1"),
        ("A", "A2"),
        ("B", "B1"),
        ("B", "B2"),
        ("C", "C1"),
        ("C", "C2"),
    ]
    dates = pd.date_range(start="2020-01-01", periods=10, freq="D")

    multiindex_tuples = [(*a, b) for a, b in itertools.product(series_ids, dates)]

    return pd.MultiIndex.from_tuples(
        multiindex_tuples, names=["level0", "level1", "dates"]
    )


@pytest.fixture
def categories(timeseries_index):
    unique_series = timeseries_index.droplevel(-1).unique()

    categories = pd.DataFrame(
        index=unique_series,
        data={"category": np.arange(len(unique_series)) % 2},
    )
    categories.loc[("C", "C1"), "category"] = 3
    return categories


@pytest.fixture
def timeseries(timeseries_index):
    return pd.DataFrame(
        index=timeseries_index,
        data={"target": np.arange(len(timeseries_index))},
    )


def test_predefined_output(timeseries):
    transform_output = pd.Series(["A"])
    transformer = PredefinedCategory(transform_output=transform_output)

    # Should completely ignore the input and return the predefined output
    output = transformer.fit_transform(X=timeseries)

    assert output.equals(transform_output)


def test_predefined_output_groupby(timeseries, categories):
    """Test if the correct forecasters are fitted for each category"""
    categorizer = PredefinedCategory(transform_output=categories)
    forecaster = GroupbyCategoryForecaster(
        forecasters={
            0: NaiveForecaster(strategy="mean"),
            1: NaiveForecaster(strategy="drift"),
        },
        transformer=categorizer,
        fallback_forecaster=NaiveForecaster(strategy="last"),
    )

    forecaster.fit(timeseries)
    fitted_params = forecaster.get_fitted_params()
    forecasters = fitted_params["forecasters"]
    assert len(forecasters) == 3
    assert forecasters[0].strategy == "mean"
    assert forecasters[1].strategy == "drift"
    assert forecasters[3].strategy == "last"


def test_series_without_panel_level():
    from sktime.datasets import load_airline
    from sktime.transformations.series.adi_cv import ADICVTransformer

    y = load_airline()
    X = pd.DataFrame(0, index=pd.date_range("2020-01-01", periods=10), columns=["B"])

    forecaster = GroupbyCategoryForecaster(
        forecasters={
            0: NaiveForecaster(strategy="mean"),
            1: NaiveForecaster(strategy="drift"),
        },
        transformer=ADICVTransformer(features=["class"]),
        fallback_forecaster=NaiveForecaster(strategy="last"),
    )

    forecaster.fit(y, X)
    y_pred = forecaster.predict(X=X, fh=[1, 2, 3])

    assert y_pred.index.nlevels == 1
