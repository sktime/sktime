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


class _DummyProbabilisticForecaster(NaiveForecaster):
    """Naive forecaster that advertises probabilistic capability.

    This class is used only for testing descriptor binding in
    GroupbyCategoryForecaster. It sets the relevant capability tags but does not
    rely on skpro distributions, so we can focus the test on method binding.
    """

    _tags = {
        **NaiveForecaster._tags,
        "capability:pred_int": True,
        "capability:pred_int:insample": True,
    }


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


def test_groupby_category_probabilistic_binding():
    """Probabilistic methods are bound correctly on GroupbyCategoryForecaster.

    This specifically verifies that the dynamically attached private methods
    `_predict_interval`, `_predict_var`, and `_predict_proba` are bound methods
    (i.e., receive `self` correctly) and delegate to
    `_iterate_predict_method_over_categories` with the expected arguments.
    """
    # construct a minimal but probabilistic-capable groupby forecaster
    dummy = _DummyProbabilisticForecaster()
    categories = pd.Series([0, 1], index=[("A", "A1"), ("B", "B1")], name="category")
    transformer = PredefinedCategory(transform_output=categories)

    gf = GroupbyCategoryForecaster(
        forecasters={0: dummy, 1: dummy.clone()},
        transformer=transformer,
    )

    # Sanity-check that the probabilistic capability tag is set, which is the
    # precondition for dynamic binding.
    assert gf.get_tags()["capability:pred_int"] is True

    captured_calls = []

    def fake_iter(self, methodname, X=None, **kwargs):
        captured_calls.append(
            {
                "self": self,
                "methodname": methodname,
                "X": X,
                "kwargs": kwargs,
            }
        )
        return "ok"

    # Bind the fake iterator as an instance method
    gf._iterate_predict_method_over_categories = fake_iter.__get__(
        gf, GroupbyCategoryForecaster
    )

    # Call the dynamically attached private probabilistic helpers.
    # If these were raw, unbound functions in the instance dict, each of these
    # calls would raise a TypeError due to missing `self`.
    result_interval = gf._predict_interval(fh="fh_int", X="X_int", coverage=[0.8, 0.9])
    result_var = gf._predict_var(fh="fh_var", X="X_var", cov=True)
    result_proba = gf._predict_proba(fh="fh_proba", X="X_proba", marginal=False)

    assert result_interval == "ok"
    assert result_var == "ok"
    assert result_proba == "ok"

    # Verify that the fake iterator saw the correct `self` and arguments
    assert len(captured_calls) == 3
    for call in captured_calls:
        assert call["self"] is gf

    assert captured_calls[0]["methodname"] == "predict_interval"
    assert captured_calls[0]["X"] == "X_int"
    assert captured_calls[0]["kwargs"] == {"fh": "fh_int", "coverage": [0.8, 0.9]}

    assert captured_calls[1]["methodname"] == "predict_var"
    assert captured_calls[1]["X"] == "X_var"
    assert captured_calls[1]["kwargs"] == {"fh": "fh_var", "cov": True}

    assert captured_calls[2]["methodname"] == "predict_proba"
    assert captured_calls[2]["X"] == "X_proba"
    assert captured_calls[2]["kwargs"] == {"fh": "fh_proba", "marginal": False}
