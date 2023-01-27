# -*- coding: utf-8 -*-
"""Test extraction of features across (shifted) windows."""
__author__ = ["danbartl"]

from sktime.utils.validation._dependencies import _check_soft_dependencies

# HistGradientBoostingRegressor requires experimental flag in old sklearn versions
if _check_soft_dependencies("sklearn<1.0", severity="none"):
    from sklearn.experimental import enable_hist_gradient_boosting  # noqa

import random

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

from sktime.datasets import load_airline
from sktime.datatypes import get_examples
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import make_reduction
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
from sktime.transformations.series.summarize import WindowSummarizer
from sktime.utils._testing.hierarchical import _make_hierarchical

# Load data that will be the basis of tests
y = load_airline()
y_multi = get_examples(mtype="pd-multiindex", as_scitype="Panel")[0]

# y Train will be univariate data set
y_train, y_test = temporal_train_test_split(y)

# y_int = y.copy()
# y_int.index = [i for i in range(len(y_int))]

# y_train_int, y_test_int = temporal_train_test_split(y_int)

# Create traina nd test panel sample data
mi = pd.MultiIndex.from_product([[0], y_train.index], names=["instances", "timepoints"])
y_group1 = pd.DataFrame(y_train.values, index=mi, columns=["y"])

mi = pd.MultiIndex.from_product([[1], y_train.index], names=["instances", "timepoints"])
y_group2 = pd.DataFrame(y_train.values, index=mi, columns=["y"])

y_train_grp = pd.concat([y_group1, y_group2])

mi = pd.MultiIndex.from_product([[0], y_test.index], names=["instances", "timepoints"])
y_group1 = pd.DataFrame(y_test.values, index=mi, columns=["y"])

mi = pd.MultiIndex.from_product([[1], y_test.index], names=["instances", "timepoints"])
y_group2 = pd.DataFrame(y_test.values, index=mi, columns=["y"])

y_test_grp = pd.concat([y_group1, y_group2])

# Get hierachical data
y_train_hier = get_examples(mtype="pd_multiindex_hier")[0]

# Create unbalanced hierachical data
X = y_train_hier.reset_index().copy()
X = X[~((X["bar"] == 2) & (X["foo"] == "b"))]
X = X[["foo", "bar"]].drop_duplicates()

time_names = y_train.index.names[-1]
timeframe = y_train.index.to_frame()

X2 = X.merge(timeframe, how="cross")

freq_inferred = y_train.index.freq

x_names = X.columns
if not isinstance(x_names, list):
    x_names = x_names.to_list()

y_train_reset = y_train.reset_index()

X3 = X2.merge(y_train_reset, on="Period")

freq_inferred = y_train.index.freq

y_train_hier_unequal = X3.groupby(x_names, as_index=True).apply(
    lambda df: df.drop(x_names, axis=1).set_index(time_names).asfreq(freq_inferred)
)

# Create integer index data
y_numeric = y_train.copy()
y_numeric.index = pd.to_numeric(y_numeric.index)


# Get different WindowSummarizer functions
kwargs = WindowSummarizer.get_test_params()[0]
kwargs_alternames = WindowSummarizer.get_test_params()[1]
kwargs_variant = WindowSummarizer.get_test_params()[2]


def check_eval(test_input, expected):
    """Test which columns are returned for different arguments.

    For a detailed description what these arguments do,
    and how theyinteract see docstring of DateTimeFeatures.
    """
    if test_input is not None:
        assert len(test_input) == len(expected)
        assert all([a == b for a, b in zip(test_input, expected)])
    else:
        assert expected is None


@pytest.mark.parametrize(
    "y, index_names",
    [
        (
            y_train_grp,
            ["instances", "timepoints"],
        ),
        (
            y_train,
            [None],
        ),
        (
            y_numeric,
            [None],
        ),
        (
            y_train_hier_unequal,
            ["foo", "bar", "Period"],
        ),
    ],
)
def test_recursive_reduction(y, index_names):
    """Test index columns match input."""
    regressor = make_pipeline(
        RandomForestRegressor(random_state=1),
    )

    forecaster2 = make_reduction(
        regressor,
        scitype="tabular-regressor",
        transformers=[WindowSummarizer(**kwargs, n_jobs=1)],
        window_length=None,
        strategy="recursive",
        pooling="global",
    )

    forecaster2.fit(y, fh=[1, 2])
    y_pred = forecaster2.predict(fh=[1, 2, 12])
    check_eval(y_pred.index.names, index_names)


@pytest.mark.parametrize(
    "y, index_names",
    [
        (
            y_train_grp,
            ["instances", "timepoints"],
        ),
        (
            y_train,
            [None],
        ),
        (
            y_numeric,
            [None],
        ),
        (
            y_train_hier_unequal,
            ["foo", "bar", "Period"],
        ),
    ],
)
def test_direct_reduction(y, index_names):
    """Test index columns match input."""
    regressor = make_pipeline(
        RandomForestRegressor(random_state=1),
    )

    forecaster2 = make_reduction(
        regressor,
        transformers=[WindowSummarizer(**kwargs, n_jobs=1)],
        window_length=None,
        strategy="recursive",
        pooling="global",
    )

    forecaster2.fit(y, fh=[1, 2])
    y_pred = forecaster2.predict(fh=[1, 2, 12])
    check_eval(y_pred.index.names, index_names)


@pytest.mark.parametrize(
    "y, index_names",
    [
        (
            y_train_grp,
            ["instances", "timepoints"],
        ),
        (
            y_train,
            [None],
        ),
        (
            y_numeric,
            [None],
        ),
        (
            y_train_hier_unequal,
            ["foo", "bar", "Period"],
        ),
    ],
)
def test_list_reduction(y, index_names):
    """Test index columns match input."""
    regressor = make_pipeline(
        RandomForestRegressor(random_state=1),
    )

    forecaster2 = make_reduction(
        regressor,
        transformers=[WindowSummarizer(**kwargs), WindowSummarizer(**kwargs_variant)],
        window_length=None,
        strategy="recursive",
        pooling="global",
    )

    forecaster2.fit(y, fh=[1, 2, 12])
    y_pred = forecaster2.predict(fh=[1, 2, 12])
    check_eval(y_pred.index.names, index_names)


@pytest.mark.parametrize(
    "regressor",
    [
        RandomForestRegressor(),
        GradientBoostingRegressor(),
        HistGradientBoostingRegressor(),
    ],
)
def test_equality_transfo_nontranso(regressor):
    """Test that recursive reducers return same results for global / local forecasts."""
    y = load_airline()
    y_train, y_test = temporal_train_test_split(y, test_size=30)
    fh = ForecastingHorizon(y_test.index, is_relative=False)

    lag_vec = [i for i in range(12, 0, -1)]
    kwargs = {
        "lag_feature": {
            "lag": lag_vec,
        }
    }

    for _i in range(1, 5):
        random_int = random.randint(1, 1000)
        regressor.random_state = random_int
        forecaster = make_reduction(
            regressor, window_length=int(12), strategy="recursive"
        )
        forecaster.fit(y_train)
        y_pred = forecaster.predict(fh)
        recursive_without = mean_absolute_percentage_error(
            y_test, y_pred, symmetric=False
        )
        forecaster = make_reduction(
            regressor,
            window_length=None,
            strategy="recursive",
            transformers=[WindowSummarizer(**kwargs, n_jobs=1)],
            pooling="global",
        )

        forecaster.fit(y_train)
        y_pred = forecaster.predict(fh)
        recursive_global = mean_absolute_percentage_error(
            y_test, y_pred, symmetric=False
        )
        np.testing.assert_almost_equal(recursive_without, recursive_global)


def test_nofreq_pass():
    """Test that recursive reducers return same results with / without freq given."""
    regressor = make_pipeline(
        LinearRegression(),
    )

    kwargs = {
        "lag_feature": {
            "lag": [1],
        }
    }

    forecaster_global = make_reduction(
        regressor,
        scitype="tabular-regressor",
        transformers=[WindowSummarizer(**kwargs, n_jobs=1, truncate="bfill")],
        window_length=None,
        strategy="recursive",
        pooling="global",
    )

    forecaster_global_freq = make_reduction(
        regressor,
        scitype="tabular-regressor",
        transformers=[WindowSummarizer(**kwargs, n_jobs=1, truncate="bfill")],
        window_length=None,
        strategy="recursive",
        pooling="global",
    )

    y = _make_hierarchical(
        hierarchy_levels=(100,), min_timepoints=1000, max_timepoints=1000
    )

    y_no_freq = y.reset_index().set_index(["h0", "time"])

    forecaster_global.fit(y)
    forecaster_global_freq.fit(y_no_freq)

    y_pred_global = forecaster_global.predict(fh=[1, 2])
    y_pred_nofreq = forecaster_global_freq.predict(fh=[1, 2])
    np.testing.assert_almost_equal(
        y_pred_global["c0"].values, y_pred_nofreq["c0"].values
    )
