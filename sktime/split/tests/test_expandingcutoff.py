# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for expanding cutoff splitter."""

import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.base import ForecastingHorizon
from sktime.split import ExpandingCutoffSplitter
from sktime.split.tests.test_split import _check_cv
from sktime.utils._testing.hierarchical import _make_hierarchical
from sktime.utils._testing.series import _make_series


def test_expandingcutoff_datelike_index_001():
    """Test datetime index with _check_cv"""
    y = _make_series(n_timepoints=10)
    cutoff = y.index[3]
    fh = ForecastingHorizon([1, 2, 3], freq=y.index.freq)
    cv = ExpandingCutoffSplitter(cutoff=cutoff, fh=fh, step_length=1)
    train_windows, test_windows, cutoffs, n_splits = _check_cv(cv, y)
    np.testing.assert_array_equal(cutoffs, cv.get_cutoffs(y))


def test_expandingcutoff_int_index_002():
    """Test int index with _check_cv"""
    y = _make_series(n_timepoints=10, index_type="int")
    cutoff = y.index[3]
    fh = ForecastingHorizon([1, 2, 3])
    cv = ExpandingCutoffSplitter(cutoff=cutoff, fh=fh, step_length=1)
    train_windows, test_windows, cutoffs, n_splits = _check_cv(cv, y)
    np.testing.assert_array_equal(cutoffs, cv.get_cutoffs(y))


def test_expandingcutoff_raises_invalid_param_combos_003():
    """Test invalid param combo"""
    y = _make_series(n_timepoints=10)
    cutoff = 3
    fh = ForecastingHorizon([1, 2, 3], freq=y.index.freq)
    cv = ExpandingCutoffSplitter(cutoff=cutoff, fh=fh, step_length=1)
    with pytest.raises(TypeError):
        _check_cv(cv, y)


def test_expandingcutoff_splitloc_004():
    """Test split loc"""
    y = _make_series(n_timepoints=10)
    cutoff = y.index[3]
    fh = ForecastingHorizon([1, 2, 3], freq=y.index.freq)
    cv = ExpandingCutoffSplitter(cutoff=cutoff, fh=fh, step_length=1)
    train_test_values = list(cv.split_loc(y))
    last_train, last_test = train_test_values[-1]
    expected_last_train = y.index[0:7]
    expected_last_test = y.index[7:]
    test_cases = [(expected_last_train, last_train), (expected_last_test, last_test)]
    for expected, actual in test_cases:
        np.testing.assert_array_equal(expected.values, actual.values)


def test_expandingcutoff_hiearchical_splitloc_005():
    """Test hiearchical splitloc with datetime"""
    y = _make_hierarchical(
        min_timepoints=6,
        max_timepoints=10,
        same_cutoff=True,
        hierarchy_levels=(1, 2),
        index_type="datetime",
    )
    y_index = y.index.get_level_values(-1)
    cutoff_index = -4
    cutoff = y_index[cutoff_index]
    fh = ForecastingHorizon([1, 2, 3])
    cv = ExpandingCutoffSplitter(cutoff=cutoff, fh=fh, step_length=1)

    train_test = list(cv.split_loc(y))

    y_train_expected = (
        y.groupby(level=-2)
        .apply(lambda row: row[0:cutoff_index])
        .index.get_level_values(-1)
    )
    y_test_expected = (
        y.groupby(level=-2)
        .apply(lambda row: row[cutoff_index : cutoff_index + fh[-1]])
        .index.get_level_values(-1)
    )
    pd.testing.assert_index_equal(
        y_train_expected, train_test[0][0].get_level_values(-1)
    )
    pd.testing.assert_index_equal(
        y_test_expected, train_test[0][1].get_level_values(-1)
    )


def test_expandingcutoff_hiearchical_forecastbylevel_006():
    """Test hiearchical with forecast by level"""
    from sktime.forecasting.compose import ForecastByLevel
    from sktime.forecasting.model_selection import ForecastingGridSearchCV
    from sktime.forecasting.naive import NaiveForecaster
    from sktime.utils._testing.hierarchical import _make_hierarchical

    y = _make_hierarchical(
        min_timepoints=6,
        max_timepoints=10,
        same_cutoff=True,
        hierarchy_levels=(1, 2),
        index_type="datetime",
    )
    y_index = y.index.get_level_values(-1)
    last_date = y_index.max()
    cutoff_index = -4
    cutoff = y_index[cutoff_index]
    fh = ForecastingHorizon([1, 2, 3])
    cv = ExpandingCutoffSplitter(cutoff=cutoff, fh=fh, step_length=1)
    forecaster = NaiveForecaster()
    param_grid = {"strategy": ["last", "mean", "drift"]}
    gscv = ForecastByLevel(
        ForecastingGridSearchCV(forecaster=forecaster, param_grid=param_grid, cv=cv)
    )
    gscv.fit(y, fh=fh)
    y_pred = gscv.predict()
    expected_last_forecast_date = last_date + pd.Timedelta(days=fh[-1])
    actual_last_forecast_date = y_pred.index.max()
    assert expected_last_forecast_date, actual_last_forecast_date


def test_expandingcutoff_fh_list_007():
    """Test fh as list with _check_cv"""
    y = _make_series(n_timepoints=10)
    cutoff = y.index[3]
    fh = [1, 2, 3]
    cv = ExpandingCutoffSplitter(cutoff=cutoff, fh=fh, step_length=1)
    train_windows, test_windows, cutoffs, n_splits = _check_cv(cv, y)
    np.testing.assert_array_equal(cutoffs, cv.get_cutoffs(y))
