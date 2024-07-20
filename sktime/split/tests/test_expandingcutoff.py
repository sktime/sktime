# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for expanding cutoff splitter."""

import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.base import ForecastingHorizon
from sktime.split import ExpandingCutoffSplitter
from sktime.split.tests.test_split import _check_cv
from sktime.tests.test_switch import run_test_for_class
from sktime.utils._testing.hierarchical import _make_hierarchical
from sktime.utils._testing.series import _make_series


@pytest.mark.skipif(
    not run_test_for_class(ExpandingCutoffSplitter),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_expandingcutoff_datelike_index_001():
    """Test datetime index with _check_cv"""
    y = _make_series(n_timepoints=10, random_state=42)
    cutoff = y.index[3]
    fh = ForecastingHorizon([1, 2, 3], freq=y.index.freq)
    cv = ExpandingCutoffSplitter(cutoff=cutoff, fh=fh, step_length=1)
    train_windows, test_windows, cutoffs, n_splits = _check_cv(cv, y)
    np.testing.assert_array_equal(cutoffs, cv.get_cutoffs(y))


@pytest.mark.skipif(
    not run_test_for_class(ExpandingCutoffSplitter),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_expandingcutoff_int_index_002():
    """Test int index with _check_cv"""
    y = _make_series(n_timepoints=10, index_type="int", random_state=42)
    cutoff = y.index[3]
    fh = ForecastingHorizon([1, 2, 3])
    cv = ExpandingCutoffSplitter(cutoff=cutoff, fh=fh, step_length=1)
    train_windows, test_windows, cutoffs, n_splits = _check_cv(cv, y)
    np.testing.assert_array_equal(cutoffs, cv.get_cutoffs(y))


@pytest.mark.skipif(
    not run_test_for_class(ExpandingCutoffSplitter),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_expandingcutoff_ytype_cutofftype_combos_003a():
    """Test invalid param combo"""
    # Datetime cutoff
    i = 3
    y1 = _make_series(n_timepoints=10, index_type="datetime", random_state=42)
    fh = ForecastingHorizon([1, 2, 3])
    cutoffs = [-7, y1.index[i], 5]
    step_lengths = [1, 2, 3]
    for cutoff in cutoffs:
        for step_length in step_lengths:
            cv1 = ExpandingCutoffSplitter(cutoff=cutoff, fh=fh, step_length=step_length)
            _check_cv(cv1, y1)


@pytest.mark.skipif(
    not run_test_for_class(ExpandingCutoffSplitter),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_expandingcutoff_ytype_cutofftype_combos_003b():
    """Test invalid param combo"""
    # Datetime cutoff
    y1 = _make_series(n_timepoints=10, index_type="int", random_state=42)
    fh = ForecastingHorizon([1, 2, 3])
    cutoffs = [-7, y1.index[3]]
    step_lengths = [1, 2, 3]
    for cutoff in cutoffs:
        for step_length in step_lengths:
            cv1 = ExpandingCutoffSplitter(cutoff=cutoff, fh=fh, step_length=step_length)
            _check_cv(cv1, y1)


@pytest.mark.skipif(
    not run_test_for_class(ExpandingCutoffSplitter),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_expandingcutoff_ytype_cutofftype_combos_003c():
    """Test invalid param combo"""
    # Datetime cutoff
    y1 = _make_series(n_timepoints=10, index_type="int", random_state=42)
    fh = ForecastingHorizon([1, 2, 3])
    # Y-index is int but cutoff is datetime
    with pytest.raises(TypeError):
        cv1 = ExpandingCutoffSplitter(
            cutoff=pd.Timestamp("2000-01-01"), fh=fh, step_length=1
        )
        _check_cv(cv1, y1)


@pytest.mark.skipif(
    not run_test_for_class(ExpandingCutoffSplitter),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_expandingcutoff_splitloc_004():
    """Test split loc"""
    y = _make_series(n_timepoints=10, random_state=42)
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


@pytest.mark.skipif(
    not run_test_for_class(ExpandingCutoffSplitter),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_expandingcutoff_hiearchical_splitloc_005():
    """Test hierarchical splitloc with datetime"""
    y = _make_hierarchical(
        min_timepoints=6,
        max_timepoints=10,
        same_cutoff=True,
        hierarchy_levels=(1, 2),
        index_type="datetime",
        random_state=42,
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


@pytest.mark.skipif(
    not run_test_for_class(ExpandingCutoffSplitter),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_expandingcutoff_hiearchical_forecastbylevel_006():
    """Test hierarchical with forecast by level"""
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
        random_state=42,
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


@pytest.mark.skipif(
    not run_test_for_class(ExpandingCutoffSplitter),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_expandingcutoff_fh_list_007():
    """Test fh as list with _check_cv"""
    y = _make_series(n_timepoints=10, random_state=42)
    cutoff = y.index[3]
    fh = [1, 2, 3]
    cv = ExpandingCutoffSplitter(cutoff=cutoff, fh=fh, step_length=1)
    train_windows, test_windows, cutoffs, n_splits = _check_cv(cv, y)
    np.testing.assert_array_equal(cutoffs, cv.get_cutoffs(y))


@pytest.mark.skipif(
    not run_test_for_class(ExpandingCutoffSplitter),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_expanding_cutoff_period_008():
    date_range = pd.date_range(
        start=pd.Timestamp("2020-Q1"), end=pd.Timestamp("2021-Q4"), freq="QS"
    )
    y = pd.DataFrame({"date": date_range})
    y = y.reset_index()
    y = y.set_index("date")
    y.index = pd.PeriodIndex(y.index, freq="Q")

    cutoff = pd.Period("2021-Q1")
    fh = ForecastingHorizon([1, 2, 3])
    cv = ExpandingCutoffSplitter(cutoff=cutoff, fh=fh, step_length=2)
    _check_cv(cv, y)

    # pandas periods and timestamps are both datelike but don't work directly
    cutoff = pd.Timestamp("2021-Q1")
    with pytest.raises(TypeError):
        cv = ExpandingCutoffSplitter(cutoff=cutoff, fh=fh, step_length=1)
        _check_cv(cv, y)


def _make_splits_listlike(splits):
    splits_new = []
    for train, test in splits:
        splits_new.append([train.tolist(), test.tolist()])
    return splits_new


@pytest.mark.skipif(
    not run_test_for_class(ExpandingCutoffSplitter),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_expanding_cutoff_docstring_examples():
    date_range = pd.date_range(start="2020-Q1", end="2021-Q3", freq="QS")
    y = pd.DataFrame(index=pd.PeriodIndex(date_range, freq="Q"))
    cv1 = ExpandingCutoffSplitter(cutoff=pd.Period("2021-Q1"), fh=[1, 2], step_length=1)
    splits1 = list(cv1.split(y))

    cv2 = ExpandingCutoffSplitter(cutoff=4, fh=[1, 2], step_length=1)
    splits2 = list(cv2.split(y))

    cv3 = ExpandingCutoffSplitter(cutoff=-3, fh=[1, 2], step_length=1)
    splits3 = list(cv3.split(y))

    assert (
        _make_splits_listlike(splits1)
        == _make_splits_listlike(splits2)
        == _make_splits_listlike(splits3)
    )
