#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file).
"""Unit tests of DateTimeFeatures functionality."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from sktime.datasets import load_airline, load_longley, load_PBS_dataset
from sktime.split import temporal_train_test_split
from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.series.date import DateTimeFeatures
from sktime.utils._testing.hierarchical import _make_hierarchical


@pytest.fixture
def load_split_data():
    y, X = load_longley()
    y_train, y_test, X_train, X_test = temporal_train_test_split(y, X)
    return X_train


@pytest.fixture
def featurescope_step(load_split_data):
    # Test that comprehensive feature_scope works for weeks
    pipe = DateTimeFeatures(
        ts_freq="W", feature_scope="comprehensive", keep_original_columns=True
    )
    test_full_featurescope = pipe.fit_transform(load_split_data)
    return test_full_featurescope


@pytest.fixture
def featurescope_step_output():
    return [
        "GNPDEFL",
        "GNP",
        "UNEMP",
        "ARMED",
        "POP",
        "year",
        "quarter_of_year",
        "month_of_year",
        "week_of_year",
        "month_of_quarter",
        "week_of_quarter",
        "week_of_month",
    ]


@pytest.fixture
def reduced_featurescope_step(load_split_data):
    # Test that minimal feature_scope works for weeks
    pipe = DateTimeFeatures(
        ts_freq="W", feature_scope="minimal", keep_original_columns=True
    )
    test_reduced_featurescope = pipe.fit_transform(load_split_data)
    return test_reduced_featurescope


@pytest.fixture
def reduced_featurescope_step_output():
    return ["GNPDEFL", "GNP", "UNEMP", "ARMED", "POP", "year", "month_of_year"]


@pytest.fixture
def test_changing_frequency_step(load_split_data):
    # Test that comprehensive feature_scope works for months
    pipe = DateTimeFeatures(
        ts_freq="M", feature_scope="comprehensive", keep_original_columns=True
    )
    test_changing_frequency = pipe.fit_transform(load_split_data)
    return test_changing_frequency


@pytest.fixture
def test_changing_frequency_step_output():
    return [
        "GNPDEFL",
        "GNP",
        "UNEMP",
        "ARMED",
        "POP",
        "year",
        "quarter_of_year",
        "month_of_year",
        "month_of_quarter",
    ]


@pytest.fixture
def test_manspec_with_tsfreq_step(load_split_data):
    # Test that manual_selection works for with provided arguments
    # Should ignore feature scope and raise warning for second_of_minute,
    # since ts_freq = "M" is provided.
    # (dummies with frequency higher than ts_freq)
    pipe = DateTimeFeatures(
        ts_freq="M",
        feature_scope="comprehensive",
        manual_selection=["year", "second_of_minute"],
        keep_original_columns=True,
    )
    test_manspec_with_tsfreq = pipe.fit_transform(load_split_data)
    return test_manspec_with_tsfreq


@pytest.fixture
def test_manspec_with_tsfreq_step_output():
    return ["GNPDEFL", "GNP", "UNEMP", "ARMED", "POP", "year", "second_of_minute"]


@pytest.fixture
def test_manspec_wo_tsfreq_step(load_split_data):
    # Test that manual_selection works for with provided arguments
    # Should ignore feature scope and raise no warning for second_of_minute,
    # since ts_freq is not provided.
    pipe = DateTimeFeatures(
        manual_selection=["year", "second_of_minute"], keep_original_columns=True
    )
    test_manspec_wo_tsfreq = pipe.fit_transform(load_split_data)
    return test_manspec_wo_tsfreq


@pytest.fixture
def test_manspec_wo_tsfreq_step_output():
    return ["GNPDEFL", "GNP", "UNEMP", "ARMED", "POP", "year", "second_of_minute"]


@pytest.mark.skipif(
    not run_test_for_class(DateTimeFeatures),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "test_input,expected",
    [
        (
            "featurescope_step",
            "featurescope_step_output",
        ),
        (
            "reduced_featurescope_step",
            "reduced_featurescope_step_output",
        ),
        (
            "test_changing_frequency_step",
            "test_changing_frequency_step_output",
        ),
        (
            "test_manspec_with_tsfreq_step",
            "test_manspec_with_tsfreq_step_output",
        ),
        (
            "test_manspec_wo_tsfreq_step",
            "test_manspec_wo_tsfreq_step_output",
        ),
    ],
)
def test_multivariate_eval(test_input, expected, request):
    """Tests which columns are returned for different arguments.

    For a detailed description of what these arguments do, and how they interact, see
    the docstring of DateTimeFeatures.
    """
    test_input = request.getfixturevalue(test_input).columns.to_list()
    expected = request.getfixturevalue(expected)
    assert len(test_input) == len(expected)
    assert all(a == b for a, b in zip(test_input, expected))


all_args = [
    "Number of airline passengers",
    "year",
    "quarter_of_year",
    "month_of_year",
    "week_of_year",
    "day_of_year",
    "month_of_quarter",
    "week_of_quarter",
    "day_of_quarter",
    "week_of_month",
    "day_of_month",
    "day_of_week",
    "hour_of_day",
    "hour_of_week",
    "minute_of_hour",
    "second_of_minute",
    "millisecond_of_second",
    "is_weekend",
]


@pytest.fixture
def test_univariate_data_step():
    y = load_airline()
    # Test that prior test works for with univariate dataset
    y_train, y_test = temporal_train_test_split(y)
    pipe = DateTimeFeatures(
        manual_selection=["year", "second_of_minute"], keep_original_columns=True
    )
    test_univariate_data = pipe.fit_transform(y_train)
    return test_univariate_data


@pytest.fixture
def test_univariate_data_step_output():
    return ["Number of airline passengers", "year", "second_of_minute"]


@pytest.fixture()
def test_diffdateformat():
    y = load_airline()
    # Test that prior test also works when Index is converted to DateTime index
    y.index = y.index.to_timestamp().astype("datetime64[ns]")
    y_train, y_test = temporal_train_test_split(y)
    pipe = DateTimeFeatures(
        manual_selection=["year", "second_of_minute"], keep_original_columns=True
    )
    test_diffdateformat = pipe.fit_transform(y_train)
    return test_diffdateformat


@pytest.fixture
def test_diffdateformat_output():
    return ["Number of airline passengers", "year", "second_of_minute"]


@pytest.fixture
def test_comprehensive_transform():
    y = load_airline()
    # Test that prior test also works when Index is converted to DateTime index
    y.index = y.index.to_timestamp().astype("datetime64[ns]")
    y_train, y_test = temporal_train_test_split(y)
    pipe = DateTimeFeatures(
        ts_freq="L", feature_scope="comprehensive", keep_original_columns=True
    )
    y_train_t = pipe.fit_transform(y_train)
    return y_train_t


@pytest.fixture
def test_comprehensive_transform_output():
    return all_args


@pytest.fixture
def test_types(test_comprehensive_transform):
    test_types = test_comprehensive_transform.select_dtypes(include=["int64"])
    return test_types


@pytest.fixture
def test_types_output():
    return all_args[1:]


@pytest.mark.skipif(
    not run_test_for_class(DateTimeFeatures),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "test_input,expected",
    [
        (
            "test_univariate_data_step",
            "test_univariate_data_step_output",
        ),
        (
            "test_diffdateformat",
            "test_diffdateformat_output",
        ),
        (
            "test_comprehensive_transform",
            "test_comprehensive_transform_output",
        ),
        (
            "test_types",
            "test_types_output",
        ),
    ],
)
def test_uniivariate_eval(test_input, expected, request):
    """Tests which columns are returned for different arguments.

    For a detailed description of what these arguments do, and how they interact, see
    the docstring of DateTimeFeatures.
    """
    test_input = request.getfixturevalue(test_input).columns.to_list()
    expected = request.getfixturevalue(expected)
    assert len(test_input) == len(expected)
    assert all(a == b for a, b in zip(test_input, expected))


@pytest.fixture
def df_datetime_daily_idx():
    """Create timeseries with Datetime index, daily frequency."""
    return pd.DataFrame(
        data={"y": [1, 1, 1, 1, 1, 1, 1]},
        index=pd.date_range(start="2000-01-01", freq="D", periods=7),
    )


@pytest.fixture
def df_panel():
    """Create panel data of two time series using pd-multiindex mtype."""
    return _make_hierarchical(hierarchy_levels=(2,), min_timepoints=3, max_timepoints=3)


@pytest.mark.skipif(
    not run_test_for_class(DateTimeFeatures),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_manual_selection_is_weekend(df_datetime_daily_idx):
    """Tests that "is_weekend" returns correct result in `manual_selection`."""
    transformer = DateTimeFeatures(
        manual_selection=["is_weekend"], keep_original_columns=True
    )

    Xt = transformer.fit_transform(df_datetime_daily_idx)
    expected = pd.DataFrame(
        data={"y": [1, 1, 1, 1, 1, 1, 1], "is_weekend": [1, 1, 0, 0, 0, 0, 0]},
        index=df_datetime_daily_idx.index,
    )
    assert_frame_equal(Xt, expected)


@pytest.mark.skipif(
    not run_test_for_class(DateTimeFeatures),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_transform_panel(df_panel):
    """Test `.transform()` on panel data."""
    transformer = DateTimeFeatures(
        manual_selection=["year", "month_of_year", "day_of_month"],
        keep_original_columns=True,
    )
    Xt = transformer.fit_transform(df_panel)

    expected = pd.DataFrame(
        index=df_panel.index,
        data={
            "c0": df_panel["c0"].values,
            "year": [2000, 2000, 2000, 2000, 2000, 2000],
            "month_of_year": [1, 1, 1, 1, 1, 1],
            "day_of_month": [1, 2, 3, 1, 2, 3],
        },
    )
    assert_frame_equal(Xt, expected)


@pytest.mark.skipif(
    not run_test_for_class(DateTimeFeatures),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_keep_original_columns(df_panel):
    """Test `.transform()` on panel data."""
    transformer = DateTimeFeatures(
        manual_selection=["year", "month_of_year", "day_of_month"],
        keep_original_columns=False,
    )
    Xt = transformer.fit_transform(df_panel)

    expected = pd.DataFrame(
        index=df_panel.index,
        data={
            "year": [2000, 2000, 2000, 2000, 2000, 2000],
            "month_of_year": [1, 1, 1, 1, 1, 1],
            "day_of_month": [1, 2, 3, 1, 2, 3],
        },
    )
    assert_frame_equal(Xt, expected)


@pytest.mark.skipif(
    not (DateTimeFeatures),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_month_of_quarter(df_panel):
    """Test month_of_quarter for correctness, failure case of bug #4541."""
    y = load_PBS_dataset()

    FEATURES = ["month_of_quarter"]
    t = DateTimeFeatures(manual_selection=FEATURES)

    yt = t.fit_transform(y)[:13]
    expected = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1])
    assert (expected == yt.values).all()


@pytest.mark.skipif(
    not run_test_for_class(DateTimeFeatures),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_manual_selection_hour_of_week(df_panel):
    """Tests that "hour_of_week" returns correct result in `manual_selection`."""
    y = pd.DataFrame(
        data={"y": range(6)},
        index=pd.date_range(start="2023-01-01", freq="H", periods=6),
    )
    transformer = DateTimeFeatures(
        manual_selection=["hour_of_week"], keep_original_columns=True
    )

    yt = transformer.fit_transform(y)
    expected = pd.DataFrame(
        data={"y": range(6), "hour_of_week": [144, 145, 146, 147, 148, 149]},
        index=y.index,
    )
    assert_frame_equal(yt, expected)
