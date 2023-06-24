"""Tests for TimeSince transformer."""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file).

__author__ = ["KishManani"]


import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from sktime.transformations.series.time_since import TimeSince
from sktime.utils._testing.hierarchical import _make_hierarchical


@pytest.fixture
def df_int_idx():
    """Create timeseries with integer index, unequally spaced."""
    return pd.DataFrame(data={"y": [1, 1, 1, 1, 1]}, index=[1, 2, 3, 5, 9])


@pytest.fixture
def df_datetime_15mins_idx():
    """Create timeseries with Datetime index, 15 minute frequency."""
    return pd.DataFrame(
        data={"y": [1, 1, 1, 1, 1]},
        index=pd.date_range(start="2000-01-01", freq="15T", periods=5),
    )


@pytest.fixture
def df_datetime_weekly_wed_idx():
    """Create timeseries with Datetime index, weekly with Wed start frequency."""
    return pd.DataFrame(
        data={"y": [1, 1, 1, 1, 1]},
        index=pd.date_range(start="2000-01-05", freq="W-WED", periods=5),
    )


@pytest.fixture
def df_datetime_monthly_idx():
    """Create timeseries with Datetime index, month start frequency."""
    return pd.DataFrame(
        data={"y": [1, 1, 1, 1, 1]},
        index=pd.date_range(start="2000-01-01", freq="MS", periods=5),
    )


@pytest.fixture
def df_period_monthly_idx():
    """Create timeseries with Period index, monthly frequency."""
    return pd.DataFrame(
        data={"y": [1, 1, 1, 1, 1]},
        index=pd.period_range(start="2000-01-01", freq="M", periods=5),
    )


@pytest.fixture()
def df_datetime_daily_idx_panel():
    """Create panel data of two time series using pd-multiindex mtype."""
    return _make_hierarchical(hierarchy_levels=(2,), min_timepoints=3, max_timepoints=3)


def test_fit_transform_int_idx_output(df_int_idx):
    """Tests that we get the expected outputs."""
    transformer = TimeSince(
        start=None, to_numeric=True, keep_original_columns=False, positive_only=False
    )

    Xt = transformer.fit_transform(df_int_idx)

    expected = pd.DataFrame(
        data={"time_since_1": [0, 1, 2, 4, 8]}, index=df_int_idx.index
    )
    assert_frame_equal(Xt, expected)


def test_fit_transform_datetime_15mins_idx_numeric_output(df_datetime_15mins_idx):
    """Tests that we get the expected outputs."""
    transformer = TimeSince(
        start=None, to_numeric=True, keep_original_columns=False, positive_only=False
    )

    Xt = transformer.fit_transform(df_datetime_15mins_idx)

    expected = pd.DataFrame(
        data={"time_since_2000-01-01 00:00:00": [0, 15, 30, 45, 60]},
        index=df_datetime_15mins_idx.index,
    )
    assert_frame_equal(Xt, expected)


def test_fit_transform_datetime_weekly_wed_idx_numeric_output(
    df_datetime_weekly_wed_idx,
):
    """Tests that we get the expected outputs."""
    transformer = TimeSince(
        start=None, to_numeric=True, keep_original_columns=False, positive_only=False
    )

    Xt = transformer.fit_transform(df_datetime_weekly_wed_idx)

    expected = pd.DataFrame(
        data={"time_since_2000-01-05 00:00:00": [0, 1, 2, 3, 4]},
        index=df_datetime_weekly_wed_idx.index,
    )
    assert_frame_equal(Xt, expected)


def test_fit_transform_datetime_monthly_idx_numeric_output(df_datetime_monthly_idx):
    """Tests that we get the expected outputs."""
    transformer = TimeSince(
        start=None, to_numeric=True, keep_original_columns=False, positive_only=False
    )

    Xt = transformer.fit_transform(df_datetime_monthly_idx)
    expected = pd.DataFrame(
        data={"time_since_2000-01-01 00:00:00": [0, 1, 2, 3, 4]},
        index=df_datetime_monthly_idx.index,
    )
    assert_frame_equal(Xt, expected)


def test_fit_transform_datetime_monthly_idx_datetime_output(df_datetime_monthly_idx):
    """Tests that we get the expected outputs."""
    transformer = TimeSince(
        start=None, to_numeric=False, keep_original_columns=False, positive_only=False
    )

    Xt = transformer.fit_transform(df_datetime_monthly_idx)
    expected = pd.DataFrame(
        data={
            "time_since_2000-01-01 00:00:00": [
                pd.Timedelta(i, unit="D") for i in (0, 31, 60, 91, 121)
            ]
        },
        index=df_datetime_monthly_idx.index,
    )
    assert_frame_equal(Xt, expected)


def test_fit_transform_period_monthly_idx_numeric_output(df_period_monthly_idx):
    """Tests that we get the expected outputs."""
    transformer = TimeSince(
        start=None, to_numeric=True, keep_original_columns=False, positive_only=False
    )

    Xt = transformer.fit_transform(df_period_monthly_idx)
    expected = pd.DataFrame(
        data={"time_since_2000-01": [0, 1, 2, 3, 4]}, index=df_period_monthly_idx.index
    )
    assert_frame_equal(Xt, expected)


def test_fit_transform_period_monthly_idx_period_output(df_period_monthly_idx):
    """Tests that we get the expected outputs."""
    transformer = TimeSince(
        start=None, to_numeric=False, keep_original_columns=False, positive_only=False
    )

    transformer.fit(df_period_monthly_idx)
    # Temporary solution to get this test to pass
    # See: https://github.com/sktime/sktime/pull/3810#issuecomment-1320969799
    transformer.set_config(**{"output_conversion": "off"})
    Xt = transformer.transform(df_period_monthly_idx)
    expected = pd.DataFrame(
        data={
            "time_since_2000-01": [i * pd.offsets.MonthEnd() for i in (0, 1, 2, 3, 4)]
        },
        index=df_period_monthly_idx.index,
    )
    assert_frame_equal(Xt, expected)


def test_fit_transform_datetime_monthly_idx_multiple_starts_output(
    df_datetime_monthly_idx,
):
    """Tests that we get the expected outputs."""
    transformer = TimeSince(
        start=["2000-01-01", "2000-02-01"],
        to_numeric=True,
        keep_original_columns=False,
        positive_only=False,
    )

    Xt = transformer.fit_transform(df_datetime_monthly_idx)
    expected = pd.DataFrame(
        data={
            "time_since_2000-01-01 00:00:00": [0, 1, 2, 3, 4],
            "time_since_2000-02-01 00:00:00": [-1, 0, 1, 2, 3],
        },
        index=df_datetime_monthly_idx.index,
    )
    assert_frame_equal(Xt, expected)


def test_fit_transform_datetime_monthly_idx_multiple_starts_positive_only_output(
    df_datetime_monthly_idx,
):
    """Tests that we get the expected outputs."""
    transformer = TimeSince(
        start=["2000-01-01", "2000-02-01"],
        to_numeric=True,
        keep_original_columns=False,
        positive_only=True,
    )

    Xt = transformer.fit_transform(df_datetime_monthly_idx)
    expected = pd.DataFrame(
        data={
            "time_since_2000-01-01 00:00:00": [0, 1, 2, 3, 4],
            "time_since_2000-02-01 00:00:00": [0, 0, 1, 2, 3],
        },
        index=df_datetime_monthly_idx.index,
    )
    assert_frame_equal(Xt, expected)


def test_fit_transform_datetime_daily_idx_panel_output(
    df_datetime_daily_idx_panel,
):
    """Tests that we get the expected outputs when input is panel data."""
    transformer = TimeSince(
        start=None,
        freq="D",
        to_numeric=True,
        keep_original_columns=False,
        positive_only=False,
    )

    Xt = transformer.fit_transform(df_datetime_daily_idx_panel)

    expected = pd.DataFrame(
        data={
            "time_since_2000-01-01 00:00:00": [0, 1, 2, 0, 1, 2],
        },
        index=df_datetime_daily_idx_panel.index,
    )
    assert_frame_equal(Xt, expected)


def test_fit_transform_datetime_daily_idx_panel_multiple_starts_output(
    df_datetime_daily_idx_panel,
):
    """Tests that we get the expected outputs when input is panel data."""
    transformer = TimeSince(
        start=["2000-01-01", "2000-01-02"],
        freq="D",
        to_numeric=True,
        keep_original_columns=False,
        positive_only=False,
    )

    Xt = transformer.fit_transform(df_datetime_daily_idx_panel)

    expected = pd.DataFrame(
        data={
            "time_since_2000-01-01 00:00:00": [0, 1, 2, 0, 1, 2],
            "time_since_2000-01-02 00:00:00": [-1, 0, 1, -1, 0, 1],
        },
        index=df_datetime_daily_idx_panel.index,
    )
    assert_frame_equal(Xt, expected)
