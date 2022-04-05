# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Test functionality of time series plotting functions."""

import re

import numpy as np
import pandas as pd
import pytest

from sktime.datasets import load_airline
from sktime.utils.plotting import plot_correlations, plot_lags, plot_series
from sktime.utils.validation._dependencies import _check_soft_dependencies
from sktime.utils.validation.series import VALID_DATA_TYPES

ALLOW_NUMPY = False
y_airline = load_airline()
y_airline_true = y_airline.iloc[y_airline.index < "1960-01"]
y_airline_test = y_airline.iloc[y_airline.index >= "1960-01"]
series_to_test = [y_airline, (y_airline_true, y_airline_test)]
invalid_input_types = [
    y_airline.values,
    pd.DataFrame({"y1": y_airline, "y2": y_airline}),
    "this_is_a_string",
]

# can be used with pytest.mark.parametrize to check plots that accept
# univariate series
univariate_plots = [plot_correlations, plot_lags]


# Need to use _plot_series to make it easy for test cases to pass either a
# single series or a tuple of multiple series to be unpacked as argss
def _plot_series(series, ax=None, **kwargs):
    if isinstance(series, tuple):
        return plot_series(*series, ax=ax, **kwargs)
    else:
        return plot_series(series, ax=ax, **kwargs)


# Can be used with pytest.mark.parametrize to run a test on all plots
all_plots = univariate_plots + [_plot_series]


@pytest.fixture
def valid_data_types():
    """Filter valid data types for those that work with plotting functions."""
    valid_data_types = tuple(filter(lambda x: x is not np.ndarray, VALID_DATA_TYPES))
    return valid_data_types


# todo: remove skip when issue #2066 has been fixed
@pytest.mark.skip(reason="sporadic failure on win CI/CD, see issue 2066")
@pytest.mark.parametrize("series_to_plot", series_to_test)
def test_plot_series_runs_without_error(series_to_plot):
    """Test whether plot_series runs without error."""
    _check_soft_dependencies("matplotlib")
    import matplotlib.pyplot as plt

    _plot_series(series_to_plot)
    plt.gcf().canvas.draw_idle()

    # Test with labels specified
    if isinstance(series_to_plot, pd.Series):
        labels = ["Series 1"]
    elif isinstance(series_to_plot, tuple):
        labels = [f"Series {i+1}" for i in range(len(series_to_plot))]
    _plot_series(series_to_plot, labels=labels)
    plt.gcf().canvas.draw_idle()
    plt.close()


# todo: remove skip when issue #2066 has been fixed
@pytest.mark.skip(reason="sporadic failure on win CI/CD, see issue 2066")
@pytest.mark.parametrize("series_to_plot", invalid_input_types)
def test_plot_series_invalid_input_type_raises_error(series_to_plot, valid_data_types):
    """Tests whether plot_series raises error for invalid input types."""
    series_type = type(series_to_plot)

    if not isinstance(series_to_plot, (pd.Series, pd.DataFrame)):
        match = (
            rf"input must be a one of {valid_data_types}, but found type: {series_type}"
        )
        with pytest.raises((TypeError), match=re.escape(match)):
            _plot_series(series_to_plot)
    else:
        match = "input must be univariate, but found 2 variables."
        with pytest.raises(ValueError, match=match):
            _plot_series(series_to_plot)


# todo: remove skip when issue #2066 has been fixed
@pytest.mark.skip(reason="sporadic failure on win CI/CD, see issue 2066")
@pytest.mark.parametrize(
    "series_to_plot", [(y_airline_true, y_airline_test.reset_index(drop=True))]
)
def test_plot_series_with_unequal_index_type_raises_error(
    series_to_plot, valid_data_types
):
    """Tests whether plot_series raises error for series with unequal index."""
    match = "Found series with inconsistent index types"
    with pytest.raises(TypeError, match=match):
        _plot_series(series_to_plot)


# todo: remove skip when issue #2066 has been fixed
@pytest.mark.skip(reason="sporadic failure on win CI/CD, see issue 2066")
@pytest.mark.parametrize("series_to_plot", series_to_test)
def test_plot_series_invalid_marker_kwarg_len_raises_error(series_to_plot):
    """Tests whether plot_series raises error for inconsistent series/markers."""
    match = """There must be one marker for each time series,
                but found inconsistent numbers of series and
                markers."""
    with pytest.raises(ValueError, match=match):
        # Generate error by creating list of markers with length that does
        # not match input number of input series
        if isinstance(series_to_plot, pd.Series):
            markers = ["o", "o"]
        elif isinstance(series_to_plot, tuple):
            markers = ["o" for _ in range(len(series_to_plot) - 1)]

        _plot_series(series_to_plot, markers=markers)


# todo: remove skip when issue #2066 has been fixed
@pytest.mark.skip(reason="sporadic failure on win CI/CD, see issue 2066")
@pytest.mark.parametrize("series_to_plot", series_to_test)
def test_plot_series_invalid_label_kwarg_len_raises_error(series_to_plot):
    """Tests whether plot_series raises error for inconsistent series/labels."""
    match = """There must be one label for each time series,
                but found inconsistent numbers of series and
                labels."""
    with pytest.raises(ValueError, match=match):
        # Generate error by creating list of labels with length that does
        # not match input number of input series
        if isinstance(series_to_plot, pd.Series):
            labels = ["Series 1", "Series 2"]
        elif isinstance(series_to_plot, tuple):
            labels = [f"Series {i}" for i in range(len(series_to_plot) - 1)]

        _plot_series(series_to_plot, labels=labels)


# todo: remove skip when issue #2066 has been fixed
@pytest.mark.skip(reason="sporadic failure on win CI/CD, see issue 2066")
@pytest.mark.parametrize("series_to_plot", series_to_test)
def test_plot_series_output_type(series_to_plot):
    """Tests whether plot_series returns plt.fig and plt.ax."""
    _check_soft_dependencies("matplotlib")
    import matplotlib.pyplot as plt

    # Test output case where kwarg ax=None
    fig, ax = _plot_series(series_to_plot)

    is_fig_figure = isinstance(fig, plt.Figure)
    is_ax_axis = isinstance(ax, plt.Axes)

    assert is_fig_figure and is_ax_axis, "".join(
        [
            "plot_series with kwarg ax=None should return plt.Figure and plt.Axes,",
            f"but returned: {type(fig)} and {type(ax)}",
        ]
    )

    # Test output case where an existing plt.Axes object is passed to kwarg ax
    fig, ax = plt.subplots(1, figsize=plt.figaspect(0.25))
    ax = _plot_series(series_to_plot, ax=ax)

    is_ax_axis = isinstance(ax, plt.Axes)

    assert is_ax_axis, "".join(
        [
            "plot_series with plt.Axes object passed to kwarg ax",
            f"should return plt.Axes, but returned: {type(ax)}",
        ]
    )


# todo: remove skip when issue #2066 has been fixed
@pytest.mark.skip(reason="sporadic failure on win CI/CD, see issue 2066")
def test_plot_series_uniform_treatment_of_int64_range_index_types():
    """Verify that plot_series treats Int64 and Range indices equally."""
    # We test that int64 and range indices are treated uniformly and do not raise an
    # error of inconsistent index types
    _check_soft_dependencies("matplotlib")
    import matplotlib.pyplot as plt

    y1 = pd.Series(np.arange(10))
    y2 = pd.Series(np.random.normal(size=10))
    y1.index = pd.Index(y1.index, dtype=int)
    y2.index = pd.RangeIndex(y2.index)
    plot_series(y1, y2)
    plt.gcf().canvas.draw_idle()
    plt.close()


# todo: remove skip when issue #2066 has been fixed
@pytest.mark.skip(reason="sporadic failure on win CI/CD, see issue 2066")
# Generically test whether plots only accepting univariate input run
@pytest.mark.parametrize("series_to_plot", [y_airline])
@pytest.mark.parametrize("plot_func", univariate_plots)
def test_univariate_plots_run_without_error(series_to_plot, plot_func):
    """Tests whether plots that accept univariate series run without error."""
    _check_soft_dependencies("matplotlib")
    import matplotlib.pyplot as plt

    plot_func(series_to_plot)
    plt.gcf().canvas.draw_idle()
    plt.close()


# todo: remove skip when issue #2066 has been fixed
@pytest.mark.skip(reason="sporadic failure on win CI/CD, see issue 2066")
# Generically test whether plots only accepting univariate input
# raise an error when invalid input type is found
@pytest.mark.parametrize("series_to_plot", invalid_input_types)
@pytest.mark.parametrize("plot_func", univariate_plots)
def test_univariate_plots_invalid_input_type_raises_error(
    series_to_plot, plot_func, valid_data_types
):
    """Tests whether plots that accept univariate series run without error."""
    if not isinstance(series_to_plot, (pd.Series, pd.DataFrame)):
        series_type = type(series_to_plot)
        match = (
            rf"input must be a one of {valid_data_types}, but found type: {series_type}"
        )
        with pytest.raises(TypeError, match=re.escape(match)):
            plot_func(series_to_plot)
    else:
        match = "input must be univariate, but found 2 variables."
        with pytest.raises(ValueError, match=match):
            plot_func(series_to_plot)


# todo: remove skip when issue #2066 has been fixed
@pytest.mark.skip(reason="sporadic failure on win CI/CD, see issue 2066")
# Generically test output of plots only accepting univariate input
@pytest.mark.parametrize("series_to_plot", [y_airline])
@pytest.mark.parametrize("plot_func", univariate_plots)
def test_univariate_plots_output_type(series_to_plot, plot_func):
    """Tests whether plots accepting univariate series have correct output types."""
    _check_soft_dependencies("matplotlib")
    import matplotlib.pyplot as plt

    fig, ax = plot_func(series_to_plot)

    is_fig_figure = isinstance(fig, plt.Figure)
    is_ax_array = isinstance(ax, np.ndarray)
    is_ax_array_axis = all([isinstance(ax_, plt.Axes) for ax_ in ax])

    assert is_fig_figure and is_ax_array and is_ax_array_axis, "".join(
        [
            f"{plot_func.__name__} should return plt.Figure and array of plt.Axes,",
            f"but returned: {type(fig)} and {type(ax)}",
        ]
    )


# todo: remove skip when issue #2066 has been fixed
@pytest.mark.skip(reason="sporadic failure on win CI/CD, see issue 2066")
# For plots that only accept univariate input, from here onwards are
# tests specific to a given plot. E.g. to test specific arguments or functionality.
@pytest.mark.parametrize("series_to_plot", [y_airline])
@pytest.mark.parametrize("lags", [2, (1, 2, 3)])
@pytest.mark.parametrize("suptitle", ["Lag Plot", None])
def test_plot_lags_arguments(series_to_plot, lags, suptitle):
    """Tests whether plot_lags run with different input arguments."""
    _check_soft_dependencies("matplotlib")
    import matplotlib.pyplot as plt

    plot_lags(series_to_plot, lags=lags, suptitle=suptitle)
    plt.gcf().canvas.draw_idle()
    plt.close()


# todo: remove skip when issue #2066 has been fixed
@pytest.mark.skip(reason="sporadic failure on win CI/CD, see issue 2066")
@pytest.mark.parametrize("series_to_plot", [y_airline])
@pytest.mark.parametrize("lags", [6, 12, 24, 36])
@pytest.mark.parametrize("suptitle", ["Correlation Plot", None])
@pytest.mark.parametrize("series_title", ["Time Series", None])
def test_plot_correlations_arguments(series_to_plot, lags, suptitle, series_title):
    """Tests whether plot_lags run with different input arguments."""
    _check_soft_dependencies("matplotlib")
    import matplotlib.pyplot as plt

    plot_correlations(
        series_to_plot, lags=lags, suptitle=suptitle, series_title=series_title
    )
    plt.gcf().canvas.draw_idle()
    plt.close()
