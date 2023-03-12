#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Common timeseries plotting functionality."""

__all__ = ["plot_series", "plot_correlations", "plot_windows"]
__author__ = ["mloning", "RNKuhns", "Drishti Bhasin", "chillerobscuro"]

import math
from warnings import simplefilter, warn

import numpy as np
import pandas as pd

from sktime.datatypes import convert_to
from sktime.utils.validation._dependencies import _check_soft_dependencies
from sktime.utils.validation.forecasting import check_interval_df, check_y
from sktime.utils.validation.series import check_consistent_index_type


def plot_series(
    *series,
    labels=None,
    markers=None,
    colors=None,
    title=None,
    x_label=None,
    y_label=None,
    ax=None,
    pred_interval=None,
):
    """Plot one or more time series.

    Parameters
    ----------
    series : pd.Series or iterable of pd.Series
        One or more time series
    labels : list, default = None
        Names of series, will be displayed in figure legend
    markers: list, default = None
        Markers of data points, if None the marker "o" is used by default.
        The length of the list has to match with the number of series.
    colors: list, default = None
        The colors to use for plotting each series. Must contain one color per series
    title: str, default = None
        The text to use as the figure's suptitle
    pred_interval: pd.DataFrame, default = None
        Output of `forecaster.predict_interval()`. Contains columns for lower
        and upper boundaries of confidence interval.

    Returns
    -------
    fig : plt.Figure
    ax : plt.Axis

    Examples
    --------
    >>> from sktime.utils.plotting import plot_series
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> fig, ax = plot_series(y)  # doctest: +SKIP
    """
    _check_soft_dependencies("matplotlib", "seaborn")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.cbook import flatten
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    for y in series:
        check_y(y)

    series = list(series)
    series = [convert_to(y, "pd.Series", "Series") for y in series]

    n_series = len(series)
    _ax_kwarg_is_none = True if ax is None else False
    # labels
    if labels is not None:
        if n_series != len(labels):
            raise ValueError(
                """There must be one label for each time series,
                but found inconsistent numbers of series and
                labels."""
            )
        legend = True
    else:
        labels = ["" for _ in range(n_series)]
        legend = False

    # markers
    if markers is not None:
        if n_series != len(markers):
            raise ValueError(
                """There must be one marker for each time series,
                but found inconsistent numbers of series and
                markers."""
            )
    else:
        markers = ["o" for _ in range(n_series)]

    # create combined index
    index = series[0].index
    for y in series[1:]:
        # check index types
        check_consistent_index_type(index, y.index)
        index = index.union(y.index)

    # generate integer x-values
    xs = [np.argwhere(index.isin(y.index)).ravel() for y in series]

    # create figure if no Axe provided for plotting
    if _ax_kwarg_is_none:
        fig, ax = plt.subplots(1, figsize=plt.figaspect(0.25))

    # colors
    if colors is None or not _check_colors(colors, n_series):
        colors = sns.color_palette("colorblind", n_colors=n_series)

    # plot series
    for x, y, color, label, marker in zip(xs, series, colors, labels, markers):

        # scatter if little data is available or index is not complete
        if len(x) <= 3 or not np.array_equal(np.arange(x[0], x[-1] + 1), x):
            plot_func = sns.scatterplot
        else:
            plot_func = sns.lineplot

        plot_func(x=x, y=y, ax=ax, marker=marker, label=label, color=color)

    # combine data points for all series
    xs_flat = list(flatten(xs))

    # set x label of data point to the matching index
    def format_fn(tick_val, tick_pos):
        if int(tick_val) in xs_flat:
            return index[int(tick_val)]
        else:
            return ""

    # dynamically set x label ticks and spacing from index labels
    ax.xaxis.set_major_formatter(FuncFormatter(format_fn))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Set the figure's title
    if title is not None:
        fig.suptitle(title, size="xx-large")

    # Label the x and y axes
    if x_label is not None:
        ax.set_xlabel(x_label)

    _y_label = y_label if y_label is not None else series[0].name
    ax.set_ylabel(_y_label)

    if legend:
        ax.legend()
    if pred_interval is not None:
        check_interval_df(pred_interval, series[-1].index)
        ax = plot_interval(ax, pred_interval)
    if _ax_kwarg_is_none:
        return fig, ax
    else:
        return ax


def plot_interval(ax, interval_df):
    cov = interval_df.columns.levels[1][0]
    ax.fill_between(
        ax.get_lines()[-1].get_xdata(),
        interval_df["Coverage"][cov]["lower"].astype("float64"),
        interval_df["Coverage"][cov]["upper"].astype("float64"),
        alpha=0.2,
        color=ax.get_lines()[-1].get_c(),
        label=f"{int(cov * 100)}% prediction interval",
    )
    ax.legend()
    return ax


def plot_lags(series, lags=1, suptitle=None):
    """Plot one or more lagged versions of a time series.

    Parameters
    ----------
    series : pd.Series
        Time series for plotting lags.
    lags : int or array-like, default=1
        The lag or lags to plot.

        - int plots the specified lag
        - array-like  plots specified lags in the array/list

    suptitle : str, default=None
        The text to use as the Figure's suptitle. If None, then the title
        will be "Plot of series against lags {lags}"

    Returns
    -------
    fig : matplotlib.figure.Figure

    axes : np.ndarray
        Array of the figure's Axe objects

    Examples
    --------
    >>> from sktime.utils.plotting import plot_lags
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> fig, ax = plot_lags(y, lags=2) # plot of y(t) with y(t-2)  # doctest: +SKIP
    >>> fig, ax = plot_lags(y, lags=[1,2,3]) # y(t) & y(t-1), y(t-2).. # doctest: +SKIP
    """
    _check_soft_dependencies("matplotlib")
    import matplotlib.pyplot as plt

    check_y(series)

    if isinstance(lags, int):
        single_lag = True
        lags = [lags]
    elif isinstance(lags, (tuple, list, np.ndarray)):
        single_lag = False
    else:
        raise ValueError("`lags should be an integer, tuple, list, or np.ndarray.")

    length = len(lags)
    n_cols = min(3, length)
    n_rows = math.ceil(length / n_cols)
    fig, ax = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(8, 6 * n_rows),
        sharex=True,
        sharey=True,
    )
    if single_lag:
        axes = ax
        pd.plotting.lag_plot(series, lag=lags[0], ax=axes)
    else:
        axes = ax.ravel()
        for i, val in enumerate(lags):
            pd.plotting.lag_plot(series, lag=val, ax=axes[i])

    if suptitle is None:
        fig.suptitle(
            f"Plot of series against lags {', '.join([str(lag) for lag in lags])}",
            size="xx-large",
        )
    else:
        fig.suptitle(suptitle, size="xx-large")

    return fig, np.array(fig.get_axes())


def plot_correlations(
    series,
    lags=24,
    alpha=0.05,
    zero_lag=True,
    acf_fft=False,
    acf_adjusted=True,
    pacf_method="ywadjusted",
    suptitle=None,
    series_title=None,
    acf_title="Autocorrelation",
    pacf_title="Partial Autocorrelation",
):
    """Plot series and its ACF and PACF values.

    Parameters
    ----------
    series : pd.Series
        A time series.

    lags : int, default = 24
        Number of lags to include in ACF and PACF plots

    alpha : int, default = 0.05
        Alpha value used to set confidence intervals. Alpha = 0.05 results in
        95% confidence interval with standard deviation calculated via
        Bartlett's formula.

    zero_lag : bool, default = True
        If True, start ACF and PACF plots at 0th lag

    acf_fft : bool,  = False
        Whether to compute ACF via FFT.

    acf_adjusted : bool, default = True
        If True, denonimator of ACF calculations uses n-k instead of n, where
        n is number of observations and k is the lag.

    pacf_method : str, default = 'ywadjusted'
        Method to use in calculation of PACF.

    suptitle : str, default = None
        The text to use as the Figure's suptitle.

    series_title : str, default = None
        Used to set the title of the series plot if provided. Otherwise, series
        plot has no title.

    acf_title : str, default = 'Autocorrelation'
        Used to set title of ACF plot.

    pacf_title : str, default = 'Partial Autocorrelation'
        Used to set title of PACF plot.

    Returns
    -------
    fig : matplotlib.figure.Figure

    axes : np.ndarray
        Array of the figure's Axe objects

    Examples
    --------
    >>> from sktime.utils.plotting import plot_correlations
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> fig, ax = plot_correlations(y)  # doctest: +SKIP
    """
    _check_soft_dependencies("matplotlib", "statsmodels")
    import matplotlib.pyplot as plt
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    series = check_y(series)
    series = convert_to(series, "pd.Series", "Series")

    # Setup figure for plotting
    fig = plt.figure(constrained_layout=True, figsize=(12, 8))
    gs = fig.add_gridspec(2, 2)
    f_ax1 = fig.add_subplot(gs[0, :])
    if series_title is not None:
        f_ax1.set_title(series_title)
    f_ax2 = fig.add_subplot(gs[1, 0])
    f_ax3 = fig.add_subplot(gs[1, 1])

    # Create expected plots on their respective Axes
    plot_series(series, ax=f_ax1)
    plot_acf(
        series,
        ax=f_ax2,
        lags=lags,
        zero=zero_lag,
        alpha=alpha,
        title=acf_title,
        adjusted=acf_adjusted,
        fft=acf_fft,
    )
    plot_pacf(
        series,
        ax=f_ax3,
        lags=lags,
        zero=zero_lag,
        alpha=alpha,
        title=pacf_title,
        method=pacf_method,
    )
    if suptitle is not None:
        fig.suptitle(suptitle, size="xx-large")

    return fig, np.array(fig.get_axes())


def _check_colors(colors, n_series):
    """Verify color list is correct length and contains only colors."""
    from matplotlib.colors import is_color_like

    if n_series == len(colors) and all([is_color_like(c) for c in colors]):
        return True
    warn(
        "Color list must be same length as `series` and contain only matplotlib colors"
    )
    return False


def _get_windows(cv, y):
    """Generate cv split windows, utility function."""
    train_windows = []
    test_windows = []
    for train, test in cv.split(y):
        train_windows.append(train)
        test_windows.append(test)
    return train_windows, test_windows


def plot_windows(cv, y, title=""):
    """Plot training and test windows.

    Parameters
    ----------
    y : pd.Series
        Time series to split
    cv : temporal cross-validation iterator object
        Temporal cross-validation iterator
    title : str
        Plot title
    """
    _check_soft_dependencies("matplotlib", "seaborn")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.ticker import MaxNLocator

    simplefilter("ignore", category=UserWarning)

    train_windows, test_windows = _get_windows(cv, y)

    def get_y(length, split):
        # Create a constant vector based on the split for y-axis."""
        return np.ones(length) * split

    n_splits = len(train_windows)
    n_timepoints = len(y)
    len_test = len(test_windows[0])

    train_color, test_color = sns.color_palette("colorblind")[:2]

    fig, ax = plt.subplots(figsize=plt.figaspect(0.3))

    for i in range(n_splits):
        train = train_windows[i]
        test = test_windows[i]

        ax.plot(
            np.arange(n_timepoints), get_y(n_timepoints, i), marker="o", c="lightgray"
        )
        ax.plot(
            train,
            get_y(len(train), i),
            marker="o",
            c=train_color,
            label="Window",
        )
        ax.plot(
            test,
            get_y(len_test, i),
            marker="o",
            c=test_color,
            label="Forecasting horizon",
        )
    ax.invert_yaxis()
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set(
        title=title,
        ylabel="Window number",
        xlabel="Time",
        xticklabels=y.index,
    )
    # remove duplicate labels/handles
    handles, labels = [(leg[:2]) for leg in ax.get_legend_handles_labels()]
    ax.legend(handles, labels)
