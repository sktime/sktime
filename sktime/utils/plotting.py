#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Common timeseries plotting functionality."""

__all__ = ["plot_series", "plot_correlations", "plot_windows"]
__author__ = ["mloning", "RNKuhns", "Dbhasin1", "chillerobscuro"]

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

    This function allows you to plot one or more
    time series on a single figure via `series`.
    Used for making comparisons between different series.

    The resulting figure includes the time series data plotted on a graph with
    x-axis as time by default and can be changed via `x_label` and
    y-axis as value of time series can be renamed via `y_label` and
    labels explaining the meaning of each series via `labels`,
    markers for data points via `markers`.
    You can also specify custom colors via `colors` for each series and
    add a title to the figure via `title`.
    If prediction intervals are available add them using `pred_interval`,
    they can be overlaid on the plot to visualize uncertainty.

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
    ax : matplotlib axes, optional
        Axes to plot on, if None, a new figure is created and returned

    Returns
    -------
    fig : plt.Figure
        It manages the final visual appearance and layout.
        Create a new figure, or activate an existing figure.
    ax : plt.Axis
        Axes containing the plot
        If ax was None, a new figure is created and returned
        If ax was not None, the same ax is returned with plot added

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

    # create figure if no ax provided for plotting
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
        ax = plot_interval(ax, pred_interval, index)
    if _ax_kwarg_is_none:
        return fig, ax
    else:
        return ax


def plot_interval(ax, interval_df, ix=None):
    cov = interval_df.columns.levels[1][0]
    var_name = interval_df.columns.levels[0][0]
    x_ix = np.argwhere(ix.isin(interval_df.index)).ravel()
    x_ix = np.array(x_ix)

    ax.fill_between(
        x_ix,
        interval_df[var_name][cov]["lower"].astype("float64").to_numpy(),
        interval_df[var_name][cov]["upper"].astype("float64").to_numpy(),
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
        If True, denominator of ACF calculations uses n-k instead of n, where
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


def plot_windows(cv, y, title="", ax=None):
    """Plot training and test windows.

    Plots the training and test windows for each split of a time series,
    subject to an sktime time series splitter.

    x-axis: time, ranging from start to end of `y`
    y-axis: window number, starting at 0
    plot elements: training split (orange) and test split (blue)
        dots indicate index in the training or test split
        will be plotted on top of each other if train/test split is not disjoint

    Parameters
    ----------
    y : pd.Series
        Time series to split
    cv : sktime splitter object, descendant of BaseSplitter
        Time series splitter, e.g., temporal cross-validation iterator
    title : str
        Plot title
    ax : matplotlib.axes.Axes, optional (default=None)
        Axes on which to plot. If None, axes will be created and returned.

    Returns
    -------
    fig : matplotlib.figure.Figure, returned only if ax is None
        matplotlib figure object
    ax : matplotlib.axes.Axes
        matplotlib axes object with the figure
    """
    _check_soft_dependencies("matplotlib", "seaborn")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.ticker import MaxNLocator

    simplefilter("ignore", category=UserWarning)

    _ax_kwarg_is_none = True if ax is None else False

    # create figure if no ax provided for plotting
    if _ax_kwarg_is_none:
        fig, ax = plt.subplots(1, figsize=plt.figaspect(0.25))

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
    xtickslocs = [tick for tick in ax.get_xticks() if tick in np.arange(n_timepoints)]
    ax.set(
        title=title,
        ylabel="Window number",
        xlabel="Time",
        xticks=xtickslocs,
        xticklabels=y.iloc[xtickslocs].index,
    )
    # remove duplicate labels/handles
    handles, labels = ((leg[:2]) for leg in ax.get_legend_handles_labels())
    ax.legend(handles, labels)

    if _ax_kwarg_is_none:
        return fig, ax
    else:
        return ax
