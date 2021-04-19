#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__all__ = ["plot_series"]
__author__ = ["Markus Löning"]

import numpy as np

from sktime.utils.validation._dependencies import _check_soft_dependencies
from sktime.utils.validation.forecasting import check_y


def plot_series(*series, labels=None, markers=None):
    """Plot one or more time series

    Parameters
    ----------
    series : pd.Series
        One or more time series
    labels : list, optional (default=None)
        Names of series, will be displayed in figure legend
    markers: list, optional (default=None)
        Markers of data points, if None the marker "o" is used by default.
        Lenght of list has to match with number of series

    Returns
    -------
    fig : plt.Figure
    ax : plt.Axis
    """
    _check_soft_dependencies("matplotlib", "seaborn")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    from matplotlib.cbook import flatten
    import seaborn as sns

    for y in series:
        check_y(y)

    n_series = len(series)

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
        # check types, note that isinstance() does not work here because index
        # types inherit from each other, hence we check for type equality
        if not type(index) is type(y.index):  # noqa
            raise TypeError("Found series with different index types.")
        index = index.union(y.index)

    # generate integer x-values
    xs = [np.argwhere(index.isin(y.index)).ravel() for y in series]

    # create figure
    fig, ax = plt.subplots(1, figsize=plt.figaspect(0.25))
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

    if legend:
        ax.legend()

    return fig, ax


def plot_lags(series, lags=1):
    """Plot one or more lagged versions of a time series

    Parameters
    ----------
    series : pd.Series
        Time series for plotting lags
    lags : int or array-like
        Number of lags to plot.
        int         - plot the specified lag
        array-like  - plot specified lags in the array/list

    Returns
    -------

    fig   :  plt.Figure
    axes  :  plt.Axis or ndarray of plt.Axis objects

    Example
    -------

    Given the following time series

        >>> np.random.seed(5)
        >>> x = np.cumsum(np.random.normal(loc=1, scale=5, size=50))
        >>> s = pd.Series(x)

     If lags is an int, plot a single lagged time series with the
     specified lag

        >>> plot_lags(s, lags=2) #plot of y(t) with y(t+2)

     If lags is an array-like , plot several lagged time series according to
     the lags specified in the

        >>> plot_lags(s, lags=[1,2,3]) #plots of y(t) with y(t+1),y(t+2)..

    """
    _check_soft_dependencies("matplotlib")
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    length = 0
    is_lag_one = isinstance(lags, int)

    if is_lag_one:
        length = 1
        lags = [lags]
    else:
        length = len(lags)

    fig = plt.figure(figsize=(8, 6 * length))
    axes = np.array([])

    for i, val in enumerate(lags, start=1):

        ax = fig.add_subplot(length, 1, i)
        if is_lag_one:
            axes = ax
        else:
            axes = np.append(axes, ax)
        pd.plotting.lag_plot(series, lag=val, ax=ax)

    return fig, axes
