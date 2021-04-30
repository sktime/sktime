#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__all__ = ["plot_series", "plot_correlations"]
__author__ = ["Markus Löning"]

import numpy as np

from sktime.utils.validation._dependencies import _check_soft_dependencies
from sktime.utils.validation.forecasting import check_y
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def plot_series(*series, labels=None, markers=None, ax=None):
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

    # create figure if no Axe provided for plotting
    if ax is None:
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
    if ax is None:
        return fig, ax
    else:
        return ax


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
    """
    _check_soft_dependencies("matplotlib", "seaborn")
    import matplotlib.pyplot as plt

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
    plot_acf(series, ax=f_ax2, lags=lags, zero=zero_lag, alpha=alpha, title=acf_title)
    plot_pacf(series, ax=f_ax3, lags=lags, zero=zero_lag, alpha=alpha, title=pacf_title)
    if suptitle is not None:
        fig.suptitle(suptitle, size="xx-large")

    return fig, np.array(fig.get_axes())
