#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

"""Common timeseries plotting functionality.

Functions
---------
check_pred_int(pred_int)
plot_series(*series, labels=None, markers=None, ax=None)
plot_correlations(
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
)
"""
__all__ = ["plot_series", "plot_correlations"]
__author__ = ["Markus Löning", "Ryan Kuhns", "Ifeanyi Eze"]

import numpy as np

from sktime.utils.validation._dependencies import _check_soft_dependencies
from sktime.utils.validation.forecasting import check_y
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pandas as pd


def check_pred_int(pred_int):
    """Check the pred_int data type.

    pred_int: pd.DataFrame
        Prediction intervals of series

    Raises
    ------
    TypeError: when pred_int is not a pd.DataFrame
    Exception: when the number of columns is less or more than 2
        and column labels are not ['lower', 'upper']
    """
    if isinstance(pred_int, pd.DataFrame):
        if pred_int.shape[1] == 2:
            if not pred_int.columns.isin(["lower", "upper"]).all():
                raise ValueError(
                    "Both DataFrame column labels must be 'lower' and 'upper'"
                )
        else:
            raise Exception(f"{pred_int} must have exactly two columns")
    else:
        raise TypeError(f"{pred_int} must be a DataFrame")


def plot_series(
    *series, labels=None, markers=None, x_label=None, y_label=None, ax=None
):
    """Plot one or more time series.

    Parameters
    ----------
    series : pd.Series
        One or more time series
    labels : list, default = None
        Names of series, will be displayed in figure legend
    markers: list, default = None
        Markers of data points, if None the marker "o" is used by default.
        Lenght of list has to match with number of series
    pred_int: pd.DataFrame, optional (default=None)
        Prediction intervals of series

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
        # check types, note that isinstance() does not work here because index
        # types inherit from each other, hence we check for type equality
        if not type(index) is type(y.index):  # noqa
            raise TypeError("Found series with different index types.")
        index = index.union(y.index)

    # generate integer x-values
    xs = [np.argwhere(index.isin(y.index)).ravel() for y in series]

    # create figure if no Axe provided for plotting
    if _ax_kwarg_is_none:
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

    # plot prediction intervals if present
    if pred_int is not None:
        check_pred_int(pred_int)
        # check same conditions as for earlier indices
        if all([x in index for x in pred_int.index]):
            ax.fill_between(
                ax.get_lines()[-1].get_xdata(),
                pred_int.lower,
                pred_int.upper,
                alpha=0.3,
                color=ax.get_lines()[-1].get_c(),
                label="prediction intervals",
            )
        else:
            raise ValueError("pred_int and x axis indices does not match")

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

    # Label the x and y axes
    if x_label is not None:
        ax.set_xlabel(x_label)

    _y_label = y_label if y_label is not None else series[0].name
    ax.set_ylabel(_y_label)

    if legend:
        ax.legend()
    if _ax_kwarg_is_none:
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
    """
    _check_soft_dependencies("matplotlib")
    import matplotlib.pyplot as plt

    series = check_y(series)

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
