#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__all__ = ["plot_series"]
__author__ = ["Markus Löning"]

import numpy as np

from sktime.utils.validation._dependencies import _check_soft_dependencies
from sktime.utils.validation.forecasting import check_y
import pandas as pd


def check_pred_int(pred_int):
    """helper function to check pred_int data type


    pred_int: pd.DataFrame
        Prediction intervals of series


    Riases
    -------
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


def plot_series(*series, labels=None, markers=None, pred_int=None):
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
