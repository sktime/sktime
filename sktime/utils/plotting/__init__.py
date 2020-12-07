#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__all__ = ["plot_series"]
__author__ = ["Markus Löning"]

import warnings

import numpy as np

from sktime.utils.check_imports import _check_soft_dependencies
from sktime.utils.validation.forecasting import check_y


def plot_series(*series, labels=None):
    """Plot one or more time series

    Parameters
    ----------
    series : pd.Series
        One or more time series
    labels : list, optional (default=None)
        Names of series, will be displayed in figure legend

    Returns
    -------
    fig : plt.Figure
    ax : plt.Axis
    """
    _check_soft_dependencies("matplotlib", "seaborn")
    import matplotlib.pyplot as plt
    import seaborn as sns

    n_series = len(series)
    if labels is not None:
        if n_series != len(labels):
            raise ValueError(
                "There must be one label for each time series, "
                "but found inconsistent numbers of series and "
                "labels."
            )
        legend = True
    else:
        labels = ["" for _ in range(n_series)]
        legend = False

    for y in series:
        check_y(y)

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
    for x, y, color, label in zip(xs, series, colors, labels):

        # scatter if little data is available or index is not complete
        if len(x) <= 3 or not np.array_equal(np.arange(x[0], x[-1] + 1), x):
            plot_func = sns.scatterplot
        else:
            plot_func = sns.lineplot

        plot_func(x=x, y=y, ax=ax, marker="o", label=label, color=color)

    # set combined index as xticklabels, suppress matplotlib warning
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        ax.set(xticklabels=index)

    if legend:
        ax.legend()

    return fig, ax
