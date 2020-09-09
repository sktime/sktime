#!/usr/bin/env python3 -u
# coding: utf-8
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Markus Löning"]
__all__ = ["plot_ys"]

from sktime.utils.validation.forecasting import check_y


def plot_ys(*ys, labels=None):
    """Plot time series

    Parameters
    ----------
    ys : pd.Series
        One or more time series
    labels : list, optional (default=None)
        Names of time series displayed in figure legend

    Returns
    -------
    fig : plt.Figure
    ax : plt.Axis
    """
    import matplotlib.pyplot as plt

    if labels is not None:
        if len(ys) != len(labels):
            raise ValueError("There must be one label for each time series, "
                             "but found inconsistent numbers of series and "
                             "labels.")
        labels_ = labels
        legend = True
    else:
        labels_ = ["" for _ in range(len(ys))]
        legend = False

    fig, ax = plt.subplots(1, figsize=plt.figaspect(.25))

    for y, label in zip(ys, labels_):
        check_y(y)

        # scatter if only a few points are available
        if len(y) <= 3 or not index_is_complete(y):
            df = y.to_frame().reset_index().reset_index()
            df.plot(x=0, y=2, kind="scatter", s=4, ax=ax, label=label,
                    legend=legend)
            ax.set(xticklabels=df.iloc[:, 1], xlabel=df.columns[1])

        # otherwise use line plot
        else:
            y.plot(ax=ax, kind="line", marker='o', markersize=4,
                   label=label, legend=legend)

    return fig, ax


def index_is_complete(y):
    index = y.index

    if hasattr(index, "is_full"):
        return index.is_full
