#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = ["Markus Löning"]
__all__ = ["plot_ys"]

import numpy as np
from sktime.utils.validation.forecasting import check_y


def plot_ys(*ys, labels=None):
    import matplotlib.pyplot as plt

    if labels is not None:
        if len(ys) != len(labels):
            raise ValueError("There must be one label for each time series")
        labels_ = labels
    else:
        labels_ = ["" for _ in range(len(ys))]

    fig, ax = plt.subplots(1, figsize=plt.figaspect(.25))

    for y, label in zip(ys, labels_):
        check_y(y)

        # scatter if only a few points are available, otherwise plot line
        continuous_index = np.arange(y.index.min(), y.index.max() + 1)
        if len(y) < 3 or not np.array_equal(y.index.values, continuous_index):
            ax.scatter(y.index.values, y.values, label=label)
        else:
            ax.plot(y.index.values, y.values, label=label)

    if labels is not None:
        plt.legend()

    return fig, ax
