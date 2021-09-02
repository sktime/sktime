# -*- coding: utf-8 -*-
"""Temporal importance curve diagram generators for interval forests."""

__author__ = ["Matthew Middlehurst"]

import numpy as np
from matplotlib import pyplot as plt

from sktime.transformations.panel import catch22


def plot_curves(curves, curve_names, top_curves_shown=None, plot_mean=True):
    """Temporal importance curve diagram generator for interval forests."""
    # find attributes to display by max information gain for any time point.
    top_curves_shown = len(curves) if top_curves_shown is None else top_curves_shown
    max_ig = [max(i) for i in curves]
    top = sorted(range(len(max_ig)), key=lambda i: max_ig[i], reverse=True)[
        :top_curves_shown
    ]

    top_curves = [curves[i] for i in top]
    top_names = [curve_names[i] for i in top]

    # plot curves with highest max and the mean information gain for each time point if
    # enabled.
    for i in range(0, top_curves_shown):
        plt.plot(
            top_curves[i],
            label=top_names[i],
        )
    if plot_mean:
        plt.plot(
            list(np.mean(curves, axis=0)),
            "--",
            linewidth=3,
            label="Mean Information Gain",
        )
    plt.legend(
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
        loc="lower left",
        ncol=2,
        mode="expand",
        borderaxespad=0.0,
    )
    plt.xlabel("Time Point")
    plt.ylabel("Information Gain")

    return plt


def plot_cif(cif, normalise_time_points=False, top_curves_shown=None, plot_mean=True):
    """Temporal importance curve diagram generator for the CanonicalIntervalForest."""
    curves = cif._temporal_importance_curves(
        normalise_time_points=normalise_time_points
    )
    curves = curves.reshape((25 * cif.n_dims, cif.series_length))
    features = catch22.feature_names + ["Mean", "Standard Deviation", "Slope"]
    curve_names = []
    for feature in features:
        for i in range(cif.n_dims):
            name = feature if cif.n_dims == 1 else feature + " Dim " + str(i)
            curve_names.append(name)
    return plot_curves(
        curves,
        curve_names,
        top_curves_shown=top_curves_shown,
        plot_mean=plot_mean,
    )
