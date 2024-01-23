"""Temporal importance curve diagram generators for interval forests."""

__author__ = ["MatthewMiddlehurst"]

import numpy as np

from sktime.classification.interval_based import CanonicalIntervalForest
from sktime.transformations.panel import catch22
from sktime.utils.validation._dependencies import _check_soft_dependencies

_check_soft_dependencies("matplotlib", severity="warning")


def plot_curves(curves, curve_names, top_curves_shown=None, plot_mean=True):
    """Temporal importance curve diagram generator for interval forests."""
    # find attributes to display by max information gain for any time point.
    _check_soft_dependencies("matplotlib")

    import matplotlib.pyplot as plt

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
    if not isinstance(cif, CanonicalIntervalForest) or not cif._is_fitted:
        raise ValueError("Input must be a fitted CanonicalIntervalForest classifier.")

    curves = cif._temporal_importance_curves(
        normalise_time_points=normalise_time_points
    )
    curves = curves.reshape((25 * cif.n_dims_, cif.series_length_))
    features = catch22.feature_names + ["Mean", "Standard Deviation", "Slope"]
    curve_names = []
    for feature in features:
        for i in range(cif.n_dims_):
            name = feature if cif.n_dims_ == 1 else feature + " Dim " + str(i)
            curve_names.append(name)
    return plot_curves(
        curves,
        curve_names,
        top_curves_shown=top_curves_shown,
        plot_mean=plot_mean,
    )
