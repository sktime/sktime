"""Temporal importance curve diagram generators for interval forests."""

__author__ = ["MatthewMiddlehurst", "mgazian000", "CTFallon"]

__all__ = ["plot_curves", "plot_cif", "plot_TSF_temporal_importance_curve"]
import numpy as np

from sktime.classification.interval_based import CanonicalIntervalForest
from sktime.series_as_features.base.estimators.interval_based import (
    BaseTimeSeriesForest,
)
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


def plot_TSF_temporal_importance_curve(tsf, normalize=False):
    """Temporal Importance curve diagram generator for TSF.

    Parameters
    ----------
    normalize : bool = False
    whether or not to normalize importance contribution to interval length. False
    matches design from [1], True is more informative of high importance
    timestamps/features.

    References
    ----------
    .. [1] H.Deng, G.Runger, E.Tuv and M.Vladimir, "A time series forest for
    classification and feature extraction",Information Sciences, 239, 2013

    Example
    -------
    >>> from sktime.classification.interval_based import (
    ...     TimeSeriesForestClassifier
    ... )  #docstep: +SKIP
    >>> from sktime.classification.plotting.temporal_importance_diagram import (
    ...     plot_TSF_temporal_importance_curve
    ... ) #docstep: +SKIP
    >>> from sktime.datasets import load_gunpoint   #docstep: +SKIP
    >>> X_train, y_train = load_gunpoint(split="train", return_X_y=True) #docstep: +SKIP
    >>> clf = TimeSeriesForestClassifier(n_estimators=50)  #docstep: +SKIP
    >>> clf.fit(X_train, y_train)  #docstep: +SKIP
    >>> fig = plot_TSF_temporal_importance_curve(clf, True)  #docstep: +SKIP
    >>> fig.title(label="normalized")  #docstep: +SKIP
    >>> fig.savefig("test_norm") #docstep: +SKIP
    """
    _check_soft_dependencies("matplotlib")

    import matplotlib.pyplot as plt

    if not isinstance(tsf, BaseTimeSeriesForest) or not tsf._is_fitted:
        raise ValueError("Input must be a fitted object that inherits from BaseTSF")

    try:
        if not (tsf.tic_norm == normalize):
            tsf.calc_temporal_curves(normalize)
    except AttributeError:
        tsf.calc_temporal_curves(normalize)

    curves = {
        "Mean": tsf.mean_curve,
        "StDev": tsf.stdev_curve,
        "Slope": tsf.slope_curve,
    }

    for curve_name, curve in curves.items():
        plt.plot(curve, label=curve_name)

    plt.legend()

    return plt
