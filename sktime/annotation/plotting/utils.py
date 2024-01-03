#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Utility class for plotting functionality."""

import numpy as np

from sktime.utils.validation._dependencies import _check_soft_dependencies
from sktime.utils.validation.forecasting import check_X

__all__ = [
    "plot_time_series_with_change_points",
    "plot_time_series_with_profiles",
]

__author__ = ["patrickzib"]


def plot_time_series_with_change_points(ts_name, ts, true_cps, font_size=16):
    """Plot the time series with the known change points.

    Parameters
    ----------
    ts_name: str
        the name of the time series (dataset) to be annotated
    ts: array-like, shape = [n]
        the univariate time series of length n to be annotated
    true_cps: array-like, dtype=int
        the known change points
        these are highlighted in the time series as vertical lines
    font_size: int
        for plotting

    Returns
    -------
    fig : matplotlib.figure.Figure

    axes : np.ndarray
        Array of the figure's Axe objects
    """
    # Checks availability of plotting libraries
    _check_soft_dependencies("matplotlib")
    import matplotlib.pyplot as plt

    ts = check_X(ts)

    fig = plt.figure(figsize=(20, 5))
    true_cps = np.sort(true_cps)
    segments = [0] + list(true_cps) + [ts.shape[0]]

    for idx in np.arange(0, len(segments) - 1):
        plt.plot(
            range(segments[idx], segments[idx + 1]),
            ts[segments[idx] : segments[idx + 1]],
        )

    lim1 = plt.ylim()[0]
    lim2 = plt.ylim()[1]

    ax = plt.gca()
    for i, idx in enumerate(true_cps):
        ax.vlines(idx, lim1, lim2, linestyles="--", label=str(i) + "-th-CPT")

    plt.legend(loc="best")
    plt.title(ts_name, fontsize=font_size)
    return fig, ax


def plot_time_series_with_profiles(
    ts_name,
    ts,
    profiles,
    true_cps=None,
    found_cps=None,
    score_name="ClaSP Score",
    font_size=16,
):
    """Plot TS with known and found change points and profiles from segmentation.

    Parameters
    ----------
    ts_name: str
        the name of the time series (dataset) to be annotated
    ts: array-like, shape=[n]
        the univariate time series of length n to be annotated.
        the time series is plotted as the first subplot.
    profiles: array-like, shape=[n-m+1, n_cpts], dtype=float
        the n_cpts profiles computed by the method used
        the profiles are plotted as subsequent subplots to the time series.
    true_cps: array-like, dtype=int
        the known change points.
        these are highlighted in the time series subplot as vertical lines
    found_cps: array-like, shape=[n_cpts], dtype=int
        the found change points
        these are highlighted in the profiles subplot as vertical lines
    score_name: str
        name of the scoring method used, i.e. 'ClaSP'
    font_size: int
        for plotting

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure with 1 + len(profiles) subplots, one for the time series
        and others for each profile
    axes : np.ndarray
        Array of the figure's Axe objects
    """
    # Checks availability of plotting libraries
    _check_soft_dependencies("matplotlib", "seaborn")
    import matplotlib.pyplot as plt

    ts = check_X(ts)

    fig, ax = plt.subplots(
        len(profiles) + 1,
        1,
        sharex=True,
        gridspec_kw={"hspace": 0.05},
        figsize=(20, 5 * len(profiles)),
    )
    ax = ax.reshape(-1)

    if true_cps is not None:
        segments = [0] + list(true_cps) + [ts.shape[0]]
        for idx in np.arange(0, len(segments) - 1):
            ax[0].plot(
                np.arange(segments[idx], segments[idx + 1]),
                ts[segments[idx] : segments[idx + 1]],
            )
    else:
        ax[0].plot(np.arange(ts.shape[0]), ts)

    for i, profile in enumerate(profiles):
        ax[i + 1].plot(np.arange(len(profile)), profile, color="b")
        ax[i + 1].set_ylabel(score_name + " " + str(i) + ". Split", fontsize=font_size)

    ax[-1].set_xlabel("split point $s$", fontsize=font_size)
    ax[0].set_title(ts_name, fontsize=font_size)

    for a in ax:
        for tick in a.xaxis.get_major_ticks():
            tick.label1.set_fontsize(font_size)

        for tick in a.yaxis.get_major_ticks():
            tick.label1.set_fontsize(font_size)

    if true_cps is not None:
        for idx, true_cp in enumerate(true_cps):
            ax[0].axvline(
                x=true_cp,
                linewidth=2,
                color="r",
                label="True Change Point" if idx == 0 else None,
            )

    if found_cps is not None:
        for idx, found_cp in enumerate(found_cps):
            ax[0].axvline(
                x=found_cp,
                linewidth=2,
                color="g",
                label="Predicted Change Point" if idx == 0 else None,
            )
            ax[idx + 1].axvline(
                x=found_cp,
                linewidth=2,
                color="g",
                label="Predicted Change Point" if idx == 0 else None,
            )

    ax[0].legend(prop={"size": font_size})

    return fig, ax
