#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Utility class for plots with sktime."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme()
sns.set_color_codes()


__all__ = [
    "plot_time_series_with_change_points",
    "plot_time_series_with_profiles",
]

__author__ = "Patrick Schäfer"


def plot_time_series_with_change_points(ts_name, ts, true_cps, font_size=16):
    """
    Plot the time series with the known change points.

    Parameters
    ----------
    ts_name: str
        the name of the time series
    ts: array
        the time series to be segmented
    true_cps: array
        the known change points
    font_size: int
        for plotting

    Returns
    -------
    ax
    """
    plt.figure(figsize=(20, 5))
    segments = [0] + true_cps.tolist() + [ts.shape[0]]

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
    return ax


def plot_time_series_with_profiles(
    ts_name,
    ts,
    profiles,
    true_cps=None,
    found_cps=None,
    score_name="ClaSP Score",
    font_size=16,
):
    """
    Plot the TS with the known and found change points and profiles from segmentation.

    Parameters
    ----------
    ts_name: str
        the name of the time series
    TS: array
        the time series to be segmented
    profiles: array
        the profiles computed by the method used
    true_cps: array
        the known change points
    found_cps: array
        the found change points
    score_name: str
        name of the method used
    font_size: int
        for plotting

    Returns
    -------
    ax
    """
    fig, ax = plt.subplots(
        len(profiles) + 1,
        1,
        sharex=True,
        gridspec_kw={"hspace": 0.05},
        figsize=(20, 5 * len(profiles)),
    )
    ax = ax.reshape(-1)

    if true_cps is not None:
        segments = [0] + true_cps.tolist() + [ts.shape[0]]
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
            tick.label.set_fontsize(font_size)

        for tick in a.yaxis.get_major_ticks():
            tick.label.set_fontsize(font_size)

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
    return ax
