#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Utility class for plotting functionality."""

import numpy as np

from sktime.utils.dependencies import _check_soft_dependencies
from sktime.utils.validation.forecasting import check_X

__all__ = [
    "plot_time_series_with_change_points",
    "plot_time_series_with_profiles",
    "plot_time_series_with_anomalies",
    "plot_time_series_with_predicted_anomalies",
    "plot_time_series_with_detrender",
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


def plot_time_series_with_anomalies(
    ts, X, y, ax=None, title=None, X_label=None, Y_label=None, subplot=False
):
    """Plot the time series with actual anomalies.

    Parameters
    ----------
    ts: array-like, shape=[n]
        the univariate time series of length n to be annotated.
    X: array-like, shape=[n]
        the x-coordinates (indices) of the timeseries.
    y: array-like, shape=[n]
        the y-coordinates (values)  of anomaly points.
    ax: np.ndarray
        Array of the figure's Axe objects to plot on
    title: str
        Title of the plot
    X_label: str
        X-axis label
    Y_label: str
        Y-axis label
    y_hat: pandas.Series
        Actual anotations of anomalies

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : np.ndarray
        Array of the figure's Axe objects
    """
    # Checks availability of plotting libraries
    _check_soft_dependencies("matplotlib", "seaborn")
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.plot(ts, label="Not Anomalous")

    ax.scatter(X, y, label="Anomalous", color="tab:orange")

    if X_label is not None:
        ax.set_xlabel(X_label)
    if Y_label is not None:
        ax.set_ylabel(Y_label)
    if title is not None:
        ax.set_title(title)

    return fig, ax


def plot_time_series_with_predicted_anomalies(
    df,
    y_hat,
    ax=None,
):
    """Create subplots comparing actual anomalies vs. predicted anomalies.

    Parameters
    ----------
    df: pandas.DataFrame
        dataframe consisting of time series and labels
    y_hat: pandas.Series
        Actual anotations of anomalies
    ax: np.ndarray
        Array of the figure's Axe objects to plot on

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : np.ndarray
        Array of the figure's Axe objects
    """
    # Checks availability of plotting libraries
    _check_soft_dependencies("matplotlib", "seaborn")
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    else:
        fig = ax.figure

    # Plot the actual anomalies in the first figure
    plot_time_series_with_anomalies(
        ts=df.iloc[:, 0],
        X=df.loc[df.iloc[:, 1] == 1.0].index,
        y=df.loc[df.iloc[:, 1] == 1.0, df.columns[0]],
        ax=ax[0],
        title="Actual Anomalies",
    )

    # Plot the predicted anomalies in the second figure
    plot_time_series_with_anomalies(
        ts=df.iloc[:, 0],
        X=df.loc[y_hat].index,
        y=df.loc[y_hat, df.columns[0]],
        ax=ax[1],
        title="Predicted Anomalies",
    )

    return fig, ax


def plot_time_series_with_detrender(
    df,
    y_hat,
    detrended=None,
    ax=None,
):
    """Plot the time series with actual anomalies using a detrender.

    Parameters
    ----------
    df: pandas.DataFrame
        dataframe consisting of time series and labels
    detrended: array-like, shape=[n]
        the detrended univariate time series of length n to be annotated.
    y_hat: pandas.Series
        Actual anotations of anomalies
    ax: np.ndarray
        Array of the figure's Axe objects to plot on

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : np.ndarray
        Array of the figure's Axe objects
    """
    # Checks availability of plotting libraries
    _check_soft_dependencies("matplotlib", "seaborn")
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    else:
        fig = ax.figure

    if detrended is None:
        X_actual = df.loc[df.iloc[:, 1] == 1.0, df.columns[0]].index
        y_actual = df.loc[df.iloc[:, 1] == 1.0, df.columns[0]]
        X_predicted = df.loc[y_hat, df.columns[0]].index
        y_predicted = df.loc[y_hat, df.columns[0]]
        ts = df.iloc[:, 0]
    else:
        X_actual = detrended.loc[df.iloc[:, 1] == 1.0].index
        y_actual = detrended.loc[df.iloc[:, 1] == 1.0]
        X_predicted = detrended.loc[y_hat].index
        y_predicted = detrended.loc[y_hat]
        ts = detrended

    plot_time_series_with_anomalies(
        ts=ts,
        X=X_actual,
        y=y_actual,
        ax=ax[0],
        title="Actual Anomalies",
    )

    plot_time_series_with_anomalies(
        ts=ts,
        X=X_predicted,
        y=y_predicted,
        ax=ax[1],
        title="Actual Anomalies",
    )

    return fig, ax


def plot_time_series_with_subsequent_outliers(
    df,
    intervals=None,
    ax=None,
):
    """Plot the time series with subsequent outliers.

    Parameters
    ----------
    df: pandas.DataFrame
        dataframe consisting of time series and labels
    intervals: array-like, shape=[n_intervals]
        the intervals of the outliers
    ax: np.ndarray
        Array of the figure's Axe objects to plot on

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : np.ndarray
        Array of the figure's Axe objects
    """
    # Checks availability of plotting libraries
    _check_soft_dependencies("matplotlib", "seaborn")
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    ax.plot(df.iloc[:, 0], label="Not Anomalous")
    ax.plot(df.loc[df.iloc[:, 1] == 1.0, df.columns[0]], label="Anomalous")

    if intervals is not None:
        for interval in intervals:
            left = interval.left
            right = interval.right
            ax.axvspan(
                left, right, color="tab:green", alpha=0.3, label="Predicted Anomalies"
            )

    ax.legend()

    return fig, ax


def plot_time_series_with_change_point_detection(
    df,
    actual_cp,
    predicted_change_points=None,
    ax=None,
):
    """Plot the time series with subsequent outliers.

    Parameters
    ----------
    df: pandas.DataFrame
        dataframe consisting of time series and labels
    actual_cp: datatime.Datetime
        The actual change point time.
    predicted_change_points: pandas.DataFrame
        predicted values for change points
    ax: np.ndarray
        Array of the figure's Axe objects to plot on

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : np.ndarray
        Array of the figure's Axe objects
    """
    # Checks availability of plotting libraries
    _check_soft_dependencies("matplotlib", "seaborn")
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    ax.plot(df.iloc[:, 0])
    ax.axvline(
        actual_cp,
        label="Wearing Seatbelts made Compulsory",
        color="tab:orange",
        linestyle="--",
    )

    if predicted_change_points is not None:
        for i, cp in enumerate(predicted_change_points.values.flatten()):
            label = "Predicted Change Points" if i == 0 else None
            ax.axvline(cp, color="tab:green", linestyle="--", label=label)

    ax.set_xlabel("Date")
    ax.set_ylabel(df.columns[0])
    ax.legend()
    return fig, ax
