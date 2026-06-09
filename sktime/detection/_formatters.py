# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Output formatting utilities for detection results.

Convert raw detection outputs (change point arrays, segment intervals) to
the ``pd.DataFrame`` format expected by sktime's ``BaseDetector.predict()``.
"""

import numpy as np
import pandas as pd


def format_changepoints(changepoint_indices):
    """Format change point indices as a Detector-compatible DataFrame.

    Parameters
    ----------
    changepoint_indices : np.ndarray
        1D integer array of iloc-based change point positions.

    Returns
    -------
    pd.DataFrame
        DataFrame with single ``"ilocs"`` column.
    """
    if len(changepoint_indices) == 0:
        return pd.DataFrame({"ilocs": pd.array([], dtype="int64")})
    return pd.DataFrame({"ilocs": changepoint_indices.astype(int)})


def format_segments(starts, ends):
    """Format segment boundaries as a Detector-compatible DataFrame.

    Parameters
    ----------
    starts : np.ndarray
        1D integer array of segment start indices (inclusive).
    ends : np.ndarray
        1D integer array of segment end indices (exclusive).

    Returns
    -------
    pd.DataFrame
        DataFrame with ``"ilocs"`` column containing ``pd.Interval`` objects.
    """
    if len(starts) == 0:
        return pd.DataFrame(columns=["ilocs"])
    intervals = [
        pd.Interval(int(s), int(e), closed="left") for s, e in zip(starts, ends)
    ]
    return pd.DataFrame({"ilocs": intervals})


def format_anomalies(anomaly_indices):
    """Format anomaly indices as a Detector-compatible DataFrame.

    Parameters
    ----------
    anomaly_indices : np.ndarray
        1D integer array of anomaly iloc positions.

    Returns
    -------
    pd.DataFrame
        DataFrame with single ``"ilocs"`` column.
    """
    if len(anomaly_indices) == 0:
        return pd.DataFrame(columns=["ilocs"])
    return pd.DataFrame({"ilocs": anomaly_indices.astype(int)})


def format_anomaly_segments(starts, ends):
    """Format anomalous segment intervals as a Detector-compatible DataFrame.

    Anomaly segments are intervals of consecutive anomalous observations.

    Parameters
    ----------
    starts : np.ndarray
        1D integer array of anomaly segment start indices (inclusive).
    ends : np.ndarray
        1D integer array of anomaly segment end indices (exclusive).

    Returns
    -------
    pd.DataFrame
        DataFrame with ``"ilocs"`` column containing ``pd.Interval`` objects.
    """
    return format_segments(starts, ends)


def changepoints_to_segments(changepoint_indices, n_samples):
    """Convert change point indices to segment boundaries.

    Parameters
    ----------
    changepoint_indices : np.ndarray
        1D sorted integer array of change point iloc positions.
    n_samples : int
        Total number of samples.

    Returns
    -------
    starts : np.ndarray
        Segment start indices (inclusive).
    ends : np.ndarray
        Segment end indices (exclusive).
    """
    if len(changepoint_indices) == 0:
        return np.array([0]), np.array([n_samples])
    cpts = np.asarray(changepoint_indices)
    starts = np.concatenate(([0], cpts))
    ends = np.concatenate((cpts, [n_samples]))
    return starts, ends


def format_labeled_anomaly_segments(anomalies):
    """Format anomaly (start, end) tuples as a labeled DataFrame.

    Parameters
    ----------
    anomalies : list of tuple(int, int)
        Each ``(start, end)`` pair represents an anomalous segment
        (start inclusive, end exclusive).

    Returns
    -------
    pd.DataFrame
        DataFrame with ``"ilocs"`` (``pd.Interval``) and ``"labels"`` columns.
    """
    if len(anomalies) == 0:
        return pd.DataFrame(columns=["ilocs", "labels"])
    intervals = [pd.Interval(int(s), int(e), closed="left") for s, e in anomalies]
    labels = list(range(1, len(anomalies) + 1))
    return pd.DataFrame({"ilocs": intervals, "labels": labels})


def format_anomaly_points(anomalies):
    """Expand anomaly segment intervals into individual point indices.

    Parameters
    ----------
    anomalies : list of tuple(int, int)
        Each ``(start, end)`` pair represents an anomalous segment
        (start inclusive, end exclusive).

    Returns
    -------
    pd.DataFrame
        DataFrame with ``"ilocs"`` column of integer indices.
    """
    if len(anomalies) == 0:
        return pd.DataFrame({"ilocs": pd.array([], dtype="int64")})
    ilocs = []
    for start, end in anomalies:
        ilocs.extend(range(int(start), int(end)))
    return pd.DataFrame({"ilocs": pd.array(ilocs, dtype="int64")})
