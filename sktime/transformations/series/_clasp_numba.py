# -*- coding: utf-8 -*-
"""Isolated numba imports for clasp."""


__author__ = ["ermshaua", "patrickzib"]

import numpy as np
import pandas as pd

from sktime.transformations.panel.matrix_profile import _sliding_dot_products
from sktime.utils.numba.njit import njit


def _sliding_window(X, m):
    """Return the sliding windows for a time series and a window size.

    Parameters
    ----------
    X : array-like, shape = [n]
        A single univariate time series of length n
    m : int
        The window size to generate sliding windows

    Returns
    -------
    windows : array of shape [n-m+1, m]
        The sliding windows of length over the time series of length n
    """
    shape = X.shape[:-1] + (X.shape[-1] - m + 1, m)
    strides = X.strides + (X.strides[-1],)
    return np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)


def _sliding_mean_std(X, m):
    """Return the sliding mean and std for a time series and a window size.

    Parameters
    ----------
    X : array-like, shape [n]
        A single univariate time series of length n
    m : int
        The window size to generate sliding windows

    Returns
    -------
    Tuple (float, float)
        The moving mean and moving std
    """
    s = np.insert(np.cumsum(X), 0, 0)
    sSq = np.insert(np.cumsum(X**2), 0, 0)
    segSum = s[m:] - s[:-m]
    segSumSq = sSq[m:] - sSq[:-m]
    movmean = segSum / m
    movstd = np.sqrt(segSumSq / m - (segSum / m) ** 2)

    # avoid dividing by too small std, like 0
    movstd = np.where(abs(movstd) < 0.001, 1, movstd)

    return [movmean, movstd]


def _compute_distances_iterative(X, m, k):
    """Compute kNN indices with dot-product.

    No-loops implementation for a time series, given
    a window size and k neighbours.

    Parameters
    ----------
    X : array-like, shape [n]
        A single univariate time series of length n
    m : int
        The window size to generate sliding windows
    k : int
        The number of nearest neighbors

    Returns
    -------
    knns : array-like, shape = [n-m+1, k], dtype=int
        The knns (offsets!) for each subsequence in X
    """
    length = len(X) - m + 1
    knns = np.zeros(shape=(length, k), dtype=np.int64)

    dot_prev = None
    means, stds = _sliding_mean_std(X, m)

    for order in range(0, length):
        # first iteration O(n log n)
        if order == 0:
            # dot_first = _sliding_dot_product(X[:m], X)
            dot_first = _sliding_dot_products(X[:m], X, len(X[:m]), len(X))
            dot_rolled = dot_first
        # O(1) further operations
        else:
            dot_rolled = (
                np.roll(dot_prev, 1)
                + X[order + m - 1] * X[m - 1 : length + m]
                - X[order - 1] * np.roll(X[:length], 1)
            )
            dot_rolled[0] = dot_first[order]

        x_mean = means[order]
        x_std = stds[order]

        dist = 2 * m * (1 - (dot_rolled - m * means * x_mean) / (m * stds * x_std))

        # self-join: exclusion zone
        trivialMatchRange = (
            int(max(0, order - np.round(m / 2, 0))),
            int(min(order + np.round(m / 2 + 1, 0), length)),
        )
        dist[trivialMatchRange[0] : trivialMatchRange[1]] = np.inf

        idx = np.argpartition(dist, k)

        knns[order, :] = idx[:k]
        dot_prev = dot_rolled

    return knns


@njit(fastmath=True, cache=True)
def _calc_knn_labels(knn_mask, split_idx, m):
    """Compute kNN indices relabeling at a given split index.

    Parameters
    ----------
    knn_mask : array-like, shape = [k, n-m+1], dtype=int
        The knn indices for each subsequence
    split_idx : int
        The split index to use
    m : int
        The window size to generate sliding windows

    Returns
    -------
    Tuple (array-like of shape=[n-m+1], array-like of shape=[n-m+1]):
        True labels and predicted labels
    """
    k_neighbours, n_timepoints = knn_mask.shape

    # create labels for given potential split
    y_true = np.concatenate(
        (
            np.zeros(split_idx, dtype=np.int64),
            np.ones(n_timepoints - split_idx, dtype=np.int64),
        )
    )

    knn_mask_labels = np.zeros(shape=(k_neighbours, n_timepoints), dtype=np.int64)

    # relabel the kNN indices
    for i_neighbor in range(k_neighbours):
        neighbours = knn_mask[i_neighbor]
        knn_mask_labels[i_neighbor] = y_true[neighbours]

    # compute kNN prediction
    ones = np.sum(knn_mask_labels, axis=0)
    zeros = k_neighbours - ones
    y_pred = np.asarray(ones > zeros, dtype=np.int64)

    # apply exclusion zone at split point
    exclusion_zone = np.arange(split_idx - m, split_idx)
    y_pred[exclusion_zone] = np.ones(m, dtype=np.int64)

    return y_true, y_pred


@njit(fastmath=True, cache=False)
def _binary_f1_score(y_true, y_pred):
    """Compute f1-score.

    Parameters
    ----------
    y_true : array-like, shape=[n-m+1], dtype = int
        True integer labels for each subsequence
    y_pred : array-like, shape=[n-m+1], dtype = int
        Predicted integer labels for each subsequence

    Returns
    -------
    F1 : float
        F1-score
    """
    f1_scores = np.zeros(shape=2, dtype=np.float64)

    for label in (0, 1):
        tp = np.sum(np.logical_and(y_true == label, y_pred == label))
        fp = np.sum(np.logical_and(y_true != label, y_pred == label))
        fn = np.sum(np.logical_and(y_true == label, y_pred != label))

        pr = tp / (tp + fp)
        re = tp / (tp + fn)

        f1 = 2 * (pr * re) / (pr + re)
        f1_scores[label] = f1

    return np.mean(f1_scores)


@njit(fastmath=True, cache=True)
def _roc_auc_score(y_score, y_true):
    """Compute roc-auc score.

    Parameters
    ----------
    y_true : array-like, shape=[n-m+1], dtype = int
        True integer labels for each subsequence
    y_pred : array-like, shape=[n-m+1], dtype = int
        Predicted integer labels for each subsequence

    Returns
    -------
    F1 : float
        ROC-AUC-score
    """
    # make y_true a boolean vector
    y_true = y_true == 1

    # sort scores and corresponding truth values (y_true is sorted by design)
    desc_score_indices = np.arange(y_score.shape[0])[::-1]

    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.concatenate(
        (distinct_value_indices, np.array([y_true.size - 1]))
    )

    # accumulate the true positives with decreasing threshold
    tps = np.cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    tps = np.concatenate((np.array([0]), tps))
    fps = np.concatenate((np.array([0]), fps))

    if fps[-1] <= 0 or tps[-1] <= 0:
        return np.nan

    fpr = fps / fps[-1]
    tpr = tps / tps[-1]

    if fpr.shape[0] < 2:
        return np.nan

    direction = 1
    dx = np.diff(fpr)

    if np.any(dx < 0):
        if np.all(dx <= 0):
            direction = -1
        else:
            return np.nan

    area = direction * np.trapz(tpr, fpr)
    return area


@njit(fastmath=True)
def _calc_profile(m, knn_mask, score, exclusion_zone):
    """Calculate ClaSP profile for the kNN indices and a score.

    Parameters
    ----------
    m : int
        The window size to generate sliding windows
    knn_mask : array-like, shape = [k, n-m+1], dtype=int
        The knn indices
    score : function
        Scoring method used
    exclusion_zone : int
        Exclusion zone

    Returns
    -------
    profile : array-like, shape=[n-m+1], dtype = float
        The ClaSP
    """
    n_timepoints = knn_mask.shape[1]
    profile = np.full(shape=n_timepoints, fill_value=np.nan, dtype=np.float64)

    for split_idx in range(exclusion_zone, n_timepoints - exclusion_zone):
        y_true, y_pred = _calc_knn_labels(knn_mask, split_idx, m)
        profile[split_idx] = score(y_true, y_pred)

    return profile


def clasp(
    X,
    m,
    k_neighbours=3,
    score=_roc_auc_score,
    interpolate=True,
    exclusion_radius=0.05,
):
    """Calculate ClaSP for a time series and a window size.

    Parameters
    ----------
    X : array-like, shape = [n]
        A single univariate time series of length n
    m : int
        The window size to generate sliding windows
    k_neighbours : int
        The number of knn to use
    score : function
        Scoring method used
    interpolate:
        Interpolate the profile
    exclusion_radius : int
        Blind spot of the profile to the corners

    Returns
    -------
    Tuple (array-like of shape [n], array-like of shape [k_neighbours, n])
        The ClaSP and the knn_mask
    """
    knn_mask = _compute_distances_iterative(X, m, k_neighbours).T

    n_timepoints = knn_mask.shape[1]
    exclusion_radius = np.int64(n_timepoints * exclusion_radius)

    profile = _calc_profile(m, knn_mask, score, exclusion_radius)

    if interpolate is True:
        profile = pd.Series(profile).interpolate(limit_direction="both").to_numpy()
    return profile, knn_mask
