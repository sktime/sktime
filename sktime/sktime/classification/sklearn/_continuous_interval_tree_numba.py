"""Continuous interval tree (CIT) vector classifier (aka Time Series Tree).

Continuous Interval Tree aka Time Series Tree, base classifier originally used in the
time series forest interval based classification algorithm. Fits sklearn conventions.
"""

__author__ = ["MatthewMiddlehurst"]

import math

import numpy as np

from sktime.utils.numba.njit import njit
from sktime.utils.numba.stats import iqr, mean, numba_max, numba_min, slope, std


def _summary_stat(X, att):
    if att == 22:
        function = mean
    elif att == 23:
        function = std
    elif att == 24:
        function = slope
    elif att == 25:
        function = np.median
    elif att == 26:
        function = iqr
    elif att == 27:
        function = numba_min
    elif att == 28:
        function = numba_max
    else:
        raise ValueError("Invalid summary stat ID.")

    return np.array([function(i) for i in X])


@njit(fastmath=True, cache=True)
def _entropy(x, s):
    """Entropy function."""
    e = 0
    for i in x:
        p = i / s if s > 0 else 0
        e += -(p * math.log(p) / 0.6931471805599453) if p > 0 else 0
    return e


def _drcif_feature(X, interval, dim, att, c22, case_id=None):
    """Compute DrCIF feature."""
    if att > 21:
        return _summary_stat(X[:, dim, interval[0] : interval[1]], att)
    else:
        return c22._transform_single_feature(
            X[:, dim, interval[0] : interval[1]], att, case_id=case_id
        )


@njit(fastmath=True, cache=True)
def information_gain(X, y, attribute, threshold, parent_entropy, n_classes):
    """Information gain function."""
    dist_left = np.zeros(n_classes)
    dist_right = np.zeros(n_classes)
    dist_missing = np.zeros(n_classes)
    for i, case in enumerate(X):
        if case[attribute] <= threshold:
            dist_left[y[i]] += 1
        elif case[attribute] > threshold:
            dist_right[y[i]] += 1
        else:
            dist_missing[y[i]] += 1

    sum_missing = 0
    for v in dist_missing:
        sum_missing += v
    sum_left = 0
    for v in dist_left:
        sum_left += v
    sum_right = 0
    for v in dist_right:
        sum_right += v

    entropy_left = _entropy(dist_left, sum_left)
    entropy_right = _entropy(dist_right, sum_right)
    entropy_missing = _entropy(dist_missing, sum_missing)

    num_cases = X.shape[0]
    info_gain = (
        parent_entropy
        - sum_left / num_cases * entropy_left
        - sum_right / num_cases * entropy_right
        - sum_missing / num_cases * entropy_missing
    )

    return (
        info_gain,
        [dist_left, dist_right, dist_missing],
        [entropy_left, entropy_right, entropy_missing],
    )


@njit(fastmath=True, cache=True)
def margin_gain(X, attribute, threshold):
    """Margin gain function."""
    margins = np.abs(X[:, attribute] - threshold)
    return np.min(margins)


@njit(fastmath=True, cache=True)
def split_data(X, best_split, best_threshold):
    """Split data."""
    left_idx = np.zeros(len(X), dtype=np.int_)
    left_count = 0
    right_idx = np.zeros(len(X), dtype=np.int_)
    right_count = 0
    missing_idx = np.zeros(len(X), dtype=np.int_)
    missing_count = 0
    for i, case in enumerate(X):
        if case[best_split] <= best_threshold:
            left_idx[left_count] = i
            left_count += 1
        elif case[best_split] > best_threshold:
            right_idx[right_count] = i
            right_count += 1
        else:
            missing_idx[missing_count] = i
            missing_count += 1

    return (
        left_idx[:left_count],
        right_idx[:right_count],
        missing_idx[:missing_count],
    )


@njit(fastmath=True, cache=True)
def remaining_classes(distribution):
    """Compute whether there are classes remaining."""
    remaining_classes = 0
    for d in distribution:
        if d > 0:
            remaining_classes += 1
    return remaining_classes > 1
