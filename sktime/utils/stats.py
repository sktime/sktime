#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Statistical functionality used throughout sktime."""

import numpy as np
from sklearn.utils.stats import _weighted_percentile
from sklearn.utils.validation import check_consistent_length

__author__ = ["RNKuhns", "GuzalBulatova"]
__all__ = [
    "_weighted_geometric_mean",
    "_weighted_median",
    "_weighted_min",
    "_weighted_max",
]


def _weighted_geometric_mean(y, weights=None, axis=None):
    """Calculate weighted version of geometric mean.

    Parameters
    ----------
    y : np.ndarray
        Values to take the weighted geometric mean of.
    weights: np.ndarray
        Weights for each value in `array`. Must be same shape as `array` or
        of shape `(array.shape[0],)` if axis=0 or `(array.shape[1], ) if axis=1.
    axis : int
        The axis of `y` to apply the weights to.

    Returns
    -------
    geometric_mean : float
        Weighted geometric mean
    """
    if weights.ndim == 1:
        if axis == 0:
            check_consistent_length(y, weights)
        elif axis == 1:
            if y.shape[1] != len(weights):
                raise ValueError(
                    f"Input features ({y.shape[1]}) do not match "
                    f"number of `weights` ({len(weights)})."
                )
        weight_sums = np.sum(weights)
    else:
        if y.shape != weights.shape:
            raise ValueError("Input data and weights have inconsistent shapes.")
        weight_sums = np.sum(weights, axis=axis)
    return np.exp(np.sum(weights * np.log(y), axis=axis) / weight_sums)


def _weighted_median(y, axis=1, weights=None):
    """Calculate weighted median.

    Parameters
    ----------
    y : np.ndarray, pd.Series or pd.DataFrame
        Values to take the weighted median of.
    weights: np.ndarray
        Weights for each value in `array`. Must be same shape as `array` or
        of shape `(array.shape[0],)` if axis=0 or `(array.shape[1], ) if axis=1.
    axis : int
        The axis of `y` to apply the weights to.

    Returns
    -------
    w_median : float
        Weighted median
    """
    w_median = np.apply_along_axis(
        func1d=_weighted_percentile,
        axis=axis,
        arr=y,
        sample_weight=weights,
        percentile=50,
    )
    return w_median


def _weighted_min(y, axis=1, weights=None):
    """Calculate weighted minimum.

    Parameters
    ----------
    y : np.ndarray, pd.Series or pd.DataFrame
        Values to take the weighted minimum of.
    weights: np.ndarray
        Weights for each value in `array`. Must be same shape as `array` or
        of shape `(array.shape[0],)` if axis=0 or `(array.shape[1], ) if axis=1.
    axis : int
        The axis of `y` to apply the weights to.

    Returns
    -------
    w_min : float
        Weighted minimum
    """
    w_min = np.apply_along_axis(
        func1d=_weighted_percentile,
        axis=axis,
        arr=y,
        sample_weight=weights,
        percentile=0,
    )
    return w_min


def _weighted_max(y, axis=1, weights=None):
    """Calculate weighted maximum.

    Parameters
    ----------
    y : np.ndarray, pd.Series or pd.DataFrame
        Values to take the weighted maximum of.
    weights: np.ndarray
        Weights for each value in `array`. Must be same shape as `array` or
        of shape `(array.shape[0],)` if axis=0 or `(array.shape[1], ) if axis=1.
    axis : int
        The axis of `y` to apply the weights to.

    Returns
    -------
    w_max : float
        Weighted maximum
    """
    w_max = np.apply_along_axis(
        func1d=_weighted_percentile,
        axis=axis,
        arr=y,
        sample_weight=weights,
        percentile=100,
    )
    return w_max
