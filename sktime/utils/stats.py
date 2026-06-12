#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Statistical functionality used throughout sktime."""

import numpy as np
from sklearn.utils.extmath import stable_cumsum
from sklearn.utils.validation import check_consistent_length

__author__ = ["RNKuhns", "GuzalBulatova"]
__all__ = [
    "_weighted_geometric_mean",
    "_weighted_median",
    "_weighted_min",
    "_weighted_max",
]


# forked from sklearn.utils to ensure compatibility with newer sklearn versions
def _weighted_percentile(array, sample_weight, percentile=50):
    """Compute lower weighted percentile.

    Computes lower weighted percentile. If `array` is a 2D array, the
    `percentile` is computed along the axis 0.

    Parameters
    ----------
    array : 1D or 2D array
        Values to take the weighted percentile of.

    sample_weight: 1D or 2D array
        Weights for each value in `array`. Must be same shape as `array` or
        of shape `(array.shape[0],)`.

    percentile: int or float, default=50
        Percentile to compute. Must be value between 0 and 100.

    Returns
    -------
    percentile : int if `array` 1D, ndarray if `array` 2D
        Weighted percentile.
    """
    n_dim = array.ndim
    if n_dim == 0:
        return array[()]
    if array.ndim == 1:
        array = array.reshape((-1, 1))
    # When sample_weight 1D, repeat for each array.shape[1]
    if array.shape != sample_weight.shape and array.shape[0] == sample_weight.shape[0]:
        sample_weight = np.tile(sample_weight, (array.shape[1], 1)).T
    sorted_idx = np.argsort(array, axis=0)
    sorted_weights = np.take_along_axis(sample_weight, sorted_idx, axis=0)

    # Find index of median prediction for each sample
    weight_cdf = stable_cumsum(sorted_weights, axis=0)
    adjusted_percentile = percentile / 100 * weight_cdf[-1]

    # For percentile=0, ignore leading observations with sample_weight=0. GH20528
    mask = adjusted_percentile == 0
    adjusted_percentile[mask] = np.nextafter(
        adjusted_percentile[mask], adjusted_percentile[mask] + 1
    )

    percentile_idx = np.array(
        [
            np.searchsorted(weight_cdf[:, i], adjusted_percentile[i])
            for i in range(weight_cdf.shape[1])
        ]
    )
    percentile_idx = np.array(percentile_idx)
    # In rare cases, percentile_idx equals to sorted_idx.shape[0]
    max_idx = sorted_idx.shape[0] - 1
    percentile_idx = np.apply_along_axis(
        lambda x: np.clip(x, 0, max_idx), axis=0, arr=percentile_idx
    )

    col_index = np.arange(array.shape[1])
    percentile_in_sorted = sorted_idx[percentile_idx, col_index]
    percentile = array[percentile_in_sorted, col_index]
    return percentile[0] if n_dim == 1 else percentile


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
