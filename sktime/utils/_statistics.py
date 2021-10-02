#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Statistical functionality used throughout sktime."""

import numpy as np
from sklearn.utils.validation import check_consistent_length

__author__ = ["RNKuhns"]
__all__ = ["weighted_geometric_mean"]


def weighted_geometric_mean(x, sample_weight=None, axis=None):
    """Calculate weighted version of geometric mean.

    Parameters
    ----------
    array : np.ndarray
        Values to take the weighted geometric mean of.
    sample_weight: np.ndarray
        Weights for each value in `array`. Must be same shape as `array` or
        of shape `(array.shape[0],)`.

    Returns
    -------
    geometric_mean : float
        Weighted geometric mean
    """
    check_consistent_length(x, sample_weight)
    return np.exp(
        np.sum(sample_weight * np.log(x), axis=axis) / np.sum(sample_weight, axis=axis)
    )
