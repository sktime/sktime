#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Markus Löning"]
__all__ = [
    "compute_expected_index_from_update_predict",
    "_generate_polynomial_series",
    "make_forecasting_problem",
    "_make_series",
]

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_random_state

from sktime.utils.validation.forecasting import check_fh
from sktime.utils.validation.forecasting import check_y


def compute_expected_index_from_update_predict(y, fh, step_length):
    """Helper function to compute expected time index from `update_predict`"""
    # time points at which to make predictions
    fh = check_fh(fh)
    y = check_y(y)
    index = y.index.values

    start = index[0] - 1  # initial cutoff
    end = index[-1]  #  last point to predict
    cutoffs = np.arange(start, end, step_length)

    # only predict at time points if all steps in fh can be predicted before
    # the end of y_test
    cutoffs = cutoffs[cutoffs + max(fh) <= max(index)]
    n_cutoffs = len(cutoffs)

    # all time points predicted, including duplicates from overlapping fhs
    fh_broadcasted = np.repeat(fh, n_cutoffs).reshape(len(fh), n_cutoffs)
    pred_index = cutoffs + fh_broadcasted

    # return only unique time points
    return np.unique(pred_index)


def _make_series(n_timepoints=75, all_positive=True, non_zero_index=False):
    """Helper function to generate single time series"""
    series = np.random.normal(size=n_timepoints)
    if all_positive:
        series -= np.min(series) - 1
    index = np.arange(n_timepoints)
    if non_zero_index:
        index += 30
    return pd.Series(series, index=pd.Int64Index(index))


def _generate_polynomial_series(n, order, coefs=None):
    """Helper function to generate polynomial series of given order and
    coefficients"""
    if coefs is None:
        coefs = np.ones((order + 1, 1))

    x = np.vander(np.arange(n), N=order + 1).dot(coefs)
    return x.ravel()


def make_forecasting_problem(n_timepoints=50, random_state=None):
    rng = check_random_state(random_state)
    return pd.Series(
        rng.random(size=n_timepoints), index=pd.Int64Index(np.arange(n_timepoints))
    )


def assert_correct_pred_time_index(y_pred, y_train, fh):
    fh = check_fh(fh)
    np.testing.assert_array_equal(y_pred.index, y_train.index[-1] + fh)
