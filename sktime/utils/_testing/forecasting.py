#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Markus Löning"]
__all__ = [
    "get_expected_index_for_update_predict",
    "_generate_polynomial_series",
    "make_forecasting_problem",
    "get_expected_index_for_update_predict",
    "make_forecasting_problem",
]

import numpy as np
import pandas as pd

from sktime.forecasting.base import ForecastingHorizon
from sktime.utils._testing.series import _make_series
from sktime.utils.validation.forecasting import check_fh
from sktime.utils.validation.forecasting import check_y


def get_expected_index_for_update_predict(y, fh, step_length):
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


def _generate_polynomial_series(n, order, coefs=None):
    """Helper function to generate polynomial series of given order and
    coefficients"""
    if coefs is None:
        coefs = np.ones((order + 1, 1))
    x = np.vander(np.arange(n), N=order + 1).dot(coefs)
    return x.ravel()


def make_forecasting_problem(
    n_timepoints=50, n_columns=1, all_positive=True, index_type=None, random_state=None
):
    return _make_series(
        n_timepoints=n_timepoints,
        n_columns=n_columns,
        all_positive=all_positive,
        index_type=index_type,
        random_state=random_state,
    )


def assert_correct_pred_time_index(y_pred_index, cutoff, fh):
    assert isinstance(y_pred_index, pd.Index)
    fh = check_fh(fh)
    expected = fh.to_absolute(cutoff).to_pandas()
    y_pred_index.equals(expected)


def _make_fh(cutoff, steps, fh_type, is_relative):
    """Helper function to construct forecasting horizons for testing"""
    from sktime.forecasting.tests._config import INDEX_TYPE_LOOKUP

    fh_class = INDEX_TYPE_LOOKUP[fh_type]

    if isinstance(steps, (int, np.integer)):
        steps = np.array([steps], dtype=np.int)

    if is_relative:
        return ForecastingHorizon(fh_class(steps), is_relative=is_relative)

    else:
        kwargs = {}

        if fh_type == "datetime":
            steps *= cutoff.freq

        if fh_type == "period":
            kwargs = {"freq": cutoff.freq}

        values = cutoff + steps
        return ForecastingHorizon(fh_class(values, **kwargs), is_relative)
