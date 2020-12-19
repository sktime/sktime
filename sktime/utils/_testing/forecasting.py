#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Markus Löning"]
__all__ = [
    "_get_expected_index_for_update_predict",
    "_generate_polynomial_series",
    "make_forecasting_problem",
    "_make_series",
    "_get_expected_index_for_update_predict",
    "make_forecasting_problem",
]

import numpy as np
import pandas as pd

from sktime.forecasting.base import ForecastingHorizon
from sktime.utils._testing.series import _make_series
from sktime.utils.validation.forecasting import check_fh


def _get_expected_index_for_update_predict(y, fh, step_length):
    """Helper function to compute expected time index from `update_predict`"""
    # time points at which to make predictions
    fh = check_fh(fh)
    index = y.index

    # only works with date-time index
    assert isinstance(index, pd.DatetimeIndex)
    assert hasattr(index, "freq") and index.freq is not None
    assert fh.is_relative

    freq = index.freq
    start = index[0] - 1 * freq  # initial cutoff
    end = index[-1]  # last point to predict

    # generate date-time range
    cutoffs = pd.date_range(start, end)

    # only predict at time points if all steps in fh can be predicted before
    # the end of y_test
    cutoffs = cutoffs[cutoffs + max(fh) * freq <= max(index)]

    # apply step length and recast to ignore inferred freq value
    cutoffs = cutoffs[::step_length]
    cutoffs = pd.DatetimeIndex(cutoffs, freq=None)

    # generate all predicted time points, including duplicates from overlapping fh steps
    pred_index = pd.DatetimeIndex([])
    for step in fh:
        values = cutoffs + step * freq
        pred_index = pred_index.append(values)

    # return unique and sorted index
    return pred_index.unique().sort_values()


def _generate_polynomial_series(n, order, coefs=None):
    """Helper function to generate polynomial series of given order and
    coefficients"""
    if coefs is None:
        coefs = np.ones((order + 1, 1))
    x = np.vander(np.arange(n), N=order + 1).dot(coefs)
    return x.ravel()


def make_forecasting_problem(
    n_timepoints=50,
    all_positive=True,
    index_type=None,
    make_X=False,
    n_columns=2,
    random_state=None,
):
    y = _make_series(
        n_timepoints=n_timepoints,
        n_columns=1,
        all_positive=all_positive,
        index_type=index_type,
        random_state=random_state,
    )

    if not make_X:
        return y

    X = _make_series(
        n_timepoints=n_timepoints,
        n_columns=n_columns,
        all_positive=all_positive,
        index_type=index_type,
        random_state=random_state,
    )
    X.index = y.index
    return y, X


def _assert_correct_pred_time_index(y_pred_index, cutoff, fh):
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
