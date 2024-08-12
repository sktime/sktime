#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["mloning"]
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


def _get_n_columns(tag):
    """Return the the number of columns to use in tests."""
    n_columns_list = []
    if tag in ["univariate", "both"]:
        n_columns_list = [1, 2]
    elif tag == "multivariate":
        n_columns_list = [2]
    else:
        raise ValueError(f"Unexpected tag {tag} in _get_n_columns.")
    return n_columns_list


def _get_expected_index_for_update_predict(y, fh, step_length, initial_window):
    """Compute expected time index from update_predict()."""
    # time points at which to make predictions
    fh = check_fh(fh)
    index = y.index

    # only works with date-time index
    assert isinstance(index, pd.DatetimeIndex)
    assert hasattr(index, "freq") and index.freq is not None
    assert fh.is_relative

    freq = index.freq
    start = index[0] + (-1 + initial_window) * freq  # initial cutoff
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
    """Generate polynomial series of given order and coefficients."""
    if coefs is None:
        coefs = np.ones((order + 1, 1))
    x = np.vander(np.arange(n), N=order + 1).dot(coefs)
    return x.ravel()


def make_forecasting_problem(
    n_timepoints=50,
    all_positive=True,
    index_type=None,
    make_X=False,
    n_columns=1,
    random_state=None,
):
    """Return test data for forecasting tests.

    Parameters
    ----------
    n_timepoints : int, optional
        Length of data, by default 50
    all_positive : bool, optional
        Only positive values or not, by default True
    index_type : e.g. pd.PeriodIndex, optional
        pandas Index type, by default None
    make_X : bool, optional
        Should X data also be returned, by default False
    n_columns : int, optional
        Number of columns of y, by default 1
    random_state : inst, str, float, optional
        Set seed of random state, by default None

    Returns
    -------
    ps.Series, pd.DataFrame
        y, if not make_X
        y, X if make_X
    """
    y = _make_series(
        n_timepoints=n_timepoints,
        n_columns=n_columns,
        all_positive=all_positive,
        index_type=index_type,
        random_state=random_state,
    )

    if not make_X:
        return y

    X = _make_series(
        n_timepoints=n_timepoints,
        n_columns=2,
        all_positive=all_positive,
        index_type=index_type,
        random_state=random_state,
    )
    X.index = y.index
    return y, X


def _assert_correct_pred_time_index(y_pred_index, cutoff, fh):
    assert isinstance(y_pred_index, pd.Index)
    fh = check_fh(fh)
    expected = fh.to_absolute_index(cutoff)
    assert y_pred_index.equals(expected)


def _assert_correct_columns(y_pred, y_train):
    """Check that forecast object has right column names."""
    if isinstance(y_pred, pd.DataFrame) and isinstance(y_train, pd.DataFrame):
        msg = (
            "forecast pd.DataFrame must have same column index as past data, "
            f"expected {y_train.columns} but found {y_pred.columns}"
        )
        assert (y_pred.columns == y_train.columns).all(), msg

    if isinstance(y_pred, pd.Series) and isinstance(y_train, pd.Series):
        msg = (
            "forecast pd.Series must have same name as past data, "
            f"expected {y_train.name} but found {y_pred.name}"
        )
        assert y_pred.name == y_train.name, msg


def _make_fh(cutoff, steps, fh_type, is_relative):
    """Construct forecasting horizons for testing."""
    from sktime.forecasting.tests._config import INDEX_TYPE_LOOKUP

    fh_class = INDEX_TYPE_LOOKUP[fh_type]

    if isinstance(steps, (int, np.integer)):
        steps = np.array([steps], dtype=int)

    elif isinstance(steps, pd.Timedelta):
        steps = [steps]

    if is_relative:
        return ForecastingHorizon(fh_class(steps), is_relative=is_relative)

    else:
        kwargs = {}

        if fh_type in ["datetime", "period"]:
            cutoff_freq = cutoff.freq
        if isinstance(cutoff, pd.Index):
            cutoff = cutoff[0]

        if fh_type == "datetime":
            steps *= cutoff_freq

        if fh_type == "period":
            kwargs = {"freq": cutoff_freq}

        values = cutoff + steps
        return ForecastingHorizon(fh_class(values, **kwargs), is_relative)
