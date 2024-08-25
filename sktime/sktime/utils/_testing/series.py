#!/usr/bin/env python3 -u

__author__ = ["mloning"]
__all__ = []

import numpy as np
import pandas as pd
from sklearn.utils import check_random_state


def _make_series(
    n_timepoints=50,
    n_columns=1,
    all_positive=True,
    index_type=None,
    return_numpy=False,
    random_state=None,
    add_nan=False,
    return_mtype=None,
):
    """Generate univariate or multivariate time series.

    Utility for interface contact testing.

    Will produce a time series in sktime compatible mtype data format, for testing.
    Samples values i.i.d. from standard normal distribution.

    Parameters
    ----------
    n_timepoints : int, default=50
        number of time points in the series
    n_columns : int, default=1
        number of columns in the series
    all_positive : bool, default=False
        if True, subtracts minimum and adds 1 from values after generating the series
        ensures all the values generated are greater equal 1
    index_type : str, one of "period", "datetime" (default), "range", "int"
        the index of the returned object, if a pandas object; otherwise has no effect
        "period" - `pd.PeriodIndex`, monthly (M) starting at Jan 2000 (incl)
        "datetime" - `pd.DatetimeIndex`, daily (D) starting at Jan 1, 2000 (incl)
        "range" - `pd.RangeIndex`, starting at 3 (incl)
        "int" - `pd.Index` of `int` dtype, starting at 3 (incl)
    random_state : None (default), `int` or `np.random.RandomState`
        random seed for sampling, if `None`, will use default `np.random` generation
    add_nan : bool, default=False
        whether to include nans in the series.
        If `True`, data will contain three `np.nan` entries, at start, end and middle
    return_mtype : str, sktime Series mtype str
        default="pd.DataFrame" if `n_columns>1`, and "pd.Series" if `n_columns==1`
        see sktime.datatypes.MTYPE_LIST_SERIES for a full list of admissible strings
        see sktime.datatypes.MTYPE_REGISTER for an short explanation of formats
        see examples/AA_datatypes_and_datasets.ipynb for a full specification

    Returns
    -------
    X : an `sktime` time series data container of mtype `return_mtype`
        with `n_columns` variables, `n_timepoints` time points
        index is as per `index_type` for `pandas` return objects
        generating distribution is all values i.i.d. standard normal
        if `all_positive=True`, subtracts minimum and adds 1
    """
    rng = check_random_state(random_state)
    data = rng.normal(size=(n_timepoints, n_columns))
    if add_nan:
        # add some nan values
        data[int(len(data) / 2)] = np.nan
        data[0] = np.nan
        data[-1] = np.nan
    if all_positive:
        data -= np.min(data, axis=0) - 1

    # np.ndarray case
    if return_numpy or return_mtype == "np.ndarray":
        if n_columns == 1:
            data = data.ravel()
        return data

    # pd.Series, pd.DataFrame case
    index = _make_index(n_timepoints, index_type)
    if n_columns == 1 and return_mtype is None or return_mtype == "pd.Series":
        return pd.Series(data.ravel(), index)
    elif return_mtype is None or return_mtype == "pd.DataFrame":
        return pd.DataFrame(data, index)

    # all other mtypes, convert from pd.DataFrame
    from sktime.datatypes import convert

    res = pd.DataFrame(data, index)
    res_conv = convert(res, "pd.DataFrame", return_mtype, "Series")
    return res_conv


def _make_index(n_timepoints, index_type=None):
    """Make indices for unit testing."""
    if index_type == "period":
        start = "2000-01"
        freq = "M"
        return pd.period_range(start=start, periods=n_timepoints, freq=freq)

    elif index_type == "datetime" or index_type is None:
        start = "2000-01-01"
        freq = "D"
        return pd.date_range(start=start, periods=n_timepoints, freq=freq)

    elif index_type == "range":
        start = 3  # check non-zero based indices
        return pd.RangeIndex(start=start, stop=start + n_timepoints)

    elif index_type == "int":
        start = 3
        return pd.Index(np.arange(start, start + n_timepoints), dtype=int)

    else:
        raise ValueError(f"index_class: {index_type} is not supported")
