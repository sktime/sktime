#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-

__author__ = ["Markus Löning"]
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
):
    """Generate univariate or multivariate time series."""
    rng = check_random_state(random_state)
    data = rng.normal(size=(n_timepoints, n_columns))
    if add_nan:
        # add some nan values
        data[int(len(data) / 2)] = np.nan
        data[0] = np.nan
        data[-1] = np.nan
    if all_positive:
        data -= np.min(data, axis=0) - 1
    if return_numpy:
        if n_columns == 1:
            data = data.ravel()
        return data
    else:
        index = _make_index(n_timepoints, index_type)
        if n_columns == 1:
            return pd.Series(data.ravel(), index)
        else:
            return pd.DataFrame(data, index)


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


def _load_solar(
    start="2021-05-01", end="2021-09-01", normalise=True, return_full_df=False
):
    """Get solar estimates for GB from Sheffield solar API for ClearSky."""
    url = "https://api0.solar.sheffield.ac.uk/pvlive/api/v4/gsp/0?"
    url = url + "start=" + start + "T00:00:00&"
    url = url + "end=" + end + "T00:00:00&"
    url = url + "extra_fields=capacity_mwp&"
    url = url + "data_format=csv"

    df = (
        pd.read_csv(
            url, index_col=["gsp_id", "datetime_gmt"], parse_dates=["datetime_gmt"]
        )
        .droplevel(0)
        .sort_index()
    )
    df = df.asfreq("30T")
    df["generation_pu"] = df["generation_mw"] / df["capacity_mwp"]

    if return_full_df:
        df["generation_pu"] = df["generation_mw"] / df["capacity_mwp"]
        return df
    else:
        if normalise:
            return df["generation_pu"].rename("solar_gen")
        else:
            return df["generation_mw"].rename("solar_gen")
