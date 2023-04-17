# -*- coding: utf-8 -*-
"""Utilities for seasonality."""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
# noqa: D100

__author__ = ["mloning", "fkiraly"]
__all__ = []

from warnings import warn

import numpy as np
import pandas as pd

from sktime.utils.validation._dependencies import _check_soft_dependencies
from sktime.utils.validation.forecasting import check_sp, check_y


def autocorrelation_seasonality_test(y, sp):
    """Seasonality test used in M4 competition.

    Parameters
    ----------
    sp : int
        Seasonal periodicity

    Returns
    -------
    is_seasonal : bool
        Test result

    References
    ----------
    .. [1]  https://github.com/Mcompetitions/M4-methods/blob/master
    /Benchmarks%20and%20Evaluation.R
    """
    _check_soft_dependencies("statsmodels")
    from statsmodels.tsa.stattools import acf

    y = check_y(y)
    sp = check_sp(sp)

    y = np.asarray(y)
    n_timepoints = len(y)

    if sp == 1:
        return False

    if n_timepoints < 3 * sp:
        warn(
            "Did not perform seasonality test, as `y`` is too short for the "
            "given `sp`, returned: False"
        )
        return False

    else:
        coefs = acf(y, nlags=sp, fft=False)  # acf coefficients
        coef = coefs[sp]  # coefficient to check

        tcrit = 1.645  # 90% confidence level
        limits = (
            tcrit
            / np.sqrt(n_timepoints)
            * np.sqrt(np.cumsum(np.append(1, 2 * coefs[1:] ** 2)))
        )
        limit = limits[sp - 1]  #  zero-based indexing
        return np.abs(coef) > limit


def _pivot_sp(df, sp, anchor=None, freq=None):

    if isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = df.index.to_period(freq=freq)
        was_datetime = True
    else:
        was_datetime = False

    if anchor is None:
        anchor = df

    if not isinstance(df.index, pd.PeriodIndex):
        period_len = anchor.index[1] - anchor.index[0]
        ix = (df.index - anchor.index[0]) / period_len
    else:
        ix = df.index
    ix = ix.astype("int64")

    df = pd.DataFrame(df)
    df_pivot = pd.pivot_table(
        data=df,
        index=ix // sp,  # Upper level
        columns=ix % sp,  # Lower level
        dropna=False,
    )
    # ix_selector = ix % sp == 0

    if isinstance(df.index, pd.PeriodIndex):
        if isinstance(anchor.index, pd.DatetimeIndex):
            aix = anchor.index.to_period(freq=freq)
        else:
            aix = anchor.index

        n = len(df_pivot)
        # need to correct for anchor being 1970 in int conversion
        offset = df_pivot.index * sp - aix[[0] * n].astype("int64")
        pivot_ix = aix[[0] * n] + offset
    else:
        n = len(df_pivot)
        pivot_ix = anchor.index[[0] * n] + df_pivot.index * sp

    df_pivot.index = pivot_ix
    # df_pivot.index = df.index[ix_selector]
    # df_pivot.index = df_pivot.index * sp

    if was_datetime:
        df_pivot.index = df_pivot.index.to_timestamp()

    df_pivot.columns = df_pivot.columns.droplevel(0)

    return df_pivot


def _unpivot_sp(df, template=None):

    if template is not None:
        if hasattr(template, "index") and hasattr(template.index, "freq"):
            freq = template.index.freq
        elif hasattr(template, "freq"):
            freq = template.freq
        else:
            freq = template

    df_melt = df.melt(ignore_index=False)

    offset = df_melt[df_melt.columns[0]]
    if isinstance(df_melt.index, pd.DatetimeIndex):
        a = df_melt.index.to_period(freq=freq)
        res = a + offset
        df_melt.index = res
        was_datetime = True
    else:
        df_melt.index = df_melt.index + offset
        was_datetime = False
    df_melt = df_melt.drop(columns=df_melt.columns[0])
    df_melt = df_melt.sort_index()
    df_melt = df_melt.dropna()
    # df_melt.columns = df.columns.get_level_values(0).unique()

    if was_datetime:
        df_melt.index = df_melt.index.to_timestamp()

    if template is not None:
        df_melt.columns = template.columns

    return df_melt
