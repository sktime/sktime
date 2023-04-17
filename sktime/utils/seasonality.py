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
    """Pivot univariate series to multivariate-by-seasonal-offset.

    For an input `df: pd.DataFrame` or `pd.Series`, with regular index,
    with one variable, outputs the following:
    a `pd.DataFrame` where:

    * row index is `anchor.index[0]` plus multiples of `sp` times periodicity of `df`,
      index elements present are those where there is at least one index in `df`
      at which a value is observed between output row index and subsequent index
    * column index is 0 ... `sp - 1`
    * the entry in row location `i`, column location `j` is the entry at
      `df.loc[i + j * period_of_df]`, where `period_of_df` is regular period of `df`

    Parameters
    ----------
    `df` : `pd.Series` or `pd.DataFrame` with `pandas` integer index,
        `pd.DatetimeIndex`, or `pd.PeriodIndex`, and with one column/variable
    `sp` : int
        seasonality/periodicity parameter of the pivot
    `anchor` : None, or `pd.Series` or `pd.DataFrame`
        anchor data frame for the pivot, equal to `df` if not provided
    `freq` : None, or `pd.Series`, `pd.DataFrame`, `pd.Index`, or `pandas` frequency
        if None, equal to df.index.freq
        if provided, will be used as frequency in offset calculations
        needed only of `df` is `pd.DatetimeIndex` or `pd.PeriodIndex` without `freq`
    """
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = df.index.to_period(freq=freq)
        was_datetime = True
    else:
        was_datetime = False

    if anchor is None:
        anchor = df

    if not isinstance(df.index, pd.PeriodIndex):
        if pd.api.types.is_integer_dtype(anchor.index) and len(anchor) <= 1:
            period_len = 1
        else:
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

    if was_datetime:
        df_pivot.index = df_pivot.index.to_timestamp()

    df_pivot.columns = df_pivot.columns.droplevel(0)

    return df_pivot


def _unpivot_sp(df, template=None):
    """Unpivot DataFrame with multivariate-by-seasonal-offset, invert _pivot_sp.

    Inverse operation to `_pivot_sp`.

    For a `pd.DataFrame` that is like the output of `_pivot_sp`, produces
    a univariate `pd.DataFrame` where:

    * row index consists of all combination sums of `df.index` and `df.columns`
    * column index is `template.index` if `template` is given, otherwise `RangeIndex`
    * the entry in row location `i + j`,
      for `i` a row and `j` a column of `df`, is `df.loc[i][j]`

    Parameters
    ----------
    `df` : `pd.Series` or `pd.DataFrame` with `pandas` integer index,
        `pd.DatetimeIndex`, or `pd.PeriodIndex`, and with one column/variable
    `template` : None, or `pd.Series` or `pd.DataFrame`
        template data frame for the unpivot, equal to `df` if not provided
    """
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

    if was_datetime:
        df_melt.index = df_melt.index.to_timestamp()

    if template is not None:
        df_melt.columns = template.columns

    return df_melt
