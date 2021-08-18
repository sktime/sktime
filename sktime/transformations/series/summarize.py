#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements transformers for summarizing a time series."""

__author__ = ["mloning", "RNKuhns"]
__all__ = ["SummaryTransformer", "MeanTransformer"]

import pandas as pd

from sktime.transformations.base import _SeriesToPrimitivesTransformer
from sktime.utils.validation.series import check_series


ALLOWED_SUM_FUNCS = (
    "mean",
    "min",
    "max",
    "sum",
    "skew",
    "kurtosis",
    "var",
    "std",
    "sem",
    "nunique",
    "count",
)


def _series_summary(series_or_df, summary_function="mean"):
    """Calculate summary value of a time series.

    For multivariate data the summary is applied to each column.

    Parameters
    ----------
    series_or_df : pd.Series or pd.DataFrame
        The series to summarize.
    summary_function : str
        One of pandas summary functions ("mean", "min", "max", "sum",
        "skew", "kurtosis", "var", "std", "sem", "nunique", "count") that is used
        to summarize each column of the dataset.

    Raises
    ------
    ValueError :
        `series_or_df` must be pd.Series or pd>DataFrame.
    ValueError :
        `summary_function must be one of ("mean", "min", "max", "mode", "sum",
        "skew", "kurtosis", "var", "std", "sem", "nunique", "count").

    Returns
    -------
    summary_value : scalar or pd.Series
        If `series_or_df` is univariate then a scalar is returned. Otherwise,
        a pd.Series is returned.
    """
    if isinstance(series_or_df, pd.Series):
        is_series = True
    elif isinstance(series_or_df, pd.DataFrame):
        is_series = False
    else:
        raise ValueError("`series_or_df` must be pd.Series or pd>DataFrame.")

    if summary_function == "mean":
        summary_value = series_or_df.mean(axis=0)
    elif summary_function == "min":
        summary_value = series_or_df.min(axis=0)
    elif summary_function == "max":
        summary_value = series_or_df.max(axis=0)
    elif summary_function == "sum":
        summary_value = series_or_df.sum(axis=0)
    elif summary_function == "skew":
        summary_value = series_or_df.skew(axis=0)
    elif summary_function == "kurtosis":
        summary_value = series_or_df.kurtosis(axis=0)
    elif summary_function == "std":
        summary_value = series_or_df.std(axis=0)
    elif summary_function == "var":
        summary_value = series_or_df.var(axis=0)
    elif summary_function == "mad":
        summary_value = series_or_df.mad(axis=0)
    elif summary_function == "sem":
        summary_value = series_or_df.sem(axis=0)
    elif summary_function == "nunique":
        if is_series:
            summary_value = series_or_df.nunique()
        else:
            summary_value = series_or_df.nunique(axis=0)
    elif summary_function == "count":
        if is_series:
            summary_value = series_or_df.count()
        else:
            summary_value = series_or_df.count(axis=0)
    else:
        raise ValueError(f"`summary_function must be one of {ALLOWED_SUM_FUNCS}.")

    # For univariate data in DataFrame return scalar like when input is pd.Series
    if is_series is False and series_or_df.shape[1] == 1:
        summary_value = summary_value.iloc[0]

    return summary_value


class SummaryTransformer(_SeriesToPrimitivesTransformer):
    """Calculate summary value of a time series.

    Parameters
    ----------
    summary_function : str
        One of pandas summary functions ("mean", "min", "max", "sum",
        "skew", "kurtosis", "var", "std", "sem", "nunique", "count") that is used
        to summarize each column of the dataset.

    See Also
    --------
    MeanTransformer :
        Calculate the mean of a timeseries.

    Notes
    -----
    This provides a wrapper around pandas DataFrame and Series summary methods.

    Examples
    --------
    >>> from sktime.transformations.series.summarize import SummaryTransformer
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> transformer = SummaryTransformer(summary_function="median")
    >>> y_mean = transformer.fit_transform(y)
    """

    _tags = {
        "univariate-only": False,
        "multivariate-only": False,
        "fit-in-transform": True,
    }

    def __init__(self, summary_function="mean"):
        self.summary_function = summary_function
        super(SummaryTransformer, self).__init__()

    def _transform(self, Z, X=None):
        """Logic to transform series.

        Parameters
        ----------
        Z : pd.Series or pd.DataFrame
            The series to transform.

        Returns
        -------
        summary_value : scalar or pd.Series
            If `series_or_df` is univariate then a scalar is returned. Otherwise,
            a pd.Series is returned.
        """
        summary_value = _series_summary(Z, self.summary_function)
        return summary_value

    def transform(self, Z, X=None):
        """Transform series.

        Parameters
        ----------
        Z : pd.Series or pd.DataFrame
            The series to transform.

        Returns
        -------
        summary_value : scalar or pd.Series
            If `series_or_df` is univariate then a scalar is returned. Otherwise,
            a pd.Series is returned.
        """
        self.check_is_fitted()
        Z = check_series(Z)
        summary_value = self._transform(Z, X=X)
        return summary_value


class MeanTransformer(SummaryTransformer):
    """Calculate mean value of a time series.

    Calculates scalar value when applied to a :term:`univariate time series`.
    If input is :term:`multivariate time series` then Scalar values for each
    column and returned as elements of a pd.Series.

    See Also
    --------
    SummaryTransformer :
        Calculate summary value of a time series.

    Notes
    -----
    This provides a wrapper around pandas DataFrame and Series summary methods.

    Examples
    --------
    >>> from sktime.transformations.series.summarize import MeanTransformer
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> transformer = MeanTransformer()
    >>> y_mean = transformer.fit_transform(y)
    """

    def __init__(self):
        super().__init__(summary_function="mean")
