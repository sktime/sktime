#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implement transformers for summarizing a time series."""

__author__ = ["mloning", "RNKuhns"]
__all__ = ["SummaryTransformer", "MeanTransformer"]

import pandas as pd
from deprecated.sphinx import deprecated

from sktime.transformations.base import _SeriesToPrimitivesTransformer

ALLOWED_SUM_FUNCS = [
    "mean",
    "min",
    "max",
    "median",
    "sum",
    "skew",
    "kurt",
    "var",
    "std",
    "mad",
    "sem",
    "nunique",
    "count",
]


def _check_summary_function(summary_function):
    """Validate summary_function.

    Parameters
    ----------
    summary_function : str, list or tuple
        Either a string or list/tuple of strings indicating the pandas summary
        functions ("mean", "min", "max", "median", "sum", "skew", "kurtosis",
        "var", "std", "mad", "sem", "nunique", "count") that is used to summarize
        each column of the dataset.

    Returns
    -------
    summary_function : list or tuple
        The summary functions that will be used to summarize the dataset.
    """
    msg = f"""`summary_function` must be str or a list or tuple made up of
          {ALLOWED_SUM_FUNCS}.
          """
    if isinstance(summary_function, str):
        if summary_function not in ALLOWED_SUM_FUNCS:
            raise ValueError(msg)
        summary_function = [summary_function]
    elif isinstance(summary_function, (list, tuple)):
        if not all([func in ALLOWED_SUM_FUNCS for func in summary_function]):
            raise ValueError(msg)
    else:
        raise ValueError(msg)
    return summary_function


def _check_quantiles(quantiles):
    """Validate quantiles.

    Parameters
    ----------
    quantiles : str, list, tuple or None
        Either a string or list/tuple of strings indicating the pandas summary
        functions ("mean", "min", "max", "median", "sum", "skew", "kurtosis",
        "var", "std", "mad", "sem", "nunique", "count") that is used to summarize
        each column of the dataset.

    Returns
    -------
    quantiles : list or tuple
        The validated quantiles that will be used to summarize the dataset.
    """
    msg = """`quantiles` must be int, float or a list or tuple made up of
          int and float values that are between 0 and 1.
          """
    if isinstance(quantiles, (int, float)):
        if not 0.0 <= quantiles <= 1.0:
            raise ValueError(msg)
        quantiles = [quantiles]
    elif isinstance(quantiles, (list, tuple)):
        if len(quantiles) == 0 or not all(
            [isinstance(q, (int, float)) and 0.0 <= q <= 1.0 for q in quantiles]
        ):
            raise ValueError(msg)
    elif quantiles is not None:
        raise ValueError(msg)
    return quantiles


class SummaryTransformer(_SeriesToPrimitivesTransformer):
    """Calculate summary value of a time series.

    For :term:`univariate time series` a combination of summary functions and
    quantiles of the input series are calculated. If the input is a
    :term:`multivariate time series` then the summary functions and quantiles
    are calculated separately for each column.

    Parameters
    ----------
    summary_function : str, list, tuple, default=("mean", "std", "min", "max")
        Either a string, or list or tuple of strings indicating the pandas
        summary functions that are used to summarize each column of the dataset.
        Must be one of ("mean", "min", "max", "median", "sum", "skew", "kurt",
        "var", "std", "mad", "sem", "nunique", "count").
    quantiles : str, list, tuple or None, default=(0.1, 0.25, 0.5, 0.75, 0.9)
        Optional list of series quantiles to calculate. If None, no quantiles
        are calculated.

    See Also
    --------
    MeanTransformer :
        Calculate the mean of a timeseries.

    Notes
    -----
    This provides a wrapper around pandas DataFrame and Series agg and
    quantile methods.

    Examples
    --------
    >>> from sktime.transformations.series.summarize import SummaryTransformer
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> transformer = SummaryTransformer()
    >>> y_mean = transformer.fit_transform(y)
    """

    _tags = {
        "fit-in-transform": True,
        "X_inner_mtype": ["pd.DataFrame", "pd.Series"],
    }

    def __init__(
        self,
        summary_function=("mean", "std", "min", "max"),
        quantiles=(0.1, 0.25, 0.5, 0.75, 0.9),
    ):
        self.summary_function = summary_function
        self.quantiles = quantiles
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
        if self.summary_function is None and self.quantiles is None:
            raise ValueError(
                "One of `summary_function` and `quantiles` must not be None."
            )
        summary_function = _check_summary_function(self.summary_function)
        quantiles = _check_quantiles(self.quantiles)

        summary_value = Z.agg(summary_function)
        if quantiles is not None:
            quantile_value = Z.quantile(quantiles)
            summary_value = pd.concat([summary_value, quantile_value])

        if isinstance(Z, pd.Series):
            summary_value.name = Z.name
            summary_value = pd.DataFrame(summary_value)

        return summary_value.T


class MeanTransformer(SummaryTransformer):
    """Calculate mean value of a time series.

    See Also
    --------
    SummaryTransformer :
        Calculate summary values of a time series.

    Examples
    --------
    >>> from sktime.transformations.series.summarize import MeanTransformer
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> transformer = MeanTransformer()
    >>> y_mean = transformer.fit_transform(y)
    """

    @deprecated(
        version="0.9.0",
        reason=(
            "MeanTransformer will be removed in release v0.10.0. Please use "
            "`SummaryTransformer` from `sktime.transformation.series.summarize` "
            "instead."
        ),
        category=FutureWarning,
    )
    def __init__(self):
        super().__init__(summary_function="mean", quantiles=None)
