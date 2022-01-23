#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Extract features across (shifted) windows from provided series."""

__author__ = ["Daniel Bartling"]
__all__ = ["LaggedWindowSummarizer"]

import pandas as pd
from joblib import Parallel, delayed

from sktime.transformations.base import _PanelToTabularTransformer
from sktime.utils.validation.series import check_series

# from sktime.utils.validation.panel import check_X
# List of native pandas rolling window function.
# danbartl: investivate different engines for pandas
pd_rolling = [
    "sum",
    "mean",
    "median",
    "std",
    "var",
    "kurt",
    "min",
    "max",
    "corr",
    "cov",
    "skew",
    "sem",
]


def _window_feature(
    Z,
    name=None,
    win_summarizer=None,
    window=None,
):
    """Compute window features and lag.

    y: either pandas.core.groupby.generic.SeriesGroupBy
        Object create by grouping across groupBy columns.
    name : str, base string of the derived features
           final name will also contain
           window shift and window length.
    win_summarizer: either str corresponding to native
    implemented pandas function, currently
            * "sum",
            * "mean",
            * "median",
            * "std",
            * "var",
            * "kurt",
            * "min",
            * "max",
            * "corr",
            * "cov",
            * "skew",
            * "sem"
         or function definition
    window: list of integers
        Contains values for window shift and window length.
    """
    shift = window[0]
    win = window[1]

    if win_summarizer in pd_rolling:
        if isinstance(Z, pd.Series):
            feat = getattr(Z.shift(shift).rolling(win), win_summarizer)()
        else:
            feat = Z.apply(
                lambda x: getattr(x.shift(shift).rolling(win), win_summarizer)()
            )
    else:
        feat = Z.shift(shift)
        if isinstance(Z, pd.Series) and callable(win_summarizer):
            feat = feat.rolling(win).apply(win_summarizer, raw=True)
        elif not isinstance(Z, pd.Series) and callable(win_summarizer):
            feat = feat.apply(lambda x: x.rolling(win).apply(win_summarizer, raw=True))

    feat.rename(name + "_" + "_".join([str(item) for item in window]), inplace=True)

    return feat


class _LaggedWindowExtractor(_PanelToTabularTransformer):
    """Base adapter class for transformations.

    The LaggedWindowSummarizer transforms input series to features
    based on a provided dictionary of window summarizer, window shifts
    and window lengths.
    """

    def __init__(
        self,
        functions,
        n_jobs=-1,
    ):

        self.functions = functions
        self.n_jobs = n_jobs

        super(_LaggedWindowExtractor, self).__init__()

    # Get extraction parameters
    def fit(self, X, y=None):
        """Fit.

        Parameters
        ----------
        X : pd.DataFrame
            nested pandas DataFrame of shape [n_samples, n_columns]
        y : pd.Series or np.array
            Target variable

        Returns
        -------
        self : an instance of self
        """
        func_dict = pd.DataFrame(self.functions).T.reset_index()
        func_dict.rename(
            columns={"index": "name", 0: "win_summarizer", 1: "window"}, inplace=True
        )
        func_dict = func_dict.explode("window")

        self._func_dict = func_dict
        self._is_fitted = True
        self._truncate_start = func_dict["window"].apply(lambda x: x[0] + x[1]).max()

        return self


class LaggedWindowSummarizer(_LaggedWindowExtractor):
    """Transformer for extracting time series features.

    The LaggedWindowSummarizer transforms input series to features
    based on a provided dictionary of window summarizer, window shifts
    and window lengths.

    Parameters
    ----------
    Dictionary with the following arguments:
    index : str, base string of the derived features
            final name will also contain
            window shift and window length.
    win_summarizer: either str corresponding to implemented pandas function, currently
            * "sum",
            * "mean",
            * "median",
            * "std",
            * "var",
            * "kurt",
            * "min",
            * "max",
            * "corr",
            * "cov",
            * "skew",
            * "sem"
         or function definition
    window: list of integers
        Contains values for window shift and window length.
    """

    def _transform(self, X, y=None):
        """Transform X.

        Parameters
        ----------
        Z : pd.DataFrame with y as columns
            and either single index or
            MultiIndex conforming to
            Panel default definition
            (instances and timepoints columm)

        Returns
        -------
        Zt : pandas DataFrame
        Transformed pandas DataFrame

        """
        # input checks

        self.check_is_fitted()
        # danbartl: currently check drops multiindex column names, therefore disabled
        X = check_series(X)

        if isinstance(X.index, pd.MultiIndex):
            X_grouped = getattr(X.groupby("instances"), X.columns[0])
            df = Parallel(n_jobs=self.n_jobs)(
                delayed(_window_feature)(X_grouped, **kwargs)
                for index, kwargs in self._func_dict.iterrows()
            )
        else:
            df = Parallel(n_jobs=self.n_jobs)(
                delayed(_window_feature)(X, **kwargs)
                for index, kwargs in self._func_dict.iterrows()
            )

        col_names = [o.name for o in df]
        Xt = pd.concat(df, axis=1)
        Xt.columns = col_names

        return Xt
