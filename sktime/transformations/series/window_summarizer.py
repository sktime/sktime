#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Extract features across (shifted) windows from provided series."""

__author__ = ["Daniel Bartling"]
__all__ = ["LaggedWindowSummarizer"]

import pandas as pd
from joblib import Parallel, delayed

# from sktime.transformations.base import _PanelToTabularTransformer
from sktime.transformations.base import BaseTransformer

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
        if isinstance(Z.index, pd.MultiIndex):
            feat = getattr(Z.shift(shift).rolling(win), win_summarizer)()
        else:
            feat = Z.apply(
                lambda x: getattr(x.shift(shift).rolling(win), win_summarizer)()
            )
    else:
        feat = Z.shift(shift)
        if isinstance(Z.index, pd.MultiIndex) and callable(win_summarizer):
            feat = feat.rolling(win).apply(win_summarizer, raw=True)
        elif not isinstance(Z.index, pd.MultiIndex) and callable(win_summarizer):
            feat = feat.apply(lambda x: x.rolling(win).apply(win_summarizer, raw=True))

    if isinstance(feat, pd.Series):
        feat.rename(name + "_" + "_".join([str(item) for item in window]), inplace=True)
    else:
        feat.rename(
            columns={
                feat.columns[0]: name + "_" + "_".join([str(item) for item in window])
            },
            inplace=True,
        )

    return feat


class _LaggedWindowExtractor(BaseTransformer):
    """Base adapter class for transformations.

    The LaggedWindowSummarizer transforms input series to features
    based on a provided dictionary of window summarizer, window shifts
    and window lengths.
    """

    # _tags = {
    #     "requires-fh-in-fit": False,  # is the forecasting horizon required in fit?
    #     "y_inner_mtype": ["pd-multiindex", "pd.DataFrame"],
    #     "X_inner_mtype": ["pd-multiindex", "pd.DataFrame"],
    # }

    def __init__(
        self,
        functions=None,
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
        if self.functions is not None:
            func_dict = pd.DataFrame(self.functions).T.reset_index()
            func_dict.rename(
                columns={"index": "name", 0: "win_summarizer", 1: "window"},
                inplace=True,
            )
            func_dict = func_dict.explode("window")
            self._func_dict = func_dict
            self._truncate_start = (
                func_dict["window"].apply(lambda x: x[0] + x[1]).max()
            )
        else:
            self._func_dict = None
            self._truncate_start = None
        self._is_fitted = True

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
        if self._func_dict is not None:
            if isinstance(X.index, pd.MultiIndex):
                X_grouped = getattr(X.groupby("instances"), X.columns[0])
                df = Parallel(n_jobs=self.n_jobs)(
                    delayed(_window_feature)(X_grouped, **kwargs)
                    for index, kwargs in self._func_dict.iterrows()
                )
            else:
                for _index, kwargs in self._func_dict.iterrows():
                    _window_feature(X, **kwargs)
                df = Parallel(n_jobs=self.n_jobs)(
                    delayed(_window_feature)(X, **kwargs)
                    for _index, kwargs in self._func_dict.iterrows()
                )
            if isinstance(df[0], pd.Series):
                col_names = [o.name for o in df]
                Xt = pd.concat(df, axis=1)
                Xt.columns = col_names
            else:
                Xt = pd.concat(df, axis=1)
        else:
            Xt = X

        return Xt
