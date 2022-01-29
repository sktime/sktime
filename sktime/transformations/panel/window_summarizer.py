#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Extract features across (shifted) windows from provided series."""

__author__ = ["Daniel Bartling"]
__all__ = ["LaggedWindowSummarizer"]

import pandas as pd
from joblib import Parallel, delayed

from sktime.transformations.base import _PanelToTabularTransformer
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

    Apply functions passed over a certain window
    of past observations, e.g. the mean of a window of length 7 days, lagged by 14 days.

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
         or function definition. See for the native functions also
         https://pandas.pydata.org/docs/reference/window.html.
    window: list of integers
        Contains values for window shift and window length.
    """
    win = window[0]
    shift = window[1] + 1
    

    if win_summarizer in pd_rolling:
        if isinstance(Z.index, pd.MultiIndex):
            feat = getattr(Z.shift(shift).rolling(win), win_summarizer)()
        else:
            feat = pd.DataFrame(Z).apply(
                lambda x: getattr(x.shift(shift).rolling(win), win_summarizer)()
            )
    else:
        feat = Z.shift(shift)
        if isinstance(Z.index, pd.MultiIndex) and callable(win_summarizer):
            feat = feat.rolling(win).apply(win_summarizer, raw=True)
        elif not isinstance(Z.index, pd.MultiIndex) and callable(win_summarizer):
            feat = feat.apply(lambda x: x.rolling(win).apply(win_summarizer, raw=True))

    if isinstance(feat, pd.Series):
        feat = pd.DataFrame(feat)
        #feat.rename(name + "_" + "_".join([str(item) for item in window]), inplace=True)
    
    feat.rename(
        columns={
            feat.columns[0]: name + "_" + "_".join([str(item) for item in window])
        },
        inplace=True,
    )

    return feat


class _LaggedWindowExtractor(_PanelToTabularTransformer):
    """Base adapter class for transformations.

    The LaggedWindowSummarizer transforms input series to features
    based on a provided dictionary of window summarizer, window shifts
    and window lengths.
    """

    _tags = {
        # todo: what is the scitype of X: Series, or Panel
        #"scitype:transform-input": "Panel",
        # todo: what scitype is returned: Primitives, Series, Panel
        "scitype:transform-output": "Panel",
        # todo: what is the scitype of y: None (not needed), Primitives, Series, Panel
        "scitype:transform-labels": "None",
        "scitype:instancewise": True,  # is this an instance-wise transform?
        # "X_inner_mtype": ["pd.DataFrame"
        # # which mtypes do _fit/_predict support for X?
        # # X_inner_mtype can be Panel mtype even if
        # transform-input is Series, vectorized
        # "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for y?
        "capability:inverse_transform": False,  # does transformer
        # have inverse transform
        "skip-inverse-transform": False,  # is inverse-transform skipped when called?
        "univariate-only": False,  # can the transformer handle multivariate X?
        "handles-missing-data": True,  # can estimator handle missing data?
        "X-y-must-have-same-index": True,  # can estimator handle different X/y index?
        "enforce_index_type": None,  # index type that needs to be enforced in X/y
        "fit-in-transform": False,  # is fit empty and can be skipped? Yes = True
        "transform-returns-same-time-index": False,
        # does transform return have the same time index as input X
    }

    def __init__(
        self,
        functions=None,
        n_jobs=-1,
        target_cols=None,
    ):

        self.functions = functions
        self.n_jobs = n_jobs
        self.target_cols = target_cols

        super(_LaggedWindowExtractor, self).__init__()

    # Get extraction parameters
    def _fit(self, X, y=None):
        """Fit transformer to X and y.

        Private _fit containing the core logic, called from fit
        Parameters
        ----------
        X : Series or Panel
        y : Series or Panel of same index

        Returns
        -------
        self: reference to self
        """
        # check if dict is empty
        if self.functions is None:
            self.functions = {
                "lag": ["lag", [[1, 0]]],
            }
        
        if self.target_cols is None:
            self.target_cols = [X.columns.to_list()[0]]


        if not all(x in X.columns.to_list() for x in self.target_cols):
            raise ValueError("Invalid target select for transformation")

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

    Examples
    --------
    >>> import sktime.transformations.series.window_summarizer as ws
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> kwargs_variant = {
    >>>     "functions": {
    >>>     "mean": ["mean", [[1, 7], [8, 7]]],
    >>>     "covar_feature": ["cov", [[1, 28]]],
    >>>     }
    >>>  }
    >>> transformer = ws.LaggedWindowSummarizer(**kwargs_variant)
    >>> y_transformed = transformer.fit_transform(y)
    """

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        Parameters
        ----------
        X : Series or Panel
        y : Series or Panel
        Returns
        -------
        transformed version of X
        """
        # input checks
        Xt_out = []
        for cols in self.target_cols:
            if isinstance(X.index, pd.MultiIndex):
                X_grouped = getattr(X.groupby("instances"), X[cols])
                df = Parallel(n_jobs=self.n_jobs)(
                    delayed(_window_feature)(X_grouped, **kwargs)
                    for index, kwargs in self._func_dict.iterrows()
                )
            else:
                # for _index, kwargs in self._func_dict.iterrows():
                #     _window_feature(X, **kwargs)
                df = Parallel(n_jobs=self.n_jobs)(
                    delayed(_window_feature)(X[cols], **kwargs)
                    for _index, kwargs in self._func_dict.iterrows()
                )
            Xt = pd.concat(df, axis=1)
            if len(self.target_cols) > 1:
                Xt = Xt.add_prefix(cols + "_")
            Xt_out.append(Xt)
        Xt_return = pd.concat(Xt_out, axis=1)

        return Xt_return
