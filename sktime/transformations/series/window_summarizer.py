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

# List of native pandas rolling window function.
# In the future  different engines for pandas will be investigated
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


def get_name_list(Z):
    """Get names of pd.Series or pd.Dataframe."""
    if isinstance(Z, pd.DataFrame):
        Z_name = Z.columns.to_list()
    else:
        if Z.name is not None:
            Z_name = [Z.name]
        else:
            Z_name = None
    return Z_name


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

    feat.rename(
        columns={
            feat.columns[0]: name + "_" + "_".join([str(item) for item in window])
        },
        inplace=True,
    )

    return feat


class LaggedWindowSummarizer(BaseTransformer):
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
    target_cols: list of str,
        Specifies which columns in y or X to target. If set to None will
        target first column in X.

    Examples
    --------
    >>> from sktime.transformations.series.window_summarizer import (
    ...    LaggedWindowSummarizer
    ... )
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> kwargs_variant = {
    ...     "functions": {
    ...     "mean": ["mean", [[1, 7], [8, 7]]],
    ...     "covar_feature": ["cov", [[1, 28]]],
    ...     }
    ...  }
    >>> transformer = LaggedWindowSummarizer(**kwargs_variant)
    >>> y_transformed = transformer.fit_transform(y)
    """

    #
    _tags = {
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Series",
        "scitype:instancewise": True,
        "capability:inverse_transform": False,
        "scitype:transform-labels": "Series",
        "X_inner_mtype": [
            "pd.Series",
            "pd.DataFrame",
        ],  # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": [
            "pd.Series",
            "pd.DataFrame",
        ],  # which mtypes do _fit/_predict support for X?
        "skip-inverse-transform": True,  # is inverse-transform skipped when called?
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

        # self._converter_store_X = dict()
        self.functions = functions
        self.n_jobs = n_jobs
        self.target_cols = target_cols

        super(LaggedWindowSummarizer, self).__init__()

    # Get extraction parameters
    def _fit(self, X, y=None):
        """Fit transformer to X and y.

        Private _fit containing the core logic, called from fit
        Parameters
        ----------
        X : pd.Series or pd.DataFrame
        y : None

        Returns
        -------
        X: pd.DataFrame
        self: reference to self
        """
        # check if dict is empty

        X_name = get_name_list(X)

        if X_name is None:
            self._X_rename = "var_0"
            X_name = ["var_0"]
        else:
            self._X_rename = None

        if y is not None:
            y_name = get_name_list(y)
            if y_name is None:
                self._y_rename = "y"
                y_name = ["y"]
            else:
                self._y_rename = None
            Z_name = X_name + y_name
        else:
            Z_name = X_name
            self._y_rename = None

        if self.target_cols is None:
            self._target_cols = [Z_name][0]
        else:
            self._target_cols = self.target_cols

        if y is not None:
            if not len(y_name + X_name) == len(set(y_name + X_name)):
                raise ValueError(
                    "Please make sure that names across X and y are not"
                    + " duplicate. If unnamed X/y Series are provided, X will be"
                    + " renamed to 'X_var_0' and y will be renamed to 'y'."
                    + " This could also be result of this error if e.g. X"
                    + " contained a named column y and an unnamed y Series"
                    + " was renamed to 'y'."
                )

        if self.target_cols is not None:
            if not all(x in Z_name for x in self.target_cols):
                raise ValueError("target_X targets non-existing column in X")

        if self.functions is None:
            func_dict = pd.DataFrame(
                {
                    "lag": ["lag", [[1, 0]]],
                }
            ).T.reset_index()
        else:
            func_dict = pd.DataFrame(self.functions).T.reset_index()

        func_dict.rename(
            columns={"index": "name", 0: "win_summarizer", 1: "window"},
            inplace=True,
        )
        func_dict = func_dict.explode("window")
        self._func_dict = func_dict
        self._truncate_start = func_dict["window"].apply(lambda x: x[0] + x[1]).max()

        return self

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
        if not isinstance(y, pd.DataFrame):
            y = pd.DataFrame(y)

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if self._y_rename is not None:
            y.columns = [self._y_rename]

        if self._X_rename is not None:
            X.columns = [self._X_rename]

        Z = pd.concat([X, y], axis=1)

        Zt_out = []
        for cols in self._target_cols:
            if isinstance(Z.index, pd.MultiIndex):
                Z_grouped = getattr(Z.groupby("instances"), Z[cols])
                df = Parallel(n_jobs=self.n_jobs)(
                    delayed(_window_feature)(Z_grouped, **kwargs)
                    for index, kwargs in self._func_dict.iterrows()
                )
            else:
                # for _index, kwargs in self._func_dict.iterrows():
                #     _window_feature(X, **kwargs)
                df = Parallel(n_jobs=self.n_jobs)(
                    delayed(_window_feature)(Z[cols], **kwargs)
                    for _index, kwargs in self._func_dict.iterrows()
                )
            Zt = pd.concat(df, axis=1)
            # if len(self._target_cols) > 1:
            Zt = Zt.add_prefix(str(cols) + "_")
            Zt_out.append(Zt)
        Zt_out_df = pd.concat(Zt_out, axis=1)
        Zt_return = pd.concat([Zt_out_df, Z.drop(self._target_cols, axis=1)], axis=1)
        return Zt_return

    @classmethod
    def get_test_params(cls):
        """Return testing parameter settings for the estimator.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        params1 = {
            "functions": {
                "lag": ["lag", [[1, 0]]],
                "mean": ["mean", [[3, 0], [12, 0]]],
                "std": ["std", [[4, 0]]],
            }
        }

        params2 = {
            "functions": {
                "lag": ["lag", [[3, 0], [6, 0]]],
            }
        }

        params3 = {
            "functions": {
                "mean": ["mean", [[7, 0], [7, 7]]],
                "covar_feature": ["cov", [[28, 0]]],
            }
        }

        return [params1, params2, params3]
