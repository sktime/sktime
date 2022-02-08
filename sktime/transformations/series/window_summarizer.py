#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Extract features across (shifted) windows from provided series."""

__author__ = ["danbartl"]
__all__ = ["WindowSummarizer"]

import pandas as pd
from joblib import Parallel, delayed

from sktime.transformations.base import BaseTransformer


class WindowSummarizer(BaseTransformer):
    """Transformer for extracting time series features.

    The WindowSummarizer transforms input series to features
    based on a provided dictionary of window summarizer, window shifts
    and window lengths.

    Parameters
    ----------
    n_jobs : int, optional (default=-1)
        The number of jobs to run in parallel for applying the window functions.
        ``-1`` means using all processors.
    target_cols: list of str, optional (default = None)
        Specifies which columns in X to target for applying the window functions.
        ``None`` will target the first column
    lag_config: Dictionary specifying which as index the `name` of the columns to be
        generated. The dict also specifies the type of function via the argument
        `summarize` as well as the length 2 list argument `window`. The internal
        function _window_feature will be resolved to `window length` - the length of
        the window across which to apply the function - as well as the argument
        `starting_at`, which will specify have far back in the past the window will
        start.

        For example for `window = [4, 3]`, we have a `window_length` of 4 and
        `starting_at` of 3 to target the four days prior to the last three days.
        Here is a representation of the selected window::

        |---------------------------------------|
        | x * * * * * * * x x x z - - - |

        ``-`` = future observations.
        ``z`` = current observation, to which the window should be relative to.
        ``x`` = past observations.
        ``*`` = selected window of past observations.


        index : str, base string of the derived features, will be appended by
                window shift and window length.
        summarizer: either str corresponding to pandas window function, currently
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
                or custom function call (to be provided by user).
                See for the native window functions also:
                https://pandas.pydata.org/docs/reference/window.html.
        window: list of integers
            Contains values for window shift and window length.


    Examples
    --------
    >>> import pandas as pd
    >>> from sktime.transformations.series.window_summarizer import WindowSummarizer
    >>> from sktime.datasets import load_airline, load_longley
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> from sktime.forecasting.base import ForecastingHorizon
    >>> from sktime.forecasting.compose import ForecastingPipeline
    >>> from sktime.forecasting.model_selection import temporal_train_test_split
    >>> y = load_airline()
    >>> kwargs = {
    ...     "lag_config": {
    ...         "lag": ["lag", [[1, 0]]],
    ...         "mean": ["mean", [[3, 0], [12, 0]]],
    ...         "std": ["std", [[4, 0]]],
    ...     }
    ... }
    >>> transformer = WindowSummarizer(**kwargs)
    >>> y_transformed = transformer.fit_transform(y)

        Example with transforming multiple columns of exogeneous features
    >>> y, X = load_longley()
    >>> y_train, y_test, X_train, X_test = temporal_train_test_split(y, X)
    >>> fh = ForecastingHorizon(X_test.index, is_relative=False)
    >>> # Example transforming only X
    >>> pipe = ForecastingPipeline(
    ...     steps=[
    ...         ("a", WindowSummarizer(n_jobs=1, target_cols=["POP", "GNPDEFL"])),
    ...         ("b", WindowSummarizer(n_jobs=1, target_cols=["GNP"], **kwargs)),
    ...         ("forecaster", NaiveForecaster(strategy="drift")),
    ...     ]
    ... )
    >>> pipe_return = pipe.fit(y_train, X_train)
    >>> y_pred1 = pipe_return.predict(fh=fh, X=X_test)

        Example with transforming multiple columns of exogeneous features
        as well as the y column
    >>> Z_train = pd.concat([X_train, y_train], axis=1)
    >>> Z_test = pd.concat([X_test, y_test], axis=1)
    >>> pipe = ForecastingPipeline(
    ...     steps=[
    ...         ("a", WindowSummarizer(n_jobs=1, target_cols=["POP", "TOTEMP"])),
    ...         ("b", WindowSummarizer(**kwargs, n_jobs=1, target_cols=["GNP"])),
    ...         ("forecaster", NaiveForecaster(strategy="drift")),
    ...     ]
    ... )
    >>> pipe_return = pipe.fit(y_train, Z_train)
    >>> y_pred2 = pipe_return.predict(fh=fh, X=Z_test)
    """

    _tags = {
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Series",
        "scitype:instancewise": True,
        "capability:inverse_transform": False,
        "scitype:transform-labels": None,
        "X_inner_mtype": "pd.DataFrame",  # which mtypes do _fit/_predict support for X?
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
        lag_config=None,
        n_jobs=-1,
        target_cols=None,
    ):

        # self._converter_store_X = dict()
        self.lag_config = lag_config
        self.n_jobs = n_jobs
        self.target_cols = target_cols

        super(WindowSummarizer, self).__init__()

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
        X_name = get_name_list(X)

        if X_name is None:
            X_name = ["var_0"]

        if self.target_cols is None:
            self._target_cols = [X_name[0]]
        else:
            self._target_cols = self.target_cols

        if self.target_cols is not None:
            if not all(x in X_name for x in self.target_cols):
                missing_cols = [x for x in self.target_cols if x not in X_name]
                raise ValueError(
                    "target_cols "
                    + " ".join(missing_cols)
                    + " specified that do neither exist in X (or y resp.)"
                )

        if self.lag_config is None:
            func_dict = pd.DataFrame(
                {
                    "lag": ["lag", [[1, 0]]],
                }
            ).T.reset_index()
        else:
            func_dict = pd.DataFrame(self.lag_config).T.reset_index()

        func_dict.rename(
            columns={"index": "name", 0: "summarizer", 1: "window"},
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
        X.columns = X.columns.map(str)

        Xt_out = []
        for cols in self._target_cols:
            if isinstance(X.index, pd.MultiIndex):
                X_grouped = getattr(X.groupby("instances"), X.loc[:, [cols]])
                df = Parallel(n_jobs=self.n_jobs)(
                    delayed(_window_feature)(X_grouped, **kwargs)
                    for index, kwargs in self._func_dict.iterrows()
                )
            else:
                df = Parallel(n_jobs=self.n_jobs)(
                    delayed(_window_feature)(X.loc[:, [cols]], **kwargs)
                    for _index, kwargs in self._func_dict.iterrows()
                )
            Xt = pd.concat(df, axis=1)
            Xt = Xt.add_prefix(str(cols) + "_")
            Xt_out.append(Xt)
        Xt_out_df = pd.concat(Xt_out, axis=1)
        Xt_return = pd.concat([Xt_out_df, X.drop(self._target_cols, axis=1)], axis=1)
        return Xt_return

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
            "lag_config": {
                "lag": ["lag", [[1, 0]]],
                "mean": ["mean", [[3, 0], [12, 0]]],
                "std": ["std", [[4, 0]]],
            }
        }

        params2 = {
            "lag_config": {
                "lag": ["lag", [[3, 0], [6, 0]]],
            }
        }

        params3 = {
            "lag_config": {
                "mean": ["mean", [[7, 0], [7, 7]]],
                "covar_feature": ["cov", [[28, 0]]],
            }
        }

        return [params1, params2, params3]


# List of native pandas rolling window function.
# In the future different engines for pandas will be investigated
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
    Z_name = [str(z) for z in Z_name]
    return Z_name


def _window_feature(
    Z,
    name=None,
    summarizer=None,
    window=None,
):
    """Compute window features and lag.

    Apply summarizer passed over a certain window
    of past observations, e.g. the mean of a window of length 7 days, lagged by 14 days.

    y: either pandas.core.groupby.generic.SeriesGroupBy
        Object create by grouping across groupBy columns.
    name : str, base string of the derived features, will be appended by
        window shift and window length.
    summarizer: either str corresponding to pandas window function, currently
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
         or custom function call. See for the native window functions also
         https://pandas.pydata.org/docs/reference/window.html.
    window: list of integers
        Contains values for `window_length` and window `starting_at` which defines
        how many observations back the windows start.
    """
    window_length = window[0]
    starting_at = window[1] + 1

    if summarizer in pd_rolling:
        if isinstance(Z.index, pd.MultiIndex):
            feat = getattr(Z.shift(starting_at).rolling(window_length), summarizer)()
        else:
            feat = Z.apply(
                lambda x: getattr(
                    x.shift(starting_at).rolling(window_length), summarizer
                )()
            )
    else:
        feat = Z.shift(starting_at)
        if isinstance(Z.index, pd.MultiIndex) and callable(summarizer):
            feat = feat.rolling(window_length).apply(summarizer, raw=True)
        elif not isinstance(Z.index, pd.MultiIndex) and callable(summarizer):
            feat = feat.apply(
                lambda x: x.rolling(window_length).apply(summarizer, raw=True)
            )

    feat.rename(
        columns={
            feat.columns[0]: name + "_" + "_".join([str(item) for item in window])
        },
        inplace=True,
    )

    return feat
