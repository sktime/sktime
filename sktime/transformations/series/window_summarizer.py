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
    lag_config: dict of str and list, optional (default = dict containing first lag)
        Dictionary specifying as key the `name` of the columns to be
        generated. As value the dict specifies the type of function via the argument
        `summarize` as well as the length 2 list argument `window`. The list `window`
        will be resolved by the internal function _window_feature to `window length`
        - the length of the window across which to apply the function - as well as
        the argument `starting_at`, which will specify how far back in the past
        the window will start.

        For example for `window = [4, 3]`, we have a `window_length` of 4 and
        `starting_at` of 3 to target the four days prior to the last three days.
        Here is a representation of the selected window:

        |-------------------------------|
        | x * * * * * * * x x x z - - - |
        |-------------------------------|

        ``-`` = future observations.
        ``z`` = current observation, to which the window should be relative to.
        ``x`` = past observations.
        ``*`` = selected window of past observations across which summarizer
                function will be applied.

        key (resolved to name) : str, name of the derived features, will be appended by
                window_length and starting_at parameter.
        first value (resolved to summarizer): either custom function call (to be
                provided by user) or str corresponding to native pandas window function:
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
                See also: https://pandas.pydata.org/docs/reference/window.html.
        second value (window): list of integers
            List containg window_length and starting_at parameters.
        truncate: str, optional (default = None)
            Defines how to deal with NAs that were created as a result of applying the
            functions in the lag_config dict across windows that are longer than
            the remaining history of data.
            For example a lag config of [7, 14] - a window_length of 7 starting at 14
            observations in the past - cannot be fully applied for the first 20
            observations of the targeted column.
            A lag_config of [[7, 14], [0, 28]] cannot be correctly applied for the
            first 21 resp. 28 observations of  the targeted column. Possible values
            to deal with those NAs:
                * None
                * "bfill"
            None will keep the NAs generated, and would leave it for the user to choose
            an estimator that can correctly deal with observations with missing values,
            "bfill" will fill the NAs by carrying the first observation backwards.


    Attributes
    ----------
    truncate_start : int
        See section Parameters - truncate for a more detailed explanation of truncation
        as a result of applying windows of certain lengths across past observations.
        A lag_config of [[7, 14], [0, 28]] cannot be correctly applied for the
        first 21 resp. 28 observations of  the targeted column. truncate_start will
        give the maximum og observations that are filled with NAs across all arguments
        of the lag_config, in this case 28.


    Returns
    -------
    X: pd.DataFrame
        Contains all transformed columns as well as non-transformed columns.
        The raw inputs to transformed columns will be dropped.
    self: reference to self

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
        "scitype:transform-labels": False,
        "X_inner_mtype": "pd.DataFrame",  # which mtypes do _fit/_predict support for X?
        "skip-inverse-transform": True,  # is inverse-transform skipped when called?
        "univariate-only": False,  # can the transformer handle multivariate X?
        "handles-missing-data": True,  # can estimator handle missing data?
        "X-y-must-have-same-index": False,  # can estimator handle different X/y index?
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
        truncate=None,
    ):

        # self._converter_store_X = dict()
        self.lag_config = lag_config
        self.n_jobs = n_jobs
        self.target_cols = target_cols
        self.truncate = truncate

        super(WindowSummarizer, self).__init__()

    def _fit(self, X, y=None):
        """Fit transformer to X and y.

        Private _fit containing the core logic, called from fit

        Attributes
        ----------
        truncate_start : int
            See section Parameters - truncate for a more detailed explanation of
            truncation as a result of applying windows of certain lengths across past
            observations.
            A lag_config of [[7, 14], [0, 28]] cannot be correctly applied for the
            first 21 resp. 28 observations of  the targeted column. truncate_start will
            give the maximum og observations that are filled with NAs across all
            arguments of the lag_config, in this case 28.

        Returns
        -------
        X: pd.DataFrame
            Contains all transformed columns as well as non-transformed columns.
            The raw inputs to transformed columns will be dropped.
        self: reference to self
        """
        X_name = get_name_list(X)

        if self.target_cols is not None:
            if not all(x in X_name for x in self.target_cols):
                missing_cols = [x for x in self.target_cols if x not in X_name]
                raise ValueError(
                    "target_cols "
                    + " ".join(missing_cols)
                    + " specified that do not exist in X."
                )

        if self.target_cols is None:
            self._target_cols = [X_name[0]]
        else:
            self._target_cols = self.target_cols

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
        self.truncate_start = func_dict["window"].apply(lambda x: x[0] + x[1]).max()

        self._func_dict = func_dict

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        Parameters
        ----------
        X : pd.DataFrame
        y : None

        Returns
        -------
        transformed version of X
        """
        func_dict = self._func_dict
        target_cols = self._target_cols

        X.columns = X.columns.map(str)
        Xt_out = []
        if self.truncate == "bfill":
            bfill = True
        else:
            bfill = False
        for cols in target_cols:
            if isinstance(X.index, pd.MultiIndex):
                X_grouped = getattr(X.groupby("instances"), X.loc[:, [cols]])
                df = Parallel(n_jobs=self.n_jobs)(
                    delayed(_window_feature)(X_grouped, **kwargs, bfill=bfill)
                    for index, kwargs in func_dict.iterrows()
                )
            else:
                df = Parallel(n_jobs=self.n_jobs)(
                    delayed(_window_feature)(X.loc[:, [cols]], **kwargs, bfill=bfill)
                    for _index, kwargs in func_dict.iterrows()
                )
            Xt = pd.concat(df, axis=1)
            Xt = Xt.add_prefix(str(cols) + "_")
            Xt_out.append(Xt)
        Xt_out_df = pd.concat(Xt_out, axis=1)
        Xt_return = pd.concat([Xt_out_df, X.drop(target_cols, axis=1)], axis=1)

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


def _window_feature(Z, name=None, summarizer=None, window=None, bfill=False):
    """Compute window features and lag.

    Apply summarizer passed over a certain window
    of past observations, e.g. the mean of a window of length 7 days, lagged by 14 days.

    Z: pandas Dataframe with a single column.
    name : str, base string of the derived features, will be appended by
        window length and starting at parameters defined in window.
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
        List containg window_length and starting_at parameters, see WindowSummarizer
        class description for in-depth explanation.
    """
    window_length = window[0]
    starting_at = window[1] + 1

    if summarizer in pd_rolling:
        if isinstance(Z.index, pd.MultiIndex):
            if bfill is False:
                feat = getattr(
                    Z.shift(starting_at).rolling(window_length), summarizer
                )()
            else:
                feat = getattr(
                    Z.shift(starting_at).fillna(method="bfill").rolling(window_length),
                    summarizer,
                )()
        else:
            if bfill is False:
                feat = Z.apply(
                    lambda x: getattr(
                        x.shift(starting_at).rolling(window_length), summarizer
                    )()
                )
            else:
                feat = Z.apply(
                    lambda x: getattr(
                        x.shift(starting_at)
                        .fillna(method="bfill")
                        .rolling(window_length),
                        summarizer,
                    )()
                )
    else:
        if bfill is False:
            feat = Z.shift(starting_at)
        else:
            feat = Z.shift(starting_at).fillna(method="bfill")
        if isinstance(Z.index, pd.MultiIndex) and callable(summarizer):
            feat = feat.rolling(window_length).apply(summarizer, raw=True)
        elif not isinstance(Z.index, pd.MultiIndex) and callable(summarizer):
            feat = feat.apply(
                lambda x: x.rolling(window_length).apply(summarizer, raw=True)
            )
    if bfill is True:
        feat = feat.fillna(method="bfill")

    feat.rename(
        columns={
            feat.columns[0]: name + "_" + "_".join([str(item) for item in window])
        },
        inplace=True,
    )

    return feat
