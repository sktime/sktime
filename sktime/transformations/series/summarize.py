#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implement transformers for summarizing a time series."""

__author__ = ["mloning", "RNKuhns", "danbartl", "grzegorzrut"]
__all__ = ["SummaryTransformer", "WindowSummarizer"]

import warnings

import pandas as pd
from joblib import Parallel, delayed

from sktime.transformations.base import BaseTransformer


class WindowSummarizer(BaseTransformer):
    """Transformer for extracting time series features.

    The WindowSummarizer transforms input series to features based
    on a provided dictionary of window summarizer, window shifts
    and window lengths.

    Parameters
    ----------
    n_jobs : int, optional (default=-1)
        The number of jobs to run in parallel for applying the window functions.
        ``-1`` means using all processors.
    target_cols: list of str, optional (default = None)
        Specifies which columns in X to target for applying the window functions.
        ``None`` will target the first column
    lag_feature: dict of str and list, optional (default = dict containing first lag)
        Dictionary specifying as key the type of function to be used and as value
        the argument `window`.

        For all keys other than `lag`, the argument `window` is a length 2 list.
        The window elements can be either integers indicate the number of
        observations, or strings specifying a time period (an offset).

        The first element of the `window` is the `lag`, which specifies how far back
        in the past the window will start. The second element, `window length`,
        is the length of the window across which to apply the function.

        For ease of notation, for the key "lag", only a single integer or offset
        specifying the `lag` argument will be provided.

        The offsets work only with datasets with time index (either pd.DateimeIndex,
        pd.RangeIndex, or pd.TimedeltaIndex) are expected to have a format of
        'NUMBER'+'UNIT', where NUMBER is an integer and UNIT is one of the following:
        * D - day,
        * H - hour,
        * T - minute,
        * S - second,
        * L - milisecond,
        * U - microsecond,
        * N - nanosecond.
        For instance, the following offsets would be acceptable: '1D', '2H', '30S'.

        Please see below a graphical representation of the logic using the following
        symbols:

        ``z`` = time stamp that the window is summarized *to*.
        Part of the window if `lag` is between 0 and `1-window_length`, otherwise
        not part of the window.
        ``*`` = (other) time stamps in the window which is summarized
        ``x`` = observations, past or future, not part of the window

        The summarization function is applied to the window consisting of * and
        potentially z.

        For `window = [1, 3]`, we have a `lag` of 1 and
        `window_length` of 3 to target the three last records (exclusive z) that were
        observed. Summarization is done across windows like this:
        |-------------------------- |
        | x x x x x x x x * * * z x |
        |---------------------------|

        For `window = [0, 3]`, we have a `lag` of 0 and
        `window_length` of 3 to target the three last records (inclusive z) that
        were observed. Summarization is done across windows like this:
        |-------------------------- |
        | x x x x x x x x * * z x x |
        |---------------------------|


        Special case ´lag´: Since lags are frequently used and window length is
        redundant, a special notation will be used for lags. You need to provide a list
        of `lag` values, and `window_length` is not available.
        So `window = [1]` will result in the first lag:

        |-------------------------- |
        | x x x x x x x x x x * z x |
        |---------------------------|

        And `window = [1, 4]` will result in the first and fourth lag:

        |-------------------------- |
        | x x x x x x x * x x * z x |
        |---------------------------|


        If, for instance, the time series would have a daily frequency and the window
        function would be specified as  `window = ['1D','4D']`, the logic with
        the offsets would be exactly the same. Note that it is possible that the time
        difference between consecutive records is not constant or the offset is
        not equal to a multiple of a time series frequency.

        key: either custom function call (to be provided by user)
        or str corresponding to native pandas window function:
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
            The column generated will be named after the key provided, followed by the
            lag parameter and the window_length (if not a lag).
        second value (window): list of integers
            List containg lag and window_length parameters.
        truncate: str, optional (default = None)
            Defines how to deal with NAs that were created as a result of applying the
            functions in the lag_feature dict across windows that are longer than
            the remaining history of data.
            For example a lag config of [14, 7] cannot be fully applied for the first 20
            observations of the targeted column.
            A lag_feature of [[8, 14], [1, 28]] cannot be correctly applied for the
            first 21 resp. 28 observations of the targeted column. Possible values
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
        Truncate_start will give the maximum of observations that are filled with NAs
        across all arguments of the lag_feature when truncate is set to None.

    Returns
    -------
    X: pd.DataFrame
        Contains all transformed columns as well as non-transformed columns.
        The raw inputs to transformed columns will be dropped.
    self: reference to self

    Examples
    --------
    >>> import pandas as pd
    >>> from sktime.transformations.series.summarize import WindowSummarizer
    >>> from sktime.datasets import load_airline, load_longley
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> from sktime.forecasting.base import ForecastingHorizon
    >>> from sktime.forecasting.compose import ForecastingPipeline
    >>> from sktime.forecasting.model_selection import temporal_train_test_split
    >>> y = load_airline()
    >>> kwargs = {
    ...     "lag_feature": {
    ...         "lag": [1,'30D'],
    ...         "mean": [[1, 3], ['30D', '60D']],
    ...         "std": [[1, 4]],
    ...     }
    ... }
    >>> transformer = WindowSummarizer(**kwargs)
    >>> y_transformed = transformer.fit_transform(y)

        Example where we transform on a different, later test set:
    >>> y = load_airline()
    >>> y_train, y_test = temporal_train_test_split(y)
    >>> kwargs = {
    ...     "lag_config": {
    ...         "lag": ["lag", [[1, 0],['60D', 0]]],
    ...         "mean": ["mean", [[3, 0], ['30D', '60D']]],
    ...         "std": ["std", [[4, 0],['30D','90D']]],
    ...     }
    ... }
    >>> transformer = WindowSummarizer(**kwargs)
    >>> y_test_transformed = transformer.fit(y_train).transform(y_test)

        Example with transforming multiple columns of exogeneous features
    >>> y, X = load_longley()
    >>> y_train, y_test, X_train, X_test = temporal_train_test_split(y, X)
    >>> kwargs = {
    ...     "lag_config": {
    ...         "lag": ["lag", [[1, 0]]],
    ...         "mean": ["mean", [[3, 0]]],
    ...         "std": ["std", [[4, 0]]],
    ...     }
    ... }
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
        "X_inner_mtype": [
            "pd-multiindex",
            "pd.DataFrame",
            "pd_multiindex_hier",
        ],  # which mtypes do _fit/_predict support for X?
        "skip-inverse-transform": True,  # is inverse-transform skipped when called?
        "univariate-only": False,  # can the transformer handle multivariate X?
        "handles-missing-data": True,  # can estimator handle missing data?
        "X-y-must-have-same-index": False,  # can estimator handle different X/y index?
        "enforce_index_type": None,  # index type that needs to be enforced in X/y
        "fit_is_empty": False,  # is fit empty and can be skipped? Yes = True
        "transform-returns-same-time-index": False,
        # does transform return have the same time index as input X
    }

    def __init__(
        self,
        lag_config=None,
        lag_feature=None,
        n_jobs=-1,
        target_cols=None,
        truncate=None,
    ):

        # self._converter_store_X = dict()
        self.lag_config = lag_config
        self.lag_feature = lag_feature
        self.n_jobs = n_jobs
        self.target_cols = target_cols
        self.truncate = truncate

        # if lag config has input other than ints,
        # pd-multiindex is disabled
        lag_conf = None
        if lag_config is not None:
            lag_conf = [x[1:] for x in lag_config.values()]
        if lag_feature is not None:
            lag_conf = lag_feature.values()
        if lag_conf is not None:
            if len([x for x in lag_conf if not isinstance(x, int)]) > 0:
                WindowSummarizer._tags["X_inner_mtype"] = [
                    # pd-multiindex", not compatible with panels
                    "pd.DataFrame"
                ]

                WindowSummarizer._tags["enforce_index_type"] = [
                    "pd.DatetimeIndex",
                    "pd.RangeIndex",
                    "pd.TimedeltaIndex",
                ]

        super(WindowSummarizer, self).__init__()

    def _fit(self, X, y=None):
        """Fit transformer to X and y.

        Private _fit containing the core logic, called from fit

        Attributes
        ----------
        truncate_start : int
            See section class WindowSummarizer - Parameters - truncate for a more
            detailed explanation of truncation as a result of applying windows of
            certain lengths across past observations.
            Truncate_start will give the maximum of observations that are filled
            with NAs across all arguments of the lag_feature when truncate is
            set to None.

        Returns
        -------
        X: pd.DataFrame
            Contains all transformed columns as well as non-transformed columns.
            The raw inputs to transformed columns will be dropped.
        self: reference to self
        """
        self._X_memory = X

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

        # Convert lag config dictionary to pandas dataframe
        if self.lag_config is not None:
            func_dict = pd.DataFrame(self.lag_config).T.reset_index()
            func_dict.rename(
                columns={"index": "name", 0: "summarizer", 1: "window"},
                inplace=True,
            )
            func_dict = func_dict.explode("window")
            func_dict["window"] = func_dict["window"].apply(lambda x: [x[1] + 1, x[0]])
            func_dict.drop("name", inplace=True, axis=1)
            warnings.warn(
                "Specifying lag features via lag_config is deprecated since 0.12.0,"
                + " and will be removed in 0.13.0. Please use the lag_feature notation"
                + " (see the documentation for the new notation)."
            )
        else:
            if self.lag_feature is None:
                func_dict = pd.DataFrame(
                    {
                        "lag": [1],
                    }
                ).T.reset_index()
            else:
                func_dict = pd.DataFrame.from_dict(
                    self.lag_feature, orient="index"
                ).reset_index()

            func_dict = pd.melt(
                func_dict, id_vars="index", value_name="window", ignore_index=False
            )
            func_dict.sort_index(inplace=True)
            func_dict.drop("variable", axis=1, inplace=True)
            func_dict.rename(
                columns={"index": "summarizer"},
                inplace=True,
            )
            func_dict = func_dict.dropna(axis=0, how="any")
            # Identify lags (since they can follow special notation)
            lags = func_dict["summarizer"] == "lag"
            # Convert lags to default list notation with window_length 1
            boost_lag = func_dict.loc[lags, "window"].apply(lambda x: [x, 1])
            func_dict.loc[lags, "window"] = boost_lag

        # self.truncate_start = func_dict["window"]\
        # .apply(lambda x: x[0] + x[1] - 1).max()

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
        idx = X.index
        X = X.combine_first(self._X_memory)

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
                hier_levels = list(range(X.index.nlevels - 1))
                X_grouped = X.groupby(level=hier_levels)[cols]
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

        Xt_return = Xt_return.loc[idx]
        return Xt_return

    def _update(self, X, y=None):
        """Update X and return a transformed version.

        Parameters
        ----------
        X : pd.DataFrame
        y : None

        Returns
        -------
        transformed version of X
        """
        self._X_memory = X.combine_first(self._X_memory)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        params1 = {
            "lag_feature": {
                "lag": [1],
                "mean": [[1, 3], [1, 12]],
                "std": [[1, 4]],
            }
        }

        params2 = {
            "lag_feature": {
                "lag": [3, 6],
            }
        }

        params3 = {
            "lag_feature": {
                "mean": [[1, 7], [8, 7]],
                "cov": [[1, 28]],
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


def find_timedelta_unit_value(timedelta):
    """Extract the number of time units and the time unit itself.

    Parameters
    ----------
    timedelta : pd.Timedelta

    Returns
    -------
    value : int
        Number of time units.
    unit : str
        Character describing the timedelta unit.
    """
    time_units = ["D", "H", "T", "S", "L", "U", "N"]
    unit = timedelta.resolution_string
    if unit in time_units:
        pass
    else:
        raise ValueError("passed incompatible time span")
    value = round(timedelta.asm8 / pd.Timedelta(1, unit))

    return value, unit


def _window_feature(Z, summarizer=None, window=None, bfill=False):
    """Compute window features and lag.

    Apply summarizer passed over a certain window
    of past observations, e.g. the mean of a window of length 7 days, lagged by 14 days.

    Z: pandas Dataframe with a single column.
    name : str, base string of the derived features, will be appended by
        `lag` and window length parameters defined in window.
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
        List containing window_length and lag parameters, see WindowSummarizer
        class description for in-depth explanation.
    """
    lag = window[0]
    window_length = window[1]

    # this could be moved to _fit in WindowSummarizer
    if isinstance(lag, str):
        lag = pd.Timedelta(lag)

    # this could be moved to _fit in WindowSummarizer
    if isinstance(window_length, str):
        window_length = pd.Timedelta(window_length)

    # this could be moved to _fit in WindowSummarizer
    timedelta_lag = isinstance(lag, pd.Timedelta)
    timedelta_window = isinstance(window_length, pd.Timedelta)
    freq = None
    if timedelta_lag or timedelta_window:

        if timedelta_lag:
            if isinstance(Z, pd.core.groupby.generic.SeriesGroupBy):
                raise TypeError(
                    "Timedelta lags are not compatible with SeriesGroupBy objects."
                )

            lag_value, freq = find_timedelta_unit_value(lag)

            # to have similar behavior as for numerical lags?
            lag = int(lag_value)
        Z_index = Z.index
        # this could be moved to _fit in WindowSummarizer
        period_index_flag = isinstance(Z_index, pd.PeriodIndex)
        if period_index_flag:
            org_freq = Z_index.freq
            try:
                Z_index = Z_index.to_timestamp()
                Z.index = Z_index
            except Exception:
                raise TypeError("Index has to be transformable to DatetimeIndex.")
        Z_index = pd.DataFrame(index=Z_index)
        Z = Z_index.shift(-lag, freq).join(Z, how="outer")
        if timedelta_window:
            window_value, window_freq = find_timedelta_unit_value(window_length)
    if not timedelta_lag:
        # procedure for integer lags
        lag = int(lag)

    if summarizer in pd_rolling:
        if isinstance(Z, pd.core.groupby.generic.SeriesGroupBy):
            if bfill is False:
                feat = getattr(Z.shift(lag, freq).rolling(window_length), summarizer)()
            else:
                feat = getattr(
                    Z.shift(lag, freq).fillna(method="bfill").rolling(window_length),
                    summarizer,
                )()
            feat = pd.DataFrame(feat)
        else:
            if bfill is False:
                feat = Z.apply(
                    lambda x: getattr(
                        x.shift(lag, freq).rolling(window_length), summarizer
                    )()
                )
            else:
                feat = Z.apply(
                    lambda x: getattr(
                        x.shift(lag, freq)
                        .fillna(method="bfill")
                        .rolling(window_length),
                        summarizer,
                    )()
                )
    else:
        if bfill is False:
            feat = Z.shift(lag, freq)
        else:
            feat = Z.shift(lag, freq).fillna(method="bfill")
        if isinstance(Z, pd.core.groupby.generic.SeriesGroupBy) and callable(
            summarizer
        ):
            feat = feat.rolling(window_length).apply(summarizer, raw=True)
        elif not isinstance(Z, pd.core.groupby.generic.SeriesGroupBy) and callable(
            summarizer
        ):
            feat = feat.apply(
                lambda x: x.rolling(window_length).apply(summarizer, raw=True)
            )
        feat = pd.DataFrame(feat)
    if bfill is True:
        feat = feat.fillna(method="bfill")

    if callable(summarizer):
        name = summarizer.__name__
    else:
        name = summarizer

    if name == "lag":
        feat.rename(
            columns={feat.columns[0]: name + "_" + str(window[0])},
            inplace=True,
        )
    else:
        feat.rename(
            columns={
                feat.columns[0]: name + "_" + "_".join([str(item) for item in window])
            },
            inplace=True,
        )

    if timedelta_lag or timedelta_window:

        feat = Z_index.join(feat, how="left")

        # if timedelta_window:
        #  mask = [
        #  x < min(Z_index.shift(lag, freq).shift(window_value, window_freq).index)
        #  for x in feat.index
        #  ]
        # else:
        #  # mark cases in the index that are not in the Z_index.shift(lag, freq) index
        #     mask = [x < min(Z_index.shift(lag, freq).index) for x in feat.index]
        # feat.loc[mask] = np.nan
        # returning initial PeriodIndex frequency
        if period_index_flag:
            feat = feat.to_period(org_freq)

    return feat


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


class SummaryTransformer(BaseTransformer):
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
    WindowSummarizer:
        Extracting features across (shifted) windows from series

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
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Primitives",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "X_inner_mtype": ["pd.DataFrame", "pd.Series"],
        # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for X?
        "fit_is_empty": True,
    }

    def __init__(
        self,
        summary_function=("mean", "std", "min", "max"),
        quantiles=(0.1, 0.25, 0.5, 0.75, 0.9),
    ):
        self.summary_function = summary_function
        self.quantiles = quantiles
        super(SummaryTransformer, self).__init__()

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing the core logic, called from transform

        Parameters
        ----------
        X : pd.Series or pd.DataFrame
            Data to be transformed
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        summary_value : scalar or pd.Series
            If `series_or_df` is univariate then a scalar is returned. Otherwise,
            a pd.Series is returned.
        """
        Z = X

        if self.summary_function is None and self.quantiles is None:
            raise ValueError(
                "One of `summary_function` and `quantiles` must not be None."
            )
        summary_function = _check_summary_function(self.summary_function)
        quantiles = _check_quantiles(self.quantiles)

        summary_value = Z.agg(summary_function)
        if quantiles is not None:
            quantile_value = Z.quantile(quantiles)
            quantile_value.index = [str(s) for s in quantile_value.index]
            summary_value = pd.concat([summary_value, quantile_value])

        if isinstance(Z, pd.Series):
            summary_value.name = Z.name
            summary_value = pd.DataFrame(summary_value)

        return summary_value.T
