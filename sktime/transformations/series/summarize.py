#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implement transformers for summarizing a time series."""

__author__ = ["mloning", "RNKuhns", "danbartl", "grzegorzrut", "BensHamza"]
__all__ = ["SummaryTransformer", "WindowSummarizer", "SplitterSummarizer"]

import pandas as pd

from sktime.split import ExpandingWindowSplitter, SlidingWindowSplitter
from sktime.transformations.base import BaseTransformer
from sktime.utils.multiindex import flatten_multiindex


class WindowSummarizer(BaseTransformer):
    r"""Transformer for extracting time series features.

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
        the argument ``window``.
        For the function ``lag``, the argument ``window`` is an integer or a list of
        integers giving the ``lag`` values to be used.
        For all other functions, the argument ``window`` is a list with the arguments
        ``lag`` and ``window length``. ``lag`` defines how far back in the past the
        window
        starts, ``window length`` gives the length of the window across which to apply
        the
        function. For multiple different windows, provide a list of lists.

        Please see below a graphical representation of the logic using the following
        symbols:

        ``z`` = time stamp that the window is summarized *to*.

        Part of the window if ``lag`` is between 0 and ``1-window_length``, otherwise
        not part of the window.

        ``x`` = (other) time stamps in the window which is summarized

        ``*`` = observations, past or future, not part of the window

        The summarization function is applied to the window consisting of x and
        potentially z.

        For ``window = [1, 3]``, we have a ``lag`` of 1 and
        ``window_length`` of 3 to target the three last days (exclusive z) that were
        observed. Summarization is done across windows like this::

        |---------------------------|
        | * * * * * * * * x x x z * |
        |---------------------------|

        For ``window = [0, 3]``, we have a ``lag`` of 0 and
        ``window_length`` of 3 to target the three last days (inclusive z) that
        were observed. Summarization is done across windows like this::

        |---------------------------|
        | * * * * * * * * x x z * * |
        |---------------------------|


        Special case ``lag``: Since lags are frequently used and window length is
        redundant, you only need to provide a list of ``lag`` values.
        So ``window = [1]`` will result in the first lag::

        |---------------------------|
        | * * * * * * * * * * x z * |
        |---------------------------|

        And ``window = [1, 4]`` will result in the first and fourth lag::

        |---------------------------|
        | * * * * * * * x * * x z * |
        |---------------------------|

        key: either custom function call (to be provided by user) or
            str corresponding to native pandas window function:
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
            List containing lag and window_length parameters.
        truncate: str, optional (default = None)
            Defines how to deal with NAs that were created as a result of applying the
            functions in the lag_feature dict across windows that are longer than
            the remaining history of data.
            For example a lag config of [14, 7] cannot be fully applied for the first 20
            observations of the targeted column.
            A lag_feature of [[8, 14], [1, 28]] cannot be correctly applied for the
            first 21 resp. 28 observations of the targeted column. Possible values
            to deal with those NAs:

            - None
            - "bfill"

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
    >>> from sktime.split import temporal_train_test_split
    >>> y = load_airline()
    >>> kwargs = {
    ...     "lag_feature": {
    ...         "lag": [1],
    ...         "mean": [[1, 3], [3, 6]],
    ...         "std": [[1, 4]],
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
        # packaging info
        # --------------
        "authors": ["danbartl", "grzegorzrut", "ltsaprounis"],
        "maintainers": ["danbartl"],
        "python_dependencies": ["joblib"],
        # estimator type
        # --------------
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
        "capability:missing_values": True,  # can estimator handle missing data?
        "X-y-must-have-same-index": False,  # can estimator handle different X/y index?
        "enforce_index_type": None,  # index type that needs to be enforced in X/y
        "fit_is_empty": False,  # is fit empty and can be skipped? Yes = True
        "transform-returns-same-time-index": False,
        # does transform return have the same time index as input X
        "remember_data": True,  # remember all data seen as _X
        # CI and test flags
        # -----------------
        "tests:core": True,  # should tests be triggered by framework changes?
    }

    def __init__(
        self,
        lag_feature=None,
        n_jobs=-1,
        target_cols=None,
        truncate=None,
    ):
        self.lag_feature = lag_feature
        self.n_jobs = n_jobs
        self.target_cols = target_cols
        self.truncate = truncate

        super().__init__()

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
        boost_lag = func_dict.loc[lags, "window"].apply(lambda x: [int(x), 1])
        func_dict["window"] = func_dict["window"].astype("object", copy=False)
        func_dict.loc[lags, "window"] = boost_lag
        self.truncate_start = func_dict["window"].apply(lambda x: x[0] + x[1] - 1).max()
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
        from joblib import Parallel, delayed

        idx = X.index
        X = X.combine_first(self._X)

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

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
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


def _window_feature(Z, summarizer=None, window=None, bfill=False):
    """Compute window features and lag.

    Apply summarizer passed over a certain window
    of past observations, e.g. the mean of a window of length 7 days, lagged by 14 days.

    Z: pandas Dataframe with a single column.
    name : str, base string of the derived features, will be appended by
        ``lag`` and window length parameters defined in window.
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
    feat: pd.DataFrame = pd.DataFrame()
    if summarizer in pd_rolling:
        feat = Z.transform(
            lambda x: getattr(
                x.rolling(window=window_length, min_periods=window_length), summarizer
            )().shift(lag)
        )
    elif summarizer == "lag":
        feat = Z.transform(lambda x: x.shift(lag))
    elif callable(summarizer):
        feat = Z.transform(
            lambda x: x.rolling(window=window_length, min_periods=window_length)
            .apply(summarizer, raw=True)
            .shift(lag)
        )
    else:
        raise ValueError("The provided summarizer is not callable.")
    feat = pd.DataFrame(feat)

    # Handle backfill
    if bfill is True:
        feat = feat.bfill()

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
    msg = f"""summary_function must be None, or str or a list or tuple made up of
          {ALLOWED_SUM_FUNCS}.
          """
    if isinstance(summary_function, str):
        if summary_function not in ALLOWED_SUM_FUNCS:
            raise ValueError(msg)
        summary_function = [summary_function]
    elif isinstance(summary_function, (list, tuple)):
        if not all([func in ALLOWED_SUM_FUNCS for func in summary_function]):
            raise ValueError(msg)
    elif summary_function is not None:
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
    msg = """quantiles must be None, int, float or a list or tuple made up of
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
    summary_function : str, list, tuple, or None, default=("mean", "std", "min", "max")
        If not None, a string, or list or tuple of strings indicating the pandas
        summary functions that are used to summarize each column of the dataset.
        Must be one of ("mean", "min", "max", "median", "sum", "skew", "kurt",
        "var", "std", "mad", "sem", "nunique", "count").
        If None, no summaries are calculated, and quantiles must be non-None.
    quantiles : str, list, tuple or None, default=(0.1, 0.25, 0.5, 0.75, 0.9)
        Optional list of series quantiles to calculate. If None, no quantiles
        are calculated, and summary_function must be non-None.
    flatten_transform_index : bool, optional (default=True)
        if True, columns of return DataFrame are flat, by "variablename__feature"
        if False, columns are MultiIndex (variablename__feature)
        has no effect if return mtype is one without column names

    See Also
    --------
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
        # packaging info
        # --------------
        "authors": ["RNKuhns", "fkiraly"],
        # estimator type
        # --------------
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
        flatten_transform_index=True,
    ):
        self.summary_function = summary_function
        self.quantiles = quantiles
        self.flatten_transform_index = flatten_transform_index

        super().__init__()

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
            If ``series_or_df`` is univariate then a scalar is returned. Otherwise,
            a pd.Series is returned.
        """
        if self.summary_function is None and self.quantiles is None:
            raise ValueError(
                "One of `summary_function` and `quantiles` must not be None."
            )
        summary_function = _check_summary_function(self.summary_function)
        quantiles = _check_quantiles(self.quantiles)

        if summary_function is not None:
            # pandas has deprecated "mad"
            # so we need to replicate the functionality here
            if "mad" in summary_function:
                mad_value = (X - X.mean()).abs().mean()
                mad_value = type(X)(mad_value)
                if isinstance(X, pd.DataFrame):
                    mad_value = mad_value.T
                mad_value.index = ["mad"]
                non_mad = set(summary_function).difference(["mad"])
                non_mad = list(non_mad)
            else:
                non_mad = summary_function
            if len(non_mad) > 0:
                summary_value = X.agg(non_mad)
                if "mad" in summary_function:
                    summary_value = pd.concat([summary_value, mad_value])
                    summary_value = summary_value.loc[list(summary_function)]
            else:
                summary_value = mad_value

        if quantiles is not None:
            quantile_value = X.quantile(quantiles)
            quantile_value.index = [str(s) for s in quantile_value.index]

        if summary_function is not None and quantiles is not None:
            summary_value = pd.concat([summary_value, quantile_value])
        elif summary_function is None:
            summary_value = quantile_value

        if isinstance(X, pd.Series):
            summary_value.name = X.name
            summary_value = pd.DataFrame(summary_value)

        Xt = summary_value.T

        if len(Xt) > 1:
            # move the row index as second level to column
            Xt = pd.DataFrame(Xt.T.unstack()).T
            if self.flatten_transform_index:
                Xt.columns = flatten_multiindex(Xt.columns)

        return Xt

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        params1 = {}
        params2 = {"summary_function": ["mean", "mad", "skew"], "quantiles": None}
        params3 = {"summary_function": ["mad"], "quantiles": (0.7,)}
        params4 = {"summary_function": None, "quantiles": (0.1, 0.2, 0.25)}

        return [params1, params2, params3, params4]


class SplitterSummarizer(BaseTransformer):
    """Create summary values of a time series' splits.

    A series-to-series transformer that applies the series-to-primitives transformer
    ``transformer`` to each train split created using the splitter ``splitter``.

    The i-th row of the resulting series is equivalent to
    ``transformer.fit(X_fit).transform(X_trafo)``,
    where ``X_fit`` and ``X_transform`` are obtained from the ``i-th`` split of
    ``splitter``, as determined by the ``fit_on`` and ``transform_on`` parameters.

    The output series aims to provide a summarization of the input series based on the
    given transformer and splitter.

    Parameters
    ----------
    transformer : ``sktime`` transformer inheriting from ``BaseTransformer``
        series-to-primitives transformer used to convert series to primitives.

    splitter : ``sktime`` splitter inheriting from ``BaseSplitter``, optional
    (default=None)
        splitter used to divide the series.
        If None, it takes ``ExpandingWindowSplitter`` with ``start_with_window=False``
        and otherwise default parameters.

    index : str, optional (default="last")
        Determines the indexing approach for the resulting series.
        If "last", the latest index of the split is used.
        If anything else, the row's number becomes the index.

    fit_on : str, optional (default="transform_train")
        What data to fit ``transformer`` on, for the ``i``-th row
        of the resulting series.

        * "all_train" : transform the ``i``-th train split obtained from
          ``splitter.split_series``, called on
          all data seen in ``fit`` and ``update`` calls,
          plus all data seen in ``transform``.
        * "all_test" : transform the ``i``-th test split obtained from
          ``splitter.split_series``, called on
          all data seen in ``fit`` and ``update`` calls,
          plus all data seen in ``transform``.
        * "transform_train" : transform the ``i``-th train split obtained from
          ``splitter.split_series``, called on the data seen in ``transform``.
        * "transform_test" : transform the ``i``-th test split obtained from
          ``splitter.split_series``, called on the data seen in ``transform``.

    transform_on : str, optional (default="transform_train")
        What data to transform with ``transformer``, for the ``i``-th row
        of the resulting series.
        Values and meaning same as for ``fit_on``.

    Methods
    -------
    transform(X) : Transforms the series according to the specified
        series-to-primitives transformer and splitter.

    See Also
    --------
    SummaryTransformer: Calculates summary value of a time series.

    Examples
    --------
    >>> from sktime.transformations.series.summarize import SplitterSummarizer
    >>> from sktime.transformations.series.summarize import SummaryTransformer
    >>> from sktime.split import ExpandingWindowSplitter
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> transformer = SplitterSummarizer(
    ...     transformer=SummaryTransformer(),
    ...     splitter=ExpandingWindowSplitter())
    >>> y_splitsummarized = transformer.fit_transform(y)
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["BensHamza", "fkiraly"],
        # estimator type
        # --------------
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "X_inner_mtype": ["pd.DataFrame", "pd.Series"],
        # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for X?
        "fit_is_empty": True,
    }

    def __init__(
        self,
        transformer,
        splitter=None,
        index="last",
        fit_on="transform_train",
        transform_on="transform_train",
    ):
        self.transformer = transformer
        self.index = index
        self.splitter = splitter
        self.fit_on = fit_on
        self.transform_on = transform_on

        if splitter is None:
            self._splitter = SlidingWindowSplitter(start_with_window=False)
        else:
            self._splitter = splitter

        super().__init__()

        if not hasattr(self.transformer, "fit_transform"):
            raise ValueError(
                f"Error in {self.__class__.__name__}, transformer parameter "
                "should be an estimator with a fit_transform method"
            )
        if not hasattr(self._splitter, "split_series"):
            raise ValueError(
                f"Error in {self.__class__.__name__}, splitter parameter, if passed, "
                "should be an BaseSplitter descendant with a split_series method"
            )

        need_to_remember_data = fit_on.startswith("all") or transform_on.startswith(
            "all"
        )

        if need_to_remember_data:
            self.set_tags(**{"remember_data": True, "fit_is_empty": False})

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
        Xt : pd.DataFrame
            The transformed Data
        """
        fit_on = self.fit_on
        transform_on = self.transform_on

        X_dict = {"transform": X}

        if fit_on.startswith("all") or transform_on.startswith("all"):
            X_all = X.combine_first(self._X)
            X_dict["all"] = X_all

        fit_on_data = fit_on.split("_")[0]
        transform_on_data = transform_on.split("_")[0]
        fit_on_ix = int(fit_on.split("_")[1] == "test")
        transform_on_ix = int(transform_on.split("_")[1] == "test")

        transformed_series = []
        splits_fit = self._splitter.split_series(X_dict[fit_on_data])
        splits_transform = self._splitter.split_series(X_dict[transform_on_data])

        for split_fit, split_transform in zip(splits_fit, splits_transform):
            tf = self.transformer.clone()
            X_fit = split_fit[fit_on_ix]
            X_transform = split_transform[transform_on_ix]
            transformed_split = tf.fit(X_fit).transform(X_transform)
            transformed_split.index = [X_transform.index[-1]]
            transformed_series.append(transformed_split)

        Xt = pd.concat(transformed_series)

        if self.index != "last":
            Xt = Xt.reset_index(drop=True)

        return Xt

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the transformer.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        params1 = {
            "transformer": SummaryTransformer(),
            "splitter": ExpandingWindowSplitter(initial_window=3),
        }

        params2 = {
            "transformer": SummaryTransformer(
                summary_function=["mad"], quantiles=(0.7,)
            ),
            "splitter": SlidingWindowSplitter(window_length=3, step_length=2),
            "index": None,
        }

        params3 = {
            "transformer": SummaryTransformer(),
            "splitter": SlidingWindowSplitter(window_length=3, step_length=2),
            "fit_on": "all_train",
            "transform_on": "all_train",
        }

        params4 = {
            "transformer": SummaryTransformer(),
            "splitter": SlidingWindowSplitter(
                window_length=3, step_length=2, fh=1, start_with_window=True
            ),
            "index": "last",
            "fit_on": "all_test",
            "transform_on": "transform_test",
        }

        return [params1, params2, params3, params4]
