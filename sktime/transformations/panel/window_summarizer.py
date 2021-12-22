# -*- coding: utf-8 -*-
"""Helper Functions for example."""
import pandas as pd
from joblib import Parallel, delayed

from sktime.transformations.base import _PanelToTabularTransformer


def find_maxlag(kwargs):
    """Find maximum lag based on provided dictionary."""
    lst = [v[1] for v in kwargs["functions"].values()]
    max_lag = []
    for i in lst:
        for j in i:
            max_lag.append(j[0] + j[1])
    return max(max_lag)


class _LaggedWindowExtractor(_PanelToTabularTransformer):
    """Base adapter class for transformations."""

    def __init__(
        self,
        functions,
        n_jobs=-1,
    ):

        self.functions = functions
        self.n_jobs = n_jobs

        super(_LaggedWindowExtractor, self).__init__()

    # Get extraction parameters
    def fit(self):
        """Fit.

        Returns
        -------
        self : an instance of self
        """
        # check_X(X, coerce_to_pandas=True)
        func_dict = pd.DataFrame(self.functions).T.reset_index()
        func_dict.rename(
            columns={"index": "name", 0: "func", 1: "window"}, inplace=True
        )
        func_dict = func_dict.explode("window")

        self._func_dict = func_dict
        self._is_fitted = True
        self._truncate_start = (
            func_dict["window"].apply(lambda x: x[0] + x[1] - 1).max()
        )

        return self


class LaggedWindowSummarizer(_LaggedWindowExtractor):
    """Transformer for extracting time series features."""

    def transform(self, Z):
        """Transform X.

        Parameters
        ----------
        Z : pd.DataFrame with y as columns
            and Time Series ID and Period as
            MultiIndex

        Returns
        -------
        Xt : pandas DataFrame
        Transformed pandas DataFrame

        """
        # input checks

        self.check_is_fitted()

        grouped = Z.groupby(level=0)["y"]
        df = Parallel(n_jobs=self.n_jobs)(
            delayed(_window_feature)(grouped, **kwargs)
            for index, kwargs in self._func_dict.iterrows()
        )
        col_names = [o.name for o in df]
        df = pd.concat(df, axis=1)
        df.columns = col_names

        Zt = pd.concat([Z, df], axis=1)

        return Zt


def _window_feature(
    grouped,
    name=None,
    func=None,
    window=None,
):
    """Compute window features and lag.

    grouped: pandas.core.groupby.generic.SeriesGroupBy
        Object create by grouping across groupBy columns.
    name: string
        Name fragmentfor column to be created.
        Name will also include window shift and size
    func: function or string
        Callable if a custom function, None otherwise.
    window: list
        Contains values for window shift and window length.
    """
    # List of native pandas rolling window function.
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

    # danbartl: investivate different engines for pandas
    if func == "lag":
        features = grouped.shift(window[0])
        features.name = name + "_" + str(window[0])
    else:
        if func in pd_rolling:
            features = grouped.apply(
                lambda x: getattr(x.shift(window[0]).rolling(window[1]), func)()
            )
        else:
            features = grouped.apply(
                lambda x: x.shift(window[0]).rolling(window[1]).apply(func, raw=True)
            )
        features.name = name + "_" + str(window[0]) + "_" + str(window[1])

    return features
