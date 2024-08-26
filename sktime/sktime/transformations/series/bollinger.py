#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Class to apply bollinger bounds to a time series."""

__author__ = ["ishanpai"]
__all__ = ["Bollinger"]

import pandas as pd

from sktime.datatypes._utilities import get_cutoff, update_data
from sktime.transformations.base import BaseTransformer


class Bollinger(BaseTransformer):
    """Apply Bollinger bounds to a timeseries.

    The transformation works for univariate and multivariate timeseries.

    Parameters
    ----------
    window : int
        The window over which to compute the moving average and the standard deviation.

    k: float, default = 1
        Multiplier to determine how many stds the upper and lower bounds are from the
        moving average.

    memory : str, optional, default = "all"
        how much of previously seen X to remember, for exact reconstruction of inverse
        "all" : estimator remembers all X, inverse is correct for all indices seen
        "latest" : estimator only remembers latest X necessary for future reconstruction
            inverses at any time stamps after fit are correct, but not past time stamps
        "none" : estimator does not remember any X, inverse is direct cumsum

    Examples
    --------
    >>> from sktime.transformations.series.bollinger import Bollinger
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> transformer = Bollinger(window=12, k=1)
    >>> y_transform = transformer.fit_transform(y)
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["ishanpai"],
        # estimator type
        # --------------
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:instancewise": False,  # is this an instance-wise transform?
        "X_inner_mtype": ["pd.DataFrame", "pd.Series"],
        # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for y?
        "fit_is_empty": False,
        "transform-returns-same-time-index": True,
        "univariate-only": False,
        "capability:inverse_transform": False,
        "remember_data": True,
    }

    def __init__(self, window, k=1, memory="all"):
        if window <= 1:
            raise ValueError(f"window must be greater than 1, passed {window}")
        if k <= 0:
            raise ValueError(f"k must be positive, passed {k}")
        self.window = window
        self.k = k
        self.memory = memory

        self._X = None
        super().__init__()

    def _fit(self, X, y=None):
        """Fit transformer to X and y.

        private _fit containing the core logic, called from fit

        Parameters
        ----------
        X : pd.Series or pd.DataFrame
            Data to fit transform to
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        self: a fitted instance of the estimator
        """
        memory = self.memory

        # remember X or part of X
        if memory == "all":
            self._X = X
        elif memory == "latest":
            n_memory = min(len(X), self.window)
            self._X = X.iloc[-n_memory:]

        self._freq = get_cutoff(X, return_index=True)
        return self

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
        Xt : pd.Series or pd.DataFrame, same type as X
            transformed version of X
        """
        X_orig_index = X.index

        X = update_data(X=self._X, X_new=X)

        Xt = _bollinger_transform(X, self.window, self.k)

        Xt = Xt.loc[X_orig_index]

        return Xt

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
            There are currently no reserved values for transformers.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        return [{"window": 12, "k": 1}, {"window": 2, "k": 1.2}]


def _bollinger_transform(df: pd.Series, window: int, k: float) -> pd.DataFrame:
    df_ma = df.rolling(window=window).mean()
    df_std = df.rolling(window=window).std()
    df_upper = df_ma + k * df_std
    df_lower = df_ma - k * df_std

    df_transformed = pd.concat(
        [df_ma, df_upper, df_lower], keys=("moving_average", "upper", "lower"), axis=1
    )

    return df_transformed
