# -*- coding: utf-8 -*-
"""Get index of time series data, helper function."""

import numpy as np
import pandas as pd


def _get_index(x):
    if hasattr(x, "index"):
        return x.index
    else:
        # select last dimension for time index
        return pd.RangeIndex(x.shape[-1])


def get_time_index(X):
    """Get index of time series data, helper function.

    Parameters
    ----------
    X : pd.DataFrame

    Returns
    -------
    time_index : pandas Index
        Index of time series
    """
    # assumes that all samples share the same the time index, only looks at
    # first row
    if isinstance(X, pd.DataFrame):
        if isinstance(X.index, pd.MultiIndex):
            return X.xs(
                X.index.get_level_values("instances")[0], level="instances"
            ).index
        else:
            return _get_index(X.iloc[0, 0])

    elif isinstance(X, pd.Series):
        if isinstance(X.index, pd.MultiIndex):
            return X.xs(
                X.index.get_level_values("instances")[0], level="instances"
            ).index
        else:
            return _get_index(X.iloc[0])

    elif isinstance(X, np.ndarray):
        return _get_index(X)

    else:
        raise ValueError(
            f"X must be a pandas DataFrame or Series, but found: {type(X)}"
        )
