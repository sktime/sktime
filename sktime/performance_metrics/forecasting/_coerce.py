# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Output coercion utilities for metric classes."""

import numpy as np
import pandas as pd


def _coerce_to_scalar(obj):
    """Coerce obj to scalar, from polymorphic input scalar or pandas."""
    if isinstance(obj, pd.DataFrame):
        assert len(obj) == 1
        assert len(obj.columns) == 1
        return obj.iloc[0, 0]
    if isinstance(obj, pd.Series):
        if len(obj) == 1:
            assert len(obj) == 1
            return obj.iloc[0]
        return obj
    if not isinstance(obj, np.float64):
        obj = np.float64(obj)
    return obj


def _coerce_to_df(obj):
    """Coerce to pd.DataFrame, from polymorphic input scalar or pandas."""
    if isinstance(obj, (float, int)):
        return pd.DataFrame([obj])
    return pd.DataFrame(obj)


def _coerce_to_series(obj):
    """Coerce to pd.Series, from polymorphic input scalar or pandas."""
    if isinstance(obj, pd.DataFrame):
        assert len(obj.columns) == 1
        return obj.iloc[:, 0]
    elif isinstance(obj, pd.Series):
        return obj
    else:
        return pd.Series(obj)


def _coerce_to_1d_numpy(obj):
    """Coerce to 1D np.ndarray, from pd.DataFrame or pd.Series."""
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        obj = obj.values
    return obj.flatten()


def _coerce_to_metric(obj):
    """
    Coerce the input into a forecasting-error-metric-like object.

    Take either:
      1. a BaseForecastingErrorMetric instance → return it unchanged, or
      2. a bare function (callable) → wrap it in _CallableForecastingErrorMetric.
    """
    from sktime.performance_metrics.forecasting._base import (
        BaseForecastingErrorMetric,
        _DynamicForecastingErrorMetric,
    )

    if isinstance(obj, BaseForecastingErrorMetric):
        return obj

    if callable(obj):
        return _DynamicForecastingErrorMetric(obj)

    raise TypeError(
        f"""_coerce_to_metric: expected a forecasting-error metric or a callable,
        got {type(obj).__name__}"""
    )
