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
        assert len(obj) == 1
        return obj.iloc[0]
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


# A tiny “callable → Metric” wrapper, in case the user passed a bare function
class _CallableForecastingErrorMetric:
    """
    Wrap any bare forecasting-error function into a BaseForecastingErrorMetric subclass.

    Signature: (y_true, y_pred, ..., by_index=…, multioutput=…, multilevel=…)

    This ensures that .evaluate(...) and .evaluate_by_index(...) work uniformly.
    """

    _tags = {
        "requires-y-train": False,
        "requires-y-pred-benchmark": False,
        "univariate-only": False,
    }

    def __init__(self, func):
        from sktime.performance_metrics.forecasting._base import (
            BaseForecastingErrorMetric,
        )

        if not callable(func):
            raise TypeError(f"""_CallableForecastingErrorMetric: expected
                            callable, got {type(func)}""")
        self._func = func
        # Dynamically subclass BaseForecastingErrorMetric so isinstance checks pass
        self.__class__ = type(
            "_WrapForecastingErrorMetric",
            (BaseForecastingErrorMetric,),
            {**self.__class__.__dict__},
        )

    def evaluate(self, y_true, y_pred, **kwargs):
        return self._func(y_true=y_true, y_pred=y_pred, **kwargs)

    def evaluate_by_index(self, y_true, y_pred, **kwargs):
        return self._func(y_true=y_true, y_pred=y_pred, by_index=True, **kwargs)


def _coerce_to_metric(obj):
    """
    Coerce the input into a forecasting-error-metric-like object.

    Take either:
      1. a BaseForecastingErrorMetric instance → return it unchanged, or
      2. a bare function (callable) → wrap it in _CallableForecastingErrorMetric.
    """
    from sktime.performance_metrics.forecasting._base import (
        BaseForecastingErrorMetric,
    )

    if isinstance(obj, BaseForecastingErrorMetric):
        return obj

    if callable(obj):
        return _CallableForecastingErrorMetric(obj)

    raise TypeError(
        f"""_coerce_to_metric: expected a forecasting-error metric or a callable,
        got {type(obj).__name__}"""
    )
