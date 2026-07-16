"""Formatting helpers for normalized foundation-model forecast results."""

import numpy as np
import pandas as pd


def coverage_to_alpha(coverage) -> tuple[float, ...]:
    """Convert interval coverage values to lower/upper quantile levels."""
    coverage = _as_tuple(coverage)
    alpha = []
    for value in coverage:
        alpha.extend([0.5 - value / 2, 0.5 + value / 2])
    return tuple(alpha)


def format_point_result(result, request, y) -> pd.DataFrame:
    """Format a normalized point forecast as an sktime prediction."""
    values = _get_point_values(result)
    values = _select_requested_rows(values, request)
    index = pd.Index(request.absolute_index)
    names = _get_variable_names(y)

    values = _as_2d(values, len(names))
    return pd.DataFrame(values, index=index, columns=names)


def format_quantile_result(result, request, y, alpha) -> pd.DataFrame:
    """Format normalized quantile forecasts as an sktime proba frame."""
    alpha = _as_tuple(alpha)
    names = _get_variable_names(y)
    index = pd.Index(request.absolute_index)

    data = []
    for variable_idx, _ in enumerate(names):
        for quantile in alpha:
            values = _get_quantile_values(result, quantile)
            values = _select_requested_rows(values, request)
            values = _as_2d(values, len(names))
            data.append(values[:, variable_idx])

    columns = pd.MultiIndex.from_product([names, alpha])
    return pd.DataFrame(np.asarray(data).T, index=index, columns=columns)


def format_interval_result(result, request, y, coverage) -> pd.DataFrame:
    """Format normalized quantile forecasts as an sktime interval frame."""
    coverage = _as_tuple(coverage)
    names = _get_variable_names(y)
    index = pd.Index(request.absolute_index)

    data = []
    for variable_idx, _ in enumerate(names):
        for cov in coverage:
            lower = 0.5 - cov / 2
            upper = 0.5 + cov / 2
            for quantile in (lower, upper):
                values = _get_quantile_values(result, quantile)
                values = _select_requested_rows(values, request)
                values = _as_2d(values, len(names))
                data.append(values[:, variable_idx])

    columns = pd.MultiIndex.from_product([names, coverage, ["lower", "upper"]])
    return pd.DataFrame(np.asarray(data).T, index=index, columns=columns)


def _as_tuple(values) -> tuple:
    """Coerce scalar or iterable to tuple."""
    if values is None:
        return ()
    if isinstance(values, tuple):
        return values
    if isinstance(values, list):
        return tuple(values)
    return (values,)


def _get_point_values(result):
    """Return the preferred point forecast from a normalized result."""
    if result.mean is not None:
        return np.asarray(result.mean)
    if result.median is not None:
        return np.asarray(result.median)
    if result.quantiles is not None and 0.5 in result.quantiles:
        return np.asarray(result.quantiles[0.5])
    raise ValueError("ForecastResult does not contain a point forecast.")


def _get_quantile_values(result, alpha: float):
    """Return quantile values, tolerating small float representation differences."""
    if result.quantiles is None:
        raise ValueError("ForecastResult does not contain quantile forecasts.")
    if alpha in result.quantiles:
        return np.asarray(result.quantiles[alpha])
    rounded = round(alpha, 12)
    for key, value in result.quantiles.items():
        if round(key, 12) == rounded:
            return np.asarray(value)
    raise ValueError(f"ForecastResult does not contain quantile alpha={alpha}.")


def _select_requested_rows(values, request):
    """Select only requested rows when a backend returned a full horizon."""
    values = np.asarray(values)
    if values.ndim == 0:
        values = values.reshape(1)

    relative = np.asarray(request.relative_fh)
    if len(values) == len(relative):
        return values

    row_idx = relative - 1
    return values[row_idx]


def _as_2d(values, n_columns: int):
    """Coerce prediction values to ``(n_timepoints, n_columns)``."""
    values = np.asarray(values)
    if values.ndim == 1:
        if n_columns == 1:
            return values.reshape(-1, 1)
        return values.reshape(-1, n_columns)
    return values


def _get_variable_names(y) -> list:
    """Return sktime variable names for fitted target data."""
    return list(y.columns)
