"""Convert model-neutral numeric forecasts to sktime pandas output.

Adapters should remove native batch, sample, and distribution dimensions before
creating a ``ForecastResult``. These helpers handle horizon row selection and the
final time-by-variable layout; they do not interpret backend-specific shapes.
"""

import numpy as np
import pandas as pd


def format_point_result(result, request, y) -> pd.DataFrame:
    """Format a normalized point forecast as an sktime prediction.

    Parameters
    ----------
    result : ForecastResult
        Normalized native output. Point values are selected in mean, median,
        0.5-quantile precedence order.
    request : ForecastRequest
        Relative horizon steps and absolute output index.
    y : pd.DataFrame
        Fitted target context. Its columns define output names and ordering.

    Returns
    -------
    pd.DataFrame
        Point forecasts with shape ``(len(request.relative_fh), n_targets)``.

    Notes
    -----
    Result rows may be either a dense future horizon or rows already selected in
    requested horizon order. Malformed dimensions fail through NumPy or pandas;
    adapters should normalize native output before constructing the result.
    """
    values = _get_point_values(result)
    values = _select_requested_rows(values, request)
    index = pd.Index(request.absolute_index)
    names = _get_variable_names(y)

    values = _as_2d(values, len(names))
    return pd.DataFrame(values, index=index, columns=names)


def format_quantile_result(result, request, y, alpha) -> pd.DataFrame:
    """Format normalized quantile forecasts as an sktime probability frame.

    Parameters
    ----------
    result : ForecastResult
        Normalized output containing every requested quantile.
    request : ForecastRequest
        Relative horizon steps and absolute output index.
    y : pd.DataFrame
        Fitted target context. Its columns define variable order.
    alpha : float or sequence of float
        Requested quantile probabilities. Input order is preserved.

    Returns
    -------
    pd.DataFrame
        Quantile forecasts with row index ``request.absolute_index`` and
        ``(variable, alpha)`` MultiIndex columns.
    """
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


def _as_tuple(values) -> tuple:
    """Coerce a scalar or supported sequence to an immutable tuple."""
    if values is None:
        return ()
    if isinstance(values, tuple):
        return values
    if isinstance(values, list):
        return tuple(values)
    return (values,)


def _get_point_values(result):
    """Return point values using mean, median, then 0.5-quantile precedence."""
    if result.mean is not None:
        return np.asarray(result.mean)
    if result.median is not None:
        return np.asarray(result.median)
    if result.quantiles is not None and 0.5 in result.quantiles:
        return np.asarray(result.quantiles[0.5])
    raise ValueError("ForecastResult does not contain a point forecast.")


def _get_quantile_values(result, alpha: float):
    """Return one quantile, tolerating small float representation differences."""
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
    """Select requested rows from direct or dense-horizon backend output.

    When the row count already equals the number of requested steps, rows are
    assumed to be selected and ordered by the adapter. Otherwise, values are
    interpreted as a dense one-based future horizon, so relative step ``k`` is
    found at row ``k - 1``.
    """
    values = np.asarray(values)
    if values.ndim == 0:
        values = values.reshape(1)

    relative = np.asarray(request.relative_fh)
    if len(values) == len(relative):
        return values

    row_idx = relative - 1
    return values[row_idx]


def _as_2d(values, n_columns: int):
    """Coerce prediction values to ``(n_timepoints, n_columns)``.

    One-dimensional multivariate input is reshaped in time-major order and must
    contain a multiple of ``n_columns`` values. Two-dimensional input is returned
    unchanged; adapters are responsible for removing other native dimensions.
    """
    values = np.asarray(values)
    if values.ndim == 1:
        if n_columns == 1:
            return values.reshape(-1, 1)
        return values.reshape(-1, n_columns)
    return values


def _get_variable_names(y) -> list:
    """Return fitted target columns in the required backend output order."""
    return list(y.columns)
