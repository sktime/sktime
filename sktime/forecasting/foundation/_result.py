"""Small data contracts shared by foundation-model adapters and the base class."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass
class ModelHandle:
    """Native backend objects loaded once and potentially shared.

    Parameters
    ----------
    model : Any or None, default=None
        Primary model object. Torch adapters should place the object exposing
        ``eval`` and ``device`` here so the shared inference context can use it.
    tokenizer : Any or None, default=None
        Tokenizer or input encoder used alongside the model.
    pipeline : Any or None, default=None
        Higher-level native prediction pipeline, wrapper, or forecaster.

    Notes
    -----
    A handle is cached process-locally and can be attached to multiple estimator
    instances. Its contents should therefore contain model-level state only and
    should be treated as read-only during inference. Per-series state belongs on
    the fitted estimator.
    """

    model: Any | None = None
    tokenizer: Any | None = None
    pipeline: Any | None = None


@dataclass(frozen=True)
class ForecastRequest:
    """Forecast horizon metadata used to format native output.

    Parameters
    ----------
    relative_fh : tuple of int
        Requested steps relative to the fitted cutoff, preserving request order.
    absolute_index : pandas index-like
        Absolute labels to use on the formatted prediction.
    alpha : tuple of float or None
        Requested quantile probabilities, or ``None`` for point prediction.
    """

    relative_fh: tuple[int, ...]
    absolute_index: Any
    alpha: tuple[float, ...] | None


@dataclass
class ForecastResult:
    """Model-family-neutral numeric output returned by ``_inference``.

    Parameters
    ----------
    mean : array-like or None, default=None
        Mean point forecast.
    median : array-like or None, default=None
        Median point forecast, used when ``mean`` is absent.
    quantiles : Mapping[float, array-like] or None, default=None
        Map from quantile probability to forecast values.

    Notes
    -----
    Every supplied array uses time on axis 0 and target variables on axis 1:
    ``(n_output_timepoints, n_targets)``. A one-dimensional array is accepted for
    one target. ``n_output_timepoints`` may be the dense ``pred_len`` horizon or
    exactly ``len(fh)`` rows in requested order. All populated summaries must use
    the same convention.

    Point formatting prefers ``mean``, then ``median``, then quantile ``0.5``.
    Quantile prediction requires an entry for every requested alpha (keys equal up
    to 12 decimal places are accepted).
    """

    mean: Any | None = None
    median: Any | None = None
    quantiles: Mapping[float, Any] | None = None
