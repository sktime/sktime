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
    samples : array-like or None, default=None
        Sample paths with canonical shape
        ``(n_output_timepoints, n_samples, n_targets)``. For a univariate
        forecast, the final target axis may be omitted, giving
        ``(n_output_timepoints, n_samples)``.

    Notes
    -----
    Summary arrays use time on axis 0 and target variables on axis 1:
    ``(n_output_timepoints, n_targets)``. A one-dimensional summary is accepted
    for one target. Sample paths retain time on axis 0, insert samples on axis 1,
    and place targets on axis 2. Thus ``samples[:, i, :]`` is the complete
    ``i``-th point-forecast path.

    ``n_output_timepoints`` may be the dense ``pred_len`` horizon or exactly
    ``len(fh)`` rows in requested order. All populated summaries and samples must
    use the same time convention.

    Point formatting prefers ``mean``, then ``median``, then quantile ``0.5``.
    If none is present, it uses the empirical sample mean. Quantile formatting
    prefers an explicitly supplied quantile and otherwise computes the empirical
    quantile from samples. Explicit quantile keys equal up to 12 decimal places
    are accepted.
    """

    mean: Any | None = None
    median: Any | None = None
    quantiles: Mapping[float, Any] | None = None
    samples: Any | None = None
