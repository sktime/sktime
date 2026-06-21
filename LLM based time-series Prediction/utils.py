"""Utility helpers for the LLM-based Time Series Assistant."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sktime.datasets import load_airline


@dataclass
class AnalysisResult:
    """Container for analysis outputs."""

    summary: str
    details: dict[str, Any]


def load_sample_data() -> pd.Series:
    """Load a sample univariate time series dataset.

    Returns:
        A monthly airline passenger series from ``sktime.datasets``.
    """
    y = load_airline()
    return y.astype(float)


def compute_trend(series: pd.Series) -> dict[str, float | str]:
    """Estimate a simple linear trend on the input series.

    Args:
        series: Time series values.

    Returns:
        Dictionary with slope and qualitative direction.
    """
    x = np.arange(len(series), dtype=float)
    y = series.to_numpy(dtype=float)

    slope, intercept = np.polyfit(x, y, 1)
    fitted_start = slope * x[0] + intercept
    fitted_end = slope * x[-1] + intercept

    if slope > 0:
        direction = "increasing"
    elif slope < 0:
        direction = "decreasing"
    else:
        direction = "flat"

    return {
        "slope": float(slope),
        "direction": direction,
        "fitted_start": float(fitted_start),
        "fitted_end": float(fitted_end),
    }


def compute_basic_stats(series: pd.Series) -> dict[str, float]:
    """Compute simple descriptive statistics for the series."""
    return {
        "mean": float(series.mean()),
        "std": float(series.std()),
        "min": float(series.min()),
        "max": float(series.max()),
    }


def detect_anomaly_hints(series: pd.Series, z_threshold: float = 2.5) -> dict[str, Any]:
    """Detect anomaly hints using z-score thresholding.

    This is a lightweight heuristic suitable for prototype usage.
    """
    values = series.to_numpy(dtype=float)
    mu = values.mean()
    sigma = values.std()

    if sigma == 0:
        return {
            "count": 0,
            "indices": [],
            "threshold": z_threshold,
            "message": "No variation detected, so anomaly hints are unavailable.",
        }

    z_scores = np.abs((values - mu) / sigma)
    anomaly_positions = np.where(z_scores > z_threshold)[0].tolist()

    # Convert positional indices to the series index for readability.
    anomaly_labels = [str(series.index[pos]) for pos in anomaly_positions]

    return {
        "count": len(anomaly_positions),
        "indices": anomaly_labels,
        "threshold": z_threshold,
        "message": (
            "Potential anomalies detected." if anomaly_positions else "No strong anomalies detected."
        ),
    }
