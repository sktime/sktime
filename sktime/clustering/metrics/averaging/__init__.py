# -*- coding: utf-8 -*-
"""Time series averaging metrics."""
__all__ = ["dba", "mean_average", "_resolve_average_callable"]
from sktime.clustering.metrics.averaging._averaging import (
    _resolve_average_callable,
    dba,
    mean_average,
)
