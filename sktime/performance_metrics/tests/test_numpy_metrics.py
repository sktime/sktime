# -*- coding: utf-8 -*-
"""Tests for numpy metrics in _functions module."""
from inspect import getmembers, isfunction

import numpy as np
import pytest

from sktime.performance_metrics.forecasting import _functions
from sktime.utils._testing.series import _make_series


numpy_metrics = getmembers(_functions, isfunction)

exclude_starts_with = ("_", "check", "gmean")
numpy_metrics = [x for x in numpy_metrics if not x[0].startswith(exclude_starts_with)]

names, metrics = zip(*numpy_metrics)

y_pred = _make_series(n_columns=2, n_timepoints=20, random_state=21)
y_true = _make_series(n_columns=2, n_timepoints=20, random_state=42)


@pytest.mark.parametrize("multioutput", ["uniform_average", "raw_values"])
@pytest.mark.parametrize("metric", metrics, ids=names)
def test_metric_output(metric, multioutput):
    """Test output is correct class."""
    res = metric(y_true=y_true, y_pred=y_pred, multioutput=multioutput)

    if multioutput == "uniform_average":
        assert isinstance(res, float)
    elif multioutput == "raw_values":
        assert isinstance(res, np.ndarray)
        assert res.ndim == 1
        assert len(res) == len(y_true.columns)
