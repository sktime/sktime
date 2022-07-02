# -*- coding: utf-8 -*-
"""Tests for classes in _classes module."""
from inspect import getmembers, isclass

import numpy as np
import pandas as pd
import pytest

from sktime.performance_metrics.forecasting import _classes
from sktime.utils._testing.series import _make_series

metric_classes = getmembers(_classes, isclass)

exclude_starts_with = ("_", "Base")
metric_classes = [x for x in metric_classes if not x[0].startswith(exclude_starts_with)]

names, metrics = zip(*metric_classes)


@pytest.mark.parametrize("n_columns", [1, 2])
@pytest.mark.parametrize("multioutput", ["uniform_average", "raw_values"])
@pytest.mark.parametrize("metric", metrics, ids=names)
def test_metric_output_direct(metric, multioutput, n_columns):
    """Test output is of correct type, dependent on multioutput.

    Also tests that four ways to call the metric yield equivalent results:
        1. using the __call__ dunder
        2. calling the evaluate method
    """
    y_pred = _make_series(n_columns=n_columns, n_timepoints=20, random_state=21)
    y_true = _make_series(n_columns=n_columns, n_timepoints=20, random_state=42)

    # coerce to DataFrame since _make_series does not return consisten output type
    y_pred = pd.DataFrame(y_pred)
    y_true = pd.DataFrame(y_true)

    res = dict()

    res[1] = metric(multioutput=multioutput)(
        y_true=y_true,
        y_pred=y_pred,
        y_pred_benchmark=y_pred,
        y_train=y_true,
    )

    res[2] = metric(multioutput=multioutput).evaluate(
        y_true=y_true,
        y_pred=y_pred,
        y_pred_benchmark=y_pred,
        y_train=y_true,
    )

    if multioutput == "uniform_average":
        assert all(isinstance(x, float) for x in res.values())
    elif multioutput == "raw_values":
        assert all(isinstance(x, np.ndarray) for x in res.values())
        assert all(x.ndim == 1 for x in res.values())
        assert all(len(x) == len(y_true.columns) for x in res.values())

    # assert results from all options are equal
    assert np.allclose(res[1], res[2])
