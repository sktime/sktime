# -*- coding: utf-8 -*-
"""Tests for probabilistic metrics for distribution predictions."""
import warnings

import pandas as pd
import pytest

from sktime.performance_metrics.forecasting.probabilistic._classes import CRPS, LogLoss
from sktime.proba.tfp import Normal
from sktime.utils.validation._dependencies import _check_soft_dependencies

warnings.filterwarnings("ignore", category=FutureWarning)

DISTR_METRICS = [CRPS, LogLoss]


@pytest.mark.skipif(
    not _check_soft_dependencies("tensorflow_probability", severity="none"),
    reason="skip test if required soft dependency is not available",
)
@pytest.mark.parametrize("metric", DISTR_METRICS)
@pytest.mark.parametrize("multivariate", [True, False])
def test_distr_evaluate(metric, multivariate):
    """Test expected output of evaluate functions."""
    y_pred = Normal.create_test_instance()
    y_true = y_pred.sample()

    m = metric(multivariate=multivariate)

    if not multivariate:
        expected_cols = y_true.columns
    else:
        expected_cols = ["score"]

    res = m.evaluate_by_index(y_true, y_pred)
    assert isinstance(res, pd.DataFrame)
    assert (res.columns == expected_cols).all()
    assert res.shape == (y_true.shape[0], len(expected_cols))

    res = m.evaluate(y_true, y_pred)
    assert isinstance(res, pd.DataFrame)
    assert (res.columns == expected_cols).all()
    assert res.shape == (1, len(expected_cols))
