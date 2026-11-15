"""Tests for probabilistic metrics for distribution predictions."""

import warnings

import pandas as pd
import pytest

from sktime.base._proba._normal import Normal
from sktime.performance_metrics.forecasting.probabilistic._classes import (
    CRPS,
    AUCalibration,
    LogLoss,
)
from sktime.tests.test_switch import run_test_module_changed

warnings.filterwarnings("ignore", category=FutureWarning)

DISTR_METRICS = [CRPS, AUCalibration, LogLoss]

normal_dists = [Normal]


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.performance_metrics"]),
    reason="Run if performance_metrics module has changed.",
)
@pytest.mark.parametrize("normal", normal_dists)
@pytest.mark.parametrize("metric", DISTR_METRICS)
@pytest.mark.parametrize("multivariate", [True, False])
def test_distr_evaluate(normal, metric, multivariate):
    """Test expected output of evaluate functions."""
    y_pred = normal.create_test_instance()
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


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.performance_metrics"]),
    reason="Run if performance_metrics module has changed.",
)
def test_squared_distr_loss_uses_pdf(monkeypatch):
    """SquaredDistrLoss must be -2 * pdf(y) + ||pdf||^2, not -2 * log_pdf(y)."""
    import numpy as np

    from sktime.performance_metrics.forecasting.probabilistic import (
        SquaredDistrLoss,
    )

    mu, sigma = 0.0, 2.0
    y_pred = Normal(
        mu=[[mu]], sigma=[[sigma]], index=pd.Index([0]), columns=pd.Index(["a"])
    )
    y_true = pd.DataFrame([[1.0]], index=[0], columns=["a"])

    # use the exact L2 norm of the normal pdf, to avoid Monte Carlo noise
    pdfnorm = 1 / (2 * sigma * np.sqrt(np.pi))
    monkeypatch.setattr(
        Normal,
        "pdfnorm",
        lambda self, a=2: pd.DataFrame(pdfnorm, index=self.index, columns=self.columns),
    )

    expected = -2 * y_pred.pdf(y_true).iloc[0, 0] + pdfnorm
    res = SquaredDistrLoss().evaluate_by_index(y_true, y_pred)
    assert np.allclose(res.to_numpy(), expected)
