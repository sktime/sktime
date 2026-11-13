"""Tests for BoxCoxBiasAdjustedForecaster."""

import numpy as np
import pytest

from sktime.datasets import load_airline
from sktime.forecasting.boxcox_biasadj import BoxCoxBiasAdjustedForecaster
from sktime.forecasting.naive import NaiveForecaster
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(BoxCoxBiasAdjustedForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("lambda_fixed", [None, 0.5])
def test_boxcox_biasadj_predict_interval(lambda_fixed):
    """Prediction intervals must be non-trivial, nested and contain predict."""
    y = load_airline()
    f = BoxCoxBiasAdjustedForecaster(NaiveForecaster(), lambda_fixed=lambda_fixed)
    f.fit(y, fh=[1, 2, 3])

    y_pred = f.predict().to_numpy()
    pred_int = f.predict_interval(coverage=[0.5, 0.9]).to_numpy()
    lo50, hi50, lo90, hi90 = pred_int.T

    assert (lo90 < lo50).all() and (lo50 < y_pred).all()
    assert (y_pred < hi50).all() and (hi50 < hi90).all()


@pytest.mark.skipif(
    not run_test_for_class(BoxCoxBiasAdjustedForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_boxcox_biasadj_lambda_fixed():
    """lambda_fixed must be used as the Box-Cox parameter, if passed."""
    y = load_airline()
    f = BoxCoxBiasAdjustedForecaster(NaiveForecaster(), lambda_fixed=0.5)
    f.fit(y, fh=[1])
    assert np.isclose(f.boxcox_transformer_.lambda_, 0.5)
