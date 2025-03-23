"""Tests for SIMDKalmanSmoother."""

import numpy as np
import pytest

from sktime.datasets import load_airline
from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.panel.simdkalman_smoother import SIMDKalmanSmoother
from sktime.transformations.series.kalman_filter import KalmanFilterTransformerSIMD
from sktime.utils._testing.panel import make_transformer_problem
from numpy.testing import assert_array_almost_equal


@pytest.mark.skipif(
    not run_test_for_class(SIMDKalmanSmoother),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("smooth", [True, False])
@pytest.mark.parametrize(
    "params, n_columns",
    [
        (
            dict(
                state_transition=np.array([[1, 1], [0, 1]]),
                process_noise=np.diag([1e-6, 0.01]),
                measurement_function=np.array([[1, 0]]),
                measurement_noise=10,
                # NOTE: currently fails without these definitions
                # TODO: document this in simdkalman
                initial_state=np.array([0, -1]),
                initial_state_covariance=np.diag([1, 1]),
                hidden=False,
            ),
            1,
        )
    ],
)
def test_basic_smoothing(smooth, params, n_columns):
    """Check that basic smoothing works and gives the same results as the Series version"""
    X = make_transformer_problem(n_columns=n_columns, n_timepoints=15)

    all_params = dict(params, denoising=smooth)
    panel_smoother = SIMDKalmanSmoother(**all_params)
    panel_Xt = panel_smoother.fit_transform(X)

    state_dim = all_params["state_transition"].shape[0]
    series_results = []
    for i in range(X.shape[0]):
        as_series = X[i, ...].transpose()
        series_smoother = KalmanFilterTransformerSIMD(state_dim=state_dim, **all_params)
        series_out = series_smoother.fit_transform(as_series).transpose()
        series_results.append(series_out)

    series_Xt = np.stack(series_results, axis=0)
    assert_array_almost_equal(panel_Xt, series_Xt)


@pytest.mark.skipif(
    not run_test_for_class(SIMDKalmanSmoother),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_series_smoothing():
    """Check that names are preserved in Series smoothing and that the output is
    exactly the same as in KalmanFilterTransformerSIMD"""
    y = load_airline()

    kf_params = dict(
        state_dim=2,
        state_transition=np.array([[1, 1], [0, 1]]),
        process_noise=np.diag([1e-6, 0.01]) ** 2,
        measurement_function=np.array([[1, 0]]),
        measurement_noise=50.0**2,
        initial_state=np.array([0, 0]),
        initial_state_covariance=np.eye(2) * 100**2,
        hidden=False,
    )

    kf_panel = SIMDKalmanSmoother(**kf_params)

    y_trans1 = kf_panel.fit_transform(y)
    assert y_trans1.name == y.name
    assert np.array_equal(y_trans1.index, y.index)

    kf_series = KalmanFilterTransformerSIMD(**kf_params)
    y_trans2 = kf_series.fit_transform(y)

    assert np.array_equal(y_trans1.to_numpy(), y_trans2.to_numpy())
    assert np.array_equal(y_trans1.index, y_trans2.index)
