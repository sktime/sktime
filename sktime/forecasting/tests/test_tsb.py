"""Tests for TSB estimator."""

import numpy as np
import pandas as pd
import pytest

from sktime.datasets import load_PBS_dataset
from sktime.forecasting.tsb import TSB
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(TSB),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "alpha, beta, fh, expected_forecast",
    [
        (0.1, 0.1, np.array([10]), 0.015194),
        (0.4, 0.05, np.array([5]), 0.106244),
        (0.5, 0.05, np.array([15]), 0.116315),
    ],
)
def test_TSB(alpha, beta, fh, expected_forecast):
    """
    Test TSB forecaster.
    """
    y = load_PBS_dataset()
    forecaster = TSB(alpha, beta)
    forecaster.fit(y)
    y_pred = forecaster.predict(fh=fh)
    np.testing.assert_almost_equal(
        y_pred, np.full(len(fh), expected_forecast), decimal=5
    )


# Golden reference series, short enough that the whole recursion can be
# written out. With alpha = 0.5, beta = 0.2 and the first non-zero at index 1,
# initialization is d = 3, p = 0.5, and the recursion runs:
#
#   t=0 (y=0): d = 3                 p = .8*.5       = 0.4
#   t=1 (y=3): d = .5*3 + .5*3 = 3   p = .2 + .8*.4  = 0.52
#   t=2 (y=0): d = 3                 p = .8*.52      = 0.416
#   t=3 (y=0): d = 3                 p = .8*.416     = 0.3328
#   t=4 (y=5): d = .5*5 + .5*3 = 4   p = .2 + .8*.3328 = 0.46624
#
# so the forecast is f = d * p = 4 * 0.46624 = 1.86496. The estimator
# accumulates these in a different order, so agreement is to floating point
# tolerance rather than bit-exact.
GOLDEN_SERIES = [0, 3, 0, 0, 5]
GOLDEN_ALPHA = 0.5
GOLDEN_BETA = 0.2
GOLDEN_D = 4.0
GOLDEN_P = 0.46624
GOLDEN_FORECAST = GOLDEN_D * GOLDEN_P  # 1.86496


def _golden_y():
    idx = pd.date_range("2020-01-01", periods=len(GOLDEN_SERIES), freq="D")
    return pd.Series(GOLDEN_SERIES, index=idx, dtype=float)


@pytest.mark.skipif(
    not run_test_for_class(TSB),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_tsb_golden_output():
    """Test a full fit against a hand-computed forecast.

    See GOLDEN_SERIES above for the full arithmetic.
    """
    y = _golden_y()
    forecaster = TSB(GOLDEN_ALPHA, GOLDEN_BETA).fit(y)
    y_pred = forecaster.predict(fh=[1, 2, 3]).to_numpy().ravel()

    np.testing.assert_allclose(y_pred, np.full(3, GOLDEN_FORECAST), rtol=1e-12)


@pytest.mark.skipif(
    not run_test_for_class(TSB),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("split", [1, 2, 3, 4])
def test_tsb_golden_output_after_update(split):
    """Test that fit-then-update reaches the same hand-computed forecast.

    Every split point is exercised, so the carried state is checked both
    mid-zero-run and immediately after a demand period.
    """
    y = _golden_y()
    forecaster = TSB(GOLDEN_ALPHA, GOLDEN_BETA).fit(y.iloc[:split])
    forecaster.update(y.iloc[split:])
    y_pred = forecaster.predict(fh=[1]).to_numpy().ravel()

    np.testing.assert_allclose(y_pred, np.full(1, GOLDEN_FORECAST), rtol=1e-12)


@pytest.mark.skipif(
    not run_test_for_class(TSB),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_tsb_forecast_decays_when_update_is_all_zeros():
    """Test that updating with only zeros decays the forecast.

    This is what separates TSB from Croston. Croston smooths only on demand
    periods, so an all-zero update leaves its forecast unchanged. TSB decays
    the demand probability every period, so the same update must reduce the
    forecast geometrically by ``(1 - beta)`` per zero, which is how TSB
    represents obsolescence risk.
    """
    y = _golden_y()
    n_zeros = 3
    zeros_idx = pd.date_range("2020-01-06", periods=n_zeros, freq="D")
    zeros = pd.Series(np.zeros(n_zeros), index=zeros_idx)

    forecaster = TSB(GOLDEN_ALPHA, GOLDEN_BETA).fit(y)
    before = forecaster.predict(fh=[1]).to_numpy().ravel()
    forecaster.update(zeros)
    after = forecaster.predict(fh=[1]).to_numpy().ravel()

    # d is untouched by zero periods; p decays by (1 - beta) each period
    expected = GOLDEN_D * GOLDEN_P * (1 - GOLDEN_BETA) ** n_zeros
    np.testing.assert_allclose(after, np.full(1, expected), rtol=1e-12)
    assert after[0] < before[0]


@pytest.mark.skipif(
    not run_test_for_class(TSB),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("alpha, beta", [(0.1, 0.1), (0.4, 0.05), (0.5, 0.2)])
@pytest.mark.parametrize("n_update", [1, 5, 20])
def test_tsb_update_equals_full_fit(alpha, beta, n_update):
    """Test that fit-then-update matches a fit on the concatenated series.

    The TSB recursion is deterministic given ``(d, p)``, so updating with the
    tail of a series must reproduce a fit on the whole series exactly.
    """
    y = load_PBS_dataset()
    split = len(y) - n_update
    fh = [1, 2, 3]

    incremental = TSB(alpha, beta).fit(y.iloc[:split])
    incremental.update(y.iloc[split:])

    full = TSB(alpha, beta).fit(y)

    np.testing.assert_allclose(
        incremental.predict(fh=fh), full.predict(fh=fh), rtol=1e-12
    )


@pytest.mark.skipif(
    not run_test_for_class(TSB),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_tsb_update_long_zero_run():
    """Test updating across a long run of zeros.

    TSB decays the demand probability on every zero period, which is what
    distinguishes it from Croston. A long trailing zero run is therefore the
    case most sensitive to an incorrectly carried probability.
    """
    idx = pd.date_range("2020-01-01", periods=40, freq="D")
    values = [3.0, 0.0, 0.0, 5.0] + [0.0] * 30 + [2.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    y = pd.Series(values, index=idx)
    fh = [1, 2]

    incremental = TSB(0.4, 0.05).fit(y.iloc[:10])
    incremental.update(y.iloc[10:])

    full = TSB(0.4, 0.05).fit(y)

    np.testing.assert_allclose(
        incremental.predict(fh=fh), full.predict(fh=fh), rtol=1e-12
    )


@pytest.mark.skipif(
    not run_test_for_class(TSB),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_tsb_update_no_demand_in_fit():
    """Test the edge case where the fit window contains no non-zero demand.

    ``_fit`` initializes the demand size from the first non-zero observation,
    so an all-zero fit window initializes it to zero, which then propagates.
    Updating with data that does contain demand must still agree with a fit on
    the concatenated series.
    """
    idx = pd.date_range("2020-01-01", periods=20, freq="D")
    y = pd.Series([0.0] * 12 + [4.0, 0.0, 0.0, 7.0, 0.0, 3.0, 0.0, 5.0], index=idx)
    fh = [1, 2]

    incremental = TSB(0.4, 0.05).fit(y.iloc[:12])
    incremental.update(y.iloc[12:])

    full = TSB(0.4, 0.05).fit(y)

    np.testing.assert_allclose(
        incremental.predict(fh=fh), full.predict(fh=fh), rtol=1e-12
    )


@pytest.mark.skipif(
    not run_test_for_class(TSB),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_tsb_update_params_false_does_not_change_forecast():
    """Test that update(update_params=False) leaves fitted parameters alone."""
    y = load_PBS_dataset()
    fh = [1, 2, 3]

    forecaster = TSB(0.4, 0.05).fit(y.iloc[:-10])
    before = forecaster.predict(fh=fh).to_numpy()
    forecaster.update(y.iloc[-10:], update_params=False)
    after = forecaster.predict(fh=fh).to_numpy()

    np.testing.assert_allclose(before, after, rtol=1e-12)
