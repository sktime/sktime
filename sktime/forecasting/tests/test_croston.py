"""Tests for Croston estimator."""

import numpy as np
import pandas as pd
import pytest

from sktime.datasets import load_PBS_dataset
from sktime.forecasting.croston import Croston
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(Croston),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "smoothing, fh, r_forecast",
    [
        (0.1, np.array([10]), 0.8688921),
        (0.5, np.array([5]), 0.6754646),
        (0.05, np.array([15]), 1.405808),
    ],
)
def test_Croston_against_r_implementation(smoothing, fh, r_forecast):
    """Test Croston estimator against the R package implementing the same algorithm.

    Testing forecasted values estimated by the R package of the Croston's method
    against the Croston method in sktime.
    R code to generate the hardcoded value for fh=10:
    ('PBS_dataset.csv' contains the data from 'load_PBS_dataset()'):

        PBS_file <- read.csv(file = '/content/PBS_dataset.csv')[,c('Scripts')]
        y <- ts(PBS_file)
        demand=ts(y)
        forecast <- croston(y,h = 10)
    Output:
        0.8688921
    """  # noqa: E501
    y = load_PBS_dataset()
    forecaster = Croston(smoothing)
    forecaster.fit(y)
    y_pred = forecaster.predict(fh=fh)
    np.testing.assert_almost_equal(y_pred, np.full(len(fh), r_forecast), decimal=5)


# Golden reference series, small enough that the whole recursion can be
# computed by hand. With smoothing = 0.1 and the first non-zero at index 2,
# initialization is q = 2, a = 3, and the three demand periods give:
#
#   t=3  (y=7, p=1): q = .1*7 + .9*2    = 2.5    a = .1*1 + .9*3    = 2.8
#   t=7  (y=5, p=4): q = .1*5 + .9*2.5  = 2.75   a = .1*4 + .9*2.8  = 2.92
#   t=12 (y=3, p=5): q = .1*3 + .9*2.75 = 2.775  a = .1*5 + .9*2.92 = 3.128
#
# so the forecast is f = q / a = 2.775 / 3.128.
GOLDEN_SERIES = [0, 0, 2, 7, 0, 0, 0, 5, 0, 0, 0, 0, 3]
GOLDEN_SMOOTHING = 0.1
GOLDEN_Q = 2.775
GOLDEN_A = 3.128
GOLDEN_FORECAST = GOLDEN_Q / GOLDEN_A  # 0.8871483375959078


def _golden_y():
    idx = pd.date_range("2020-01-01", periods=len(GOLDEN_SERIES), freq="D")
    return pd.Series(GOLDEN_SERIES, index=idx, dtype=float)


@pytest.mark.skipif(
    not run_test_for_class(Croston),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_croston_golden_output():
    """Test a full fit against a hand-computed forecast.

    See GOLDEN_SERIES above for the full arithmetic.
    """
    y = _golden_y()
    forecaster = Croston(GOLDEN_SMOOTHING).fit(y)
    y_pred = forecaster.predict(fh=[1, 2, 3]).to_numpy().ravel()

    np.testing.assert_allclose(y_pred, np.full(3, GOLDEN_FORECAST), rtol=1e-12)


@pytest.mark.skipif(
    not run_test_for_class(Croston),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("split", [3, 6, 9])
def test_croston_golden_output_after_update(split):
    """Test that fit-then-update reaches the same hand-computed forecast.

    The split points straddle both demand periods and zero runs, so the carried
    state is exercised in both regimes.
    """
    y = _golden_y()
    forecaster = Croston(GOLDEN_SMOOTHING).fit(y.iloc[:split])
    forecaster.update(y.iloc[split:])
    y_pred = forecaster.predict(fh=[1]).to_numpy().ravel()

    np.testing.assert_allclose(y_pred, np.full(1, GOLDEN_FORECAST), rtol=1e-12)


@pytest.mark.skipif(
    not run_test_for_class(Croston),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_croston_forecast_unchanged_when_update_is_all_zeros():
    """Test that updating with only zeros leaves the forecast unchanged.

    Croston smooths the demand size and interval only on non-zero periods, so
    an update consisting entirely of zeros must not move the forecast. The
    periods-since-last-demand counter does advance, which is what makes the
    *next* non-zero observation smooth against the correct interval.
    """
    y = _golden_y()
    zeros_idx = pd.date_range("2020-01-14", periods=5, freq="D")
    zeros = pd.Series(np.zeros(5), index=zeros_idx)

    forecaster = Croston(GOLDEN_SMOOTHING).fit(y)
    before = forecaster.predict(fh=[1]).to_numpy().ravel()
    forecaster.update(zeros)
    after = forecaster.predict(fh=[1]).to_numpy().ravel()

    np.testing.assert_allclose(before, after, rtol=1e-12)
    np.testing.assert_allclose(after, np.full(1, GOLDEN_FORECAST), rtol=1e-12)

    # the counter advanced even though the forecast did not, so a subsequent
    # demand is smoothed against the longer interval
    demand_idx = pd.date_range("2020-01-19", periods=1, freq="D")
    forecaster.update(pd.Series([4.0], index=demand_idx))
    expected_a = 0.1 * 6 + 0.9 * GOLDEN_A
    expected_q = 0.1 * 4 + 0.9 * GOLDEN_Q
    np.testing.assert_allclose(
        forecaster.predict(fh=[1]).to_numpy().ravel(),
        np.full(1, expected_q / expected_a),
        rtol=1e-12,
    )


@pytest.mark.skipif(
    not run_test_for_class(Croston),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("smoothing", [0.1, 0.5])
@pytest.mark.parametrize("n_update", [1, 5, 20])
def test_croston_update_equals_full_fit(smoothing, n_update):
    """Test that fit-then-update matches a fit on the concatenated series.

    Croston's recursion is deterministic given its state, so updating with the
    tail of a series must reproduce a fit on the whole series exactly.
    """
    y = load_PBS_dataset()
    split = len(y) - n_update
    fh = [1, 2, 3]

    incremental = Croston(smoothing).fit(y.iloc[:split])
    incremental.update(y.iloc[split:])

    full = Croston(smoothing).fit(y)

    np.testing.assert_allclose(
        incremental.predict(fh=fh), full.predict(fh=fh), rtol=1e-12
    )


@pytest.mark.skipif(
    not run_test_for_class(Croston),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_croston_update_no_demand_in_fit():
    """Test the edge case where the fit window contains no non-zero demand.

    ``_fit`` initializes from the first non-zero observation, so a fit window of
    all zeros initializes degenerately. Updating with data that does contain
    demand must still agree with a fit on the concatenated series.
    """
    idx = pd.date_range("2020-01-01", periods=20, freq="D")
    y = pd.Series([0.0] * 12 + [4.0, 0.0, 0.0, 7.0, 0.0, 3.0, 0.0, 5.0], index=idx)
    fh = [1, 2]

    incremental = Croston(0.1).fit(y.iloc[:12])
    incremental.update(y.iloc[12:])

    full = Croston(0.1).fit(y)

    np.testing.assert_allclose(
        incremental.predict(fh=fh), full.predict(fh=fh), rtol=1e-12
    )


@pytest.mark.skipif(
    not run_test_for_class(Croston),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_croston_update_params_false_does_not_change_forecast():
    """Test that update(update_params=False) leaves fitted parameters alone."""
    y = load_PBS_dataset()
    fh = [1, 2, 3]

    forecaster = Croston(0.1).fit(y.iloc[:-10])
    before = forecaster.predict(fh=fh).to_numpy()
    forecaster.update(y.iloc[-10:], update_params=False)
    after = forecaster.predict(fh=fh).to_numpy()

    np.testing.assert_allclose(before, after, rtol=1e-12)
