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
