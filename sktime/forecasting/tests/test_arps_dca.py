# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for Arps Decline Curve Analysis (DCA) forecasters."""

__author__ = ["OfficialAbhinavSingh"]

import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.arps_dca import ArpsExponential, ArpsHarmonic, ArpsHyperbolic
from sktime.tests.test_switch import run_test_for_class

ARPS_CLASSES = [ArpsExponential, ArpsHyperbolic, ArpsHarmonic]


def _make_decline_series(n=20, noise=3.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    q = 1000 * np.exp(-0.1 * t) + rng.normal(0, noise, size=n)
    return pd.Series(q, index=t)


@pytest.mark.skipif(
    not run_test_for_class(ARPS_CLASSES),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("cls", ARPS_CLASSES)
def test_cumulative_output_anchored_at_cutoff_not_first_fh(cls):
    """Cumulative forecasts must not depend on which fh values are requested.

    Regression test: ``_apply_forecast_transforms`` used to rebase the
    cumulative curve to the model's value at the *first requested* horizon
    point instead of the cutoff, so predicting a later horizon on its own
    (e.g. ``fh=[3]``) silently changed the whole curve's baseline. Both
    calls below must agree, since the underlying trained model and cutoff
    are identical.
    """
    y = _make_decline_series()
    forecaster = cls(output="cumulative", base_np=5000.0, random_state=0)
    forecaster.fit(y)

    joint = forecaster.predict(fh=[1, 2, 3, 5, 8])

    late_only = cls(output="cumulative", base_np=5000.0, random_state=0)
    late_only.fit(y)
    single = late_only.predict(fh=[8])

    np.testing.assert_allclose(
        np.asarray(joint).ravel()[-1],
        np.asarray(single).ravel()[-1],
        rtol=1e-8,
    )


@pytest.mark.skipif(
    not run_test_for_class(ARPS_CLASSES),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("cls", ARPS_CLASSES)
def test_cumulative_base_np_offset_applied_at_cutoff(cls):
    """``base_np`` should offset the whole cumulative curve, not just fh[0].

    Before the fix, the first predicted cumulative value was always exactly
    ``base_np`` (since the rebase point coincided with the first fh value),
    regardless of how far that first fh point was from the cutoff.
    """
    y = _make_decline_series()
    near = cls(output="cumulative", base_np=5000.0, random_state=0)
    near.fit(y)
    y_near = near.predict(fh=[1]).values.ravel()[0]

    far = cls(output="cumulative", base_np=5000.0, random_state=0)
    far.fit(y)
    y_far = far.predict(fh=[8]).values.ravel()[0]

    # further-out forecasts should have accumulated more production than
    # the immediate next-step forecast -- both must exceed base_np, and the
    # near-term one must NOT collapse to exactly base_np.
    assert y_near > 5000.0
    assert y_far > y_near


@pytest.mark.skipif(
    not run_test_for_class(ARPS_CLASSES),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("cls", ARPS_CLASSES)
def test_cumulative_predict_interval_nonzero_width(cls):
    """Prediction intervals for cumulative output must not degenerate to zero width.

    Before the fix, requesting a single, non-first horizon point (e.g.
    ``fh=[3]``) rebased that point's Monte Carlo samples to their own
    baseline, collapsing lower == upper.
    """
    y = _make_decline_series()
    forecaster = cls(output="cumulative", base_np=0.0, random_state=0)
    forecaster.fit(y)

    pred_int = forecaster.predict_interval(fh=[3], coverage=[0.9])
    lower = pred_int.xs("lower", level=-1, axis=1).values.ravel()
    upper = pred_int.xs("upper", level=-1, axis=1).values.ravel()

    assert np.all(upper - lower > 0)


@pytest.mark.skipif(
    not run_test_for_class(ARPS_CLASSES),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("cls", ARPS_CLASSES)
def test_rate_output_unaffected_by_cumulative_fix(cls):
    """``output="rate"`` forecasts must be identical regardless of requested fh."""
    y = _make_decline_series()
    forecaster = cls(output="rate", random_state=0)
    forecaster.fit(y)

    joint = forecaster.predict(fh=[1, 2, 3])
    single = cls(output="rate", random_state=0).fit(y).predict(fh=[3])

    np.testing.assert_allclose(
        np.asarray(joint).ravel()[-1],
        np.asarray(single).ravel()[-1],
        rtol=1e-8,
    )
