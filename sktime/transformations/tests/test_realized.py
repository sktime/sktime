"""Tests for realized variance, bipower variation and jump variation."""

__author__ = ["Mattbusel"]

import numpy as np
import pandas as pd
import pytest

from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.realized import (
    BipowerVariation,
    JumpVariation,
    RealizedVariance,
)

ALL = [RealizedVariance, BipowerVariation, JumpVariation]
skip_reason = "run test only if softdeps are present and incrementally (if requested)"


def _intraday(days=3, bars=78, columns=("a", "b"), seed=0):
    """Five-minute returns over ``days`` US equity sessions of ``bars`` bars."""
    index = pd.DatetimeIndex([])
    for day in pd.bdate_range("2024-03-04", periods=days):
        start = day + pd.Timedelta("9h35min")
        index = index.append(pd.date_range(start, periods=bars, freq="5min"))
    rng = np.random.default_rng(seed)
    values = rng.normal(0, 1e-3, (len(index), len(columns)))
    return pd.DataFrame(values, index=index, columns=list(columns))


@pytest.mark.skipif(not run_test_for_class(ALL), reason=skip_reason)
def test_hand_computed_values():
    """RV, BV and JV of one period against values computed by hand."""
    r = pd.Series([1.0, -2.0, 3.0])
    rv = RealizedVariance(freq=3).fit_transform(r)
    bv = BipowerVariation(freq=3).fit_transform(r)
    jv = JumpVariation(freq=3).fit_transform(r)
    # RV = 1 + 4 + 9, BV = (pi / 2) * (|1||-2| + |-2||3|) = 4 pi
    assert rv.iloc[0] == pytest.approx(14.0)
    assert bv.iloc[0] == pytest.approx(4 * np.pi)
    assert jv.iloc[0] == pytest.approx(14.0 - 4 * np.pi)


@pytest.mark.skipif(not run_test_for_class(JumpVariation), reason=skip_reason)
def test_jump_variation_is_never_negative():
    """BV > RV for constant returns, JV is truncated at zero."""
    r = pd.Series([1.0, 1.0, 1.0, 1.0])  # BV = 3 pi / 2 > RV = 4
    assert JumpVariation(freq=4).fit_transform(r).iloc[0] == 0.0
    assert JumpVariation(freq=4, relative=True).fit_transform(r).iloc[0] == 0.0


@pytest.mark.skipif(not run_test_for_class(ALL), reason=skip_reason)
def test_int_freq_periods_and_labels():
    """Integer freq: one row per complete period, labelled by its last bar."""
    X = _intraday()
    X = pd.concat([X, X.iloc[:5]])  # 5 trailing bars, an incomplete period
    X.index = pd.RangeIndex(len(X))
    rv = RealizedVariance(freq=78).fit_transform(X)
    bv = BipowerVariation(freq=78).fit_transform(X)
    jv = JumpVariation(freq=78).fit_transform(X)
    assert list(rv.index) == [77, 155, 233]
    assert list(rv.columns) == ["a", "b"]
    for k in range(3):
        day = X.iloc[78 * k : 78 * (k + 1)].to_numpy()
        a = np.abs(day)
        bv_day = (a[1:] * a[:-1]).sum(axis=0) * np.pi / 2
        np.testing.assert_allclose(rv.iloc[k], (day**2).sum(axis=0))
        np.testing.assert_allclose(bv.iloc[k], bv_day)
        np.testing.assert_allclose(
            jv.iloc[k], np.maximum((day**2).sum(axis=0) - bv_day, 0)
        )


@pytest.mark.skipif(not run_test_for_class(RealizedVariance), reason=skip_reason)
def test_calendar_freq_groups_by_day():
    """String freq groups a DatetimeIndex by calendar day, like int freq."""
    X = _intraday(days=4)
    rv_day = RealizedVariance(freq="D").fit_transform(X)
    rv_int = RealizedVariance(freq=78).fit_transform(X)
    np.testing.assert_allclose(rv_day.to_numpy(), rv_int.to_numpy())
    assert list(rv_day.index) == list(pd.bdate_range("2024-03-04", periods=4))

    X_tz = X.tz_localize("America/New_York")
    rv_tz = RealizedVariance(freq="D").fit_transform(X_tz)
    np.testing.assert_allclose(rv_tz.to_numpy(), rv_int.to_numpy())

    X_period = X.copy()
    X_period.index = X.index.to_period("5min")
    rv_period = RealizedVariance(freq="D").fit_transform(X_period)
    np.testing.assert_allclose(rv_period.to_numpy(), rv_int.to_numpy())
    assert isinstance(rv_period.index, pd.PeriodIndex)


@pytest.mark.skipif(not run_test_for_class(RealizedVariance), reason=skip_reason)
def test_calendar_freq_needs_time_index():
    """String freq on an integer index raises an informative error."""
    with pytest.raises(TypeError, match="DatetimeIndex or PeriodIndex"):
        RealizedVariance(freq="D").fit_transform(pd.Series(np.ones(10)))
    with pytest.raises(ValueError, match="positive integer"):
        RealizedVariance(freq=0).fit_transform(pd.Series(np.ones(10)))


@pytest.mark.skipif(not run_test_for_class(ALL), reason=skip_reason)
def test_annualize_and_relative_jump():
    """annualize scales linearly; the relative jump measure is not scaled."""
    X = _intraday(days=2)
    a = RealizedVariance(freq=78, annualize=True).fit_transform(X)
    b = RealizedVariance(freq=78).fit_transform(X)
    np.testing.assert_allclose(a, 252 * b)

    rel = JumpVariation(freq=78, relative=True).fit_transform(X)
    rel_ann = JumpVariation(freq=78, relative=True, annualize=True).fit_transform(X)
    np.testing.assert_allclose(rel, rel_ann)
    assert ((rel >= 0) & (rel <= 1)).all().all()


@pytest.mark.skipif(not run_test_for_class(ALL), reason=skip_reason)
def test_bns_consistency_without_jumps():
    """Without jumps, RV and BV both estimate integrated variance.

    With r_i ~ N(0, s2 / n), E[RV] = s2 and E[BV] = s2 (n - 1) / n,
    see Barndorff-Nielsen and Shephard (2004).
    """
    rng = np.random.default_rng(7)
    n, days, s2 = 390, 4000, 1.0
    r = pd.Series(rng.normal(0.0, np.sqrt(s2 / n), size=n * days))
    rv = RealizedVariance(freq=n).fit_transform(r)
    bv = BipowerVariation(freq=n).fit_transform(r)
    # sd of the 4000-day mean is about 0.0011 for RV, 0.0013 for BV
    assert rv.mean() == pytest.approx(s2, rel=5e-3)
    assert bv.mean() == pytest.approx(s2 * (n - 1) / n, rel=5e-3)


@pytest.mark.skipif(not run_test_for_class(ALL), reason=skip_reason)
def test_bns_jump_robustness():
    """A jump of size J adds about J^2 to RV and leaves BV almost unchanged.

    Jump variation then recovers J^2, see Barndorff-Nielsen and Shephard (2006).
    """
    rng = np.random.default_rng(11)
    n, days, s2, J = 390, 500, 1e-4, 0.01
    base = rng.normal(0.0, np.sqrt(s2 / n), size=(days, n))
    jumped = base.copy()
    jumped[:, n // 2] += J
    base, jumped = pd.Series(base.ravel()), pd.Series(jumped.ravel())

    rv0 = RealizedVariance(freq=n).fit_transform(base)
    bv0 = BipowerVariation(freq=n).fit_transform(base)
    rv1 = RealizedVariance(freq=n).fit_transform(jumped)
    bv1 = BipowerVariation(freq=n).fit_transform(jumped)
    jv1 = JumpVariation(freq=n).fit_transform(jumped)

    assert (rv1 - rv0).mean() == pytest.approx(J**2, rel=0.02)
    # BV moves only through the two cross products touching the jump,
    # E[shift] = pi * mu1 * sb * (J - mu1 * sb) with sb = sqrt(s2 / n)
    mu1, sb = np.sqrt(2 / np.pi), np.sqrt(s2 / n)
    shift = (bv1 - bv0).mean()
    assert shift == pytest.approx(np.pi * mu1 * sb * (J - mu1 * sb), rel=0.1)
    assert shift < 0.15 * J**2
    assert jv1.mean() == pytest.approx(J**2 - shift, rel=0.1)
