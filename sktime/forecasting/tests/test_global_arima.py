# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for GlobalARIMA."""

__author__ = ["renuprasads"]

import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.arima._global import (
    _BOUND,
    _MAX_LOG_ERROR_GROWTH,
    GlobalARIMA,
)
from sktime.tests.test_switch import run_test_for_class
from sktime.utils.dependencies import _check_soft_dependencies

# Every warning is an error in this module. Without this, a regression that
# only shows up as a warning would pass unnoticed. Tests that expect a warning
# capture it and assert on the message.
#
# Two pandas deprecations are exempt. Both are raised by framework code, affect
# every forecaster, and have nothing to do with this estimator:
#
# * _format_moving_cutoff_predictions in the forecaster base class calls
#   pd.concat without passing sort, deprecated in pandas 3. NaiveForecaster
#   fails 40 of the same contract checks under warnings as errors.
# * _index_equally_spaced in sktime.datatypes calls PeriodIndex.is_full,
#   deprecated in pandas 3.1. NaiveForecaster fails 10 contract checks.
pytestmark = [
    pytest.mark.skipif(
        not run_test_for_class(GlobalARIMA),
        reason="run test only if softdeps are present and incrementally (if requested)",
    ),
    pytest.mark.skipif(
        not _check_soft_dependencies("statsmodels", severity="none"),
        reason="skip test if statsmodels is not available",
    ),
    pytest.mark.filterwarnings("error"),
    pytest.mark.filterwarnings("ignore:Sorting by default when concatenating"),
    pytest.mark.filterwarnings("ignore:PeriodIndex.is_full is deprecated"),
]


# ----------------------------------------------------------------------
# data helpers
# ----------------------------------------------------------------------


def _arma(n, ar=0.6, ma=0.3, seed=0, burn=100, scale=1.0, level=0.0):
    """One ARMA(1,1) series, burnt in."""
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n + burn)
    x = np.zeros(n + burn)
    for t in range(1, n + burn):
        x[t] = ar * x[t - 1] + e[t] + ma * e[t - 1]
    return level + scale * x[burn:]


def _series(values, start=0):
    """A pd.Series on a range index."""
    values = np.asarray(values, dtype=float)
    return pd.Series(values, index=pd.RangeIndex(start, start + len(values)))


def _panel(series_list, name="c0"):
    """A panel DataFrame from a list of arrays."""
    frames = []
    for i, values in enumerate(series_list):
        index = pd.MultiIndex.from_product(
            [[f"s{i}"], pd.RangeIndex(len(values))], names=["instance", "time"]
        )
        frames.append(
            pd.DataFrame({name: np.asarray(values, dtype=float)}, index=index)
        )
    return pd.concat(frames)


def _shared_panel(n_series=4, n=40, ar=0.6, ma=0.3, scales=None, levels=None, seed=0):
    """A panel of series that share dynamics but not scale or level."""
    scales = [1.0] * n_series if scales is None else scales
    levels = [0.0] * n_series if levels is None else levels
    return _panel(
        [
            _arma(n, ar=ar, ma=ma, seed=seed + i, scale=scales[i], level=levels[i])
            for i in range(n_series)
        ]
    )


def _integrated(order_d, n=60, seed=0):
    """A series integrated ``order_d`` times."""
    rng = np.random.default_rng(seed)
    values = rng.standard_normal(n)
    for _ in range(order_d):
        values = np.cumsum(values)
    return _series(values)


# ----------------------------------------------------------------------
# single series behaviour: fit without pretrain is maximum likelihood ARIMA
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "order", [(1, 0, 0), (0, 0, 1), (1, 0, 1), (2, 0, 1), (1, 1, 1)]
)
def test_fit_without_pretrain_is_at_least_as_good_as_statsmodels(order):
    """Without pretraining, fit must reach the statsmodels likelihood or better."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    p, d, q = order
    values = _arma(120, seed=11)
    y = _series(values)

    forecaster = GlobalARIMA(order=order).fit(y)

    z = values - values.mean() if d == 0 else np.diff(values, n=d)
    mod = SARIMAX(z, order=(p, 0, q), trend="n", concentrate_scale=True)
    reference = mod.fit(disp=False, maxiter=500)

    mine = mod.loglike(mod.untransform_params(forecaster._params), transformed=False)
    assert mine >= reference.llf - 1e-6


@pytest.mark.parametrize("d", [1, 2])
def test_forecasts_are_reintegrated_correctly(d):
    """Forecasts for d >= 1 must undo the differencing against the observed series."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    values = _arma(100, seed=5)
    values = np.cumsum(values) if d == 1 else np.cumsum(np.cumsum(values))
    y = _series(values)

    forecaster = GlobalARIMA(order=(1, d, 1)).fit(y)
    y_pred = forecaster.predict(fh=[1, 2, 3]).to_numpy()

    z = np.diff(values, n=d)
    mod = SARIMAX(z, order=(1, 0, 1), trend="n", concentrate_scale=True)
    z_pred = np.asarray(mod.filter(forecaster._params).forecast(3), dtype=float)
    expected = z_pred
    for level in range(d, 0, -1):
        expected = np.diff(values, n=level - 1)[-1] + np.cumsum(expected)

    np.testing.assert_allclose(y_pred, expected, rtol=1e-9, atol=1e-9)


def test_differencing_removes_dependence_on_the_level():
    """For d >= 1 the estimate must not move when the series is shifted.

    statsmodels' default initialisation for integrated models lets the level
    enter the likelihood, which moves the estimate by order 0.1. What is left
    here is optimiser noise in the last few digits.
    """
    values = np.cumsum(_arma(80, seed=2))
    low = GlobalARIMA(order=(1, 1, 1)).fit(_series(values))
    high = GlobalARIMA(order=(1, 1, 1)).fit(_series(values + 1e7))
    np.testing.assert_allclose(low._params, high._params, rtol=0, atol=1e-5)


def test_mean_is_estimated_per_series_when_d_is_zero():
    """With d = 0 the forecast reverts toward that series' own mean."""
    values = _arma(200, ar=0.5, ma=0.0, seed=4, level=37.0)
    forecaster = GlobalARIMA(order=(1, 0, 0)).fit(_series(values))
    assert forecaster._mean == pytest.approx(values.mean())
    far = forecaster.predict(fh=[400]).to_numpy()[0]
    assert far == pytest.approx(values.mean(), abs=1e-3)


# ----------------------------------------------------------------------
# pretraining
# ----------------------------------------------------------------------


def test_pretrain_sets_the_documented_attributes():
    """pretrain must set exactly the attributes named in the tag."""
    forecaster = GlobalARIMA(order=(1, 0, 1))
    forecaster.pretrain(_shared_panel(seed=20))

    assert forecaster.global_ar_params_.shape == (1,)
    assert forecaster.global_ma_params_.shape == (1,)
    assert forecaster.n_pretrain_instances_ == 4
    assert np.isfinite(forecaster.pretrain_loglike_)
    assert forecaster.pretrain_order_ == (1, 0, 1)

    reported = forecaster.get_pretrained_params()
    for name in forecaster.get_tag("pretrain:fitted_params"):
        assert name in reported


def test_pretrain_then_fit_forecasts_an_unseen_series():
    """A series absent from the pretraining data must still be forecastable."""
    panel = _shared_panel(seed=30)
    forecaster = GlobalARIMA(order=(1, 0, 1)).pretrain(panel)
    shared = np.r_[forecaster.global_ar_params_, forecaster.global_ma_params_]

    unseen = _series(_arma(25, seed=999, level=100.0))
    forecaster.fit(unseen)

    np.testing.assert_allclose(forecaster._params, shared, rtol=0, atol=0)
    assert forecaster._uses_pretrained
    y_pred = forecaster.predict(fh=[1, 2, 3])
    assert len(y_pred) == 3
    assert np.all(np.isfinite(y_pred.to_numpy()))


def test_fit_without_pretrain_does_not_claim_pretrained_state():
    """The pretrained flag must reflect what actually happened."""
    forecaster = GlobalARIMA(order=(1, 0, 0)).fit(_series(_arma(60, seed=6)))
    assert not forecaster._uses_pretrained


@pytest.mark.parametrize(
    "scales,levels",
    [
        ([1e-3, 1.0, 1e4, 1.0], [0.0, 0.0, 0.0, 0.0]),
        ([1.0, 1.0, 1.0, 1.0], [0.0, 1e7, -50.0, 1e3]),
        ([1e-2, 1e2, 1.0, 1e3], [1e6, -1e6, 0.0, 10.0]),
    ],
)
def test_pooled_estimate_is_invariant_to_scale_and_level(scales, levels):
    """Concentrating the variance per series is what makes pooling scale free."""
    plain = GlobalARIMA(order=(1, 0, 1)).pretrain(_shared_panel(seed=40))
    altered = GlobalARIMA(order=(1, 0, 1)).pretrain(
        _shared_panel(seed=40, scales=scales, levels=levels)
    )
    np.testing.assert_allclose(
        np.r_[plain.global_ar_params_, plain.global_ma_params_],
        np.r_[altered.global_ar_params_, altered.global_ma_params_],
        rtol=0,
        atol=1e-5,
    )


def test_series_with_no_variation_does_not_affect_the_pooled_estimate():
    """A noise free trend differences to rounding error and must be ignored.

    Without the tolerance in _transform, the rounding error left by
    differencing a smooth trend is read as data and takes over the estimate.
    """
    clean = [_arma(40, seed=50 + i) for i in range(3)]
    trend = 0.1 * np.arange(40, dtype=float)

    without = GlobalARIMA(order=(1, 1, 0)).pretrain(_panel(clean))
    with_trend = GlobalARIMA(order=(1, 1, 0)).pretrain(_panel(clean + [trend]))

    np.testing.assert_allclose(
        without.global_ar_params_, with_trend.global_ar_params_, rtol=0, atol=1e-10
    )
    assert with_trend.n_pretrain_instances_ == 4


def test_pretrain_raises_when_no_series_carries_variation():
    """There is nothing to estimate from constant series."""
    flat = [np.full(30, 5.0), np.full(30, -2.0)]
    with pytest.raises(ValueError, match="nothing to estimate"):
        GlobalARIMA(order=(1, 0, 0)).pretrain(_panel(flat))


def test_hierarchical_panel_is_pooled_by_leaf_series():
    """Grouping must use every index level but the last."""
    from sktime.utils._testing.hierarchical import _make_hierarchical

    panel = _make_hierarchical(
        hierarchy_levels=(2, 3), min_timepoints=30, max_timepoints=30, random_state=0
    )
    forecaster = GlobalARIMA(order=(1, 0, 0)).pretrain(panel)
    assert forecaster.n_pretrain_instances_ == 6


def test_multicolumn_panel_is_treated_as_one_series_per_column():
    """The framework hands multivariate panels to pretrain even here.

    GlobalARIMA is univariate, but pretrain receives panels with several
    columns, so each column of each instance has to be taken as a series.
    """
    panel = _shared_panel(n_series=2, seed=60)
    panel["c1"] = _arma(len(panel) // 2, seed=61).tolist() * 2

    forecaster = GlobalARIMA(order=(1, 0, 0)).pretrain(panel)
    assert np.isfinite(forecaster.pretrain_loglike_)
    assert forecaster.n_pretrain_instances_ == 2


def test_pooling_recovers_shared_coefficients_better_than_separate_fits():
    """The reason to pool: short series, shared dynamics."""
    true_ar = 0.7
    n_series, n = 20, 25
    series = [_arma(n, ar=true_ar, ma=0.0, seed=300 + i) for i in range(n_series)]

    pooled = GlobalARIMA(order=(1, 0, 0)).pretrain(_panel(series))
    pooled_error = abs(float(pooled.global_ar_params_[0]) - true_ar)

    separate = [
        float(GlobalARIMA(order=(1, 0, 0)).fit(_series(s))._params[0]) for s in series
    ]
    separate_error = float(np.sqrt(np.mean((np.array(separate) - true_ar) ** 2)))

    assert pooled_error < separate_error


# ----------------------------------------------------------------------
# input validation
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "order",
    [
        (1, 0),
        (1, 0, 0, 0),
        "abc",
        None,
        (1.5, 0, 0),
        (-1, 0, 0),
        (1, -1, 0),
        (0, 1, 0),
        (0, 0, 0),
    ],
)
def test_invalid_order_raises(order):
    """A bad order must be rejected before anything is fitted."""
    with pytest.raises(ValueError):
        GlobalARIMA(order=order).fit(_series(_arma(40, seed=7)))


@pytest.mark.parametrize("d", [1, 2, 3])
def test_series_too_short_to_difference_raises(d):
    """A series must have more observations than differences taken."""
    y = _series(np.arange(d + 1, dtype=float) * 1.0)
    with pytest.raises(ValueError, match="cannot be differenced"):
        GlobalARIMA(order=(1, d, 0)).fit(y.iloc[:d])


def test_fit_after_pretrain_still_checks_the_series():
    """The pretrained path must not skip the length check."""
    forecaster = GlobalARIMA(order=(1, 2, 0)).pretrain(_shared_panel(seed=70))
    with pytest.raises(ValueError, match="cannot be differenced"):
        forecaster.fit(_series([1.0, 2.0]))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_values_in_fit_raise(bad):
    """Missing or infinite observations must be rejected.

    The framework rejects nan ahead of _fit, because the estimator does not
    claim capability:missing_values. Infinities reach _fit, so the estimator
    has to catch those itself.
    """
    values = _arma(40, seed=8)
    values[10] = bad
    with pytest.raises(ValueError, match="missing data|infinite or missing"):
        GlobalARIMA(order=(1, 0, 0)).fit(_series(values))


@pytest.mark.parametrize("update_params", [True, False])
@pytest.mark.parametrize("pretrained", [True, False])
def test_non_finite_values_in_update_raise(update_params, pretrained):
    """Validation must run on every update path.

    When update skipped validation for update_params=False or for a pretrained
    forecaster, an infinite observation produced silent NaN forecasts.
    """
    y = _series(_arma(50, seed=9))
    forecaster = GlobalARIMA(order=(1, 0, 0))
    if pretrained:
        forecaster.pretrain(_shared_panel(seed=80))
    forecaster.fit(y)

    bad = pd.Series([np.inf], index=pd.RangeIndex(50, 51))
    with pytest.raises(ValueError, match="infinite or missing"):
        forecaster.update(bad, update_params=update_params)


def test_stale_pretrained_coefficients_are_refused():
    """Changing order after pretraining invalidates the shared coefficients.

    set_params resets the forecaster to its unpretrained state, so that route
    is already safe. Assigning the attribute directly is not, and that is the
    path this guards.
    """
    forecaster = GlobalARIMA(order=(1, 0, 1)).pretrain(_shared_panel(seed=90))
    forecaster.order = (2, 0, 0)
    with pytest.raises(ValueError, match="do not match this model"):
        forecaster.fit(_series(_arma(40, seed=91)))


def test_set_params_after_pretrain_discards_the_shared_coefficients():
    """The framework's own route out of a stale order."""
    forecaster = GlobalARIMA(order=(1, 0, 1)).pretrain(_shared_panel(seed=92))
    forecaster.set_params(order=(2, 0, 0))
    assert not hasattr(forecaster, "global_ar_params_")
    forecaster.fit(_series(_arma(40, seed=93)))
    assert not forecaster._uses_pretrained
    assert len(forecaster._params) == 2


# ----------------------------------------------------------------------
# update
# ----------------------------------------------------------------------


def test_update_with_params_equals_a_fit_on_the_whole_series():
    """Re-estimating on update must agree with fitting the joined series."""
    values = _arma(90, seed=12)
    first, rest = _series(values[:70]), _series(values[70:], start=70)

    updated = GlobalARIMA(order=(1, 0, 1)).fit(first)
    updated.update(rest, update_params=True)
    refit = GlobalARIMA(order=(1, 0, 1)).fit(_series(values))

    np.testing.assert_allclose(updated._params, refit._params, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(
        updated.predict(fh=[1, 2]).to_numpy(),
        refit.predict(fh=[1, 2]).to_numpy(),
        rtol=1e-4,
        atol=1e-6,
    )


def test_update_without_params_leaves_the_fitted_variance_alone():
    """update_params=False must not write to the fitted state.

    sigma2 was being recomputed on every update, so a forecaster asked not to
    update its parameters still changed.
    """
    values = _arma(80, seed=13)
    forecaster = GlobalARIMA(order=(1, 0, 1)).fit(_series(values[:60]))
    before = (forecaster._sigma2, forecaster._mean, np.array(forecaster._params))

    forecaster.update(_series(values[60:], start=60), update_params=False)

    assert forecaster._sigma2 == before[0]
    assert forecaster._mean == before[1]
    np.testing.assert_array_equal(forecaster._params, before[2])


def test_update_keeps_pretrained_coefficients():
    """Shared coefficients belong to the collection, not to one series."""
    values = _arma(60, seed=14)
    forecaster = GlobalARIMA(order=(1, 0, 1)).pretrain(_shared_panel(seed=100))
    shared = np.r_[forecaster.global_ar_params_, forecaster.global_ma_params_]
    forecaster.fit(_series(values[:40]))

    forecaster.update(_series(values[40:], start=40), update_params=True)

    np.testing.assert_allclose(forecaster._params, shared, rtol=0, atol=0)


def test_update_moves_the_forecast_origin():
    """After update, forecasts must start from the new last observation."""
    values = np.cumsum(_arma(70, seed=15))
    forecaster = GlobalARIMA(order=(1, 1, 0)).fit(_series(values[:50]))
    forecaster.update(_series(values[50:], start=50), update_params=False)
    y_pred = forecaster.predict(fh=[1])
    assert y_pred.index[0] == 70
    assert abs(float(y_pred.iloc[0]) - values[-1]) < 10 * float(np.std(np.diff(values)))


# ----------------------------------------------------------------------
# the accuracy region near the unit circle
# ----------------------------------------------------------------------


def _fit_recording_warnings(order, y, maxiter=100):
    """Fit and return the forecaster together with the warnings it raised."""
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        forecaster = GlobalARIMA(order=order, maxiter=maxiter).fit(y)
    return forecaster, [str(w.message) for w in caught]


def test_estimate_at_the_accuracy_limit_warns():
    """Too little differencing drives the estimate to the edge of the region.

    This case reports both that the search stopped short and that the estimate
    sits at the limit, so the messages are checked directly rather than
    through pytest.warns.
    """
    y = _integrated(2, n=60, seed=25)
    _, messages = _fit_recording_warnings((4, 0, 0), y)
    assert any("accurately" in m for m in messages), messages


def test_forecast_at_the_accuracy_limit_stays_near_the_data():
    """The limit is what stops the silent wild forecast.

    Before the limit, an AR(4) fitted to doubly integrated data reached a point
    where the reported likelihood was wrong by more than a hundred units and
    the forecast left the range of the data entirely.
    """
    y = _integrated(2, n=60, seed=25)
    forecaster, messages = _fit_recording_warnings((4, 0, 0), y)
    assert messages
    y_pred = forecaster.predict(fh=[1, 2, 3]).to_numpy()

    last = float(y.iloc[-1])
    spread = float(np.ptp(y.to_numpy()))
    assert np.all(np.abs(y_pred - last) < spread)


@pytest.mark.parametrize("seed", [26, 27, 28, 29, 30, 32, 33])
def test_misspecified_fits_never_forecast_wildly_without_warning(seed):
    """Every estimate is either sane or flagged."""
    import warnings

    y = _integrated(2, n=60, seed=seed)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        forecaster = GlobalARIMA(order=(4, 0, 0), maxiter=100).fit(y)
        y_pred = forecaster.predict(fh=[1, 2, 3]).to_numpy()

    warned = any("GlobalARIMA" in str(w.message) for w in caught)
    spread = float(np.ptp(y.to_numpy()))
    wild = bool(np.any(np.abs(y_pred - float(y.iloc[-1])) > 3 * spread))
    assert warned or not wild


def test_log_error_growth_is_zero_at_the_origin():
    """No loss of accuracy where the model is white noise."""
    forecaster = GlobalARIMA(order=(2, 0, 1))
    mod = forecaster._model(_arma(60, seed=16), 2, 1)
    assert forecaster._log_error_growth(np.zeros(3), 2, mod) == pytest.approx(0.0)


def test_log_error_growth_rises_toward_the_unit_circle():
    """The measure has to be monotone in the thing it measures."""
    forecaster = GlobalARIMA(order=(1, 0, 0))
    mod = forecaster._model(_arma(60, seed=17), 1, 0)
    values = [
        forecaster._log_error_growth(np.array([u]), 1, mod) for u in (0.5, 5.0, 50.0)
    ]
    assert values[0] < values[1] < values[2]


@pytest.mark.parametrize("seed", range(8))
def test_shrink_terminates_and_lands_inside_the_region(seed):
    """The projection used to loop without bound on some parameter vectors."""
    rng = np.random.default_rng(seed)
    forecaster = GlobalARIMA(order=(3, 0, 1))
    mod = forecaster._model(_arma(80, seed=18), 3, 1)

    u = rng.normal(scale=10.0 ** rng.integers(0, 4), size=4)
    projected, distance = forecaster._shrink(u, 3, mod)

    assert np.all(np.isfinite(projected))
    assert distance >= 0.0
    assert forecaster._log_error_growth(projected, 3, mod) <= _MAX_LOG_ERROR_GROWTH


def test_shrink_leaves_points_inside_the_region_untouched():
    """A point that already satisfies the limit must not be moved."""
    forecaster = GlobalARIMA(order=(1, 0, 1))
    mod = forecaster._model(_arma(60, seed=19), 1, 1)
    u = np.array([0.3, -0.2])
    projected, distance = forecaster._shrink(u, 1, mod)
    np.testing.assert_array_equal(projected, u)
    assert distance == 0.0


def test_objective_is_continuous_across_the_limit():
    """A step at the boundary is what made the optimiser stop short.

    The first version of the limit was a hard penalty. L-BFGS-B stopped at the
    wall and reported success, as much as 49 log-likelihood units below the
    maximum.
    """
    forecaster = GlobalARIMA(order=(1, 0, 0))
    models = [forecaster._model(_integrated(2, n=60, seed=25).to_numpy(), 1, 0)]

    direction = np.array([1.0])
    low, high = 0.0, _BOUND
    for _ in range(200):
        mid = 0.5 * (low + high)
        if (
            forecaster._log_error_growth(mid * direction, 1, models[0])
            <= _MAX_LOG_ERROR_GROWTH
        ):
            low = mid
        else:
            high = mid

    inside = forecaster._objective((low * 0.999999) * direction, models, 1)
    at_edge = forecaster._objective(low * direction, models, 1)
    outside = forecaster._objective((high * 1.000001) * direction, models, 1)

    assert abs(at_edge - inside) < 1.0
    assert outside >= at_edge


def test_objective_equals_negative_loglike_inside_the_region():
    """No penalty may leak into the interior."""
    forecaster = GlobalARIMA(order=(1, 0, 1))
    models = [forecaster._model(_arma(70, seed=21), 1, 1)]
    u = np.array([0.4, -0.3])
    expected = -forecaster._total_loglike(u, models)
    assert forecaster._objective(u, models, 1) == pytest.approx(expected)


def test_search_matches_a_derivative_free_reference():
    """The search must find the maximum, not merely stop somewhere."""
    from scipy.optimize import minimize

    forecaster = GlobalARIMA(order=(2, 0, 0))
    series = [_integrated(1, n=40, seed=200 + i).to_numpy() for i in range(4)]
    z_list = [s - s.mean() for s in series]
    models = [forecaster._model(z, 2, 0) for z in z_list]

    found, _ = forecaster._search(models, 2, 0)

    best = found.fun
    for start in ([0.0, 0.0], [1.0, -1.0], [-1.0, 1.0], [5.0, 5.0], [0.5, 0.1]):
        reference = minimize(
            forecaster._objective,
            np.array(start),
            args=(models, 2),
            method="Nelder-Mead",
            options={"maxiter": 4000, "fatol": 1e-12, "xatol": 1e-12},
        )
        best = min(best, reference.fun)

    assert found.fun <= best + 1e-6


@pytest.mark.parametrize("status", [0, 1, 2])
@pytest.mark.parametrize(
    "x,jac,expected",
    [
        (np.array([0.0, 0.0]), np.array([0.0, 0.0]), True),
        (np.array([0.0, 0.0]), np.array([1e-9, -1e-9]), True),
        (np.array([0.0, 0.0]), np.array([1.0, 0.0]), False),
        (np.array([-_BOUND, 0.0]), np.array([5.0, 0.0]), True),
        (np.array([_BOUND, 0.0]), np.array([-5.0, 0.0]), True),
    ],
)
def test_convergence_is_judged_by_the_projected_gradient(status, x, jac, expected):
    """The exit status alone is not a reliable test of optimality.

    L-BFGS-B reports status 2 at points that are maxima, and status 0 at points
    that are not, so the gradient decides. A component held against a bound by
    the gradient is not evidence of non-convergence.
    """

    class _Result:
        pass

    res = _Result()
    res.x = x
    res.jac = jac
    res.status = status
    res.fun = 1.0

    assert GlobalARIMA(order=(1, 0, 0))._converged(res) is expected


def test_convergence_rejects_a_non_finite_gradient():
    """An unusable gradient is not a converged one."""

    class _Result:
        pass

    res = _Result()
    res.x = np.array([0.0])
    res.jac = np.array([np.nan])
    res.status = 0
    res.fun = 1.0

    assert GlobalARIMA(order=(1, 0, 0))._converged(res) is False


# ----------------------------------------------------------------------
# series with no variation
# ----------------------------------------------------------------------


def test_constant_series_forecasts_the_constant():
    """A flat series has no dynamics to estimate, and no excuse to fail."""
    y = _series(np.full(40, 7.0))
    forecaster = GlobalARIMA(order=(1, 0, 0)).fit(y)
    y_pred = forecaster.predict(fh=[1, 2, 3]).to_numpy()
    np.testing.assert_allclose(y_pred, 7.0, rtol=0, atol=1e-9)


def test_perfect_trend_is_extrapolated_linearly():
    """One difference of a straight line leaves nothing to fit."""
    slope = 0.1
    y = _series(slope * np.arange(40, dtype=float))
    forecaster = GlobalARIMA(order=(1, 1, 0)).fit(y)
    y_pred = forecaster.predict(fh=[1, 2, 3]).to_numpy()
    expected = slope * np.arange(40, 43, dtype=float)
    np.testing.assert_allclose(y_pred, expected, rtol=0, atol=1e-8)


# ----------------------------------------------------------------------
# output shape and reported state
# ----------------------------------------------------------------------


def test_predict_returns_a_series_for_series_input():
    """Output type follows input type."""
    y = _series(_arma(50, seed=22))
    y.name = "demand"
    y_pred = GlobalARIMA(order=(1, 0, 0)).fit(y).predict(fh=[1, 2])
    assert isinstance(y_pred, pd.Series)
    assert y_pred.name == "demand"


def test_predict_handles_a_gapped_horizon():
    """A horizon need not be contiguous."""
    y = _series(_arma(60, seed=23))
    forecaster = GlobalARIMA(order=(1, 0, 1)).fit(y)
    gapped = forecaster.predict(fh=[1, 5, 9])
    dense = forecaster.predict(fh=list(range(1, 10)))
    assert list(gapped.index) == [60, 64, 68]
    np.testing.assert_allclose(
        gapped.to_numpy(), dense.to_numpy()[[0, 4, 8]], rtol=0, atol=0
    )


def test_get_fitted_params_reports_the_fitted_state():
    """Fitted parameters have to be inspectable, and consistent."""
    y = _series(_arma(70, seed=24, level=12.0))
    forecaster = GlobalARIMA(order=(2, 0, 1)).fit(y)
    params = forecaster.get_fitted_params()

    assert params["ar_params"].shape == (2,)
    assert params["ma_params"].shape == (1,)
    assert params["mean"] == pytest.approx(float(np.mean(y.to_numpy())))
    assert params["sigma2"] > 0
    assert params["uses_pretrained"] is False


def test_tags_declare_the_capabilities_that_are_implemented():
    """The pretrain tags are what the framework dispatches on."""
    tags = GlobalARIMA.get_class_tags()
    assert tags["capability:pretrain"] is True
    assert tags["capability:update"] is True
    assert tags["capability:multivariate"] is False
    assert set(tags["pretrain:fitted_params"]) == {
        "global_ar_params_",
        "global_ma_params_",
        "n_pretrain_instances_",
        "pretrain_loglike_",
        "pretrain_order_",
    }


# ----------------------------------------------------------------------
# full estimator contract
# ----------------------------------------------------------------------


def test_estimator_contract():
    """GlobalARIMA must satisfy sktime's estimator contract.

    The contract fits short random series, with few iterations in one of the
    test parameter sets, so the search can stop short of a maximum or end at
    the accuracy limit. The estimator reports that with its own fit quality
    warnings, which are expected here. Any other warning is still an error.
    """
    import warnings

    from sktime.utils.estimator_checks import check_estimator

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"GlobalARIMA\.\w+: the (likelihood search did not reach"
            r"|estimate sits at the limit)",
            category=UserWarning,
        )
        results = check_estimator(GlobalARIMA, raise_exceptions=False, verbose=False)
    failed = {k: v for k, v in results.items() if v != "PASSED"}
    assert not failed, f"{len(failed)} contract checks failed: {list(failed)[:5]}"
