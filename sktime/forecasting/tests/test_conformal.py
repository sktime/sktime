"""Tests for the ConformalIntervals probability wrapper."""

import numpy as np
import pytest

from sktime.datasets import load_airline
from sktime.datatypes import check_is_mtype
from sktime.forecasting.conformal import ConformalIntervals, _conformal_quantile
from sktime.forecasting.naive import NaiveForecaster
from sktime.tests.test_switch import run_test_for_class

__author__ = ["fkiraly"]


@pytest.mark.skipif(
    not run_test_for_class(ConformalIntervals),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_conformal_standard():
    """Tests standard use of the conformal intervals adapter."""
    y = load_airline()
    forecaster = NaiveForecaster(strategy="drift")

    conformal_forecaster = ConformalIntervals(forecaster)
    conformal_forecaster.fit(y, fh=[1, 2, 3])
    pred_int = conformal_forecaster.predict_interval()

    assert check_is_mtype(pred_int, "pred_interval", "Proba", msg_return_dict="list")


@pytest.mark.skipif(
    not run_test_for_class(ConformalIntervals),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_conformal_with_gscv():
    """With ForecastingGridSearchCV and parameter plugin."""
    from sktime.forecasting.model_selection import ForecastingGridSearchCV
    from sktime.param_est.plugin import PluginParamsForecaster
    from sktime.split import ExpandingWindowSplitter

    y = load_airline()

    # part 1 = grid search
    cv = ExpandingWindowSplitter(fh=[1, 2, 3])
    forecaster = NaiveForecaster()
    param_grid = {"strategy": ["last", "mean", "drift"]}
    gscv = ForecastingGridSearchCV(
        forecaster=forecaster,
        param_grid=param_grid,
        cv=cv,
    )

    # part 2 = plug in results of grid search into conformal intervals estimator
    conformal_with_fallback = ConformalIntervals(NaiveForecaster())
    gscv_with_conformal = PluginParamsForecaster(
        gscv,
        conformal_with_fallback,
        params={"forecaster": "best_forecaster"},
    )

    gscv_with_conformal.fit(y, fh=[1, 2, 3])

    y_pred_quantiles = gscv_with_conformal.predict_quantiles()

    assert check_is_mtype(
        y_pred_quantiles, "pred_quantiles", "Proba", msg_return_dict="list"
    )


@pytest.mark.skipif(
    not run_test_for_class(ConformalIntervals),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_conformal_with_hierarchical():
    """Test ConformalIntervals with a wrapped estimator of hierarchical mtype.

    Failure case of bug #5092.
    """
    from sklearn.linear_model import LinearRegression

    from sktime.forecasting.compose import ForecastX, make_reduction
    from sktime.split import temporal_train_test_split
    from sktime.utils._testing.hierarchical import _make_hierarchical

    df = _make_hierarchical(
        hierarchy_levels=(2, 3),
        max_timepoints=25,
        min_timepoints=25,
        n_columns=4,
        all_positive=True,
        random_state=0,
    )

    y = df[["c0", "c1"]]
    X = df[["c2", "c3"]]

    y_train, y_test, X_train, X_test = temporal_train_test_split(y, X=X, test_size=5)

    regressor = LinearRegression()
    endogenous_model = make_reduction(regressor, pooling="global")

    exogenous_model = NaiveForecaster()

    forecaster = ConformalIntervals(
        ForecastX(endogenous_model.clone(), exogenous_model.clone()), initial_window=15
    )

    forecaster.fit(y_train, X=X_train, fh=range(1, 3))

    forecaster.predict(X=X_test)
    forecaster.predict_interval(X=X_test)


def test_conformal_quantile_picks_corrected_order_statistic():
    """_conformal_quantile returns the ceil((n+1)*(1-alpha))-th smallest score."""
    scores = np.arange(19, 0, -1, dtype=float)  # 19, 18, ..., 1, unsorted

    # n=19: rank = ceil(20 * 0.95) = 19, ceil(20 * 0.9) = 18, ceil(20 * 0.5) = 10
    quantile, too_few = _conformal_quantile(scores, np.array([0.05, 0.1, 0.5]))

    np.testing.assert_array_equal(quantile, [19, 18, 10])
    assert not too_few.any()

    # result is an element of the sample, not an interpolation
    quantile, _ = _conformal_quantile([1.0, 2.0, 4.0, 8.0], 0.4)
    assert quantile in [1.0, 2.0, 4.0, 8.0]
    assert quantile == 4.0  # rank = ceil(5 * 0.6) = 3


def test_conformal_quantile_too_few_scores():
    """If ceil((n+1)*(1-alpha)) > n, the largest score is used and flagged."""
    scores = np.arange(1, 9, dtype=float)  # n=8, coverage 0.9 needs n >= 9

    quantile, too_few = _conformal_quantile(scores, 1 - 0.9)

    assert quantile == 8
    assert too_few

    # exactly enough scores: n=9 at coverage 0.9 is valid, rank 9 = largest score
    # (guards against floating point error in (n+1)*(1-alpha))
    quantile, too_few = _conformal_quantile(np.arange(1, 10, dtype=float), 1 - 0.9)

    assert quantile == 9
    assert not too_few

    # coverage 1 can never be attained with finitely many scores
    _, too_few = _conformal_quantile(scores, 0.0)
    assert too_few


def test_conformal_quantile_rank_below_one_uses_smallest():
    """A rank below 1, e.g., at coverage 0, is raised to the smallest score."""
    quantile, too_few = _conformal_quantile(np.array([3.0, 1.0, 2.0]), 1.0)

    assert quantile == 1.0
    assert not too_few


@pytest.mark.skipif(
    not run_test_for_class(ConformalIntervals),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("method", ["conformal", "conformal_bonferroni"])
def test_conformal_uses_finite_sample_correction(method):
    """Half-widths are the ceil((m+1)*(1-alpha))-th smallest absolute residual."""
    y = load_airline()
    fh = [1, 2, 3]
    coverage = 0.9

    forecaster = ConformalIntervals(NaiveForecaster(strategy="drift"), method=method)
    forecaster.fit(y, fh=fh)
    y_pred = forecaster.predict()
    pred_int = forecaster.predict_interval(coverage=coverage)

    alpha = 1 - coverage
    if method == "conformal_bonferroni":
        alpha = alpha / len(fh)

    for h in fh:
        # same residuals as used by the estimator: diagonal at offset h
        # (this test does not concern which diagonal is the right one, see #5245)
        resids = np.diagonal(forecaster.residuals_matrix_, offset=h)
        abs_resids = np.sort(np.abs(resids[~np.isnan(resids)]))
        m = len(abs_resids)
        rank = int(np.ceil((m + 1) * (1 - alpha) - 1e-9))
        assert 1 <= rank <= m
        expected = abs_resids[rank - 1]

        idx = y_pred.index[h - 1]
        upper = pred_int.loc[idx, (y.name, coverage, "upper")]
        lower = pred_int.loc[idx, (y.name, coverage, "lower")]

        np.testing.assert_allclose(upper - y_pred.loc[idx], expected)
        np.testing.assert_allclose(y_pred.loc[idx] - lower, expected)


@pytest.mark.skipif(
    not run_test_for_class(ConformalIntervals),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_conformal_bonferroni_equals_conformal_at_split_alpha():
    """Bonferroni at coverage c with H horizons is conformal at 1-(1-c)/H."""
    y = load_airline()
    fh = [1, 2, 3]
    coverage = 0.9
    split_coverage = 1 - (1 - coverage) / len(fh)

    bonferroni = ConformalIntervals(
        NaiveForecaster(strategy="drift"), method="conformal_bonferroni"
    ).fit(y, fh=fh)
    conformal = ConformalIntervals(
        NaiveForecaster(strategy="drift"), method="conformal"
    ).fit(y, fh=fh)

    int_bonferroni = bonferroni.predict_interval(coverage=coverage)
    int_conformal = conformal.predict_interval(coverage=split_coverage)

    np.testing.assert_allclose(
        int_bonferroni.to_numpy(dtype=float), int_conformal.to_numpy(dtype=float)
    )


@pytest.mark.skipif(
    not run_test_for_class(ConformalIntervals),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_conformal_too_few_residuals_warns_and_uses_largest_residual():
    """With too few residuals for the coverage, warn and use the largest residual."""
    y = load_airline().iloc[:25]  # initial_window=10 -> few residuals per horizon
    forecaster = ConformalIntervals(
        NaiveForecaster(strategy="drift"), method="conformal"
    )
    forecaster.fit(y, fh=[1, 2])

    with pytest.warns(UserWarning, match="too few residuals"):
        pred_int = forecaster.predict_interval(coverage=0.99)

    y_pred = forecaster.predict()
    for h in [1, 2]:
        resids = np.diagonal(forecaster.residuals_matrix_, offset=h)
        largest = np.abs(resids[~np.isnan(resids)]).max()
        idx = y_pred.index[h - 1]
        upper = pred_int.loc[idx, (y.name, 0.99, "upper")]
        np.testing.assert_allclose(upper - y_pred.loc[idx], largest)
