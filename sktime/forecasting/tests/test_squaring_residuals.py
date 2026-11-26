"""Tests of residual scale, variance and probabilistic forecast consistency."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.squaring_residuals import SquaringResiduals
from sktime.tests.test_switch import run_test_for_class

pytestmark = pytest.mark.skipif(
    not run_test_for_class(SquaringResiduals),
    reason="run only if softdeps are present and incrementally (if requested)",
)


def _fit_residuals(strategy, scale=1, period_index=False, distr="norm", kwargs=None):
    index = (
        pd.period_range("2020-01", periods=10, freq="M")
        if period_index
        else pd.RangeIndex(10)
    )
    y = pd.Series(2 * scale * np.arange(10, dtype=float), index=index, name="y")
    return SquaringResiduals(
        forecaster=NaiveForecaster(),
        residual_forecaster=NaiveForecaster(strategy="mean"),
        initial_window=2,
        strategy=strategy,
        distr=distr,
        distr_kwargs=kwargs,
    ).fit(y, fh=[1, 2])


@pytest.mark.parametrize("strategy", ["square", "abs"])
@pytest.mark.parametrize("scale", [1, 3, 10])
@pytest.mark.parametrize("period_index", [False, True])
def test_residual_variance_units(strategy, scale, period_index):
    """Normal variance is the square of residual scale and scales quadratically."""
    model = _fit_residuals(strategy, scale, period_index)
    actual = model.predict_var()
    expected = (2 * scale * np.array([1, 2], dtype=float)) ** 2
    assert isinstance(actual, pd.DataFrame)
    pd.testing.assert_index_equal(actual.index, model.predict().index)
    assert list(actual.columns) == ["y"]
    np.testing.assert_allclose(actual["y"], expected)
    alpha = [0.025, 0.5, 0.975]
    quantiles = model.predict_quantiles(alpha=alpha)
    expected_quantiles = model.predict().to_numpy()[:, None] + np.sqrt(expected)[
        :, None
    ] * stats.norm.ppf(alpha)
    np.testing.assert_allclose(quantiles, expected_quantiles)


@pytest.mark.parametrize("strategy", ["square", "abs"])
@pytest.mark.parametrize(
    "distr, kwargs, variance_factor",
    [
        ("norm", None, 1),
        ("norm", {"loc": 1.5, "scale": 3}, 9),
        ("laplace", None, 2),
        ("laplace", {"loc": -1, "scale": 2}, 8),
        ("t", {"df": 5}, 5 / 3),
        ("t", {"df": 5, "loc": 1, "scale": 2}, 20 / 3),
        ("t", {"df": 2}, np.inf),
        ("t", {"df": 1}, np.inf),
        ("cauchy", None, np.inf),
    ],
)
def test_residual_distribution_variance(strategy, distr, kwargs, variance_factor):
    """Variance accounts for the selected distribution's scale and tail weight."""
    model = _fit_residuals(strategy, distr=distr, kwargs=kwargs)
    np.testing.assert_allclose(
        model.predict_var()["y"], np.array([4, 16]) * variance_factor
    )


@pytest.mark.skipif(
    not _check_soft_dependencies("skpro", severity="none"),
    reason="distribution forecasts require skpro",
)
@pytest.mark.parametrize("strategy", ["square", "abs"])
@pytest.mark.parametrize("period_index", [False, True])
@pytest.mark.parametrize(
    "distr, kwargs",
    [
        ("norm", None),
        ("norm", {"loc": 1, "scale": 3}),
        ("laplace", None),
        ("t", {"df": 5}),
        ("t", {"df": 1}),
        ("cauchy", {"loc": -1, "scale": 2}),
    ],
)
def test_residual_distribution_consistency(strategy, period_index, distr, kwargs):
    """Variance, quantiles and intervals describe the same multi-horizon law."""
    model = _fit_residuals(
        strategy, period_index=period_index, distr=distr, kwargs=kwargs
    )
    alpha = [0.025, 0.5, 0.975]
    dist = model.predict_proba()
    pd.testing.assert_frame_equal(model.predict_var(), dist.var(), check_dtype=False)
    np.testing.assert_allclose(
        model.predict_quantiles(alpha=alpha), dist.quantile(alpha=alpha), rtol=1e-7
    )
    np.testing.assert_allclose(
        model.predict_interval(coverage=[0.95]),
        dist.quantile(alpha=[0.025, 0.975]),
        rtol=1e-7,
    )
