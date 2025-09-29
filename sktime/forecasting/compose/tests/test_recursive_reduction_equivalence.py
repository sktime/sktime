"""Equivalence tests for fast vs legacy recursive reduction paths.

These tests simulate time series data and verify that the optimized
``_predict_out_of_sample`` logic (fast local path + tail optimization) produces
identical numerical results to the legacy v1 implementation under its guard
conditions.

Scope (initial):
  * Local pooling, single-index, no exogenous data (fast path active)
  * Contiguous and gappy forecasting horizons
  * Constant-mean fallback scenario (insufficient lag rows)

Out-of-scope (future potential additions):
  * Global/panel pooling optimized path (currently not invoked by dispatcher)
  * Exogenous X parity (guard forces fallback to v1 so trivial)
  * MultiIndex scenarios (fast path disabled)

If these tests fail, it indicates a numerical deviation between optimized and
baseline logic; fast path must maintain exact parity for correctness.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose._reduce import RecursiveReductionForecaster


def eprint(*args, **kwargs):
    print(*args, flush=True, **kwargs)


# Verbose wrapper for LinearRegression
class VerboseLinearRegression(LinearRegression):
    def fit(self, X, y, sample_weight=None):
        eprint(
            "[VerboseLinearRegression.fit] X shape=",
            getattr(X, "shape", None),
            " y shape=",
            getattr(y, "shape", None),
        )
        try:
            eprint("X:\n", X)
        except Exception:
            raise ValueError("Failed to print X")
        try:
            eprint("y:\n", y)
        except Exception:
            raise ValueError("Failed to print y")
        return super().fit(X, y, sample_weight=sample_weight)

    def predict(self, X):
        eprint("[VerboseLinearRegression.predict] X shape=", getattr(X, "shape", None))
        try:
            eprint("X:\n", X)
        except Exception:
            raise ValueError("Failed to print X")

        preds = super().predict(X)
        eprint("Pred shape=", getattr(preds, "shape", None))
        eprint(f"preds = {preds}")
        return preds


RANDOM_SEED = 42


def _make_series(n=40, cols=1, freq="D", seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    idx = pd.period_range(start="2000-01-01", periods=n, freq=freq)
    data = {}
    for c in range(cols):
        # smooth-ish signal + noise
        trend = np.linspace(0.0, 1.0, n)
        noise = rng.normal(0, 0.1, n)
        data[f"y{c}"] = trend + noise
    return pd.DataFrame(data, index=idx)


def _fit_local_model(y, window_length=4):
    forecaster = RecursiveReductionForecaster(
        estimator=VerboseLinearRegression(),
        window_length=window_length,
        pooling="local",
        impute_method=None,  # keep simple to avoid side effects
    )
    forecaster.fit(y=y, fh=ForecastingHorizon([1]))  # minimal fh just to satisfy API
    return forecaster


def _compare_fast_vs_v1(y, fh_list):
    """Fit model and compare public predict (fast path) with legacy v1 output.

    Parameters
    ----------
    y : pd.DataFrame
        Training series (single-index, local pooling scenario)
    fh_list : list[int]
        Forecasting horizon steps (relative, positive)
    """
    print("_compare_fast_vs_v1: entered")
    pd.set_option("display.max_rows", None)
    print(f"y = {y}")

    fh = ForecastingHorizon(fh_list, is_relative=True, freq=y.index)
    f = _fit_local_model(y)

    # public prediction (dispatcher chooses path)
    y_pred_public = f.predict(fh=fh)

    # direct legacy path call (no exogenous passed) .. always returns only requested fh
    y_pred_v1 = f._predict_out_of_sample_v1(X_pool=None, _fh=fh)

    # shape & index equality
    assert list(y_pred_public.index) == list(y_pred_v1.index)
    # column equality
    assert list(y_pred_public.columns) == list(y_pred_v1.columns)
    # numerical closeness
    np.testing.assert_allclose(
        y_pred_public.to_numpy(), y_pred_v1.to_numpy(), rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "fh_list",
    [
        [1, 2, 3, 4, 5],  # contiguous
        [1, 2, 4, 7, 10],  # gappy
        [3],  # single step beyond window
    ],
)
def test_recursive_reduction_local_fast_vs_v1_equivalence(fh_list):
    y = _make_series(n=50)
    _compare_fast_vs_v1(y, fh_list)


def test_recursive_reduction_local_constant_mean_fallback():
    # window longer than series -> estimator_ becomes Series (mean fallback)
    y = _make_series(n=3)
    window_len = 10
    f = _fit_local_model(y, window_length=window_len)
    fh = ForecastingHorizon([1, 2, 5], is_relative=True, freq=y.index)

    y_pred_public = f.predict(fh=fh)
    # simulate legacy constant-mean production manually
    # (v1 path invoked internally already; still check stable behaviour)
    mean_vals = y.mean()
    expected = pd.DataFrame(
        np.repeat(mean_vals.values.reshape(1, -1), len(fh), axis=0),
        index=fh.to_absolute(f.cutoff).to_pandas(),
        columns=y.columns,
    )
    np.testing.assert_allclose(
        y_pred_public.to_numpy(), expected.to_numpy(), rtol=0, atol=1e-12
    )


# def test_recursive_reduction_local_tail_optimization_parity():
#     """Explicitly exercise tail optimization path vs legacy.

#     Tail optimization (_predict_out_of_sample_v1_fasttail) should be numerically
#     identical to full v1; we indirectly verify by temporarily disabling fast
#     tail and comparing.
#     """
#     y = _make_series(n=60)
#     fh = ForecastingHorizon([1, 2, 3, 6, 9], is_relative=True, freq=y.index)
#     f = _fit_local_model(y, window_length=6)

#     # public path (may use tail optimization) .. baseline
#     pred_public = f.predict(fh=fh)

#     # Force disable tail optimization by monkeypatching helper to always return None
#     # (so dispatcher falls back to full v1). This keeps state identical.
#     original_fasttail = f._predict_out_of_sample_v1_fasttail

#     def _no_fasttail(X_pool, fh):  # noqa: D401
#         return None

#     f._predict_out_of_sample_v1_fasttail = _no_fasttail  # type: ignore
#     try:
#         pred_legacy = f.predict(fh=fh)
#     finally:
#         f._predict_out_of_sample_v1_fasttail = original_fasttail  # restore

#     np.testing.assert_allclose(
#         pred_public.to_numpy(), pred_legacy.to_numpy(), rtol=0, atol=0
#     )


@pytest.mark.skip(reason="Global/panel optimized path not yet dispatched; add later")
def test_recursive_reduction_global_v2_vs_v1_future():  # pragma: no cover
    pass
