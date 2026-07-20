"""Tests for LagLlamaForecaster probabilistic forecast consistency.

Regression tests for the probabilistic API being consistent, see issue #10566:
``predict_proba`` must be the single source of truth, with ``predict_quantiles``,
``predict_interval``, and ``predict_var`` derived from the same Monte Carlo
samples via an ``Empirical`` distribution.

Uses a deterministic stub predictor at the GluonTS ``Predictor`` interface, so
the tests exercise the full sktime and GluonTS data path without checkpoint
downloads, following the pattern of the TimesFM2 tests.
"""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.lagllama import LagLlamaForecaster
from sktime.utils.dependencies import _check_estimator_deps

NUM_SAMPLES = 10
FH = [1, 2, 3]

_softdep_skip = pytest.mark.skipif(
    not _check_estimator_deps(LagLlamaForecaster, severity="none"),
    reason="skip test if required soft dependencies not available",
)


class _StubLagLlamaPredictor:
    """Deterministic stand-in for the GluonTS LagLlama predictor.

    Emits ``SampleForecast`` objects with
    ``samples[i, h] = 1000 * k + 100 * (h + 1) + i`` for sample ``i``,
    0-based horizon step ``h``, and series number ``k``, so means and
    quantiles are closed-form: the point forecast at 1-based step ``h``
    of series ``k`` is ``1000 * k + 100 * h + (num_samples - 1) / 2``.
    """

    def __init__(self, prediction_length, num_samples):
        self.prediction_length = prediction_length
        self.lead_time = 0
        self.num_samples = num_samples

    def predict(self, dataset, num_samples=None):
        from gluonts.model.forecast import SampleForecast

        n = num_samples if num_samples is not None else self.num_samples
        steps = np.arange(1, self.prediction_length + 1, dtype=float)
        for k, entry in enumerate(dataset):
            start = entry["start"] + len(entry["target"])
            samples = (
                1000.0 * k + 100.0 * steps[None, :] + np.arange(n, dtype=float)[:, None]
            )
            yield SampleForecast(
                samples=samples,
                start_date=start,
                item_id=entry.get("item_id"),
            )


def _fit_stubbed(y, fh, monkeypatch):
    """Fit a LagLlamaForecaster with the stub predictor, no checkpoint load."""
    import sktime.forecasting.lagllama as lagllama_mod

    class _StubCache:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def load_predictor(self):
            return None, _StubLagLlamaPredictor(
                prediction_length=self.kwargs["prediction_length"],
                num_samples=self.kwargs["num_samples"],
            )

    monkeypatch.setattr(lagllama_mod, "_CachedLagLlama", _StubCache)
    monkeypatch.setattr(
        LagLlamaForecaster, "_ensure_checkpoint", lambda self: "stub.ckpt"
    )

    forecaster = LagLlamaForecaster(context_length=32, num_samples=NUM_SAMPLES)
    forecaster.fit(y, fh=fh)
    return forecaster


def _check_proba_consistency(forecaster):
    """Check predict_proba is Empirical and quantiles/intervals derive from it."""
    from skpro.distributions.empirical import Empirical

    alpha = [0.1, 0.5, 0.9]

    pred_dist = forecaster.predict_proba()
    assert isinstance(pred_dist, Empirical)

    # quantiles from predict_quantiles and from the distribution must agree,
    # see https://github.com/sktime/sktime/issues/10566
    q_direct = forecaster.predict_quantiles(alpha=alpha)
    q_from_proba = pred_dist.quantile(alpha=alpha)
    np.testing.assert_allclose(q_direct.to_numpy(), q_from_proba.to_numpy())

    # intervals must equal the corresponding quantiles
    pred_int = forecaster.predict_interval(coverage=0.8)
    np.testing.assert_allclose(
        pred_int.to_numpy(),
        q_direct.to_numpy()[:, [0, 2]],
    )

    # variance must derive from the same distribution
    pred_var = forecaster.predict_var()
    np.testing.assert_allclose(pred_var.to_numpy(), pred_dist.var().to_numpy())

    # point predictions and quantiles must share the same row index
    y_pred = forecaster.predict()
    assert q_direct.index.equals(pd.DataFrame(y_pred).index)
    assert len(pred_dist.index) == len(y_pred)

    # quantiles must be monotonically increasing in alpha
    q_np = q_direct.to_numpy()
    assert (np.diff(q_np, axis=1) >= 0).all()

    return y_pred, q_direct


@_softdep_skip
def test_lagllama_proba_consistency_series(monkeypatch):
    """Probabilistic forecasts are consistent for a single series."""
    y = pd.Series(
        np.sin(np.arange(48) / 4),
        index=pd.period_range("2020-01", periods=48, freq="M"),
        name="passengers",
    )
    forecaster = _fit_stubbed(y, FH, monkeypatch)
    y_pred, _ = _check_proba_consistency(forecaster)

    # closed-form stub means: 100 * step + (NUM_SAMPLES - 1) / 2
    expected = 100.0 * np.array(FH) + (NUM_SAMPLES - 1) / 2
    np.testing.assert_allclose(np.asarray(y_pred).ravel(), expected)


@_softdep_skip
def test_lagllama_proba_consistency_range_index(monkeypatch):
    """Probabilistic forecasts are consistent for a series with RangeIndex."""
    y = pd.DataFrame(
        {"value": np.sin(np.arange(48) / 4)},
        index=pd.RangeIndex(48),
    )
    forecaster = _fit_stubbed(y, FH, monkeypatch)
    y_pred, _ = _check_proba_consistency(forecaster)

    expected = 100.0 * np.array(FH) + (NUM_SAMPLES - 1) / 2
    np.testing.assert_allclose(np.asarray(y_pred).ravel(), expected)


@_softdep_skip
def test_lagllama_proba_consistency_panel(monkeypatch):
    """Probabilistic forecasts are consistent for panel data."""
    from sktime.utils._testing.hierarchical import _make_hierarchical

    y = _make_hierarchical(
        hierarchy_levels=(2,), min_timepoints=40, max_timepoints=40, random_state=0
    )
    forecaster = _fit_stubbed(y, FH, monkeypatch)
    _check_proba_consistency(forecaster)


@_softdep_skip
def test_lagllama_proba_consistency_hierarchical(monkeypatch):
    """Probabilistic forecasts are consistent for hierarchical (3-level) data."""
    from sktime.utils._testing.hierarchical import _make_hierarchical

    y = _make_hierarchical(
        hierarchy_levels=(2, 2), min_timepoints=40, max_timepoints=40, random_state=0
    )
    forecaster = _fit_stubbed(y, FH, monkeypatch)
    _check_proba_consistency(forecaster)


@_softdep_skip
def test_lagllama_proba_sparse_fh(monkeypatch):
    """Quantiles and proba subset correctly for a non-contiguous horizon."""
    y = pd.Series(
        np.sin(np.arange(48) / 4),
        index=pd.period_range("2020-01", periods=48, freq="M"),
        name="passengers",
    )
    forecaster = _fit_stubbed(y, [1, 3], monkeypatch)
    y_pred, q_direct = _check_proba_consistency(forecaster)

    expected = 100.0 * np.array([1, 3]) + (NUM_SAMPLES - 1) / 2
    np.testing.assert_allclose(np.asarray(y_pred).ravel(), expected)
    assert len(q_direct) == 2
