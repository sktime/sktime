"""Tests for the Tabby forecaster adapter."""

import numpy as np
import pandas as pd
import pytest

from sktime.datasets import load_airline
from sktime.forecasting.tabby import TabbyForecaster, _CachedTabby
from sktime.tests.test_switch import run_test_for_class

pytestmark = pytest.mark.skipif(
    not run_test_for_class(TabbyForecaster),
    reason="run test only if torch and transformers are installed",
)

_TABBY_REVISION = "1b2381d7f1b0982aa1d5f09ce01b99eb4ee57e01"
_GOLDEN_POINT = np.array([448.55990601, 442.19024658, 494.63415527])
_GOLDEN_QUANTILES = np.array(
    [
        [429.79977417, 448.55990601, 468.48461914],
        [417.43212891, 442.19024658, 465.74475098],
        [463.70422363, 494.63415527, 524.9329834],
    ]
)
_FH = [1, 2, 3]
_ALPHA = [0.1, 0.5, 0.9]


def _golden_forecaster():
    """Fit the real Tabby checkpoint used for golden outputs."""
    return TabbyForecaster(revision=_TABBY_REVISION).fit(load_airline())


def test_tabby_golden_outputs():
    """Real Tabby outputs remain stable for a fixed input and horizon."""
    forecaster = _golden_forecaster()

    point = forecaster.predict(fh=_FH)
    quantiles = forecaster.predict_quantiles(fh=_FH, alpha=_ALPHA)

    np.testing.assert_allclose(point.to_numpy(), _GOLDEN_POINT, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(
        quantiles.to_numpy(), _GOLDEN_QUANTILES, rtol=1e-5, atol=1e-6
    )


def test_tabby_predict_proba_matches_quantiles():
    """Tabby's HistogramQPD preserves index, columns, and quantile interpolation."""
    forecaster = _golden_forecaster()
    distribution = forecaster.predict_proba(fh=_FH)

    quantiles = forecaster.predict_quantiles(fh=_FH, alpha=_ALPHA)
    assert distribution.index.equals(quantiles.index)
    var_name = load_airline().name
    assert list(distribution.columns) == [var_name]
    for alpha in _ALPHA:
        probabilities = pd.DataFrame(
            alpha, index=distribution.index, columns=[var_name]
        )
        ppf = distribution.ppf(probabilities)
        expected = quantiles[(var_name, alpha)].to_numpy().reshape(-1, 1)
        np.testing.assert_allclose(ppf.to_numpy(), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "loading_info",
    [
        {"missing_keys": ["backbone.blocks.0.weight"], "unexpected_keys": []},
        {"missing_keys": [], "unexpected_keys": ["ctx_generator.mlp.0.weight"]},
    ],
)
def test_tabby_rejects_incomplete_checkpoint(monkeypatch, loading_info):
    """A checkpoint with missing or unexpected weights must not be used."""
    import sktime.libs.tabby as tabby_lib

    class FakeModel:
        def to(self, device):
            return self

        def eval(self):
            return self

    def fake_from_pretrained(*args, **kwargs):
        return FakeModel(), loading_info

    monkeypatch.setattr(
        tabby_lib.TabbyForForecasting, "from_pretrained", fake_from_pretrained
    )
    monkeypatch.setattr(
        tabby_lib.TabbyConfig,
        "from_pretrained",
        classmethod(lambda cls, *args, **kwargs: cls()),
    )

    loader = _CachedTabby(
        key=("incomplete-checkpoint", repr(loading_info)),
        model_path="test",
        revision=None,
        device="cpu",
        dtype=None,
    )
    with pytest.raises(RuntimeError, match="did not load exactly"):
        loader.load()
