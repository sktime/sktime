"""Tests for the Tabby forecaster adapter."""

import numpy as np
import pandas as pd
import pytest

from sktime.datasets import load_airline
from sktime.forecasting.tabby import TabbyForecaster
from sktime.tests.test_switch import run_test_for_class

pytestmark = pytest.mark.skipif(
    not run_test_for_class(TabbyForecaster),
    reason="run test only if torch and transformers are installed",
)

# Golden values generated with the raw upstream HF remote-code model, on CPU:
#
#   pip install sktime torch==2.15.0 transformers==5.22.1 einops numpy pandas
#
#   import numpy as np
#   import torch
#   from sktime.datasets import load_airline
#   from transformers import AutoModel
#
#   revision = "1b2381d7f1b0982aa1d5f09ce01b99eb4ee57e01"
#   model = AutoModel.from_pretrained(
#       "paris-noah/Tabby", revision=revision, trust_remote_code=True
#   ).eval()
#   history = load_airline().to_numpy(dtype=np.float32)[None, :]
#   with torch.no_grad():
#       output = model.predict(history, prediction_length=3).cpu().numpy()
#   print(output[0, 49, :])  # median, _GOLDEN_POINT
#   print(output[0, [9, 49, 89], :].T)  # alpha 0.1, 0.5, 0.9, _GOLDEN_QUANTILES
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
    return TabbyForecaster(revision=_TABBY_REVISION, license_accepted=True).fit(
        load_airline()
    )


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
